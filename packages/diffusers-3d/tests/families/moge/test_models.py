from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from diffusers_3d.families.moge import MoGe2Model, MoGeModel
from diffusers_3d.families.moge.geometry import (
    depth_to_points,
    focal_to_intrinsics,
    normalized_view_plane_uv,
    recover_focal_shift,
    solve_shift,
)
from diffusers_3d.families.moge.models import MoGeDinov2Embeddings

from .configs import TINY_BACKBONE, tiny_moge2_config, tiny_moge_config

VARIANTS = {"moge": (MoGeModel, tiny_moge_config), "moge2": (MoGe2Model, tiny_moge2_config)}


def _model(name: str) -> MoGeModel | MoGe2Model:
    model_type, config = VARIANTS[name]
    torch.manual_seed(0)
    return model_type(**config()).eval()


def _image(height: int = 30, width: int = 44) -> torch.Tensor:
    return torch.rand(3, height, width, generator=torch.Generator().manual_seed(1))


def _perspective_points(focal: float, shift: float, height: int = 48, width: int = 64) -> torch.Tensor:
    """An affine point map: a tilted plane seen through ``focal`` (relative to the half diagonal), z shifted by -shift."""

    uv = normalized_view_plane_uv(width, height, dtype=torch.float64)
    depth = 2.0 + 0.5 * uv[..., 0] - 0.3 * uv[..., 1]
    return torch.cat([uv * depth[..., None] / focal, (depth - shift)[..., None]], dim=-1).float()


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_infer_returns_the_upstream_keys_and_shapes(name):
    model = _model(name)
    output = model.infer(_image(), use_fp16=False)
    expected = {"points", "intrinsics", "depth", "mask"} | ({"mask_prob"} if name == "moge" else set())
    assert set(output) == expected
    assert output["points"].shape == (30, 44, 3)
    assert output["depth"].shape == output["mask"].shape == (30, 44)
    assert output["intrinsics"].shape == (3, 3)
    assert output["mask"].dtype == torch.bool
    torch.testing.assert_close(output["intrinsics"][:, 2], torch.tensor([0.5, 0.5, 1.0]))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_infer_batches_like_single_images(name):
    model = _model(name)
    images = torch.stack([_image(), _image().flip(-1)])
    batched = model.infer(images, use_fp16=False, apply_mask=False)
    single = model.infer(images[1], use_fp16=False, apply_mask=False)
    for key, value in single.items():
        torch.testing.assert_close(batched[key][1], value, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_masking_and_projection(name):
    model = _model(name)
    masked = model.infer(_image(), use_fp16=False)
    assert torch.isinf(masked["depth"][~masked["mask"]]).all()
    assert torch.isinf(masked["points"][~masked["mask"]]).all()

    projected = model.infer(_image(), use_fp16=False, apply_mask=False)
    points, intrinsics = projected["points"], projected["intrinsics"]
    torch.testing.assert_close(points[..., 2], projected["depth"])
    # ``force_projection`` makes the points the unprojected depth: x / z = (u - cx) / fx at pixel centres.
    u = (torch.arange(44) + 0.5) / 44
    torch.testing.assert_close(points[..., 0] / points[..., 2], ((u - 0.5) / intrinsics[0, 0]).expand(30, 44))

    shifted = model.infer(_image(), use_fp16=False, apply_mask=False, force_projection=False)
    torch.testing.assert_close(shifted["depth"], projected["depth"])
    torch.testing.assert_close(shifted["points"][..., 2], projected["points"][..., 2])


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_known_fov_sets_the_intrinsics(name):
    output = _model(name).infer(_image(), use_fp16=False, fov_x=55.0)
    fov_x = 2 * math.atan(1 / (2 * output["intrinsics"][0, 0].item()))
    assert math.degrees(fov_x) == pytest.approx(55.0, abs=1e-4)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_save_and_load_round_trip(name, tmp_path):
    model = _model(name)
    model.save_pretrained(tmp_path)
    loaded = type(model).from_pretrained(tmp_path).eval()
    first, second = model.infer(_image(), use_fp16=False), loaded.infer(_image(), use_fp16=False)
    for key, value in first.items():
        torch.testing.assert_close(second[key], value, rtol=0, atol=0)


def test_position_interpolation_differs_from_transformers_off_grid():
    config = Dinov2Config(**TINY_BACKBONE)
    torch.manual_seed(0)
    stock = Dinov2Model(config).embeddings
    ours = MoGeDinov2Embeddings(config)
    ours.load_state_dict(stock.state_dict())
    native = torch.zeros(1, 65, 32)
    torch.testing.assert_close(
        ours.interpolate_pos_encoding(native, 112, 112), stock.interpolate_pos_encoding(native, 112, 112)
    )
    tokens = torch.zeros(1, 1 + 6 * 9, 32)
    difference = (
        ours.interpolate_pos_encoding(tokens, 84, 126) - stock.interpolate_pos_encoding(tokens, 84, 126)
    ).abs()
    assert difference.max() > 1e-3


def test_recover_focal_shift_inverts_a_perspective_point_map():
    points = _perspective_points(focal=1.3, shift=0.4)
    mask = torch.ones(points.shape[:-1], dtype=torch.bool)
    focal, shift = recover_focal_shift(points, mask)
    assert focal.item() == pytest.approx(1.3, rel=1e-5)
    assert shift.item() == pytest.approx(0.4, rel=1e-5)
    _, shift = recover_focal_shift(points[None], mask[None], focal=torch.tensor([1.3]))
    assert shift.shape == (1,)
    assert shift.item() == pytest.approx(0.4, rel=1e-5)


def test_recover_focal_shift_falls_back_without_enough_points():
    points = _perspective_points(focal=1.3, shift=0.4)
    focal, shift = recover_focal_shift(points, torch.zeros(points.shape[:-1], dtype=torch.bool))
    assert (focal.item(), shift.item()) == (1.0, 0.0)


def test_solve_shift_reports_minpack_status():
    generator = np.random.default_rng(0)
    uv = generator.uniform(-0.8, 0.8, (200, 2)).astype(np.float32)
    z = generator.uniform(1.5, 4.0, 200)
    xyz = np.concatenate([uv * z[:, None] / 1.5, (z - 0.7)[:, None]], axis=1).astype(np.float32)
    shift, nfev, status = solve_shift(uv, xyz[:, :2], xyz[:, 2])
    assert shift == pytest.approx(0.7, rel=1e-5)
    assert 1 <= status <= 4 and nfev < 100


def test_intrinsics_and_unprojection_conventions():
    intrinsics = focal_to_intrinsics(torch.tensor([2.0]), aspect_ratio=2.0)
    # fx is normalized by the width and fy by the height, so fx * width == fy * height.
    assert intrinsics[0, 0, 0].item() * 2 == pytest.approx(intrinsics[0, 1, 1].item())
    depth = torch.full((1, 4, 8), 3.0)
    for homogeneous in (False, True):
        points = depth_to_points(depth, intrinsics, homogeneous=homogeneous)
        assert points.shape == (1, 4, 8, 3)
        torch.testing.assert_close(points[..., 2], depth)
        assert (points[0, :, :4, 0] < 0).all() and (points[0, :2, :, 1] < 0).all()


@pytest.mark.parametrize("naming", ["legacy", "current"])
def test_folders_load_with_either_dinov2_attention_naming(tmp_path, naming):
    from safetensors.torch import load_file, save_file

    from diffusers_3d.families.moge.models import _DINOV2_ATTENTION_NAMES

    model = MoGeModel(**tiny_moge_config()).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    state_dict = load_file(path)
    renamed = {}
    for key, value in state_dict.items():
        for legacy, current in _DINOV2_ATTENTION_NAMES:
            key = key.replace(current, legacy) if naming == "legacy" else key.replace(legacy, current)
        renamed[key] = value
    save_file(renamed, path, metadata={"format": "pt"})
    loaded = MoGeModel.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
