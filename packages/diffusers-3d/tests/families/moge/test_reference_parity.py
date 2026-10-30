from __future__ import annotations

import copy
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)

from diffusers_3d.families.moge import (
    MOGE2_REFERENCE_REVISION,
    MOGE2_WEIGHTS_REPOSITORY,
    MOGE2_WEIGHTS_REVISION,
    MOGE_REFERENCE_REVISION,
    MOGE_WEIGHTS_REPOSITORY,
    MOGE_WEIGHTS_REVISION,
    convert_moge_checkpoint,
)
from diffusers_3d.families.moge.conversion import convert_dinov2_state_dict
from diffusers_3d.families.moge.geometry import recover_focal_shift, solve_shift
from diffusers_3d.families.moge.models import _dinov2_backbone, _intermediate_layers, match_dinov2_attention_names

from .configs import TINY_BACKBONE

pytestmark = pytest.mark.reference_parity

MOGE_REPOSITORY = "https://github.com/microsoft/MoGe.git"
# SAM 3D Objects pins MoGe a8c3734 (v1 only); Pixal3D installs unpinned MoGe, HEAD 74fbce0 at its revision.
CHECKOUTS = {
    "v1": {
        "root": Path(os.environ.get("DIFFUSERS_3D_MOGE_REFERENCE_ROOT", "/tmp/MoGe-a8c3734")),
        "revision": MOGE_REFERENCE_REVISION,
        "paths": ("moge/model/v1.py", "moge/utils/geometry_torch.py", "moge/utils/geometry_numpy.py"),
        "vit": "model.dinov2.models.vision_transformer",
        "utils3d": "utils3d",
    },
    "v2": {
        "root": Path(os.environ.get("DIFFUSERS_3D_MOGE2_REFERENCE_ROOT", "/tmp/MoGe")),
        "revision": MOGE2_REFERENCE_REVISION,
        "paths": ("moge/model/v2.py", "moge/utils/geometry_torch.py", "moge/utils/geometry_numpy.py"),
        "vit": "model.modules.dinov2.models.vision_transformer",
        "utils3d": "utils3d_moge",
    },
}
# MoGe a8c3734 requires utils3d 3913c65; MoGe 74fbce0 requires utils3d-moge 62f09d5 (and falls back to plain utils3d
# when that is missing, which the module check below rejects).
UTILS3D = {
    "utils3d": (
        Path(os.environ.get("DIFFUSERS_3D_UTILS3D_REFERENCE_ROOT", "/tmp/utils3d-3913c65")),
        "3913c65d81e05e47b9f367250cf8c0f7462a0900",
        "https://github.com/EasternJournalist/utils3d.git",
    ),
    "utils3d_moge": (
        Path(os.environ.get("DIFFUSERS_3D_UTILS3D_MOGE_REFERENCE_ROOT", "/tmp/utils3d-moge-62f09d5")),
        "62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
        "https://github.com/EasternJournalist/utils3d-moge.git",
    ),
}
_REFERENCES: dict[str, types.SimpleNamespace] = {}

# Small upstream ``model_config`` dicts on the real ViT-S/14 loader (native 37x37 position grid).
UPSTREAM_CONFIGS = {
    "v1": {
        "encoder": "dinov2_vits14",
        "remap_output": "exp",
        "output_mask": True,
        "split_head": True,
        "intermediate_layers": 4,
        "dim_proj": 32,
        "dim_upsample": [32, 16, 16],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 2,
        "trained_area_range": [40 * 196, 80 * 196],
        "last_conv_channels": 8,
        "last_conv_size": 1,
    },
    "v2": {
        "encoder": {"backbone": "dinov2_vits14", "intermediate_layers": [2, 5, 8, 11], "dim_out": 64},
        "neck": {
            "dim_in": [66, 2, 2, 2, 2],
            "dim_out": None,
            "dim_res_blocks": [64, 32, 16, 16, 8],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        },
        "points_head": {
            "dim_in": [64, 32, 16, 16, 8],
            "dim_out": [None, None, None, None, 3],
            "dim_res_blocks": [64, 32, 16, 16, 8],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        },
        "scale_head": {"dims": [384, 32, 1]},
        "remap_output": "exp",
        "num_tokens_range": [40, 80],
    },
}
UPSTREAM_CONFIGS["v2"]["mask_head"] = {**UPSTREAM_CONFIGS["v2"]["points_head"], "dim_out": [None] * 4 + [1]}


def _unavailable(error: ReferenceCheckoutError) -> None:
    try:
        reason = reference_unavailable(error)
    except ReferenceCheckoutError as required_error:
        pytest.fail(str(required_error), pytrace=False)
    pytest.skip(reason)


def _reference(version: str) -> types.SimpleNamespace:
    """Import a pinned MoGe checkout as a private package, so the two revisions never share ``moge``."""

    if version in _REFERENCES:
        return _REFERENCES[version]
    checkout = CHECKOUTS[version]
    utils3d_root, utils3d_revision, utils3d_repository = UTILS3D[checkout["utils3d"]]
    package = f"_diffusers_3d_moge_{checkout['revision'][:7]}"
    try:
        validate_reference_checkout(
            checkout["root"],
            expected_revision=checkout["revision"],
            expected_repository=MOGE_REPOSITORY,
            expected_paths=checkout["paths"],
        )
        validate_reference_checkout(
            utils3d_root,
            expected_revision=utils3d_revision,
            expected_repository=utils3d_repository,
            expected_paths=(f"{checkout['utils3d']}/__init__.py",),
        )
        for dependency in ("cv2", "scipy.optimize", "scipy.signal"):
            import_reference_dependency(dependency)
        if str(utils3d_root) not in sys.path:
            sys.path.insert(0, str(utils3d_root))
        utils3d = import_reference_dependency(checkout["utils3d"])
        if not Path(utils3d.__file__).resolve().is_relative_to(utils3d_root.resolve()):
            raise ReferenceCheckoutError(f"{checkout['utils3d']} imports from {utils3d.__file__}, not {utils3d_root}")
        spec = importlib.util.spec_from_file_location(
            package,
            checkout["root"] / "moge" / "__init__.py",
            submodule_search_locations=[str(checkout["root"] / "moge")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
        reference = types.SimpleNamespace(
            model=importlib.import_module(f"{package}.model.{version}"),
            utils=importlib.import_module(f"{package}.model.utils"),
            vit=importlib.import_module(f"{package}.{checkout['vit']}"),
            geometry=importlib.import_module(f"{package}.utils.geometry_torch"),
            geometry_numpy=importlib.import_module(f"{package}.utils.geometry_numpy"),
        )
    except ReferenceCheckoutError as error:
        _unavailable(error)
    except ImportError as error:
        _unavailable(ReferenceCheckoutError(f"MoGe {version} reference is not importable: {error}"))
    _REFERENCES[version] = reference
    return reference


def _perturb(module: torch.nn.Module, seed: int, std: float = 0.02) -> torch.nn.Module:
    """Upstream initialization plus noise, so LayerScale, norms and biases are not at their identity values."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * std)
    return module.eval()


def _models(version: str, tmp_path: Path):
    """A small random upstream model, and ours converted from its saved ``model.pt``."""

    reference = _reference(version)
    torch.manual_seed(0)
    config = copy.deepcopy(UPSTREAM_CONFIGS[version])
    upstream = _perturb(reference.model.MoGeModel(**config), seed=1)
    # Centre the mask logits on the threshold so about half of each image is masked.
    with torch.no_grad():
        mask = upstream.forward(_image()[None], num_tokens=60)["mask"]
        if version == "v1":
            upstream.head.output_block[1][-1].bias += 0.5 - mask.median()
        else:
            upstream.mask_head.output_blocks[-1].bias -= torch.logit(mask.median().double()).float()
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model": upstream.state_dict()}, path)
    return upstream, convert_moge_checkpoint(path)


def _image(height: int = 50, width: int = 70) -> torch.Tensor:
    return torch.rand(3, height, width, generator=torch.Generator().manual_seed(2))


def _assert_outputs_match(ours: dict, upstream: dict, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    assert set(ours) == set(upstream)
    for key, value in upstream.items():
        if value.dtype == torch.bool:
            assert torch.equal(ours[key], value), key
        else:
            torch.testing.assert_close(ours[key], value, rtol=rtol, atol=atol, msg=key)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_dinov2_backbone_matches_vendored_torch_hub_off_grid(version):
    """MoGe's grids are non-square and non-native, where Transformers' own interpolation differs."""

    reference = _reference(version)
    torch.manual_seed(0)
    vit = reference.vit.DinoVisionTransformer(
        img_size=TINY_BACKBONE["image_size"],
        patch_size=14,
        embed_dim=TINY_BACKBONE["hidden_size"],
        depth=TINY_BACKBONE["num_hidden_layers"],
        num_heads=TINY_BACKBONE["num_attention_heads"],
        init_values=1.0,
        block_chunks=0,
    )
    if version == "v1":  # a8c3734's MoGeModel swaps every block's attention for SDPA.
        for block in vit.blocks:
            block.attn = reference.utils.wrap_dinov2_attention_with_sdpa(block.attn)
    vit = _perturb(vit, seed=3)
    ours = _dinov2_backbone(TINY_BACKBONE).eval()
    state_dict = convert_dinov2_state_dict(vit.state_dict())
    match_dinov2_attention_names(state_dict, ours.state_dict().keys())
    ours.load_state_dict(state_dict)
    stock = copy.deepcopy(ours)
    stock.embeddings.__class__ = type(stock.embeddings).__mro__[1]  # transformers' Dinov2Embeddings

    layers = [1, 3]
    pixels = torch.randn(2, 3, 6 * 14, 9 * 14, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = vit.get_intermediate_layers(pixels, n=layers, return_class_token=True)
        tokens = _intermediate_layers(ours, pixels, layers)
        stock_tokens = _intermediate_layers(stock, pixels, layers)
    for (patch_tokens, class_token), layer_tokens in zip(expected, tokens):
        torch.testing.assert_close(layer_tokens[:, 1:], patch_tokens, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(layer_tokens[:, 0], class_token, rtol=1e-5, atol=1e-6)
    assert (stock_tokens[-1][:, 1:] - expected[-1][0]).abs().max() > 1e-3


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_forward_matches_upstream(version, tmp_path):
    upstream, ours = _models(version, tmp_path)
    image = _image()[None]
    with torch.no_grad():
        expected = upstream.forward(image, num_tokens=60)
        output = ours.forward(image, num_tokens=60)
    _assert_outputs_match(output, expected)


CASES = {
    "v1": [{}, {"force_projection": False}, {"fov_x": 50.0}, {"apply_mask": False, "resolution_level": 3}],
    "v2": [{}, {"force_projection": False, "apply_mask": False}, {"fov_x": 50.0}, {"num_tokens": 50}],
}


@pytest.mark.parametrize(("version", "kwargs"), [(version, kwargs) for version in CASES for kwargs in CASES[version]])
def test_infer_matches_upstream(version, kwargs, tmp_path):
    """Float32 on CPU on both sides; ``{"force_projection": False}`` is SAM 3D Objects' call, ``{}`` Pixal3D's."""

    upstream, ours = _models(version, tmp_path)
    for image in (_image(), _image(64, 48)[None].expand(2, -1, -1, -1)):
        expected = upstream.infer(image, use_fp16=False, **kwargs)
        assert 0 < expected["mask"].float().mean() < 1  # the focal and shift solve runs on a partial mask
        _assert_outputs_match(ours.infer(image, use_fp16=False, **kwargs), expected)


def test_half_precision_default_matches_upstream_on_cpu(tmp_path):
    """Upstream's ``use_fp16=True`` autocasts on every device. SAM 3D Objects' MoGe call runs in float16 on CPU too."""

    upstream, ours = _models("v1", tmp_path)
    _assert_outputs_match(
        ours.infer(_image(), force_projection=False), upstream.infer(_image(), force_projection=False)
    )


@pytest.mark.parametrize(("version", "kwargs"), [("v1", {}), ("v2", {})])
def test_half_precision_defaults_fail_on_cpu_like_upstream(version, kwargs, tmp_path):
    """Upstream defaults mix float16 activations with float32 weights or intrinsics on CPU, where autocast cannot run
    float32; ours raises the same error (on CUDA, autocast accepts float32 and both run)."""

    upstream, ours = _models(version, tmp_path)
    for model in (upstream, ours):
        with pytest.raises(RuntimeError, match="Half"):
            model.infer(_image(), **kwargs)


def _shift_problems(count: int = 240):
    """Point maps of a random scene seen through a known focal and shift, with increasing prediction noise."""

    generator = np.random.default_rng(0)
    for index in range(count):
        size = int(generator.integers(2, 400))
        uv = generator.uniform(-0.8, 0.8, (size, 2)).astype(np.float32)
        depth = generator.uniform(0.3, 5.0, size)
        noise = (0.0, 0.01, 0.5)[index % 3]
        xy = uv * depth[:, None] / generator.uniform(0.5, 3.0) + generator.normal(0, noise, (size, 2))
        shift = generator.uniform(-2.0, 2.0)
        focal = None if index % 4 else np.float32(generator.uniform(0.5, 3.0))
        yield uv, np.concatenate([xy, (depth - shift)[:, None]], axis=1).astype(np.float32), focal


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_shift_solver_matches_scipy_least_squares(version):
    """Upstream solves with ``scipy.optimize.least_squares(method="lm", ftol=1e-3)``.

    SciPy's compiled MINPACK fuses multiply-adds on some platforms (clang on arm64 contracts ``r += a * b`` into one
    FMA), so its last float64 bits depend on the build; ours rounds every operation. The float32 shifts upstream keeps
    almost always agree exactly.
    """

    reference = _reference(version)
    identical, worst = 0, 0.0
    problems = list(_shift_problems())
    for uv, xyz, focal in problems:
        if focal is None:
            expected, _ = reference.geometry_numpy.solve_optimal_focal_shift(uv, xyz)
        else:
            expected = reference.geometry_numpy.solve_optimal_shift(uv, xyz, focal)
        shift, _, _ = solve_shift(uv, xyz[:, :2], xyz[:, 2], None if focal is None else float(focal))
        identical += np.float32(shift) == expected
        worst = max(worst, abs(float(np.float32(shift)) - float(expected)) / max(abs(float(expected)), 1e-6))
    assert identical >= 0.95 * len(problems)
    assert worst < 1e-5


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_recover_focal_shift_matches_upstream(version):
    reference = _reference(version)
    generator = torch.Generator().manual_seed(5)
    uv = reference.geometry.normalized_view_plane_uv(80, 60, dtype=torch.float32)
    depth = 2.0 + 0.5 * uv[..., 0] + 0.3 * torch.rand(60, 80, generator=generator)
    points = torch.cat([uv * depth[..., None] / 1.2, (depth - 0.6)[..., None]], dim=-1)
    points = torch.stack([points, points * 1.5 + 0.01 * torch.randn(points.shape, generator=generator)])
    mask = torch.rand(2, 60, 80, generator=generator) > 0.2
    mask[1] = False
    mask[1, :1, :1] = True
    for focal in (None, torch.tensor([1.1, 0.9])):
        expected = reference.geometry.recover_focal_shift(points, mask, focal=focal)
        output = recover_focal_shift(points, mask, focal=focal)
        for value, reference_value in zip(output, expected):
            assert torch.equal(value, reference_value)


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("DIFFUSERS_3D_MOGE_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_MOGE_REAL_WEIGHTS=1"
)
@pytest.mark.parametrize(
    ("version", "repository", "revision", "kwargs"),
    [
        ("v1", MOGE_WEIGHTS_REPOSITORY, MOGE_WEIGHTS_REVISION, {"force_projection": False}),
        ("v1", MOGE_WEIGHTS_REPOSITORY, MOGE_WEIGHTS_REVISION, {}),
        ("v2", MOGE2_WEIGHTS_REPOSITORY, MOGE2_WEIGHTS_REVISION, {}),
    ],
)
def test_released_weights_match_upstream(version, repository, revision, kwargs):
    """The released ViT-L checkpoints on one 320x240 photo, float32 on CPU (about 10 s per model)."""

    from huggingface_hub import hf_hub_download
    from PIL import Image

    reference = _reference(version)
    path = hf_hub_download(repository, "model.pt", revision=revision)
    upstream = reference.model.MoGeModel.from_pretrained(path).eval()
    ours = convert_moge_checkpoint(path)
    photo = CHECKOUTS["v2"]["root"] / "example_images" / "01_HouseIndoor.jpg"
    if not photo.is_file():
        _unavailable(ReferenceCheckoutError(f"example image {photo} is missing"))
    image = torch.from_numpy(np.asarray(Image.open(photo).convert("RGB").resize((320, 240))).copy())
    image = image.permute(2, 0, 1).float() / 255
    _assert_outputs_match(ours.infer(image, use_fp16=False, **kwargs), upstream.infer(image, use_fp16=False, **kwargs))
