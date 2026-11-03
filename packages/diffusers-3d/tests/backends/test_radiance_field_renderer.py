from __future__ import annotations

import ast
import importlib
import importlib.util
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from diffusers_3d import CameraRig, CoordinateSystem, RadianceFieldAsset
from diffusers_3d.backends import radiance_field
from diffusers_3d.backends.radiance_field import SH_C0, render_radiance_field, sample_radiance_field
from diffusers_3d.families.trellis.conversion import TRELLIS_REFERENCE_REVISION
from diffusers_3d.families.trellis.decoders import TrellisSLatRadianceFieldDecoder
from diffusers_3d.families.trellis.sparse import TrellisSparseTensor, trellis_grid_transform

RESOLUTION = 8
REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT", "/tmp/TRELLIS"))


def _asset(coordinates, density, colors, *, trivec=None):
    count = len(coordinates)
    return RadianceFieldAsset(
        coordinates=torch.tensor(coordinates),
        trivec=torch.ones(count, 1, 3, 8) if trivec is None else trivec,
        density=torch.tensor(density).reshape(count, 1),
        color_coefficients=torch.tensor(colors).reshape(count, 1, 3),
        resolution=RESOLUTION,
        grid_transform=trellis_grid_transform(RESOLUTION),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )


def _camera_above(target_xy, height=2.0, focal=64.0, size=32):
    """OpenCV camera on the +z axis looking straight down at ``target_xy``."""

    camera_to_world = torch.eye(4)
    camera_to_world[:3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    camera_to_world[:3, 3] = torch.tensor([target_xy[0], target_xy[1], height])
    intrinsics = torch.tensor([[focal, 0.0, size / 2], [0.0, focal, size / 2], [0.0, 0.0, 1.0]])
    return CameraRig(
        world_to_camera=torch.linalg.inv(camera_to_world).unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
        image_sizes=torch.tensor([[size, size]]),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )


def _render(asset, cameras, seed=0, **kwargs):
    torch.manual_seed(seed)
    kwargs.setdefault("ssaa", 1)
    return render_radiance_field(asset, cameras, **kwargs)


def test_constant_voxel_matches_the_closed_form_transmittance():
    # Voxel (3, 3, 3) spans [-0.125, 0]^3; the centre ray crosses it in 16 steps of 0.5 / dim voxel sizes.
    asset = _asset([[3, 3, 3]], [2.0], [[3.0, -3.0, 0.0]])
    cameras = _camera_above((-0.0625, -0.0625))

    rendered = _render(asset, cameras)

    alpha = rendered["alpha"][0, 0, 16, 16]
    step = 0.125 / 16
    expected_alpha = 1.0 - math.exp(-float(F.softplus(torch.tensor(2.0))) * step) ** 16
    assert abs(float(alpha) - expected_alpha) < 1e-4
    expected_color = torch.sigmoid(SH_C0 * torch.tensor([3.0, -3.0, 0.0])) * alpha
    torch.testing.assert_close(rendered["color"][0, :, 16, 16], expected_color, atol=1e-5, rtol=0)
    # Depth is sum(w * z), not normalised by alpha: depth / alpha lies in the voxel's z range [2.0, 2.125].
    assert 2.0 <= float(rendered["depth"][0, 0, 16, 16] / alpha) <= 2.125
    assert float(rendered["alpha"][0, 0, 2, 2]) == 0.0 and float(rendered["depth"][0, 0, 2, 2]) == 0.0
    # 0.125 world units at focal 64 from 2 units away covers 4 pixels per side.
    assert int((rendered["alpha"][0, 0] > 0).sum()) == 16
    with_background = _render(asset, cameras, background=(1.0, 1.0, 1.0))
    torch.testing.assert_close(with_background["color"][0, :, 2, 2], torch.ones(3))
    torch.testing.assert_close(
        with_background["color"][0, :, 16, 16], expected_color + (1.0 - alpha), atol=1e-5, rtol=0
    )


def test_early_termination_stops_compositing_once_transmittance_drops_below_1e3():
    # Per sample the front voxel keeps exp(-100 / 128) = 0.458 of the light: 9 samples take it below 1e-3.
    front = _asset([[3, 3, 4]], [100.0], [[20.0, -20.0, -20.0]])
    both = _asset([[3, 3, 4], [3, 3, 3]], [100.0, 100.0], [[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]])
    cameras = _camera_above((-0.0625, -0.0625))

    only_front, stacked = _render(front, cameras), _render(both, cameras)

    for name in ("color", "depth", "alpha"):
        torch.testing.assert_close(stacked[name], only_front[name], atol=0, rtol=0)
    transmittance = 1.0 - float(stacked["alpha"][0, 0, 16, 16])
    survive = math.exp(-100.0 / 128)
    assert survive**9 * 0.99 < transmittance < survive**8, transmittance


def test_voxels_within_0_2_of_the_camera_plane_are_culled():
    asset = _asset([[3, 3, 3]], [2.0], [[0.0, 0.0, 0.0]])
    # The voxel centre sits at z = -0.0625; a camera at z = 0.125 sees it at depth 0.1875 <= 0.2 and drops it.
    near = _render(asset, _camera_above((-0.0625, -0.0625), height=0.125, focal=4.0))
    far = _render(asset, _camera_above((-0.0625, -0.0625), height=0.16, focal=4.0))

    assert float(near["alpha"].max()) == 0.0
    assert float(far["alpha"].max()) > 0.0


def test_ssaa_renders_larger_and_downsamples_like_trellis():
    asset = _asset([[3, 3, 3], [4, 3, 3]], [2.0, 5.0], [[3.0, -3.0, 0.0], [0.0, 1.0, 2.0]])
    cameras = _camera_above((0.0, -0.0625), focal=48.0, size=16)
    large = _camera_above((0.0, -0.0625), focal=96.0, size=32)

    supersampled = _render(asset, cameras, ssaa=2)
    direct = _render(asset, large)

    options = {"size": (16, 16), "mode": "bilinear", "align_corners": False, "antialias": True}
    for name in ("color", "depth", "alpha"):
        torch.testing.assert_close(supersampled[name], F.interpolate(direct[name], **options), atol=0, rtol=0)


def test_sample_phase_comes_from_the_global_generator():
    trivec = torch.ones(1, 1, 3, 8)
    trivec[0, 0, 2] = torch.linspace(0.2, 2.0, 8)  # density varies along z, so the phase changes the result
    asset = _asset([[3, 3, 3]], [2.0], [[3.0, -3.0, 0.0]], trivec=trivec)
    cameras = _camera_above((-0.0625, -0.0625))

    first, again, other = _render(asset, cameras, seed=0), _render(asset, cameras, seed=0), _render(asset, cameras, 1)

    torch.testing.assert_close(first["color"], again["color"], atol=0, rtol=0)
    assert not torch.equal(first["depth"], other["depth"])


def test_trivec_interpolation_varies_inside_a_voxel_and_extrapolates_at_the_border():
    trivec = torch.ones(1, 1, 3, 8)
    trivec[0, 0, 0] = torch.linspace(0.0, 7.0, 8)  # component grows along x, flat along y and z
    asset = _asset([[0, 0, 0]], [1.0], [[0.0, 0.0, 0.0]], trivec=trivec)
    rows = torch.zeros(4, dtype=torch.int64)
    local = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [0.75, 0.2, 0.9]])

    density, color = sample_radiance_field(asset, rows, local)

    # Cell centres sit at (i + 0.5) / 8, so x = 0.5 reads 3.5; the border reads the extrapolated 0-th segment.
    component = torch.tensor([3.5, -0.5, 7.5, 5.5])
    torch.testing.assert_close(density, F.softplus(component, threshold=10.0))
    torch.testing.assert_close(color, torch.full((4, 3), 0.5))


def test_renderer_rejects_mismatched_frames_and_batches_cameras():
    asset = _asset([[3, 3, 3]], [2.0], [[0.0, 0.0, 0.0]])
    cameras = _camera_above((-0.0625, -0.0625))
    two = CameraRig(
        world_to_camera=cameras.world_to_camera.repeat(2, 1, 1),
        intrinsics=cameras.intrinsics.repeat(2, 1, 1),
        image_sizes=torch.tensor([[32, 32], [32, 32]]),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    rendered = render_radiance_field(asset, two)
    assert rendered["color"].shape == (2, 3, 32, 32) and rendered["depth"].shape == (2, 1, 32, 32)
    torch.testing.assert_close(rendered["alpha"][0], rendered["alpha"][1], atol=1e-3, rtol=0)
    y_up = CameraRig(
        world_to_camera=cameras.world_to_camera,
        intrinsics=cameras.intrinsics,
        image_sizes=cameras.image_sizes,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
    )
    with pytest.raises(ValueError, match="same coordinate system"):
        render_radiance_field(asset, y_up)
    with pytest.raises(ValueError, match="ssaa"):
        render_radiance_field(asset, cameras, ssaa=0)


# ------------------------------------------------------------------ diffoctreerast arguments vs pinned TRELLIS


class _EasyDict(dict):
    """Stand-in for ``easydict.EasyDict``, which the reference renderer imports."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _CapturingRasterizer:
    """Records what ``OctreeTrivecRasterizer`` receives and returns a fixed pseudo-random image."""

    calls: list = []

    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, **inputs):
        type(self).calls.append((self.raster_settings, inputs))
        height, width = self.raster_settings.image_height, self.raster_settings.image_width
        generator = torch.Generator().manual_seed(len(type(self).calls))
        rgb, depth, alpha, percent_depth = torch.rand(6, height, width, generator=generator).split([3, 1, 1, 1])
        return rgb, depth[0], alpha[0], percent_depth[0]


def _load_reference(monkeypatch):
    """Pinned TRELLIS ``Strivec``, ``to_representation`` and ``OctreeRenderer`` without the heavy package imports."""

    if not REFERENCE_ROOT.is_dir():
        reason = f"TRELLIS reference checkout is unavailable at {REFERENCE_ROOT}"
    else:
        result = subprocess.run(
            ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        revision = result.stdout.strip()
        reason = None if revision == TRELLIS_REFERENCE_REVISION else f"TRELLIS checkout is at {revision!r}"
    if reason is not None:
        if os.environ.get("DIFFUSERS_3D_REQUIRE_REFERENCE") == "1":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    if importlib.util.find_spec("easydict") is None:
        monkeypatch.setitem(sys.modules, "easydict", types.SimpleNamespace(EasyDict=_EasyDict))
    root = REFERENCE_ROOT / "trellis"
    package = "_diffusers_3d_radiance_field_reference"
    for name, path in ((package, root), (f"{package}.representations", root / "representations")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def load(name: str, path: Path):
        # Upstream hard-codes CUDA tensors; everything else runs unchanged on the CPU.
        source = path.read_text().replace("'cuda'", "'cpu'").replace('"cuda"', '"cpu"')
        module = types.ModuleType(name)
        module.__file__, module.__package__ = str(path), name.rpartition(".")[0]
        sys.modules[name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module

    strivec = importlib.import_module(f"{package}.representations.radiance_field").Strivec
    renderers = types.ModuleType(f"{package}.renderers")
    renderers.__path__ = [str(root / "renderers")]
    sys.modules[renderers.__name__] = renderers
    octree_renderer = load(f"{package}.renderers.octree_renderer", root / "renderers" / "octree_renderer.py")

    decoder_path = root / "models" / "structured_latent_vae" / "decoder_rf.py"
    tree = ast.parse(decoder_path.read_text().replace("'cuda'", "'cpu'"))
    decoder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SLatRadianceFieldDecoder"
    )
    methods = [node for node in decoder.body if isinstance(node, ast.FunctionDef)]
    methods = [node for node in methods if node.name in ("_calc_layout", "to_representation")]
    namespace = {
        "Strivec": strivec,
        "np": np,
        "torch": torch,
        "List": list,
        "sp": types.SimpleNamespace(SparseTensor=None),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(decoder_path), "exec"), namespace)
    return namespace["_calc_layout"], namespace["to_representation"], octree_renderer


def _look_at(yaw: float, pitch: float, radius: float) -> torch.Tensor:
    """OpenCV world-to-camera extrinsics looking at the origin with +z up (``utils3d.extrinsics_look_at``)."""

    eye = (
        torch.tensor(
            [math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)], dtype=torch.float32
        )
        * radius
    )
    forward = -eye / eye.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0]))
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])
    extrinsics = torch.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = -rotation @ eye
    return extrinsics


@pytest.mark.reference_parity
def test_diffoctreerast_arguments_match_pinned_trellis(monkeypatch):
    calc_layout, to_representation, octree_renderer = _load_reference(monkeypatch)
    # Both sides import the rasterizer from the diffoctreerast module; record what they hand it.
    names = (
        "OctreeVoxelRasterizer",
        "OctreeGaussianRasterizer",
        "OctreeTrivecRasterizer",
        "OctreeDecoupolyRasterizer",
    )
    monkeypatch.setitem(
        sys.modules, "diffoctreerast", types.SimpleNamespace(**dict.fromkeys(names, _CapturingRasterizer))
    )
    resolution, rank, dim, size, ssaa = 64, 4, 8, 16, 4
    generator = torch.Generator().manual_seed(0)
    coordinates = torch.unique(torch.randint(16, 48, (200, 3), generator=generator), dim=0)
    features = torch.randn(coordinates.shape[0], rank * 3 * dim + rank + rank * 3, generator=generator)
    sparse_coordinates = torch.cat([torch.zeros_like(coordinates[:, :1]), coordinates], dim=1).int()

    # Upstream: SLatRadianceFieldDecoder.to_representation, then render_utils' OctreeRenderer settings.
    decoder = types.SimpleNamespace(resolution=resolution, rep_config={"rank": rank, "dim": dim})
    calc_layout(decoder)
    sparse = types.SimpleNamespace(
        coords=sparse_coordinates,
        feats=features,
        layout=[slice(0, coordinates.shape[0])],
        shape=(1, features.shape[1]),
    )
    (strivec,) = to_representation(decoder, sparse)
    renderer = octree_renderer.OctreeRenderer()
    renderer.unsupported = False
    renderer.rendering_options.update(
        {"resolution": size, "near": 0.8, "far": 1.6, "bg_color": (0, 0, 0), "ssaa": ssaa}
    )
    renderer.pipe.primitive = strivec.primitive
    fov = math.radians(40.0)
    focal = 0.5 / math.tan(fov / 2)
    intrinsics = torch.tensor([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]])
    views = [_look_at(0.3, 0.4, 2.0), _look_at(2.1, -0.2, 2.0)]
    _CapturingRasterizer.calls = []
    reference = [renderer.render(strivec, view, intrinsics) for view in views]
    reference_calls = _CapturingRasterizer.calls

    # Package: the decoder's asset through render_radiance_field's diffoctreerast path.
    asset_decoder = types.SimpleNamespace(rank=rank, dim=dim, resolution=resolution)
    hidden_states = TrellisSparseTensor(coordinates=sparse_coordinates.long(), features=features)
    (asset,) = TrellisSLatRadianceFieldDecoder._to_assets(asset_decoder, hidden_states, features)
    cameras = CameraRig(
        world_to_camera=torch.stack(views),
        intrinsics=(intrinsics * torch.tensor([[size], [size], [1.0]])).expand(2, 3, 3),
        image_sizes=torch.tensor([[size, size]] * 2),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    _CapturingRasterizer.calls = []
    with pytest.warns(UserWarning, match="non-commercial"):
        ours = radiance_field._render_diffoctreerast(asset, cameras, background=(0.0, 0.0, 0.0), ssaa=ssaa)
    ours_calls = _CapturingRasterizer.calls

    for (reference_settings, reference_inputs), (settings, inputs) in zip(reference_calls, ours_calls, strict=True):
        for name in ("positions", "trivecs", "densities", "shs", "depths", "aabb"):
            assert torch.equal(inputs[name], reference_inputs[name]), name
            assert inputs[name].dtype == reference_inputs[name].dtype, name
        for name in ("colors_precomp", "colors_overwrite", "aux"):
            assert reference_inputs[name] is None and inputs.get(name) is None, name
        for name in ("image_height", "image_width", "tanfovx", "tanfovy", "scale_modifier", "sh_degree", "jitter"):
            assert getattr(settings, name) == reference_settings[name], name
        for name in ("viewmatrix", "projmatrix", "campos", "bg"):
            assert torch.equal(getattr(settings, name), reference_settings[name]), name
        assert settings.density_shift == reference_settings.density_shift
        assert settings.used_rank == reference_settings.used_rank
    for index, expected in enumerate(reference):
        assert torch.equal(ours["color"][index], expected.color)
        assert torch.equal(ours["depth"][index, 0], expected.depth)
        assert torch.equal(ours["alpha"][index, 0], expected.alpha)
