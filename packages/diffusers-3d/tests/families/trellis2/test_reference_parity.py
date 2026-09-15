from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers_3d import (
    TRELLIS2_REFERENCE_REVISION,
    ImageCondition,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
    TrellisSparseTensor,
    preprocess_image_condition,
)
from diffusers_3d._reference import ReferenceCheckoutError, reference_unavailable, validate_reference_checkout

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRELLIS2_REFERENCE_ROOT", "/tmp/TRELLIS.2"))
REFERENCE_REPOSITORY = "https://github.com/microsoft/TRELLIS.2.git"
REFERENCE_PATHS = (
    "trellis2/pipelines/trellis2_image_to_3d.py",
    "trellis2/models/sparse_structure_flow.py",
    "trellis2/models/sparse_structure_vae.py",
    "trellis2/modules/attention/__init__.py",
    "trellis2/modules/norm.py",
    "trellis2/modules/spatial.py",
    "trellis2/modules/transformer/__init__.py",
    "trellis2/modules/utils.py",
    "trellis2/models/structured_latent_flow.py",
    "trellis2/models/sparse_elastic_mixin.py",
    "trellis2/models/sc_vaes/sparse_unet_vae.py",
    "trellis2/models/sc_vaes/fdg_vae.py",
    "trellis2/modules/sparse/__init__.py",
    "trellis2/utils/elastic_utils.py",
)
REFERENCE_PACKAGE = "_diffusers_3d_trellis2_reference"
_REFERENCE_TYPES: tuple[type[torch.nn.Module], type[torch.nn.Module]] | None = None
_SPARSE_REFERENCE: types.SimpleNamespace | None = None


def _reference_unavailable(error: ReferenceCheckoutError) -> None:
    try:
        reason = reference_unavailable(error)
    except ReferenceCheckoutError as required_error:
        pytest.fail(str(required_error), pytrace=False)
    pytest.skip(reason)


def _validate_reference() -> None:
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=TRELLIS2_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
    except ReferenceCheckoutError as error:
        _reference_unavailable(error)


def _load_module(name: str, path: Path, *, package: bool = False):
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned TRELLIS.2 module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_pinned_reference():
    global _REFERENCE_TYPES
    if _REFERENCE_TYPES is not None:
        return _REFERENCE_TYPES
    source_root = REFERENCE_ROOT / "trellis2"
    _validate_reference()

    os.environ["ATTN_BACKEND"] = "sdpa"
    _package(REFERENCE_PACKAGE, source_root)
    modules_package = _package(f"{REFERENCE_PACKAGE}.modules", source_root / "modules")
    _package(f"{REFERENCE_PACKAGE}.models", source_root / "models")

    # The dense reference utility module names sparse primitive classes in a
    # conversion tuple even though the sparse-structure path never invokes
    # them. Stubbing only those type names keeps this parity test CPU-only and
    # avoids importing spconv/FlexGEMM.
    sparse_module = types.ModuleType(f"{REFERENCE_PACKAGE}.modules.sparse")
    sparse_module.SparseConv3d = type("SparseConv3d", (torch.nn.Module,), {})
    sparse_module.SparseInverseConv3d = type("SparseInverseConv3d", (torch.nn.Module,), {})
    sparse_module.SparseLinear = type("SparseLinear", (torch.nn.Module,), {})
    sys.modules[sparse_module.__name__] = sparse_module
    modules_package.sparse = sparse_module

    try:
        _load_module(f"{REFERENCE_PACKAGE}.modules.utils", source_root / "modules" / "utils.py")
        _load_module(f"{REFERENCE_PACKAGE}.modules.spatial", source_root / "modules" / "spatial.py")
        _load_module(f"{REFERENCE_PACKAGE}.modules.norm", source_root / "modules" / "norm.py")
        _load_module(
            f"{REFERENCE_PACKAGE}.modules.attention",
            source_root / "modules" / "attention" / "__init__.py",
            package=True,
        )
        _load_module(
            f"{REFERENCE_PACKAGE}.modules.transformer",
            source_root / "modules" / "transformer" / "__init__.py",
            package=True,
        )
        flow_module = _load_module(
            f"{REFERENCE_PACKAGE}.models.sparse_structure_flow",
            source_root / "models" / "sparse_structure_flow.py",
        )
        decoder_module = _load_module(
            f"{REFERENCE_PACKAGE}.models.sparse_structure_vae",
            source_root / "models" / "sparse_structure_vae.py",
        )
    except (ImportError, RuntimeError) as error:
        _reference_unavailable(ReferenceCheckoutError(f"optional pinned reference dependency unavailable: {error}"))
    _REFERENCE_TYPES = flow_module.SparseStructureFlowModel, decoder_module.SparseStructureDecoder
    return _REFERENCE_TYPES


def _checkpoint_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    # The pinned dense flow keeps RoPE phases as a persistent buffer; the released safetensors do
    # not store them and neither does the port.
    return {key: value for key, value in module.state_dict().items() if key != "rope_phases"}


def _randomize_state(module: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype) * 0.02)


def _assert_soft_alpha_preprocessing_parity() -> None:
    path = REFERENCE_ROOT / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trellis2ImageTo3DPipeline"
    )
    function_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "preprocess_image"
    )
    function_node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
    namespace = {"Image": Image, "np": np}
    exec(compile(module, str(path), "exec"), namespace)

    rgba = np.zeros((9, 11, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.arange(11, dtype=np.uint8) * 23
    rgba[:, :, 1] = np.arange(9, dtype=np.uint8)[:, None] * 29
    rgba[:, :, 2] = 191
    rgba[1:8, 2:10, 3] = np.linspace(1, 255, 56, dtype=np.uint8).reshape(7, 8)
    rgba[0, 0, 3] = 204
    rgba[8, 10, 3] = 205
    reference = namespace["preprocess_image"](object(), Image.fromarray(rgba))
    reference = reference.resize((512, 512), Image.Resampling.LANCZOS)
    expected = torch.from_numpy(np.array(reference, copy=True)).permute(2, 0, 1).float().div(255)
    condition = ImageCondition(torch.from_numpy(rgba.copy()).permute(2, 0, 1).float().div(255))
    actual = preprocess_image_condition(
        condition,
        image_size=512,
        foreground_scale=1.0,
        premultiply_before_resize=True,
    ).image
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_tiny_sparse_structure_components_match_pinned_reference():
    reference_flow_type, reference_decoder_type = _load_pinned_reference()
    _assert_soft_alpha_preprocessing_parity()
    flow_config = Trellis2SparseStructureFlowModel.tiny_config()
    torch.manual_seed(0)
    reference_flow = reference_flow_type(**flow_config).eval()
    _randomize_state(reference_flow, seed=1)
    flow = Trellis2SparseStructureFlowModel(**flow_config).eval()
    flow.load_state_dict(_checkpoint_state(reference_flow), strict=True)
    assert tuple(flow.state_dict()) == tuple(_checkpoint_state(reference_flow))

    generator = torch.Generator().manual_seed(2)
    hidden_states = torch.randn(2, 2, 2, 2, 2, generator=generator)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)
    with torch.no_grad():
        expected_flow = reference_flow(hidden_states, timesteps, context)
        actual_flow = flow(hidden_states, timesteps, context).sample
    torch.testing.assert_close(actual_flow, expected_flow, atol=1e-6, rtol=1e-5)

    decoder_config = Trellis2SparseStructureDecoder.tiny_config()
    torch.manual_seed(3)
    reference_decoder = reference_decoder_type(**decoder_config).eval()
    _randomize_state(reference_decoder, seed=4)
    decoder = Trellis2SparseStructureDecoder(**decoder_config).eval()
    decoder.load_state_dict(reference_decoder.state_dict(), strict=True)
    assert tuple(decoder.state_dict()) == tuple(reference_decoder.state_dict())
    with torch.no_grad():
        expected_logits = reference_decoder(actual_flow)
        actual_logits = decoder(actual_flow).sample
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-5)


def test_tiny_sparse_structure_flow_backward_matches_pinned_reference():
    reference_flow_type, _ = _load_pinned_reference()
    config = Trellis2SparseStructureFlowModel.tiny_config()
    torch.manual_seed(5)
    reference = reference_flow_type(**config)
    _randomize_state(reference, seed=6)
    model = Trellis2SparseStructureFlowModel(**config)
    model.load_state_dict(_checkpoint_state(reference), strict=True)

    generator = torch.Generator().manual_seed(7)
    reference_input = torch.randn(2, 2, 2, 2, 2, generator=generator, requires_grad=True)
    model_input = reference_input.detach().clone().requires_grad_(True)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)
    target = torch.randn(2, 2, 2, 2, 2, generator=generator)

    reference_loss = torch.nn.functional.mse_loss(reference(reference_input, timesteps, context), target)
    model_loss = torch.nn.functional.mse_loss(model(model_input, timesteps, context).sample, target)
    reference_loss.backward()
    model_loss.backward()
    torch.testing.assert_close(model_loss, reference_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model_input.grad, reference_input.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        model.blocks[0].self_attn.to_qkv.weight.grad,
        reference.blocks[0].self_attn.to_qkv.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )


# --- Sparse (SLAT) reference -------------------------------------------------
#
# The pinned sparse modules dispatch attention to xformers / flash-attn and convolution to
# FlexGEMM / spconv. Both are swapped for CPU references here: block-diagonal attention becomes
# per-sequence SDPA, and submanifold convolution becomes a dense ``F.conv3d`` over the voxel grid.
# Everything else (sparse tensors, RoPE, modulation, ConvNeXt / C2S blocks, subdivision, dual-grid
# heads) runs the upstream code unchanged.


class _BlockDiagonalMask:
    def __init__(self, q_seqlens, kv_seqlens):
        self.q_seqlens = [int(value) for value in q_seqlens]
        self.kv_seqlens = [int(value) for value in kv_seqlens]

    @classmethod
    def from_seqlens(cls, q_seqlens, kv_seqlens=None):
        return cls(q_seqlens, q_seqlens if kv_seqlens is None else kv_seqlens)


def _memory_efficient_attention(q, k, v, attn_bias):
    # q, k, v: [1, T, H, C] packed sequences.
    outputs = []
    q_start = kv_start = 0
    for q_len, kv_len in zip(attn_bias.q_seqlens, attn_bias.kv_seqlens):
        query = q[0, q_start : q_start + q_len].transpose(0, 1)
        key = k[0, kv_start : kv_start + kv_len].transpose(0, 1)
        value = v[0, kv_start : kv_start + kv_len].transpose(0, 1)
        outputs.append(torch.nn.functional.scaled_dot_product_attention(query, key, value).transpose(0, 1))
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outputs, dim=0).unsqueeze(0)


@contextlib.contextmanager
def _fake_xformers():
    """The pinned attention code re-imports ``xformers.ops`` on every call; serve the SDPA stand-in."""

    ops = types.ModuleType("xformers.ops")
    ops.fmha = types.SimpleNamespace(BlockDiagonalMask=_BlockDiagonalMask)
    ops.memory_efficient_attention = _memory_efficient_attention
    package = types.ModuleType("xformers")
    package.ops = ops
    saved = {name: sys.modules.get(name) for name in ("xformers", "xformers.ops")}
    sys.modules["xformers"], sys.modules["xformers.ops"] = package, ops
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _dense_conv_backend() -> types.ModuleType:
    module = types.ModuleType(f"{REFERENCE_PACKAGE}.modules.sparse.conv.conv_none")

    def sparse_conv3d_init(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None
    ):
        assert stride == 1 and padding is None
        self.kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.dilation = dilation
        dense = torch.nn.Conv3d(in_channels, out_channels, self.kernel_size, bias=bias)
        # FlexGEMM layout: (Co, Kd, Kh, Kw, Ci).
        self.weight = torch.nn.Parameter(dense.weight.detach().permute(0, 2, 3, 4, 1).contiguous())
        self.bias = torch.nn.Parameter(dense.bias.detach()) if bias else None

    def sparse_conv3d_forward(self, x):
        coords = x.coords.long()
        pad = (self.kernel_size[0] // 2) * self.dilation
        extent = [int(coords[:, axis].max()) + 1 for axis in range(1, 4)]
        batch = int(coords[:, 0].max()) + 1
        grid = x.feats.new_zeros(batch, x.feats.shape[1], *extent)
        grid[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = x.feats
        weight = self.weight.permute(0, 4, 1, 2, 3)
        out = torch.nn.functional.conv3d(grid, weight, self.bias, padding=pad, dilation=self.dilation)
        return x.replace(out[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]])

    def unsupported(*args, **kwargs):
        raise NotImplementedError

    module.sparse_conv3d_init = sparse_conv3d_init
    module.sparse_conv3d_forward = sparse_conv3d_forward
    module.sparse_inverse_conv3d_init = unsupported
    module.sparse_inverse_conv3d_forward = unsupported
    return module


class _CapturedMesh:
    """Stands in for ``trellis2.representations.Mesh``; keeps the raw dual-grid tensors."""

    def __init__(self, coords, vertices, intersected, quad_lerp):
        self.coords, self.vertices, self.intersected, self.quad_lerp = coords, vertices, intersected, quad_lerp


def _load_pinned_sparse_reference() -> types.SimpleNamespace:
    global _SPARSE_REFERENCE
    if _SPARSE_REFERENCE is not None:
        return _SPARSE_REFERENCE
    _load_pinned_reference()
    source_root = REFERENCE_ROOT / "trellis2"

    os.environ["SPARSE_CONV_BACKEND"] = "none"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    modules_package = sys.modules[f"{REFERENCE_PACKAGE}.modules"]
    sparse_name = f"{REFERENCE_PACKAGE}.modules.sparse"
    sys.modules.pop(sparse_name, None)
    _package(f"{REFERENCE_PACKAGE}.utils", source_root / "utils")
    _package(f"{REFERENCE_PACKAGE}.models.sc_vaes", source_root / "models" / "sc_vaes")
    representations = types.ModuleType(f"{REFERENCE_PACKAGE}.representations")
    representations.Mesh = _CapturedMesh
    sys.modules[representations.__name__] = representations
    o_voxel_convert = types.ModuleType("o_voxel.convert")
    o_voxel_convert.flexible_dual_grid_to_mesh = lambda coords, vertices, intersected, quad_lerp, **kwargs: (
        coords,
        vertices,
        intersected,
        quad_lerp,
    )
    o_voxel = types.ModuleType("o_voxel")
    o_voxel.convert = o_voxel_convert
    saved = {name: sys.modules.get(name) for name in ("o_voxel", "o_voxel.convert")}
    sys.modules["o_voxel"], sys.modules["o_voxel.convert"] = o_voxel, o_voxel_convert
    try:
        sparse = _load_module(sparse_name, source_root / "modules" / "sparse" / "__init__.py", package=True)
        modules_package.sparse = sparse
        sys.modules[f"{sparse_name}.conv.conv_none"] = _dense_conv_backend()
        _load_module(f"{REFERENCE_PACKAGE}.utils.elastic_utils", source_root / "utils" / "elastic_utils.py")
        _load_module(
            f"{REFERENCE_PACKAGE}.models.sparse_elastic_mixin", source_root / "models" / "sparse_elastic_mixin.py"
        )
        flow_module = _load_module(
            f"{REFERENCE_PACKAGE}.models.structured_latent_flow", source_root / "models" / "structured_latent_flow.py"
        )
        unet_module = _load_module(
            f"{REFERENCE_PACKAGE}.models.sc_vaes.sparse_unet_vae",
            source_root / "models" / "sc_vaes" / "sparse_unet_vae.py",
        )
        fdg_module = _load_module(
            f"{REFERENCE_PACKAGE}.models.sc_vaes.fdg_vae", source_root / "models" / "sc_vaes" / "fdg_vae.py"
        )
    except (ImportError, RuntimeError) as error:
        _reference_unavailable(ReferenceCheckoutError(f"optional pinned reference dependency unavailable: {error}"))
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    _SPARSE_REFERENCE = types.SimpleNamespace(
        sparse=sparse,
        SLatFlowModel=flow_module.SLatFlowModel,
        SparseUnetVaeDecoder=unet_module.SparseUnetVaeDecoder,
        FlexiDualGridVaeDecoder=fdg_module.FlexiDualGridVaeDecoder,
    )
    return _SPARSE_REFERENCE


def _sparse_inputs(reference, coordinates: torch.Tensor, features: torch.Tensor):
    return reference.sparse.SparseTensor(features, coordinates.to(torch.int32)), TrellisSparseTensor(
        coordinates, features
    )


def test_tiny_slat_flow_matches_pinned_reference():
    reference = _load_pinned_sparse_reference()
    config = Trellis2SLatFlowModel.tiny_config(texture=True)
    torch.manual_seed(20)
    reference_model = reference.SLatFlowModel(**config).eval()
    _randomize_state(reference_model, seed=21)
    model = Trellis2SLatFlowModel(**config).eval()
    model.load_state_dict(reference_model.state_dict(), strict=True)
    assert tuple(model.state_dict()) == tuple(reference_model.state_dict())

    generator = torch.Generator().manual_seed(22)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 7, 0, 5], [1, 2, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    latents = torch.randn(5, 4, generator=generator, requires_grad=True)
    concat = torch.randn(5, 4, generator=generator)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)
    reference_x, x = _sparse_inputs(reference, coordinates, latents)
    reference_c, c = _sparse_inputs(reference, coordinates, concat)

    with _fake_xformers():
        expected = reference_model(reference_x, timesteps, context, concat_cond=reference_c)
    actual = model(x, timesteps, context, concat_cond=c).sample
    torch.testing.assert_close(actual.features, expected.feats, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual.coordinates, coordinates)

    (expected_grad,) = torch.autograd.grad(expected.feats.square().sum(), latents, retain_graph=True)
    (actual_grad,) = torch.autograd.grad(actual.features.square().sum(), latents)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_tiny_shape_and_pbr_decoders_match_pinned_reference():
    reference = _load_pinned_sparse_reference()
    shape_config = Trellis2ShapeDualGridDecoder.tiny_config()
    pbr_config = Trellis2PBRSparseDecoder.tiny_config()
    torch.manual_seed(30)
    reference_shape = reference.FlexiDualGridVaeDecoder(**shape_config).eval()
    reference_pbr = reference.SparseUnetVaeDecoder(
        **{key: value for key, value in pbr_config.items() if key != "channel_layout"}
    ).eval()
    _randomize_state(reference_shape, seed=31)
    _randomize_state(reference_pbr, seed=32)
    with torch.no_grad():
        # Randomised subdivision logits pick a mixed set of children per voxel.
        reference_shape.blocks[0][-1].to_subdiv.weight.mul_(40.0)
    shape_decoder = Trellis2ShapeDualGridDecoder(**shape_config).eval()
    pbr_decoder = Trellis2PBRSparseDecoder(**pbr_config).eval()
    shape_decoder.load_state_dict(reference_shape.state_dict(), strict=True)
    pbr_decoder.load_state_dict(reference_pbr.state_dict(), strict=True)
    assert tuple(shape_decoder.state_dict()) == tuple(reference_shape.state_dict())
    assert tuple(pbr_decoder.state_dict()) == tuple(reference_pbr.state_dict())

    generator = torch.Generator().manual_seed(33)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 3, 3, 3], [1, 2, 1, 0], [1, 5, 6, 7], [1, 6, 6, 7]],
        dtype=torch.int64,
    )
    shape_latents = torch.randn(6, 4, generator=generator)
    texture_latents = torch.randn(6, 4, generator=generator)
    reference_shape_input, shape_input = _sparse_inputs(reference, coordinates, shape_latents)
    reference_texture_input, texture_input = _sparse_inputs(reference, coordinates, texture_latents)

    with torch.no_grad():
        reference_shape.set_resolution(16)
        meshes, reference_subs = reference_shape(reference_shape_input, return_subs=True)
        output = shape_decoder(shape_input, resolution=16)
    assert len(meshes) == len(output.assets) == 2
    assert len(reference_subs) == len(output.subdivisions) == 1
    assert torch.equal(output.subdivisions[0], reference_subs[0].feats > 0)
    assert int(output.subdivisions[0].sum()) not in (0, 8 * coordinates.shape[0])
    for mesh, asset in zip(meshes, output.assets):
        assert torch.equal(asset.active_coordinates, mesh.coords.to(torch.int64))
        torch.testing.assert_close(asset.dual_grid_vertex_offsets, mesh.vertices, atol=1e-6, rtol=1e-5)
        assert torch.equal(asset.intersection_data, mesh.intersected)
        torch.testing.assert_close(asset.split_weights, mesh.quad_lerp, atol=1e-6, rtol=1e-5)

    with torch.no_grad():
        expected_pbr = reference_pbr(reference_texture_input, guide_subs=reference_subs) * 0.5 + 0.5
        pbr_assets = pbr_decoder(texture_input, output.assets, output.subdivisions).assets
    actual = torch.cat(
        [torch.cat([asset.base_color, asset.metallic, asset.roughness, asset.opacity], dim=1) for asset in pbr_assets]
    )
    torch.testing.assert_close(actual, expected_pbr.feats.clamp(0, 1), atol=1e-6, rtol=1e-5)
    for index, asset in enumerate(pbr_assets):
        expected_coordinates = expected_pbr.coords[expected_pbr.coords[:, 0] == index][:, 1:]
        assert torch.equal(asset.active_coordinates, expected_coordinates.to(torch.int64))
