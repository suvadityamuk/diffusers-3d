from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers_3d import (
    TRELLIS_REFERENCE_REVISION,
    ImageCondition,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    TrellisSparseTensor,
    preprocess_image_condition,
)
from diffusers_3d._reference import ReferenceCheckoutError, reference_unavailable, validate_reference_checkout
from diffusers_3d.families.trellis.flexicubes import FLEXICUBES_REFERENCE_REVISION
from diffusers_3d.objects import CoordinateSystem

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT", "/tmp/TRELLIS"))
REFERENCE_REPOSITORY = "https://github.com/microsoft/TRELLIS.git"
REFERENCE_PATHS = (
    "trellis/pipelines/trellis_image_to_3d.py",
    "trellis/models/sparse_structure_flow.py",
    "trellis/models/sparse_structure_vae.py",
    "trellis/modules/attention/__init__.py",
    "trellis/modules/norm.py",
    "trellis/modules/spatial.py",
    "trellis/modules/transformer/__init__.py",
    "trellis/modules/utils.py",
    "trellis/models/structured_latent_flow.py",
    "trellis/models/sparse_elastic_mixin.py",
    "trellis/models/structured_latent_vae/base.py",
    "trellis/models/structured_latent_vae/decoder_gs.py",
    "trellis/models/structured_latent_vae/decoder_mesh.py",
    "trellis/models/structured_latent_vae/decoder_rf.py",
    "trellis/representations/mesh/cube2mesh.py",
    "trellis/representations/mesh/utils_cube.py",
    "trellis/modules/sparse/__init__.py",
    "trellis/utils/elastic_utils.py",
    "trellis/utils/random_utils.py",
)
REFERENCE_PACKAGE = "_diffusers_3d_trellis_reference"
_REFERENCE_TYPES: tuple[type[torch.nn.Module], type[torch.nn.Module]] | None = None


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
            expected_revision=TRELLIS_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
    except ReferenceCheckoutError as error:
        _reference_unavailable(error)


def _load_module(name: str, path: Path, *, package: bool = False):
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned TRELLIS module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_pinned_reference():
    global _REFERENCE_TYPES
    if _REFERENCE_TYPES is not None:
        return _REFERENCE_TYPES
    source_root = REFERENCE_ROOT / "trellis"
    _validate_reference()

    os.environ["ATTN_BACKEND"] = "sdpa"
    for name, path in (
        (REFERENCE_PACKAGE, source_root),
        (f"{REFERENCE_PACKAGE}.modules", source_root / "modules"),
        (f"{REFERENCE_PACKAGE}.models", source_root / "models"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    try:
        _load_module(
            f"{REFERENCE_PACKAGE}.modules.utils",
            source_root / "modules" / "utils.py",
        )
        _load_module(
            f"{REFERENCE_PACKAGE}.modules.spatial",
            source_root / "modules" / "spatial.py",
        )
        _load_module(
            f"{REFERENCE_PACKAGE}.modules.norm",
            source_root / "modules" / "norm.py",
        )
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


def _randomize_state(module: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype) * 0.02)


def _assert_soft_alpha_preprocessing_parity() -> None:
    path = REFERENCE_ROOT / "trellis" / "pipelines" / "trellis_image_to_3d.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrellisImageTo3DPipeline"
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
    expected = torch.from_numpy(np.array(reference, copy=True)).permute(2, 0, 1).float().div(255)
    condition = ImageCondition(torch.from_numpy(rgba.copy()).permute(2, 0, 1).float().div(255))
    actual = preprocess_image_condition(condition, image_size=518, foreground_scale=1.2).image
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_tiny_sparse_structure_components_match_pinned_reference():
    reference_flow_type, reference_decoder_type = _load_pinned_reference()
    _assert_soft_alpha_preprocessing_parity()

    flow_config = TrellisSparseStructureFlowModel.tiny_config()
    torch.manual_seed(0)
    reference_flow = reference_flow_type(**flow_config).eval()
    _randomize_state(reference_flow, seed=1)
    flow = TrellisSparseStructureFlowModel(**flow_config).eval()
    flow.load_state_dict(reference_flow.state_dict(), strict=True)
    assert tuple(flow.state_dict()) == tuple(reference_flow.state_dict())

    generator = torch.Generator().manual_seed(2)
    hidden_states = torch.randn(2, 2, 4, 4, 4, generator=generator)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)
    with torch.no_grad():
        expected_flow = reference_flow(hidden_states, timesteps, context)
        actual_flow = flow(hidden_states, timesteps, context).sample
    torch.testing.assert_close(actual_flow, expected_flow, atol=1e-6, rtol=1e-5)

    decoder_config = TrellisSparseStructureDecoder.tiny_config()
    torch.manual_seed(3)
    reference_decoder = reference_decoder_type(**decoder_config).eval()
    _randomize_state(reference_decoder, seed=4)
    decoder = TrellisSparseStructureDecoder(**decoder_config).eval()
    decoder.load_state_dict(reference_decoder.state_dict(), strict=True)
    assert tuple(decoder.state_dict()) == tuple(reference_decoder.state_dict())
    with torch.no_grad():
        expected_logits = reference_decoder(actual_flow)
        actual_logits = decoder(actual_flow).sample
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-5)


def test_tiny_sparse_structure_flow_backward_matches_pinned_reference():
    reference_flow_type, _ = _load_pinned_reference()
    config = TrellisSparseStructureFlowModel.tiny_config()
    torch.manual_seed(5)
    reference = reference_flow_type(**config)
    _randomize_state(reference, seed=6)
    model = TrellisSparseStructureFlowModel(**config)
    model.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(7)
    reference_input = torch.randn(2, 2, 4, 4, 4, generator=generator, requires_grad=True)
    model_input = reference_input.detach().clone().requires_grad_(True)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)
    target = torch.randn(2, 2, 4, 4, 4, generator=generator)

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
# The pinned sparse modules need spconv for storage / submanifold convolution and xformers for
# block-diagonal attention. Both are replaced by CPU references: a plain tensor container whose
# ``SubMConv3d`` is a dense ``F.conv3d`` over the voxel grid, and per-sequence SDPA. Everything else
# (sparse tensors, res blocks, skips, swin windowing, Gaussian heads) is the upstream code unchanged.

_SPARSE_REFERENCE: types.SimpleNamespace | None = None


class _BlockDiagonalMask:
    def __init__(self, q_seqlens, kv_seqlens):
        self.q_seqlens = [int(value) for value in q_seqlens]
        self.kv_seqlens = [int(value) for value in kv_seqlens]

    @classmethod
    def from_seqlens(cls, q_seqlens, kv_seqlens=None):
        return cls(q_seqlens, q_seqlens if kv_seqlens is None else kv_seqlens)


def _memory_efficient_attention(q, k, v, attn_bias):
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


class _SparseConvTensor:
    """Minimal stand-in for ``spconv.pytorch.SparseConvTensor``."""

    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None):
        self._features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.grid, self.voxel_num, self.indice_dict = grid, voxel_num, indice_dict or {}
        self.benchmark, self.benchmark_record = False, {}
        self.thrust_allocator = self._timer = self.force_algo = self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value


class _SubMConv3d(torch.nn.Module):
    """Dense-reference submanifold convolution with the spconv 2.x ``(Co, Kd, Kh, Kw, Ci)`` weight layout."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, indice_key=None, algo=None):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.dilation = dilation
        dense = torch.nn.Conv3d(in_channels, out_channels, self.kernel_size, bias=bias)
        self.weight = torch.nn.Parameter(dense.weight.detach().permute(0, 2, 3, 4, 1).contiguous())
        self.bias = torch.nn.Parameter(dense.bias.detach()) if bias else None

    def forward(self, data):
        coords = data.indices.long()
        pad = (self.kernel_size[0] // 2) * self.dilation
        extent = [int(coords[:, axis].max()) + 1 for axis in range(1, 4)]
        grid = data.features.new_zeros(data.batch_size, data.features.shape[1], *extent)
        grid[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = data.features
        weight = self.weight.permute(0, 4, 1, 2, 3)
        out = torch.nn.functional.conv3d(grid, weight, self.bias, padding=pad, dilation=self.dilation)
        return _SparseConvTensor(
            out[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]],
            data.indices,
            data.spatial_shape,
            data.batch_size,
        )


@contextlib.contextmanager
def _fake_backends():
    """Serve the CPU stand-ins for ``spconv.pytorch`` and ``xformers.ops`` while pinned code imports them."""

    spconv = types.ModuleType("spconv.pytorch")
    spconv.SparseConvTensor = _SparseConvTensor
    spconv.SubMConv3d = _SubMConv3d
    spconv.ConvAlgo = types.SimpleNamespace(Native="native", MaskImplicitGemm="implicit_gemm")
    spconv_package = types.ModuleType("spconv")
    spconv_package.pytorch = spconv
    ops = types.ModuleType("xformers.ops")
    ops.fmha = types.SimpleNamespace(BlockDiagonalMask=_BlockDiagonalMask)
    ops.memory_efficient_attention = _memory_efficient_attention
    xformers = types.ModuleType("xformers")
    xformers.ops = ops
    # FlexiCubes only uses kaolin for a shape assertion and cube2mesh only uses EasyDict as a dict.
    kaolin_testing = types.ModuleType("kaolin.utils.testing")
    kaolin_testing.check_tensor = lambda *args, **kwargs: True
    kaolin_utils = types.ModuleType("kaolin.utils")
    kaolin_utils.testing = kaolin_testing
    kaolin = types.ModuleType("kaolin")
    kaolin.utils = kaolin_utils
    easydict = types.ModuleType("easydict")
    easydict.EasyDict = dict
    fakes = {
        "spconv": spconv_package,
        "spconv.pytorch": spconv,
        "xformers": xformers,
        "xformers.ops": ops,
        "kaolin": kaolin,
        "kaolin.utils": kaolin_utils,
        "kaolin.utils.testing": kaolin_testing,
        "easydict": easydict,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _CapturedGaussian:
    """Stands in for ``trellis.representations.Gaussian``; keeps the raw attributes the decoder sets."""

    def __init__(self, **kwargs):
        self.config = kwargs


class _CapturedStrivec(_CapturedGaussian):
    """Stands in for ``trellis.representations.Strivec`` (upstream builds it on ``cuda``)."""


class _CpuTorch:
    """``torch`` with ``full`` ignoring ``device``: upstream ``to_representation`` hard-codes ``device='cuda'``."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def full(*args, device=None, **kwargs):
        del device
        return torch.full(*args, **kwargs)


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_pinned_sparse_reference() -> types.SimpleNamespace:
    global _SPARSE_REFERENCE
    if _SPARSE_REFERENCE is not None:
        return _SPARSE_REFERENCE
    _load_pinned_reference()
    source_root = REFERENCE_ROOT / "trellis"

    os.environ["SPARSE_BACKEND"] = "spconv"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    modules_package = sys.modules[f"{REFERENCE_PACKAGE}.modules"]
    _package(f"{REFERENCE_PACKAGE}.utils", source_root / "utils")
    _package(f"{REFERENCE_PACKAGE}.models.structured_latent_vae", source_root / "models" / "structured_latent_vae")
    representations = _package(f"{REFERENCE_PACKAGE}.representations", source_root / "representations")
    representations.Gaussian = _CapturedGaussian
    representations.Strivec = _CapturedStrivec
    flexicubes_root = source_root / "representations" / "mesh" / "flexicubes"
    try:
        with _fake_backends():
            sparse = _load_module(
                f"{REFERENCE_PACKAGE}.modules.sparse", source_root / "modules" / "sparse" / "__init__.py", package=True
            )
            modules_package.sparse = sparse
            importlib.import_module(f"{REFERENCE_PACKAGE}.modules.sparse.attention")
            _load_module(f"{REFERENCE_PACKAGE}.utils.random_utils", source_root / "utils" / "random_utils.py")
            _load_module(f"{REFERENCE_PACKAGE}.utils.elastic_utils", source_root / "utils" / "elastic_utils.py")
            _load_module(
                f"{REFERENCE_PACKAGE}.models.sparse_elastic_mixin", source_root / "models" / "sparse_elastic_mixin.py"
            )
            flow_module = _load_module(
                f"{REFERENCE_PACKAGE}.models.structured_latent_flow",
                source_root / "models" / "structured_latent_flow.py",
            )
            _load_module(
                f"{REFERENCE_PACKAGE}.models.structured_latent_vae.base",
                source_root / "models" / "structured_latent_vae" / "base.py",
            )
            gs_module = _load_module(
                f"{REFERENCE_PACKAGE}.models.structured_latent_vae.decoder_gs",
                source_root / "models" / "structured_latent_vae" / "decoder_gs.py",
            )
            rf_module = _load_module(
                f"{REFERENCE_PACKAGE}.models.structured_latent_vae.decoder_rf",
                source_root / "models" / "structured_latent_vae" / "decoder_rf.py",
            )
            rf_module.torch = _CpuTorch()
            mesh_decoder = None
            if (flexicubes_root / "flexicubes.py").is_file():
                # The FlexiCubes submodule is not part of the superproject tree; pin it separately.
                revision = subprocess.run(
                    ["git", "-C", str(flexicubes_root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                if revision != FLEXICUBES_REFERENCE_REVISION:
                    raise ReferenceCheckoutError(
                        f"FlexiCubes submodule is at {revision!r}; expected {FLEXICUBES_REFERENCE_REVISION!r}"
                    )
                # FlexiCubes does not ignore ``__pycache__``; writing bytecode would dirty the pinned checkout.
                dont_write_bytecode = sys.dont_write_bytecode
                sys.dont_write_bytecode = True
                try:
                    mesh_representation = importlib.import_module(f"{REFERENCE_PACKAGE}.representations.mesh")
                finally:
                    sys.dont_write_bytecode = dont_write_bytecode
                representations.MeshExtractResult = mesh_representation.MeshExtractResult
                mesh_module = _load_module(
                    f"{REFERENCE_PACKAGE}.models.structured_latent_vae.decoder_mesh",
                    source_root / "models" / "structured_latent_vae" / "decoder_mesh.py",
                )
                mesh_decoder = types.SimpleNamespace(
                    module=mesh_module, SparseFeatures2Mesh=mesh_representation.SparseFeatures2Mesh
                )
    except (ImportError, RuntimeError) as error:
        _reference_unavailable(ReferenceCheckoutError(f"optional pinned reference dependency unavailable: {error}"))
    _SPARSE_REFERENCE = types.SimpleNamespace(
        sparse=sparse,
        SLatFlowModel=flow_module.SLatFlowModel,
        SLatGaussianDecoder=gs_module.SLatGaussianDecoder,
        SLatRadianceFieldDecoder=rf_module.SLatRadianceFieldDecoder,
        mesh_decoder=mesh_decoder,
    )
    return _SPARSE_REFERENCE


def _sparse_inputs(reference, coordinates: torch.Tensor, features: torch.Tensor):
    return reference.sparse.SparseTensor(features, coordinates.to(torch.int32)), TrellisSparseTensor(
        coordinates, features
    )


def test_slat_flow_io_stages_match_pinned_reference():
    reference = _load_pinned_sparse_reference()
    config = TrellisSLatFlowModel.small_config()
    generator = torch.Generator().manual_seed(40)
    # Two batches on an 8^3 grid; the IO stage pools them to 4^3 and back.
    coordinates = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 2, 3],
            [0, 7, 0, 5],
            [0, 7, 1, 5],
            [1, 2, 1, 0],
            [1, 3, 1, 0],
            [1, 7, 7, 7],
        ],
        dtype=torch.int64,
    )
    latents = torch.randn(coordinates.shape[0], 4, generator=generator, requires_grad=True)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = torch.randn(2, 6, 12, generator=generator)

    with _fake_backends():
        torch.manual_seed(41)
        reference_model = reference.SLatFlowModel(**config).eval()
        _randomize_state(reference_model, seed=42)
        model = TrellisSLatFlowModel(**config).eval()
        model.load_state_dict(reference_model.state_dict(), strict=True)
        assert set(model.state_dict()) == set(reference_model.state_dict())
        reference_x, x = _sparse_inputs(reference, coordinates, latents)
        expected = reference_model(reference_x, timesteps, context)
        assert torch.equal(expected.coords.to(torch.int64), coordinates)
        (expected_grad,) = torch.autograd.grad(expected.feats.square().sum(), latents, retain_graph=True)

    actual = model(x, timesteps, context).sample
    assert torch.equal(actual.coordinates, coordinates)
    torch.testing.assert_close(actual.features, expected.feats, atol=1e-6, rtol=1e-5)
    (actual_grad,) = torch.autograd.grad(actual.features.square().sum(), latents)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_swin_gaussian_decoder_matches_pinned_reference():
    reference = _load_pinned_sparse_reference()
    config = {**TrellisSLatGaussianDecoder.tiny_config(), "attn_mode": "swin", "window_size": 2}
    config["representation_config"] = {**config["representation_config"], "perturb_offset": True}
    generator = torch.Generator().manual_seed(50)
    coordinates = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 2, 3],
            [0, 7, 0, 5],
            [0, 7, 1, 5],
            [1, 2, 1, 0],
            [1, 3, 1, 0],
            [1, 7, 7, 7],
        ],
        dtype=torch.int64,
    )
    latents = torch.randn(coordinates.shape[0], 4, generator=generator)

    with _fake_backends():
        torch.manual_seed(51)
        reference_model = reference.SLatGaussianDecoder(**config).eval()
        _randomize_state(reference_model, seed=52)
        model = TrellisSLatGaussianDecoder(**config).eval()
        model.load_state_dict(reference_model.state_dict(), strict=True)
        assert set(model.state_dict()) == set(reference_model.state_dict())
        reference_x, x = _sparse_inputs(reference, coordinates, latents)
        with torch.no_grad():
            expected = reference_model(reference_x)
            actual = model(x).assets
    assert len(expected) == len(actual) == 2
    rep_config = config["representation_config"]
    for gaussian, asset in zip(expected, actual):
        torch.testing.assert_close(asset.means + 0.5, gaussian._xyz, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(asset.sh_coefficients, gaussian._features_dc, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            asset.extras["trellis_raw_scaling"] * rep_config["lr"]["_scaling"], gaussian._scaling, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            asset.extras["trellis_raw_rotation"] * rep_config["lr"]["_rotation"],
            gaussian._rotation,
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            asset.extras["trellis_raw_opacity"] * rep_config["lr"]["_opacity"], gaussian._opacity, atol=1e-6, rtol=1e-5
        )
        assert gaussian.config["scaling_bias"] == rep_config["scaling_bias"]


def test_swin_radiance_field_decoder_matches_pinned_reference():
    reference = _load_pinned_sparse_reference()
    config = {**TrellisSLatRadianceFieldDecoder.tiny_config(), "attn_mode": "swin", "window_size": 2}
    generator = torch.Generator().manual_seed(70)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 2, 3], [0, 7, 0, 5], [1, 2, 1, 0], [1, 3, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    latents = torch.randn(coordinates.shape[0], 4, generator=generator)

    with _fake_backends():
        torch.manual_seed(71)
        reference_model = reference.SLatRadianceFieldDecoder(**config).eval()
        _randomize_state(reference_model, seed=72)
        model = TrellisSLatRadianceFieldDecoder(**config).eval()
        model.load_state_dict(reference_model.state_dict(), strict=True)
        assert set(model.state_dict()) == set(reference_model.state_dict())
        reference_x, x = _sparse_inputs(reference, coordinates, latents)
        with torch.no_grad():
            expected = reference_model(reference_x)
            actual = model(x).assets
    assert len(expected) == len(actual) == 2
    rank, dim = config["representation_config"]["rank"], config["representation_config"]["dim"]
    for strivec, asset in zip(expected, actual):
        assert strivec.config["rank"] == rank and strivec.config["dim"] == dim
        assert strivec.config["aabb"] == [-0.5, -0.5, -0.5, 1, 1, 1] and strivec.density_shift == 0.0
        assert asset.density_shift == 0.0 and asset.resolution == config["resolution"]
        # Upstream stores unit-cube positions; the asset keeps grid coordinates plus a centre-mapping transform.
        centres = asset.coordinates.float() @ asset.grid_transform[:3, :3].T + asset.grid_transform[:3, 3]
        torch.testing.assert_close(centres + 0.5, strivec.position, atol=1e-6, rtol=0)
        assert bool((strivec.depth == 3).all())  # log2(8)
        torch.testing.assert_close(asset.trivec, strivec.trivec, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(asset.density, strivec.density, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            asset.color_coefficients, strivec.features_dc.reshape(-1, rank, 3), atol=1e-6, rtol=1e-5
        )


def _canonical_mesh(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor):
    """Sort vertices lexicographically and faces by their (rotated) index triple so vertex order does not matter."""

    order = torch.tensor(sorted(range(vertices.shape[0]), key=lambda index: tuple(vertices[index].tolist())))
    remap = torch.empty_like(order)
    remap[order] = torch.arange(order.numel())
    faces = remap[faces]
    first = faces.argmin(dim=1)
    rows = torch.arange(faces.shape[0])
    faces = torch.stack([faces[rows, (first + shift) % 3] for shift in range(3)], dim=1)
    face_order = torch.tensor(sorted(range(faces.shape[0]), key=lambda index: tuple(faces[index].tolist())))
    return vertices[order], faces[face_order], colors[order]


def _group_norm_with(cls: type, num_groups: int) -> type:
    """Upstream hardcodes 32 groups in the subdivide blocks; the tiny config needs fewer channels than that."""

    class _Patched(cls):
        def __init__(self, _num_groups, num_channels, *args, **kwargs):
            super().__init__(num_groups, num_channels, *args, **kwargs)

    return _Patched


def test_swin_mesh_decoder_matches_pinned_reference():
    reference = _load_pinned_sparse_reference()
    if reference.mesh_decoder is None:
        _reference_unavailable(
            ReferenceCheckoutError(
                "FlexiCubes submodule is not checked out; run `git submodule update --init` in the reference root"
            )
        )
    config = {**TrellisSLatMeshDecoder.tiny_config(), "attn_mode": "swin", "window_size": 2}
    generator = torch.Generator().manual_seed(60)
    coordinates = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 2, 3],
            [0, 3, 0, 2],
            [0, 3, 1, 2],
            [1, 2, 1, 0],
            [1, 3, 1, 0],
            [1, 3, 3, 3],
        ],
        dtype=torch.int64,
    )
    latents = torch.randn(coordinates.shape[0], 4, generator=generator)

    with _fake_backends():
        module = reference.mesh_decoder.module
        # Upstream hardcodes ``SparseFeatures2Mesh(device="cuda")``; the extractor is otherwise device-agnostic.
        module.SparseFeatures2Mesh = functools.partial(reference.mesh_decoder.SparseFeatures2Mesh, device="cpu")
        group_norm = module.sp.SparseGroupNorm32
        module.sp.SparseGroupNorm32 = _group_norm_with(group_norm, config["num_groups"])
        torch.manual_seed(61)
        try:
            reference_model = module.SLatMeshDecoder(**{k: v for k, v in config.items() if k != "num_groups"}).eval()
        finally:
            module.sp.SparseGroupNorm32 = group_norm
        _randomize_state(reference_model, seed=62)
        model = TrellisSLatMeshDecoder(**config).eval()
        model.load_state_dict(reference_model.state_dict(), strict=True)
        assert set(model.state_dict()) == set(reference_model.state_dict())
        reference_x, x = _sparse_inputs(reference, coordinates, latents)
        with torch.no_grad():
            expected = reference_model(reference_x)
            actual = model(x).assets
    assert len(expected) == len(actual) == 2
    for mesh, asset in zip(expected, actual):
        assert mesh.success and mesh.vertices.shape[0] == asset.vertices.shape[0]
        expected_vertices, expected_faces, expected_colors = _canonical_mesh(
            mesh.vertices, mesh.faces, mesh.vertex_attrs
        )
        colors = torch.cat([asset.colors, asset.extras["normal_map"]], dim=1)
        actual_vertices, actual_faces, actual_colors = _canonical_mesh(asset.vertices, asset.faces, colors)
        torch.testing.assert_close(actual_vertices, expected_vertices, atol=1e-6, rtol=1e-5)
        assert torch.equal(actual_faces, expected_faces)
        torch.testing.assert_close(actual_colors, expected_colors, atol=1e-6, rtol=1e-5)
        assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
