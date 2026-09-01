from __future__ import annotations

import ast
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
    TRELLIS_REFERENCE_REVISION,
    ImageCondition,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    preprocess_image_condition,
)
from diffusers_3d._reference import ReferenceCheckoutError, reference_unavailable, validate_reference_checkout

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
