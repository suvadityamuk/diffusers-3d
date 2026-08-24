from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from diffusers_3d import (
    TRELLIS2_REFERENCE_REVISION,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
)

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path("/tmp/TRELLIS.2")
REFERENCE_PACKAGE = "_diffusers_3d_trellis2_reference"
_REFERENCE_TYPES: tuple[type[torch.nn.Module], type[torch.nn.Module]] | None = None


def _assert_pinned_revision() -> None:
    git_directory = REFERENCE_ROOT / ".git"
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        pytest.skip("the TRELLIS.2 reference checkout has no verifiable revision metadata")
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        revision_path = git_directory / reference
        if revision_path.is_file():
            head = revision_path.read_text(encoding="utf-8").strip()
        else:
            packed_refs = git_directory / "packed-refs"
            if not packed_refs.is_file():
                pytest.skip("the TRELLIS.2 reference checkout revision is not resolvable")
            revisions = {
                name: revision
                for revision, name in (
                    line.split(" ", maxsplit=1)
                    for line in packed_refs.read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith(("#", "^"))
                )
            }
            head = revisions.get(reference, "")
    assert head == TRELLIS2_REFERENCE_REVISION


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
    if not source_root.is_dir():
        pytest.skip("the pinned TRELLIS.2 reference checkout is unavailable")
    _assert_pinned_revision()

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
        pytest.skip(f"optional pinned reference dependency unavailable: {error}")
    _REFERENCE_TYPES = flow_module.SparseStructureFlowModel, decoder_module.SparseStructureDecoder
    return _REFERENCE_TYPES


def _randomize_state(module: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype) * 0.02)


def test_tiny_sparse_structure_components_match_pinned_reference():
    reference_flow_type, reference_decoder_type = _load_pinned_reference()
    flow_config = Trellis2SparseStructureFlowModel.tiny_config()
    torch.manual_seed(0)
    reference_flow = reference_flow_type(**flow_config).eval()
    _randomize_state(reference_flow, seed=1)
    flow = Trellis2SparseStructureFlowModel(**flow_config).eval()
    flow.load_state_dict(reference_flow.state_dict(), strict=True)
    assert tuple(flow.state_dict()) == tuple(reference_flow.state_dict())

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
    model.load_state_dict(reference.state_dict(), strict=True)

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
