from __future__ import annotations

from diffusers_3d import (
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_trellis2_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.trellis2.")


def test_registries_contain_exact_trellis2_classes():
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen

    model_registrations = tuple(
        registration for registration in _MODEL_REGISTRY if _is_trellis2_type(registration.model_class)
    )
    pipeline_registrations = tuple(
        registration for registration in _PIPELINE_REGISTRY if _is_trellis2_type(registration.pipeline_class)
    )
    assert {registration.model_class for registration in model_registrations} == {
        Trellis2SparseStructureDecoder,
        Trellis2SparseStructureFlowModel,
        Trellis2SLatFlowModel,
        Trellis2ShapeDualGridDecoder,
        Trellis2PBRSparseDecoder,
    }
    assert {registration.pipeline_class for registration in pipeline_registrations} == {Trellis2ImageTo3DPipeline}
    assert set(pipeline_registrations[0].metadata.output_representations) == {
        "mesh",
        "o-voxel",
        "slat",
        "sparse-structure",
    }
    assert (
        _PIPELINE_REGISTRY.resolve(
            Trellis2ImageTo3DPipeline.object3d_model_index(),
            "image-to-3d",
        )
        is Trellis2ImageTo3DPipeline
    )
