from __future__ import annotations

from diffusers_3d import (
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    TrellisTextTo3DPipeline,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_trellis_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.trellis.")


def test_registries_contain_exact_trellis_classes():
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen

    model_registrations = tuple(
        registration for registration in _MODEL_REGISTRY if _is_trellis_type(registration.model_class)
    )
    pipeline_registrations = tuple(
        registration for registration in _PIPELINE_REGISTRY if _is_trellis_type(registration.pipeline_class)
    )
    assert {registration.model_class for registration in model_registrations} == {
        TrellisSparseStructureDecoder,
        TrellisSparseStructureFlowModel,
        TrellisSLatFlowModel,
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
    }
    assert {registration.pipeline_class for registration in pipeline_registrations} == {
        TrellisImageTo3DPipeline,
        TrellisTextTo3DPipeline,
    }
    for pipeline_registration in pipeline_registrations:
        assert set(pipeline_registration.metadata.output_representations) == {
            "gaussian-splat",
            "mesh",
            "radiance-field",
            "slat",
            "sparse-structure",
        }
    assert (
        _PIPELINE_REGISTRY.resolve(TrellisImageTo3DPipeline.object3d_model_index(), "image-to-3d")
        is TrellisImageTo3DPipeline
    )
    assert (
        _PIPELINE_REGISTRY.resolve(TrellisTextTo3DPipeline.object3d_model_index(), "text-to-3d")
        is TrellisTextTo3DPipeline
    )
