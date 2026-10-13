from __future__ import annotations

from typing import Any


def sam3d_objects_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the image-to-3D pipeline and its new components; the reused TRELLIS stages are registered by TRELLIS."""

    from .conditioner import Sam3DConditionEmbedder
    from .models import Sam3DSparseStructureFlowModel
    from .pipeline import Sam3DObjectsPipeline

    models = (Sam3DSparseStructureFlowModel, Sam3DConditionEmbedder)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (pipeline_registration_type(Sam3DObjectsPipeline, Sam3DObjectsPipeline.object3d_model_index()),),
    )


__all__ = ["sam3d_objects_execution_registrations"]
