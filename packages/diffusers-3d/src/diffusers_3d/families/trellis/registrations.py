from __future__ import annotations

from typing import Any


def trellis_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the image and text pipelines and every reviewed component (sparse structure, SLAT, three decoders)."""

    from .conditioner import TrellisDinov2Conditioner
    from .decoders import (
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
        TrellisSparseStructureDecoder,
    )
    from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
    from .pipeline import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
    from .text_conditioner import TrellisClipTextConditioner

    models = (
        TrellisSparseStructureFlowModel,
        TrellisSparseStructureDecoder,
        TrellisSLatFlowModel,
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
        TrellisDinov2Conditioner,
        TrellisClipTextConditioner,
    )
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                TrellisImageTo3DPipeline,
                TrellisImageTo3DPipeline.object3d_model_index(),
            ),
            pipeline_registration_type(
                TrellisTextTo3DPipeline,
                TrellisTextTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


__all__ = ["trellis_execution_registrations"]
