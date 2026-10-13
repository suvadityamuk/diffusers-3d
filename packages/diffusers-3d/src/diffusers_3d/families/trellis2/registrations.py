from __future__ import annotations

from typing import Any


def trellis2_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the schema-v2 pipeline and every reviewed component (sparse structure, SLAT, O-Voxel decoders)."""

    from .decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
    from .models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
    from .pipeline import Trellis2ImageTo3DPipeline

    models = (
        Trellis2SparseStructureFlowModel,
        Trellis2SparseStructureDecoder,
        Trellis2SLatFlowModel,
        Trellis2ShapeDualGridDecoder,
        Trellis2PBRSparseDecoder,
    )
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                Trellis2ImageTo3DPipeline,
                Trellis2ImageTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


__all__ = ["trellis2_execution_registrations"]
