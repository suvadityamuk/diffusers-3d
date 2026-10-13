from __future__ import annotations

from typing import Any


def pixal3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the pipeline and the components Pixal3D adds; decoders and scheduler are TRELLIS.2's."""

    from .conditioner import Pixal3DNAFUpsampler
    from .models import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
    from .pipeline import Pixal3DImageTo3DPipeline

    models = (Pixal3DSparseStructureFlowModel, Pixal3DSLatFlowModel, Pixal3DNAFUpsampler)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (pipeline_registration_type(Pixal3DImageTo3DPipeline, Pixal3DImageTo3DPipeline.object3d_model_index()),),
    )


__all__ = ["pixal3d_execution_registrations"]
