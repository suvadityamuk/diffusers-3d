from __future__ import annotations

from typing import Any


def reviewed_family_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the family's models and pipeline; add this function to ``families/registrations.py``."""

    from .model import ReviewedDenoiser
    from .pipeline import ReviewedObject3DPipeline

    return (
        (model_registration_type(ReviewedDenoiser, ReviewedDenoiser.object3d_metadata()),),
        (pipeline_registration_type(ReviewedObject3DPipeline, ReviewedObject3DPipeline.object3d_model_index()),),
    )
