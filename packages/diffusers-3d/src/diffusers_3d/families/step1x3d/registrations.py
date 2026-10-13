from __future__ import annotations

from typing import Any


def step1x3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the image-to-geometry pipeline and its flow transformer and shape decoder."""

    from .models import Step1X3DTransformer1DModel
    from .pipeline import Step1X3DImageTo3DPipeline
    from .vae import Step1X3DShapeDecoder

    models = (Step1X3DTransformer1DModel, Step1X3DShapeDecoder)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (pipeline_registration_type(Step1X3DImageTo3DPipeline, Step1X3DImageTo3DPipeline.object3d_model_index()),),
    )


__all__ = ["step1x3d_execution_registrations"]
