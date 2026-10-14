from __future__ import annotations

from typing import Any


def step1x3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the image-to-geometry and mesh-texturing pipelines and their package models."""

    from .models import Step1X3DTransformer1DModel
    from .pipeline import Step1X3DImageTo3DPipeline
    from .texture_attention import Step1X3DMultiViewAdapter
    from .texture_pipeline import Step1X3DMeshTexturePipeline
    from .vae import Step1X3DShapeDecoder

    models = (Step1X3DTransformer1DModel, Step1X3DShapeDecoder, Step1X3DMultiViewAdapter)
    pipelines = (Step1X3DImageTo3DPipeline, Step1X3DMeshTexturePipeline)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        tuple(pipeline_registration_type(pipeline, pipeline.object3d_model_index()) for pipeline in pipelines),
    )


__all__ = ["step1x3d_execution_registrations"]
