from __future__ import annotations

from typing import Any


def hunyuan3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the Hunyuan3D-2.1 shape and Paint pipelines and their reviewed components."""

    from .models import Hunyuan3DDiTModel
    from .paint_models import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
    from .paint_pipeline import Hunyuan3DPaintPipeline
    from .pipeline import Hunyuan3DImageTo3DPipeline
    from .vae import Hunyuan3DShapeVAE

    models = (Hunyuan3DDiTModel, Hunyuan3DShapeVAE, Hunyuan3DPaintUNet, Hunyuan3DPaintRRDBNet)
    pipelines = (Hunyuan3DImageTo3DPipeline, Hunyuan3DPaintPipeline)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        tuple(pipeline_registration_type(pipeline, pipeline.object3d_model_index()) for pipeline in pipelines),
    )


__all__ = ["hunyuan3d_execution_registrations"]
