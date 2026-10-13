from __future__ import annotations

from typing import Any


def hunyuan3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the Hunyuan3D-2.1 image-to-shape pipeline and its reviewed components."""

    from .models import Hunyuan3DDiTModel
    from .pipeline import Hunyuan3DImageTo3DPipeline
    from .vae import Hunyuan3DShapeVAE

    models = (Hunyuan3DDiTModel, Hunyuan3DShapeVAE)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                Hunyuan3DImageTo3DPipeline,
                Hunyuan3DImageTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


__all__ = ["hunyuan3d_execution_registrations"]
