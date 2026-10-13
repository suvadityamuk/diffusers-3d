from __future__ import annotations

from typing import Any


def triposg_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the TripoSG image-to-shape pipeline and its flow transformer and VAE decoder."""

    from .models import TripoSGTransformerModel
    from .pipeline import TripoSGImageTo3DPipeline
    from .vae import TripoSGVAEDecoder

    return (
        tuple(
            model_registration_type(model_type, model_type.object3d_metadata())
            for model_type in (TripoSGTransformerModel, TripoSGVAEDecoder)
        ),
        (pipeline_registration_type(TripoSGImageTo3DPipeline, TripoSGImageTo3DPipeline.object3d_model_index()),),
    )


__all__ = ["triposg_execution_registrations"]
