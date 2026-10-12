from diffusers_3d import (
    Object3DModelRegistration,
    Object3DPipelineRegistration,
)

from .model import ReviewedDenoiser
from .pipeline import ReviewedObject3DPipeline

REVIEWED_MODEL_REGISTRATION = Object3DModelRegistration(
    model_class=ReviewedDenoiser,
    metadata=ReviewedDenoiser.object3d_metadata(),
)

REVIEWED_PIPELINE_REGISTRATION = Object3DPipelineRegistration(
    pipeline_class=ReviewedObject3DPipeline,
    metadata=ReviewedObject3DPipeline.object3d_model_index(),
)
