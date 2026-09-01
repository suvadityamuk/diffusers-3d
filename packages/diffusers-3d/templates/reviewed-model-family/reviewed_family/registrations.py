from diffusers_3d import (
    Object3DExample,
    Object3DModelRegistration,
    Object3DPipelineRegistration,
    ReviewStatus,
    TrainingRecipeRegistration,
)

from .model import ReviewedDenoiser
from .pipeline import ReviewedObject3DPipeline
from .training_recipe import REVIEWED_DENOISER_POLICY, ReviewedBatch, ReviewedTrainingRecipe

REVIEWED_MODEL_REGISTRATION = Object3DModelRegistration(
    model_class=ReviewedDenoiser,
    metadata=ReviewedDenoiser.object3d_metadata(),
)

REVIEWED_PIPELINE_REGISTRATION = Object3DPipelineRegistration(
    pipeline_class=ReviewedObject3DPipeline,
    metadata=ReviewedObject3DPipeline.object3d_model_index(),
)

REVIEWED_TRAINING_REGISTRATION = TrainingRecipeRegistration(
    recipe_type=ReviewedTrainingRecipe,
    target_type=ReviewedObject3DPipeline,
    example_type=Object3DExample,
    batch_type=ReviewedBatch,
    recipe_id=ReviewedTrainingRecipe.recipe_id,
    recipe_version=ReviewedTrainingRecipe.recipe_version,
    family_id=ReviewedTrainingRecipe.family_id,
    component_policies=(REVIEWED_DENOISER_POLICY,),
    review_status=ReviewStatus.REVIEWED,
    frozen_component_policies=ReviewedTrainingRecipe.frozen_component_policies,
)
