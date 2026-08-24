from __future__ import annotations

from typing import Any


def hunyuan3d_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build exact reviewed Hunyuan3D execution registrations."""

    from .conditioner import Hunyuan3DDinov2Conditioner
    from .models import Hunyuan3DShapeDiTModel
    from .pipeline import Hunyuan3DImageToShapePipeline
    from .vae import Hunyuan3DShapeVAE

    models = (Hunyuan3DShapeDiTModel, Hunyuan3DShapeVAE, Hunyuan3DDinov2Conditioner)
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                Hunyuan3DImageToShapePipeline,
                Hunyuan3DImageToShapePipeline.object3d_model_index(),
            ),
        ),
    )


def hunyuan3d_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Build the exact reviewed released-evidence training registration."""

    from ...execution.metadata import ReviewStatus
    from .training import Hunyuan3DShapeFlowMatchingRecipe

    recipe = Hunyuan3DShapeFlowMatchingRecipe
    return (
        training_registration_type(
            recipe_type=recipe,
            target_type=recipe.target_type,
            example_type=recipe.example_type,
            batch_type=recipe.batch_type,
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.recipe_version,
            family_id=recipe.family_id,
            component_policies=recipe.component_policies,
            review_status=ReviewStatus.REVIEWED,
        ),
    )


__all__ = ["hunyuan3d_execution_registrations", "hunyuan3d_training_registrations"]
