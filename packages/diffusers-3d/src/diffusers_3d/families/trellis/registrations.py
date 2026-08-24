from __future__ import annotations

from typing import Any


def trellis_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register only the reviewed portable sparse-structure TRELLIS path."""

    from .conditioner import TrellisDinov2Conditioner
    from .decoders import TrellisSparseStructureDecoder
    from .models import TrellisSparseStructureFlowModel
    from .pipeline import TrellisImageTo3DPipeline

    models = (
        TrellisSparseStructureFlowModel,
        TrellisSparseStructureDecoder,
        TrellisDinov2Conditioner,
    )
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                TrellisImageTo3DPipeline,
                TrellisImageTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


def trellis_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Register the released-evidence FULL-only dense sparse-structure recipe."""

    from ...execution.metadata import ReviewStatus
    from .training import TrellisSparseStructureFlowRecipe

    recipe = TrellisSparseStructureFlowRecipe
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


__all__ = ["trellis_execution_registrations", "trellis_training_registrations"]
