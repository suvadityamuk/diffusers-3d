from __future__ import annotations

from typing import Any


def trellis2_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register only the reviewed portable TRELLIS.2 sparse-structure path."""

    from .conditioner import Trellis2Dinov3Conditioner
    from .decoders import Trellis2SparseStructureDecoder
    from .models import Trellis2SparseStructureFlowModel
    from .pipeline import Trellis2ImageTo3DPipeline

    models = (
        Trellis2SparseStructureFlowModel,
        Trellis2SparseStructureDecoder,
        Trellis2Dinov3Conditioner,
    )
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                Trellis2ImageTo3DPipeline,
                Trellis2ImageTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


def trellis2_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Register the released-evidence FULL-only sparse-structure flow recipe."""

    from ...execution.metadata import ReviewStatus
    from .training import Trellis2SparseStructureFlowRecipe

    recipe = Trellis2SparseStructureFlowRecipe
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
            frozen_component_policies=recipe.frozen_component_policies,
        ),
    )


__all__ = ["trellis2_execution_registrations", "trellis2_training_registrations"]
