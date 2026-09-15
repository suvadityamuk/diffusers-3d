from __future__ import annotations

from typing import Any


def trellis2_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the schema-v2 pipeline and every reviewed component (sparse structure, SLAT, O-Voxel decoders)."""

    from .conditioner import Trellis2Dinov3Conditioner
    from .decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
    from .models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
    from .pipeline import Trellis2ImageTo3DPipeline

    models = (
        Trellis2SparseStructureFlowModel,
        Trellis2SparseStructureDecoder,
        Trellis2SLatFlowModel,
        Trellis2ShapeDualGridDecoder,
        Trellis2PBRSparseDecoder,
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
    """Register the FULL-only sparse-structure, shape-SLAT, and texture-SLAT flow recipes."""

    from ...execution.metadata import ReviewStatus
    from .training import (
        Trellis2ShapeSLatFlowRecipe,
        Trellis2SparseStructureFlowRecipe,
        Trellis2TextureSLatFlowRecipe,
    )

    return tuple(
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
        )
        for recipe in (Trellis2SparseStructureFlowRecipe, Trellis2ShapeSLatFlowRecipe, Trellis2TextureSLatFlowRecipe)
    )


__all__ = ["trellis2_execution_registrations", "trellis2_training_registrations"]
