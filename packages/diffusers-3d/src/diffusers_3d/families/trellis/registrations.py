from __future__ import annotations

from typing import Any


def trellis_execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Register the image and text pipelines and every reviewed component (sparse structure, SLAT, three decoders)."""

    from .conditioner import TrellisDinov2Conditioner
    from .decoders import (
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
        TrellisSparseStructureDecoder,
    )
    from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
    from .pipeline import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
    from .text_conditioner import TrellisClipTextConditioner

    models = (
        TrellisSparseStructureFlowModel,
        TrellisSparseStructureDecoder,
        TrellisSLatFlowModel,
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
        TrellisDinov2Conditioner,
        TrellisClipTextConditioner,
    )
    return (
        tuple(model_registration_type(model_type, model_type.object3d_metadata()) for model_type in models),
        (
            pipeline_registration_type(
                TrellisImageTo3DPipeline,
                TrellisImageTo3DPipeline.object3d_model_index(),
            ),
            pipeline_registration_type(
                TrellisTextTo3DPipeline,
                TrellisTextTo3DPipeline.object3d_model_index(),
            ),
        ),
    )


def trellis_training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Register the FULL-only sparse-structure and SLAT flow recipes."""

    from ...execution.metadata import ReviewStatus
    from .training import TrellisSLatFlowRecipe, TrellisSparseStructureFlowRecipe

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
        for recipe in (TrellisSparseStructureFlowRecipe, TrellisSLatFlowRecipe)
    )


__all__ = ["trellis_execution_registrations", "trellis_training_registrations"]
