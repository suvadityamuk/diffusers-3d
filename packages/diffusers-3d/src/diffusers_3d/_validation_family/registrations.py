# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

from typing import Any


def execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build the exact reviewed execution registrations without import cycles."""

    from .models import ContractReferenceDenoiser, ContractReferenceMeshDecoder
    from .pipeline import ContractReferencePipeline

    model_registrations = tuple(
        model_registration_type(model_type, model_type.object3d_metadata())
        for model_type in (ContractReferenceDenoiser, ContractReferenceMeshDecoder)
    )
    pipeline_registrations = (
        pipeline_registration_type(
            ContractReferencePipeline,
            ContractReferencePipeline.object3d_model_index(),
        ),
    )
    return model_registrations, pipeline_registrations


def training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Build the exact reviewed recipe registration without public mutation."""

    from ..execution.metadata import ReviewStatus
    from .training import ContractReferenceRecipe

    return (
        training_registration_type(
            recipe_type=ContractReferenceRecipe,
            target_type=ContractReferenceRecipe.target_type,
            batch_type=ContractReferenceRecipe.batch_type,
            recipe_id=ContractReferenceRecipe.recipe_id,
            recipe_version=ContractReferenceRecipe.recipe_version,
            family_id=ContractReferenceRecipe.family_id,
            component_policies=ContractReferenceRecipe.component_policies,
            review_status=ReviewStatus.REVIEWED,
        ),
    )


__all__ = ["execution_registrations", "training_registrations"]
