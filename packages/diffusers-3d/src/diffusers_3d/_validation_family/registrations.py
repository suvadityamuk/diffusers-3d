# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

from typing import Any


def execution_registrations(
    model_registration_type: type[Any],
    pipeline_registration_type: type[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Build validation and production execution registrations without import cycles."""

    from ..families.hunyuan3d.registrations import hunyuan3d_execution_registrations
    from .models import ContractReferenceDenoiser, ContractReferenceMeshDecoder
    from .pipeline import ContractReferencePipeline

    validation_models = tuple(
        model_registration_type(model_type, model_type.object3d_metadata())
        for model_type in (ContractReferenceDenoiser, ContractReferenceMeshDecoder)
    )
    validation_pipelines = (
        pipeline_registration_type(
            ContractReferencePipeline,
            ContractReferencePipeline.object3d_model_index(),
        ),
    )
    family_models, family_pipelines = hunyuan3d_execution_registrations(
        model_registration_type,
        pipeline_registration_type,
    )
    return validation_models + family_models, validation_pipelines + family_pipelines


def training_registrations(training_registration_type: type[Any]) -> tuple[Any, ...]:
    """Build validation and production recipe registrations without public mutation."""

    from ..execution.metadata import ReviewStatus
    from ..families.hunyuan3d.registrations import hunyuan3d_training_registrations
    from .training import ContractReferenceRecipe

    validation_registrations = (
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
    return validation_registrations + hunyuan3d_training_registrations(training_registration_type)


__all__ = ["execution_registrations", "training_registrations"]
