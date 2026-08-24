from __future__ import annotations

from pathlib import Path

from diffusers_3d import (
    BackendLicenseClass,
    Hunyuan3DDinov2Conditioner,
    Hunyuan3DImageToShapePipeline,
    Hunyuan3DShapeDiTModel,
    Hunyuan3DShapeFlowMatchingRecipe,
    Hunyuan3DShapeVAE,
    IntegrationManifest3D,
    validate_integration_manifest,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

FAMILY_ROOT = Path(__file__).parents[3] / "src" / "diffusers_3d" / "families" / "hunyuan3d"


def _qualified_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _is_hunyuan_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.hunyuan3d.")


def test_hunyuan_manifest_is_valid_with_expected_restricted_license_warnings():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    report = validate_integration_manifest(manifest)
    assert report.is_valid, report.to_dict()
    assert not report.errors
    assert {warning.code for warning in report.warnings} == {"licenses.restricted"}

    assert manifest.upstream.revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert manifest.licenses.model.classification is BackendLicenseClass.RESTRICTED
    licenses = {artifact.artifact: artifact.license for artifact in manifest.licenses.artifacts}
    assert licenses["hunyuan-derived-code"].classification is BackendLicenseClass.RESTRICTED
    assert licenses["package-glue-code"].classification is BackendLicenseClass.PERMISSIVE
    pipeline = next(component for component in manifest.components if component.role == "pipeline")
    assert pipeline.parity[0].passed
    assert "surface extraction and official-checkpoint quality are excluded" in pipeline.parity[0].reference
    assert (FAMILY_ROOT / "LICENSE-TENCENT-HUNYUAN-3D-2.1").is_file()
    assert (FAMILY_ROOT / "NOTICE").is_file()


def test_manifest_matches_exact_production_registrations_and_evidence():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    component_classes = {component.class_name for component in manifest.components}
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen
    assert _TRAINING_RECIPE_REGISTRY.frozen
    registered_model_types = {
        registration.model_class for registration in _MODEL_REGISTRY if _is_hunyuan_type(registration.model_class)
    }
    registered_pipeline_types = {
        registration.pipeline_class
        for registration in _PIPELINE_REGISTRY
        if _is_hunyuan_type(registration.pipeline_class)
    }
    registered_recipe_types = {
        registration.recipe_type
        for registration in _TRAINING_RECIPE_REGISTRY
        if _is_hunyuan_type(registration.recipe_type)
    }
    assert registered_model_types == {
        Hunyuan3DDinov2Conditioner,
        Hunyuan3DShapeDiTModel,
        Hunyuan3DShapeVAE,
    }
    assert registered_pipeline_types == {Hunyuan3DImageToShapePipeline}
    assert registered_recipe_types == {Hunyuan3DShapeFlowMatchingRecipe}
    registered_model_classes = {
        registration.metadata.model_class
        for registration in _MODEL_REGISTRY
        if _is_hunyuan_type(registration.model_class)
    }
    registered_pipeline_classes = {
        registration.metadata.pipeline_class
        for registration in _PIPELINE_REGISTRY
        if _is_hunyuan_type(registration.pipeline_class)
    }
    assert registered_model_classes.issubset(component_classes)
    assert registered_pipeline_classes == {_qualified_name(Hunyuan3DImageToShapePipeline)}
    assert (
        _PIPELINE_REGISTRY.resolve(
            Hunyuan3DImageToShapePipeline.object3d_model_index(),
            "image-to-3d",
        )
        is Hunyuan3DImageToShapePipeline
    )

    training = manifest.training
    registration = _TRAINING_RECIPE_REGISTRY.resolve(
        Hunyuan3DShapeFlowMatchingRecipe,
        Hunyuan3DImageToShapePipeline,
        Hunyuan3DShapeFlowMatchingRecipe.recipe_id,
    )
    assert training.recipe_class == _qualified_name(registration.recipe_type)
    assert training.target_class == _qualified_name(registration.target_type)
    assert training.example_class == _qualified_name(registration.example_type)
    assert training.batch_class == _qualified_name(registration.batch_type)
    assert training.recipe_version == registration.recipe_version
    assert training.components == tuple(policy.key for policy in registration.component_policies)
    assert training.backward_parity.test.endswith("::test_tiny_denoiser_backward_matches_pinned_reference")
    assert training.checkpoint_parity.test.endswith("::test_object3d_trainer_full_step_and_checkpoint_roundtrip")
