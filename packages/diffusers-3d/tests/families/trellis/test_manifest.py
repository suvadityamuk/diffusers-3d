from __future__ import annotations

from pathlib import Path

from diffusers_3d import (
    BackendLicenseClass,
    BackendSupportLevel,
    IntegrationManifest3D,
    TrellisDinov2Conditioner,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatFlowRecipe,
    TrellisSLatGaussianDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    TrellisSparseStructureFlowRecipe,
    validate_integration_manifest,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

FAMILY_ROOT = Path(__file__).parents[3] / "src" / "diffusers_3d" / "families" / "trellis"
REFERENCE_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"


def _qualified_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _is_trellis_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.trellis.")


def test_trellis_manifest_is_valid_and_separates_research_backend_warnings():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    report = validate_integration_manifest(manifest)
    assert report.is_valid, report.to_dict()
    assert not report.errors
    assert {warning.code for warning in report.warnings} == {
        "backend.research_only",
        "backend.restricted_license",
    }

    assert manifest.upstream.revision == REFERENCE_REVISION
    assert manifest.licenses.model.classification is BackendLicenseClass.PERMISSIVE
    artifacts = {artifact.artifact: artifact.license for artifact in manifest.licenses.artifacts}
    assert artifacts["trellis-derived-code"].identifier == "MIT"
    assert artifacts["converted-checkpoints"].identifier == "MIT"
    assert artifacts["package-glue-code"].identifier == "Apache-2.0"

    backends = {backend.name: backend for backend in manifest.backends}
    for name in ("gsplat", "kaolin", "spconv"):
        assert backends[name].license_class is BackendLicenseClass.PERMISSIVE
        assert backends[name].support_level is BackendSupportLevel.ACCELERATED
    for name in ("diffoctreerast", "mip_gaussian", "nvdiffrast"):
        assert backends[name].license_class is BackendLicenseClass.RESTRICTED
        assert backends[name].support_level is BackendSupportLevel.RESEARCH_ONLY
        assert not backends[name].required
        assert backends[name].source is not None
    assert backends["pillow"].license_class is BackendLicenseClass.PERMISSIVE
    assert backends["pillow"].support_level is BackendSupportLevel.PORTABLE
    assert backends["pillow"].required

    pipeline = next(component for component in manifest.components if component.role == "pipeline")
    execution_evidence = next(evidence for evidence in pipeline.parity if "test_pipeline.py" in evidence.test)
    preprocessing_evidence = next(
        evidence for evidence in pipeline.parity if "test_image_processing.py" in evidence.test
    )
    assert execution_evidence.passed
    assert "SLAT and rendering quality are excluded" in execution_evidence.reference
    assert preprocessing_evidence.passed
    assert "1.2-scale square recenter crop" in preprocessing_evidence.reference
    assert (FAMILY_ROOT / "LICENSE-MIT").is_file()
    assert (FAMILY_ROOT / "NOTICE").is_file()
    assert (FAMILY_ROOT / "README.md").is_file()


def test_manifest_matches_exact_reviewed_trellis_registrations():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    components = {component.role: component.class_name for component in manifest.components}
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen
    assert _TRAINING_RECIPE_REGISTRY.frozen

    model_registrations = tuple(
        registration for registration in _MODEL_REGISTRY if _is_trellis_type(registration.model_class)
    )
    pipeline_registrations = tuple(
        registration for registration in _PIPELINE_REGISTRY if _is_trellis_type(registration.pipeline_class)
    )
    recipe_registrations = tuple(
        registration for registration in _TRAINING_RECIPE_REGISTRY if _is_trellis_type(registration.recipe_type)
    )
    assert {registration.model_class for registration in model_registrations} == {
        TrellisDinov2Conditioner,
        TrellisSparseStructureDecoder,
        TrellisSparseStructureFlowModel,
    }
    assert {registration.pipeline_class for registration in pipeline_registrations} == {TrellisImageTo3DPipeline}
    assert {registration.recipe_type for registration in recipe_registrations} == {TrellisSparseStructureFlowRecipe}
    assert TrellisSLatFlowModel not in {registration.model_class for registration in _MODEL_REGISTRY}
    assert TrellisSLatGaussianDecoder not in {registration.model_class for registration in _MODEL_REGISTRY}
    assert TrellisSLatFlowRecipe not in {registration.recipe_type for registration in _TRAINING_RECIPE_REGISTRY}

    for registration in model_registrations:
        assert components[registration.metadata.component_role] == registration.metadata.model_class
    pipeline_registration = pipeline_registrations[0]
    assert components["pipeline"] == pipeline_registration.metadata.pipeline_class
    assert pipeline_registration.metadata.output_representations == ("sparse-structure",)
    assert (
        _PIPELINE_REGISTRY.resolve(
            TrellisImageTo3DPipeline.object3d_model_index(),
            "image-to-3d",
        )
        is TrellisImageTo3DPipeline
    )

    training = manifest.training
    registration = _TRAINING_RECIPE_REGISTRY.resolve(
        TrellisSparseStructureFlowRecipe,
        TrellisImageTo3DPipeline,
        TrellisSparseStructureFlowRecipe.recipe_id,
    )
    assert training.recipe_class == _qualified_name(registration.recipe_type)
    assert training.target_class == _qualified_name(registration.target_type)
    assert training.example_class == _qualified_name(registration.example_type)
    assert training.batch_class == _qualified_name(registration.batch_type)
    assert training.recipe_version == registration.recipe_version
    assert training.components == tuple(policy.key for policy in registration.component_policies)
    assert training.backward_parity.test.endswith(
        "::test_tiny_sparse_structure_flow_backward_matches_pinned_reference"
    )
    assert training.checkpoint_parity.test.endswith(
        "::test_sparse_structure_recipe_registration_collation_and_full_trainer_checkpoint"
    )
