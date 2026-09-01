from __future__ import annotations

from pathlib import Path

from diffusers_3d import (
    TRELLIS2_REFERENCE_REVISION,
    BackendLicenseClass,
    BackendSupportLevel,
    IntegrationManifest3D,
    Trellis2Dinov3Conditioner,
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2ShapeSLatFlowRecipe,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
    Trellis2SparseStructureFlowRecipe,
    Trellis2TextureSLatFlowRecipe,
    validate_integration_manifest,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

FAMILY_ROOT = Path(__file__).parents[3] / "src" / "diffusers_3d" / "families" / "trellis2"


def _qualified_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _is_trellis2_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.trellis2.")


def test_trellis2_schema_v2_manifest_records_exact_source_license_and_capability_boundaries():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    report = validate_integration_manifest(manifest)
    assert report.is_valid, report.to_dict()
    assert not report.errors
    assert {warning.code for warning in report.warnings} == {
        "backend.research_only",
        "backend.restricted_license",
        "licenses.restricted",
    }

    assert manifest.schema_version == 2
    assert manifest.upstream.revision == TRELLIS2_REFERENCE_REVISION
    assert manifest.licenses.model.classification is BackendLicenseClass.PERMISSIVE
    artifacts = {artifact.artifact: artifact.license for artifact in manifest.licenses.artifacts}
    assert artifacts["trellis2-derived-code"].identifier == "MIT"
    assert artifacts["converted-checkpoints"].identifier == "MIT"
    assert artifacts["package-glue-code"].identifier == "Apache-2.0"
    assert artifacts["dinov3-conditioner-checkpoints"].classification is BackendLicenseClass.RESTRICTED

    backends = {backend.name: backend for backend in manifest.backends}
    for name in ("cumesh", "flex_gemm", "o_voxel"):
        assert backends[name].license_class is BackendLicenseClass.PERMISSIVE
        assert backends[name].support_level is BackendSupportLevel.ACCELERATED
        assert not backends[name].required
    assert backends["o_voxel"].source.revision == TRELLIS2_REFERENCE_REVISION
    assert backends["nvdiffrast"].license_class is BackendLicenseClass.RESTRICTED
    assert backends["nvdiffrast"].support_level is BackendSupportLevel.RESEARCH_ONLY
    assert backends["nvdiffrast"].source.revision == "253ac4fcea7de5f396371124af597e6cc957bfae"
    assert backends["pillow"].license_class is BackendLicenseClass.PERMISSIVE
    assert backends["pillow"].support_level is BackendSupportLevel.PORTABLE
    assert backends["pillow"].required

    pipeline = next(component for component in manifest.components if component.role == "pipeline")
    execution_evidence = next(evidence for evidence in pipeline.parity if "test_pipeline.py" in evidence.test)
    preprocessing_evidence = next(
        evidence for evidence in pipeline.parity if "test_image_processing.py" in evidence.test
    )
    assert execution_evidence.passed
    assert (
        "production SLAT cascade, O-Voxel conversion, rendering, and quality are excluded"
        in execution_evidence.reference
    )
    assert preprocessing_evidence.passed
    assert "1.0-scale square recenter crop" in preprocessing_evidence.reference
    for name in ("LICENSE-MIT", "NOTICE", "README.md"):
        assert (FAMILY_ROOT / name).is_file()


def test_manifest_matches_only_reviewed_trellis2_registrations_and_training():
    manifest = IntegrationManifest3D.load(FAMILY_ROOT / "diffusers_3d_integration.json")
    components = {component.role: component.class_name for component in manifest.components}
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen
    assert _TRAINING_RECIPE_REGISTRY.frozen

    model_registrations = tuple(
        registration for registration in _MODEL_REGISTRY if _is_trellis2_type(registration.model_class)
    )
    pipeline_registrations = tuple(
        registration for registration in _PIPELINE_REGISTRY if _is_trellis2_type(registration.pipeline_class)
    )
    recipe_registrations = tuple(
        registration for registration in _TRAINING_RECIPE_REGISTRY if _is_trellis2_type(registration.recipe_type)
    )
    assert {registration.model_class for registration in model_registrations} == {
        Trellis2Dinov3Conditioner,
        Trellis2SparseStructureDecoder,
        Trellis2SparseStructureFlowModel,
    }
    assert {registration.pipeline_class for registration in pipeline_registrations} == {Trellis2ImageTo3DPipeline}
    assert {registration.recipe_type for registration in recipe_registrations} == {Trellis2SparseStructureFlowRecipe}

    registered_models = {registration.model_class for registration in _MODEL_REGISTRY}
    registered_recipes = {registration.recipe_type for registration in _TRAINING_RECIPE_REGISTRY}
    for experimental in (Trellis2SLatFlowModel, Trellis2ShapeDualGridDecoder, Trellis2PBRSparseDecoder):
        assert experimental not in registered_models
    for experimental in (Trellis2ShapeSLatFlowRecipe, Trellis2TextureSLatFlowRecipe):
        assert experimental not in registered_recipes

    for registration in model_registrations:
        assert components[registration.metadata.component_role] == registration.metadata.model_class
    pipeline_registration = pipeline_registrations[0]
    assert components["pipeline"] == pipeline_registration.metadata.pipeline_class
    assert pipeline_registration.metadata.output_representations == ("sparse-structure",)
    assert (
        _PIPELINE_REGISTRY.resolve(
            Trellis2ImageTo3DPipeline.object3d_model_index(),
            "image-to-3d",
        )
        is Trellis2ImageTo3DPipeline
    )

    training = manifest.training
    registration = _TRAINING_RECIPE_REGISTRY.resolve(
        Trellis2SparseStructureFlowRecipe,
        Trellis2ImageTo3DPipeline,
        Trellis2SparseStructureFlowRecipe.recipe_id,
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
        "::test_sparse_structure_recipe_registration_collation_full_step_and_checkpoint"
    )
