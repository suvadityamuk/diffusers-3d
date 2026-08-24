from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from diffusers_3d import (
    BACKEND_REGISTRY,
    ContributionStatus,
    IntegrationManifest3D,
    ReviewStatus,
    validate_integration_manifest,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

pytestmark = pytest.mark.release

PACKAGE_ROOT = Path(__file__).parents[2]
FAMILIES_ROOT = PACKAGE_ROOT / "src" / "diffusers_3d" / "families"
ALLOWED_WARNING_CODES = {
    "backend.research_only",
    "backend.restricted_license",
    "licenses.restricted",
}


def _qualified_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _resolve_qualified_name(name: str):
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[index:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(f"could not resolve qualified name {name!r}")


def _test_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _shipped_manifests() -> tuple[IntegrationManifest3D, ...]:
    paths = tuple(sorted(FAMILIES_ROOT.glob("*/diffusers_3d_integration.json")))
    assert paths
    return tuple(IntegrationManifest3D.load(path) for path in paths)


def test_every_shipped_family_manifest_validates_and_references_executable_evidence():
    manifests = _shipped_manifests()
    assert {manifest.integration_id for manifest in manifests} == {"hunyuan3d-2.1", "trellis", "trellis2"}
    test_functions: dict[Path, set[str]] = {}

    for manifest in manifests:
        report = validate_integration_manifest(manifest)
        assert report.is_valid
        assert {warning.code for warning in report.warnings}.issubset(ALLOWED_WARNING_CODES)

        evidence_nodes = []
        for component in manifest.components:
            resolved_class = _resolve_qualified_name(component.class_name)
            assert resolved_class.__module__ in component.class_name
            if component.checkpoint_conversion is not None:
                assert callable(_resolve_qualified_name(component.checkpoint_conversion.converter))
                evidence_nodes.append(component.checkpoint_conversion.test)
            evidence_nodes.extend(evidence.test for evidence in component.parity)
        if manifest.training is not None:
            assert callable(_resolve_qualified_name(manifest.training.trainer_registration))
            evidence_nodes.extend(
                evidence.test
                for evidence in (
                    manifest.training.backward_parity,
                    manifest.training.checkpoint_parity,
                    manifest.training.objective_parity,
                )
                if evidence is not None
            )

        for node_id in evidence_nodes:
            relative_path, separator, function_name = node_id.partition("::")
            assert separator and function_name.startswith("test_")
            test_path = PACKAGE_ROOT / relative_path
            assert test_path.is_file(), node_id
            functions = test_functions.setdefault(test_path, _test_function_names(test_path))
            assert function_name in functions, node_id


def test_reviewed_execution_and_training_registries_have_exact_manifest_evidence():
    manifests = _shipped_manifests()
    manifests_by_family = {manifest.integration_id: manifest for manifest in manifests}
    components_by_class = {
        component.class_name: component for manifest in manifests for component in manifest.components
    }

    for registration in (*_MODEL_REGISTRY.list(), *_PIPELINE_REGISTRY.list()):
        registered_type = getattr(registration, "model_class", None) or registration.pipeline_class
        class_name = _qualified_name(registered_type)
        component = components_by_class[class_name]
        assert component.checkpoint_conversion is not None
        assert component.parity
        assert registered_type.contribution_status is ContributionStatus.REVIEWED_PACKAGE
        assert registered_type.review_status is ReviewStatus.REVIEWED
        assert registration.metadata.review_status is ReviewStatus.REVIEWED

    for registration in _TRAINING_RECIPE_REGISTRY:
        qualification = manifests_by_family[registration.family_id].training
        assert qualification is not None
        assert qualification.recipe_class == _qualified_name(registration.recipe_type)
        assert qualification.target_class == _qualified_name(registration.target_type)
        assert qualification.example_class == _qualified_name(registration.example_type)
        assert qualification.batch_class == _qualified_name(registration.batch_type)
        assert qualification.recipe_id == registration.recipe_id
        assert qualification.recipe_version == registration.recipe_version
        assert set(qualification.components) == {policy.key for policy in registration.component_policies}
        assert registration.review_status is ReviewStatus.REVIEWED
        assert registration.target_type.review_status is ReviewStatus.REVIEWED

    for manifest in manifests:
        for requirement in manifest.backends:
            if requirement.name in BACKEND_REGISTRY:
                spec = BACKEND_REGISTRY.get(requirement.name)
                assert requirement.support_level is spec.support_level
                assert requirement.license_class is spec.license_class
