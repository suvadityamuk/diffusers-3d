from __future__ import annotations

from collections import Counter
from pathlib import Path

from diffusers_3d import (
    DEFAULT_FORBIDDEN_MARKER,
    IntegrationManifest3D,
    scan_forbidden_marker,
    validate_integration_manifest,
)

PACKAGE_ROOT = Path(__file__).parents[2]
TEMPLATES = PACKAGE_ROOT / "templates"


def test_template_manifests_are_strict_and_have_expected_policy_status():
    manifest_paths = sorted(TEMPLATES.glob("*/integration_manifest.json"))

    assert {path.parent.name for path in manifest_paths} == {
        "experimental-custom-block",
        "reviewed-model-family",
    }
    manifests = {path.parent.name: IntegrationManifest3D.load(path) for path in manifest_paths}

    experimental_report = validate_integration_manifest(manifests["experimental-custom-block"])
    assert experimental_report.is_valid, experimental_report.to_dict()

    reviewed = manifests["reviewed-model-family"]
    reviewed_report = validate_integration_manifest(reviewed)
    assert not reviewed_report.is_valid
    assert Counter(issue.code for issue in reviewed_report.errors) == Counter(
        {
            "parity.failed": 2,
            "training.failed_backward_parity": 1,
            "training.failed_checkpoint_parity": 1,
            "training.failed_objective_parity": 1,
        }
    )
    component_evidence = tuple(evidence for component in reviewed.components for evidence in component.parity)
    training_evidence = (
        reviewed.training.backward_parity,
        reviewed.training.checkpoint_parity,
        reviewed.training.objective_parity,
    )
    all_evidence = (*component_evidence, *training_evidence)
    assert all(evidence is not None and not evidence.passed for evidence in all_evidence)
    assert all(evidence is not None and evidence.reference.startswith("NOT RUN:") for evidence in all_evidence)


def test_template_python_skeletons_compile():
    python_paths = sorted(TEMPLATES.rglob("*.py"))

    assert python_paths
    for path in python_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_templates_are_release_marker_clean():
    report = scan_forbidden_marker((TEMPLATES,), marker=DEFAULT_FORBIDDEN_MARKER)

    assert report.is_clean
    assert report.scanned_files > 0
