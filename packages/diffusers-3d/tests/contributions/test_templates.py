from __future__ import annotations

from pathlib import Path

from diffusers_3d import (
    DEFAULT_FORBIDDEN_MARKER,
    IntegrationManifest3D,
    scan_forbidden_marker,
    validate_integration_manifest,
)

PACKAGE_ROOT = Path(__file__).parents[2]
TEMPLATES = PACKAGE_ROOT / "templates"


def test_template_manifests_are_strict_and_policy_valid():
    manifest_paths = sorted(TEMPLATES.glob("*/integration_manifest.json"))

    assert {path.parent.name for path in manifest_paths} == {
        "experimental-custom-block",
        "reviewed-model-family",
    }
    for path in manifest_paths:
        manifest = IntegrationManifest3D.load(path)
        report = validate_integration_manifest(manifest)
        assert report.is_valid, report.to_dict()


def test_template_python_skeletons_compile():
    python_paths = sorted(TEMPLATES.rglob("*.py"))

    assert python_paths
    for path in python_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_templates_are_release_marker_clean():
    report = scan_forbidden_marker((TEMPLATES,), marker=DEFAULT_FORBIDDEN_MARKER)

    assert report.is_clean
    assert report.scanned_files > 0
