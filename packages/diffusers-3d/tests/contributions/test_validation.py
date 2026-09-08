from __future__ import annotations

from dataclasses import replace

from diffusers_3d import (
    BackendLicenseClass,
    BackendRequirement3D,
    BackendSupportLevel,
    ComponentIntegration3D,
    ContributionLevel,
    LicenseDeclarations3D,
    LicenseRecord3D,
    ParityEvidence3D,
    ParityKind,
    validate_integration_manifest,
)


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def test_valid_reviewed_trainable_manifest_has_no_errors(manifest_factory):
    report = validate_integration_manifest(manifest_factory())

    assert report.is_valid
    assert not report.errors
    assert not report.warnings


def test_experimental_hub_is_inference_only(manifest_factory):
    manifest = manifest_factory(level=ContributionLevel.EXPERIMENTAL_HUB)
    report = validate_integration_manifest(manifest)

    assert "training.experimental_forbidden" in codes(report.errors)
    assert "level.experimental_inference_only" in codes(report.warnings)


def test_reviewed_and_upstream_levels_require_immutable_revision(manifest_factory):
    for level in (ContributionLevel.REVIEWED_PACKAGE, ContributionLevel.UPSTREAM_DIFFUSERS):
        manifest = replace(
            manifest_factory(level=level),
            upstream=replace(manifest_factory().upstream, revision="main"),
        )
        report = validate_integration_manifest(manifest)
        assert "upstream.mutable_revision" in codes(report.errors)


def test_upstream_primitive_does_not_gain_training_qualification(manifest_factory):
    manifest = manifest_factory(
        level=ContributionLevel.UPSTREAM_DIFFUSERS,
        include_training=False,
    )
    report = validate_integration_manifest(manifest)

    assert report.is_valid
    assert "training.not_qualified" in codes(report.warnings)


def test_reviewed_integration_reports_missing_backend_license_and_parity(manifest_factory):
    manifest = manifest_factory(include_training=False)
    incomplete_component = ComponentIntegration3D(
        role="denoiser",
        class_name="example.model.ExampleDenoiser",
        checkpoint_conversion=None,
        parity=(),
    )
    manifest = replace(
        manifest,
        components=(incomplete_component,),
        backends=(),
        licenses=None,
    )
    report = validate_integration_manifest(manifest)
    error_codes = codes(report.errors)

    assert {
        "backends.missing",
        "component.missing_checkpoint_conversion",
        "component.missing_parity",
        "licenses.missing",
    }.issubset(error_codes)


def test_trainability_requires_exact_components_target_and_three_parity_records(manifest_factory):
    manifest = manifest_factory()
    qualification = replace(
        manifest.training,
        target_class="example.pipeline.OtherPipeline",
        components=("unknown-component",),
        backward_parity=None,
        checkpoint_parity=ParityEvidence3D(
            kind=ParityKind.OBJECTIVE,
            reference="wrong evidence kind",
            test="tests/example/test_training.py::test_checkpoint",
            passed=False,
            atol=0,
            rtol=0,
        ),
        objective_parity=None,
    )
    report = validate_integration_manifest(replace(manifest, training=qualification))
    error_codes = codes(report.errors)

    assert {
        "training.failed_checkpoint_parity",
        "training.missing_backward_parity",
        "training.missing_objective_parity",
        "training.unknown_components",
        "training.unknown_target",
        "training.wrong_parity_kind",
    }.issubset(error_codes)


def test_backend_and_license_policy_findings_are_warnings(manifest_factory):
    manifest = manifest_factory()
    backend = BackendRequirement3D(
        name="research-renderer",
        distribution="research-renderer",
        version="1.0",
        capabilities=("mesh-rasterization",),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        license_identifier="Research-Only",
        license_class=BackendLicenseClass.RESTRICTED,
        required=False,
        install_hint="Install only after license review.",
        source=None,
    )
    unknown = LicenseRecord3D(
        identifier="LicenseRef-Unknown",
        classification=BackendLicenseClass.UNKNOWN,
        url="https://example.com/license",
    )
    report = validate_integration_manifest(
        replace(
            manifest,
            backends=(backend,),
            licenses=LicenseDeclarations3D(
                model=unknown,
                artifacts=manifest.licenses.artifacts,
            ),
        )
    )
    warning_codes = codes(report.warnings)

    assert report.is_valid
    assert {
        "backend.research_only",
        "backend.restricted_license",
        "licenses.unknown",
    }.issubset(warning_codes)
