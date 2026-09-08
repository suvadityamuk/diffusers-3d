from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..backends import BackendLicenseClass, BackendSupportLevel
from .manifest import (
    ContributionLevel,
    IntegrationManifest3D,
    IntegrationManifestError,
    ParityEvidence3D,
    ParityKind,
)

_IMMUTABLE_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ValidationSeverity(str, Enum):
    """Stable severity values emitted by the contribution validator."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue3D:
    """One machine-readable manifest validation finding."""

    severity: ValidationSeverity
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity.value,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport3D:
    """Immutable errors and warnings from an offline validation pass."""

    errors: tuple[ValidationIssue3D, ...] = ()
    warnings: tuple[ValidationIssue3D, ...] = ()

    def __post_init__(self) -> None:
        if any(issue.severity is not ValidationSeverity.ERROR for issue in self.errors):
            raise ValueError("errors must contain only error issues")
        if any(issue.severity is not ValidationSeverity.WARNING for issue in self.warnings):
            raise ValueError("warnings must contain only warning issues")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def issues(self) -> tuple[ValidationIssue3D, ...]:
        return self.errors + self.warnings

    def to_dict(self) -> dict[str, object]:
        return {
            "errors": [issue.to_dict() for issue in self.errors],
            "valid": self.is_valid,
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def validate_integration_manifest(manifest: IntegrationManifest3D) -> ValidationReport3D:
    """Validate lifecycle, parity, backend, license, and trainability policy."""

    if not isinstance(manifest, IntegrationManifest3D):
        raise TypeError("manifest must be an IntegrationManifest3D")

    errors: list[ValidationIssue3D] = []
    warnings: list[ValidationIssue3D] = []

    def error(code: str, path: str, message: str) -> None:
        errors.append(ValidationIssue3D(ValidationSeverity.ERROR, code, path, message))

    def warning(code: str, path: str, message: str) -> None:
        warnings.append(ValidationIssue3D(ValidationSeverity.WARNING, code, path, message))

    reviewed = manifest.level in (
        ContributionLevel.REVIEWED_PACKAGE,
        ContributionLevel.UPSTREAM_DIFFUSERS,
    )
    immutable_revision = bool(_IMMUTABLE_REVISION_PATTERN.fullmatch(manifest.upstream.revision))
    if reviewed and not immutable_revision:
        error(
            "upstream.mutable_revision",
            "upstream.revision",
            "Reviewed and upstream integrations must pin a full lowercase 40- or 64-character commit digest.",
        )
    elif manifest.level is ContributionLevel.EXPERIMENTAL_HUB and not immutable_revision:
        warning(
            "upstream.unpinned_experimental",
            "upstream.revision",
            "Experimental Hub code should pin a full immutable commit digest before sharing.",
        )

    if not manifest.components:
        error(
            "components.missing",
            "components",
            "An integration must declare at least one exact component role and class.",
        )
    component_roles = [component.role for component in manifest.components]
    component_classes = [component.class_name for component in manifest.components]
    if len(set(component_roles)) != len(component_roles):
        error(
            "components.duplicate_role",
            "components",
            "Component roles must be unique within an integration.",
        )
    if len(set(component_classes)) != len(component_classes):
        error(
            "components.duplicate_class",
            "components",
            "Each exact component class may be declared only once.",
        )

    for index, component in enumerate(manifest.components):
        component_path = f"components[{index}]"
        if reviewed and component.checkpoint_conversion is None:
            error(
                "component.missing_checkpoint_conversion",
                f"{component_path}.checkpoint_conversion",
                "Reviewed components must declare their exact checkpoint converter and round-trip test.",
            )
        if reviewed and not component.parity:
            error(
                "component.missing_parity",
                f"{component_path}.parity",
                "Reviewed components must include reproducible component parity evidence.",
            )
        for parity_index, evidence in enumerate(component.parity):
            if not evidence.passed:
                error(
                    "parity.failed",
                    f"{component_path}.parity[{parity_index}].passed",
                    "Parity evidence must pass before package or upstream review.",
                )

    backend_names = [backend.name for backend in manifest.backends]
    if reviewed and not manifest.backends:
        error(
            "backends.missing",
            "backends",
            "Reviewed integrations must declare runtime backends, including dependency-free or PyTorch paths.",
        )
    if len(set(backend_names)) != len(backend_names):
        error("backends.duplicate", "backends", "Backend names must be unique.")
    for index, backend in enumerate(manifest.backends):
        backend_path = f"backends[{index}]"
        if backend.source is not None and not _IMMUTABLE_REVISION_PATTERN.fullmatch(backend.source.revision):
            finding = error if reviewed else warning
            finding(
                "backend.mutable_source_revision",
                f"{backend_path}.source.revision",
                "Source-built backends must pin a full immutable commit digest.",
            )
        if backend.license_class is BackendLicenseClass.UNKNOWN:
            warning(
                "backend.unknown_license",
                f"{backend_path}.license_class",
                f"Backend {backend.name!r} requires a completed license classification.",
            )
        elif backend.license_class is BackendLicenseClass.RESTRICTED:
            warning(
                "backend.restricted_license",
                f"{backend_path}.license_class",
                f"Backend {backend.name!r} has restricted licensing and needs explicit deployment review.",
            )
        if backend.support_level is BackendSupportLevel.RESEARCH_ONLY:
            warning(
                "backend.research_only",
                f"{backend_path}.support_level",
                f"Backend {backend.name!r} is research-only and must not be selected implicitly.",
            )

    if reviewed and manifest.licenses is None:
        error(
            "licenses.missing",
            "licenses",
            "Reviewed integrations must declare model and artifact licenses.",
        )
    elif manifest.licenses is None:
        warning(
            "licenses.missing_experimental",
            "licenses",
            "License declarations are required before package review.",
        )
    else:
        if manifest.licenses.model is None:
            finding = error if reviewed else warning
            finding(
                "licenses.missing_model",
                "licenses.model",
                "The upstream model license must be declared.",
            )
        if not manifest.licenses.artifacts:
            finding = error if reviewed else warning
            finding(
                "licenses.missing_artifacts",
                "licenses.artifacts",
                "Converted checkpoints and other shipped artifacts must have explicit license declarations.",
            )
        license_records = []
        if manifest.licenses.model is not None:
            license_records.append(("licenses.model", manifest.licenses.model))
        license_records.extend(
            (f"licenses.artifacts[{index}].license", artifact.license)
            for index, artifact in enumerate(manifest.licenses.artifacts)
        )
        for path, license_record in license_records:
            if license_record.classification is BackendLicenseClass.UNKNOWN:
                warning(
                    "licenses.unknown",
                    f"{path}.classification",
                    f"License {license_record.identifier!r} still has an unknown classification.",
                )
            elif license_record.classification is BackendLicenseClass.RESTRICTED:
                warning(
                    "licenses.restricted",
                    f"{path}.classification",
                    f"License {license_record.identifier!r} requires an explicit use and redistribution review.",
                )

    training = manifest.training
    if manifest.level is ContributionLevel.EXPERIMENTAL_HUB:
        warning(
            "level.experimental_inference_only",
            "level",
            "Experimental Hub blocks are remote-code inference staging and are not stable trainer registrations.",
        )
        if training is not None:
            error(
                "training.experimental_forbidden",
                "training",
                "Experimental Hub contributions cannot declare a stable training qualification.",
            )
    elif training is None:
        warning(
            "training.not_qualified",
            "training",
            "This integration is inference-only until an exact training recipe receives separate review.",
        )

    if training is not None and manifest.level is not ContributionLevel.EXPERIMENTAL_HUB:
        declared_roles = set(component_roles)
        undeclared_roles = set(training.components).difference(declared_roles)
        if undeclared_roles:
            error(
                "training.unknown_components",
                "training.components",
                "Training components are not declared integration roles: " + ", ".join(sorted(undeclared_roles)),
            )
        if training.target_class not in component_classes:
            error(
                "training.unknown_target",
                "training.target_class",
                "The exact training target class must be one of the integration's reviewed component classes.",
            )

        required_evidence: tuple[tuple[str, ParityKind, ParityEvidence3D | None], ...] = (
            ("backward_parity", ParityKind.BACKWARD, training.backward_parity),
            ("checkpoint_parity", ParityKind.CHECKPOINT, training.checkpoint_parity),
            ("objective_parity", ParityKind.OBJECTIVE, training.objective_parity),
        )
        for field_name, expected_kind, evidence in required_evidence:
            evidence_path = f"training.{field_name}"
            if evidence is None:
                error(
                    f"training.missing_{expected_kind.value}_parity",
                    evidence_path,
                    f"Trainability requires {expected_kind.value} parity evidence.",
                )
                continue
            if evidence.kind is not expected_kind:
                error(
                    "training.wrong_parity_kind",
                    f"{evidence_path}.kind",
                    f"{field_name} must contain {expected_kind.value!r} evidence.",
                )
            if not evidence.passed:
                error(
                    f"training.failed_{expected_kind.value}_parity",
                    f"{evidence_path}.passed",
                    f"{expected_kind.value.capitalize()} parity must pass before trainer registration.",
                )

    return ValidationReport3D(
        errors=tuple(sorted(errors, key=lambda issue: (issue.path, issue.code))),
        warnings=tuple(sorted(warnings, key=lambda issue: (issue.path, issue.code))),
    )


def validate_manifest_file(path: str | Path) -> ValidationReport3D:
    """Load and validate a local manifest without network access."""

    try:
        manifest = IntegrationManifest3D.load(path)
    except IntegrationManifestError as error:
        issue = ValidationIssue3D(
            severity=ValidationSeverity.ERROR,
            code="manifest.invalid",
            path=str(path),
            message=str(error),
        )
        return ValidationReport3D(errors=(issue,))
    return validate_integration_manifest(manifest)


__all__ = [
    "ValidationIssue3D",
    "ValidationReport3D",
    "ValidationSeverity",
    "validate_integration_manifest",
    "validate_manifest_file",
]
