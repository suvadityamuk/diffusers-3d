from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..backends import BackendLicenseClass, BackendSupportLevel

INTEGRATION_MANIFEST_NAME = "diffusers_3d_integration.json"
INTEGRATION_MANIFEST_SCHEMA = "diffusers-3d-integration"
INTEGRATION_MANIFEST_VERSION = 2

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_QUALIFIED_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


class IntegrationManifestError(ValueError):
    """A contribution manifest is not valid schema-versioned JSON."""


class ContributionLevel(str, Enum):
    """The three supported contribution lifecycle levels."""

    EXPERIMENTAL_HUB = "experimental_hub"
    REVIEWED_PACKAGE = "reviewed_package"
    UPSTREAM_DIFFUSERS = "upstream_diffusers"


class ParityKind(str, Enum):
    """Closed evidence categories used by integration and training review."""

    INFERENCE = "inference"
    CHECKPOINT = "checkpoint"
    BACKWARD = "backward"
    OBJECTIVE = "objective"


class FineTuneStrategy(str, Enum):
    """Strategies that a reviewed training recipe may qualify."""

    FULL = "full"
    LORA = "lora"


def _as_object(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IntegrationManifestError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise IntegrationManifestError(f"{context} field names must be strings")
    return value


def _strict_fields(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> Mapping[str, object]:
    data = _as_object(value, context=context)
    missing = expected.difference(data)
    unknown = set(data).difference(expected)
    if missing:
        raise IntegrationManifestError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise IntegrationManifestError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    return data


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationManifestError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field_name=field_name)


def _identifier(value: object, *, field_name: str) -> str:
    value = _nonempty_string(value, field_name=field_name)
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise IntegrationManifestError(
            f"{field_name} must start with a lowercase letter and contain only lowercase letters, "
            "digits, '.', '_', or '-'"
        )
    return value


def _version(value: object, *, field_name: str) -> str:
    value = _nonempty_string(value, field_name=field_name)
    if not _VERSION_PATTERN.fullmatch(value):
        raise IntegrationManifestError(f"{field_name} must be a stable version identifier")
    return value


def _qualified_name(value: object, *, field_name: str) -> str:
    value = _nonempty_string(value, field_name=field_name)
    if not _QUALIFIED_NAME_PATTERN.fullmatch(value):
        raise IntegrationManifestError(f"{field_name} must be a fully-qualified Python name")
    return value


def _string_tuple(
    value: object,
    *,
    field_name: str,
    identifiers: bool = False,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise IntegrationManifestError(f"{field_name} must be a JSON array of strings")
    items = tuple(value)
    if not items and not allow_empty:
        raise IntegrationManifestError(f"{field_name} must not be empty")
    normalized = tuple(
        _identifier(item, field_name=field_name) if identifiers else _nonempty_string(item, field_name=field_name)
        for item in items
    )
    if len(set(normalized)) != len(normalized):
        raise IntegrationManifestError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _record_tuple(
    value: object,
    record_type: type[Any],
    *,
    field_name: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise IntegrationManifestError(f"{field_name} must be a JSON array")
    records = tuple(value)
    if any(not isinstance(record, record_type) for record in records):
        raise IntegrationManifestError(f"{field_name} must contain only {record_type.__name__} records")
    return records


def _optional_finite_number(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise IntegrationManifestError(f"{field_name} must be null or a finite non-negative number")
    return float(value)


def _parse_enum(value: object, enum_type: type[Enum], *, field_name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise IntegrationManifestError(f"{field_name} must be one of {choices}") from error


@dataclass(frozen=True, slots=True)
class UpstreamSource3D:
    """Immutable source repository identity for an integration."""

    repository: str
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", _nonempty_string(self.repository, field_name="upstream.repository"))
        object.__setattr__(self, "revision", _nonempty_string(self.revision, field_name="upstream.revision"))

    def to_dict(self) -> dict[str, object]:
        return {"repository": self.repository, "revision": self.revision}

    @classmethod
    def from_dict(cls, value: object) -> UpstreamSource3D:
        data = _strict_fields(value, {"repository", "revision"}, context="upstream")
        return cls(repository=data["repository"], revision=data["revision"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ParityEvidence3D:
    """One reproducible local parity comparison."""

    kind: ParityKind
    reference: str
    test: str
    passed: bool
    atol: float | None
    rtol: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _parse_enum(self.kind, ParityKind, field_name="parity.kind"))
        object.__setattr__(self, "reference", _nonempty_string(self.reference, field_name="parity.reference"))
        object.__setattr__(self, "test", _nonempty_string(self.test, field_name="parity.test"))
        if not isinstance(self.passed, bool):
            raise IntegrationManifestError("parity.passed must be a boolean")
        object.__setattr__(self, "atol", _optional_finite_number(self.atol, field_name="parity.atol"))
        object.__setattr__(self, "rtol", _optional_finite_number(self.rtol, field_name="parity.rtol"))

    def to_dict(self) -> dict[str, object]:
        return {
            "atol": self.atol,
            "kind": self.kind.value,
            "passed": self.passed,
            "reference": self.reference,
            "rtol": self.rtol,
            "test": self.test,
        }

    @classmethod
    def from_dict(cls, value: object) -> ParityEvidence3D:
        data = _strict_fields(
            value,
            {"atol", "kind", "passed", "reference", "rtol", "test"},
            context="parity evidence",
        )
        return cls(
            kind=data["kind"],  # type: ignore[arg-type]
            reference=data["reference"],  # type: ignore[arg-type]
            test=data["test"],  # type: ignore[arg-type]
            passed=data["passed"],  # type: ignore[arg-type]
            atol=data["atol"],  # type: ignore[arg-type]
            rtol=data["rtol"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CheckpointConversion3D:
    """Exact checkpoint converter and its executable round-trip test."""

    source_format: str
    target_format: str
    converter: str
    test: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_format",
            _identifier(self.source_format, field_name="checkpoint_conversion.source_format"),
        )
        object.__setattr__(
            self,
            "target_format",
            _identifier(self.target_format, field_name="checkpoint_conversion.target_format"),
        )
        object.__setattr__(
            self,
            "converter",
            _qualified_name(self.converter, field_name="checkpoint_conversion.converter"),
        )
        object.__setattr__(
            self,
            "test",
            _nonempty_string(self.test, field_name="checkpoint_conversion.test"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "converter": self.converter,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "test": self.test,
        }

    @classmethod
    def from_dict(cls, value: object) -> CheckpointConversion3D:
        data = _strict_fields(
            value,
            {"converter", "source_format", "target_format", "test"},
            context="checkpoint conversion",
        )
        return cls(
            source_format=data["source_format"],  # type: ignore[arg-type]
            target_format=data["target_format"],  # type: ignore[arg-type]
            converter=data["converter"],  # type: ignore[arg-type]
            test=data["test"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ComponentIntegration3D:
    """One exact model, pipeline, scheduler, or processor role."""

    role: str
    class_name: str
    checkpoint_conversion: CheckpointConversion3D | None
    parity: tuple[ParityEvidence3D, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, field_name="component.role"))
        object.__setattr__(
            self,
            "class_name",
            _qualified_name(self.class_name, field_name="component.class_name"),
        )
        if self.checkpoint_conversion is not None and not isinstance(
            self.checkpoint_conversion, CheckpointConversion3D
        ):
            raise IntegrationManifestError("component.checkpoint_conversion must be a CheckpointConversion3D or None")
        parity = _record_tuple(self.parity, ParityEvidence3D, field_name="component.parity")
        object.__setattr__(self, "parity", tuple(sorted(parity, key=lambda item: (item.kind.value, item.test))))

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_conversion": (
                None if self.checkpoint_conversion is None else self.checkpoint_conversion.to_dict()
            ),
            "class_name": self.class_name,
            "parity": [evidence.to_dict() for evidence in self.parity],
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: object) -> ComponentIntegration3D:
        data = _strict_fields(
            value,
            {"checkpoint_conversion", "class_name", "parity", "role"},
            context="component",
        )
        conversion = data["checkpoint_conversion"]
        parity = data["parity"]
        if isinstance(parity, (str, bytes)) or not isinstance(parity, Sequence):
            raise IntegrationManifestError("component.parity must be a JSON array")
        return cls(
            role=data["role"],  # type: ignore[arg-type]
            class_name=data["class_name"],  # type: ignore[arg-type]
            checkpoint_conversion=(None if conversion is None else CheckpointConversion3D.from_dict(conversion)),
            parity=tuple(ParityEvidence3D.from_dict(item) for item in parity),
        )


@dataclass(frozen=True, slots=True)
class TaskWorkflow3D:
    """Task routing and object representation contract."""

    task_ids: tuple[str, ...]
    workflow: str
    input_representations: tuple[str, ...]
    output_representations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_ids",
            _string_tuple(self.task_ids, field_name="workflow.task_ids", identifiers=True),
        )
        object.__setattr__(self, "workflow", _identifier(self.workflow, field_name="workflow.workflow"))
        object.__setattr__(
            self,
            "input_representations",
            _string_tuple(
                self.input_representations,
                field_name="workflow.input_representations",
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "output_representations",
            _string_tuple(
                self.output_representations,
                field_name="workflow.output_representations",
                identifiers=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_representations": list(self.input_representations),
            "output_representations": list(self.output_representations),
            "task_ids": list(self.task_ids),
            "workflow": self.workflow,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskWorkflow3D:
        data = _strict_fields(
            value,
            {"input_representations", "output_representations", "task_ids", "workflow"},
            context="workflow",
        )
        return cls(
            task_ids=data["task_ids"],  # type: ignore[arg-type]
            workflow=data["workflow"],  # type: ignore[arg-type]
            input_representations=data["input_representations"],  # type: ignore[arg-type]
            output_representations=data["output_representations"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class BackendRequirement3D:
    """A backend dependency with explicit support and license policy."""

    name: str
    distribution: str | None
    version: str | None
    capabilities: tuple[str, ...]
    support_level: BackendSupportLevel
    license_identifier: str
    license_class: BackendLicenseClass
    required: bool
    install_hint: str
    source: UpstreamSource3D | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="backend.name"))
        object.__setattr__(
            self,
            "distribution",
            _optional_string(self.distribution, field_name="backend.distribution"),
        )
        object.__setattr__(self, "version", _optional_string(self.version, field_name="backend.version"))
        object.__setattr__(
            self,
            "capabilities",
            _string_tuple(self.capabilities, field_name="backend.capabilities", identifiers=True),
        )
        object.__setattr__(
            self,
            "support_level",
            _parse_enum(self.support_level, BackendSupportLevel, field_name="backend.support_level"),
        )
        object.__setattr__(
            self,
            "license_identifier",
            _nonempty_string(self.license_identifier, field_name="backend.license_identifier"),
        )
        object.__setattr__(
            self,
            "license_class",
            _parse_enum(self.license_class, BackendLicenseClass, field_name="backend.license_class"),
        )
        if not isinstance(self.required, bool):
            raise IntegrationManifestError("backend.required must be a boolean")
        object.__setattr__(
            self,
            "install_hint",
            _nonempty_string(self.install_hint, field_name="backend.install_hint"),
        )
        if self.source is not None and not isinstance(self.source, UpstreamSource3D):
            raise IntegrationManifestError("backend.source must be an UpstreamSource3D or None")
        if self.distribution is None and self.source is None:
            raise IntegrationManifestError("a backend without a distribution must declare a pinned source")

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities": list(self.capabilities),
            "distribution": self.distribution,
            "install_hint": self.install_hint,
            "license_class": self.license_class.value,
            "license_identifier": self.license_identifier,
            "name": self.name,
            "required": self.required,
            "source": None if self.source is None else self.source.to_dict(),
            "support_level": self.support_level.value,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: object) -> BackendRequirement3D:
        data = _strict_fields(
            value,
            {
                "capabilities",
                "distribution",
                "install_hint",
                "license_class",
                "license_identifier",
                "name",
                "required",
                "source",
                "support_level",
                "version",
            },
            context="backend",
        )
        source = data["source"]
        return cls(
            name=data["name"],  # type: ignore[arg-type]
            distribution=data["distribution"],  # type: ignore[arg-type]
            version=data["version"],  # type: ignore[arg-type]
            capabilities=data["capabilities"],  # type: ignore[arg-type]
            support_level=data["support_level"],  # type: ignore[arg-type]
            license_identifier=data["license_identifier"],  # type: ignore[arg-type]
            license_class=data["license_class"],  # type: ignore[arg-type]
            required=data["required"],  # type: ignore[arg-type]
            install_hint=data["install_hint"],  # type: ignore[arg-type]
            source=None if source is None else UpstreamSource3D.from_dict(source),
        )


@dataclass(frozen=True, slots=True)
class LicenseRecord3D:
    """Exact model or artifact license and its policy classification."""

    identifier: str
    classification: BackendLicenseClass
    url: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identifier",
            _nonempty_string(self.identifier, field_name="license.identifier"),
        )
        object.__setattr__(
            self,
            "classification",
            _parse_enum(self.classification, BackendLicenseClass, field_name="license.classification"),
        )
        object.__setattr__(self, "url", _nonempty_string(self.url, field_name="license.url"))

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "identifier": self.identifier,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, value: object) -> LicenseRecord3D:
        data = _strict_fields(value, {"classification", "identifier", "url"}, context="license")
        return cls(
            identifier=data["identifier"],  # type: ignore[arg-type]
            classification=data["classification"],  # type: ignore[arg-type]
            url=data["url"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ArtifactLicense3D:
    """License declaration for one converted or generated artifact set."""

    artifact: str
    license: LicenseRecord3D

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact",
            _identifier(self.artifact, field_name="artifact_license.artifact"),
        )
        if not isinstance(self.license, LicenseRecord3D):
            raise IntegrationManifestError("artifact_license.license must be a LicenseRecord3D")

    def to_dict(self) -> dict[str, object]:
        return {"artifact": self.artifact, "license": self.license.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactLicense3D:
        data = _strict_fields(value, {"artifact", "license"}, context="artifact license")
        return cls(
            artifact=data["artifact"],  # type: ignore[arg-type]
            license=LicenseRecord3D.from_dict(data["license"]),
        )


@dataclass(frozen=True, slots=True)
class LicenseDeclarations3D:
    """Model and artifact license declarations reviewed together."""

    model: LicenseRecord3D | None
    artifacts: tuple[ArtifactLicense3D, ...]

    def __post_init__(self) -> None:
        if self.model is not None and not isinstance(self.model, LicenseRecord3D):
            raise IntegrationManifestError("licenses.model must be a LicenseRecord3D or None")
        artifacts = _record_tuple(self.artifacts, ArtifactLicense3D, field_name="licenses.artifacts")
        names = [artifact.artifact for artifact in artifacts]
        if len(set(names)) != len(names):
            raise IntegrationManifestError("licenses.artifacts must not contain duplicate artifact names")
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.artifact)))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "model": None if self.model is None else self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> LicenseDeclarations3D:
        data = _strict_fields(value, {"artifacts", "model"}, context="licenses")
        artifacts = data["artifacts"]
        if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
            raise IntegrationManifestError("licenses.artifacts must be a JSON array")
        model = data["model"]
        return cls(
            model=None if model is None else LicenseRecord3D.from_dict(model),
            artifacts=tuple(ArtifactLicense3D.from_dict(item) for item in artifacts),
        )


@dataclass(frozen=True, slots=True)
class TrainingRecipeQualification3D:
    """Separately reviewed exact recipe registration and parity record."""

    recipe_id: str
    recipe_version: str
    recipe_class: str
    target_class: str
    example_class: str
    batch_class: str
    trainer_registration: str
    strategies: tuple[FineTuneStrategy, ...]
    components: tuple[str, ...]
    backward_parity: ParityEvidence3D | None
    checkpoint_parity: ParityEvidence3D | None
    objective_parity: ParityEvidence3D | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe_id", _identifier(self.recipe_id, field_name="training.recipe_id"))
        object.__setattr__(
            self,
            "recipe_version",
            _version(self.recipe_version, field_name="training.recipe_version"),
        )
        for field_name in ("recipe_class", "target_class", "example_class", "batch_class", "trainer_registration"):
            object.__setattr__(
                self,
                field_name,
                _qualified_name(getattr(self, field_name), field_name=f"training.{field_name}"),
            )
        if isinstance(self.strategies, (str, FineTuneStrategy)) or not isinstance(self.strategies, Sequence):
            raise IntegrationManifestError("training.strategies must be a JSON array")
        strategies = tuple(
            _parse_enum(strategy, FineTuneStrategy, field_name="training.strategies") for strategy in self.strategies
        )
        if not strategies or len(set(strategies)) != len(strategies):
            raise IntegrationManifestError("training.strategies must be non-empty and contain no duplicates")
        object.__setattr__(self, "strategies", tuple(sorted(strategies, key=lambda item: item.value)))
        object.__setattr__(
            self,
            "components",
            _string_tuple(self.components, field_name="training.components", identifiers=True),
        )
        for field_name in ("backward_parity", "checkpoint_parity", "objective_parity"):
            evidence = getattr(self, field_name)
            if evidence is not None and not isinstance(evidence, ParityEvidence3D):
                raise IntegrationManifestError(f"training.{field_name} must be ParityEvidence3D or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "backward_parity": (None if self.backward_parity is None else self.backward_parity.to_dict()),
            "batch_class": self.batch_class,
            "checkpoint_parity": (None if self.checkpoint_parity is None else self.checkpoint_parity.to_dict()),
            "components": list(self.components),
            "example_class": self.example_class,
            "objective_parity": (None if self.objective_parity is None else self.objective_parity.to_dict()),
            "recipe_class": self.recipe_class,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "strategies": [strategy.value for strategy in self.strategies],
            "target_class": self.target_class,
            "trainer_registration": self.trainer_registration,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainingRecipeQualification3D:
        data = _strict_fields(
            value,
            {
                "backward_parity",
                "batch_class",
                "checkpoint_parity",
                "components",
                "example_class",
                "objective_parity",
                "recipe_class",
                "recipe_id",
                "recipe_version",
                "strategies",
                "target_class",
                "trainer_registration",
            },
            context="training qualification",
        )

        def parse_evidence(field_name: str) -> ParityEvidence3D | None:
            evidence = data[field_name]
            return None if evidence is None else ParityEvidence3D.from_dict(evidence)

        return cls(
            recipe_id=data["recipe_id"],  # type: ignore[arg-type]
            recipe_version=data["recipe_version"],  # type: ignore[arg-type]
            recipe_class=data["recipe_class"],  # type: ignore[arg-type]
            target_class=data["target_class"],  # type: ignore[arg-type]
            example_class=data["example_class"],  # type: ignore[arg-type]
            batch_class=data["batch_class"],  # type: ignore[arg-type]
            trainer_registration=data["trainer_registration"],  # type: ignore[arg-type]
            strategies=data["strategies"],  # type: ignore[arg-type]
            components=data["components"],  # type: ignore[arg-type]
            backward_parity=parse_evidence("backward_parity"),
            checkpoint_parity=parse_evidence("checkpoint_parity"),
            objective_parity=parse_evidence("objective_parity"),
        )


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise IntegrationManifestError(f"JSON object contains duplicate field {key!r}")
        data[key] = value
    return data


def _reject_non_finite(value: str) -> None:
    raise IntegrationManifestError(f"JSON contains non-finite number {value}")


@dataclass(frozen=True, slots=True)
class IntegrationManifest3D:
    """Versioned immutable integration review record."""

    schema: str
    schema_version: int
    integration_id: str
    level: ContributionLevel
    upstream: UpstreamSource3D
    components: tuple[ComponentIntegration3D, ...]
    workflow: TaskWorkflow3D
    backends: tuple[BackendRequirement3D, ...]
    licenses: LicenseDeclarations3D | None
    training: TrainingRecipeQualification3D | None

    def __post_init__(self) -> None:
        if self.schema != INTEGRATION_MANIFEST_SCHEMA:
            raise IntegrationManifestError(f"schema must be {INTEGRATION_MANIFEST_SCHEMA!r}, got {self.schema!r}")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != INTEGRATION_MANIFEST_VERSION
        ):
            raise IntegrationManifestError(
                f"schema_version must be {INTEGRATION_MANIFEST_VERSION}, got {self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "integration_id",
            _identifier(self.integration_id, field_name="integration_id"),
        )
        object.__setattr__(
            self,
            "level",
            _parse_enum(self.level, ContributionLevel, field_name="level"),
        )
        if not isinstance(self.upstream, UpstreamSource3D):
            raise IntegrationManifestError("upstream must be an UpstreamSource3D")
        components = _record_tuple(self.components, ComponentIntegration3D, field_name="components")
        object.__setattr__(self, "components", tuple(sorted(components, key=lambda item: item.role)))
        if not isinstance(self.workflow, TaskWorkflow3D):
            raise IntegrationManifestError("workflow must be a TaskWorkflow3D")
        backends = _record_tuple(self.backends, BackendRequirement3D, field_name="backends")
        object.__setattr__(self, "backends", tuple(sorted(backends, key=lambda item: item.name)))
        if self.licenses is not None and not isinstance(self.licenses, LicenseDeclarations3D):
            raise IntegrationManifestError("licenses must be a LicenseDeclarations3D or None")
        if self.training is not None and not isinstance(self.training, TrainingRecipeQualification3D):
            raise IntegrationManifestError("training must be a TrainingRecipeQualification3D or None")

    @classmethod
    def create(
        cls,
        *,
        integration_id: str,
        level: ContributionLevel,
        upstream: UpstreamSource3D,
        components: tuple[ComponentIntegration3D, ...],
        workflow: TaskWorkflow3D,
        backends: tuple[BackendRequirement3D, ...],
        licenses: LicenseDeclarations3D | None,
        training: TrainingRecipeQualification3D | None = None,
    ) -> IntegrationManifest3D:
        return cls(
            schema=INTEGRATION_MANIFEST_SCHEMA,
            schema_version=INTEGRATION_MANIFEST_VERSION,
            integration_id=integration_id,
            level=level,
            upstream=upstream,
            components=components,
            workflow=workflow,
            backends=backends,
            licenses=licenses,
            training=training,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backends": [backend.to_dict() for backend in self.backends],
            "components": [component.to_dict() for component in self.components],
            "integration_id": self.integration_id,
            "level": self.level.value,
            "licenses": None if self.licenses is None else self.licenses.to_dict(),
            "schema": self.schema,
            "schema_version": self.schema_version,
            "training": None if self.training is None else self.training.to_dict(),
            "upstream": self.upstream.to_dict(),
            "workflow": self.workflow.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> IntegrationManifest3D:
        data = _strict_fields(
            value,
            {
                "backends",
                "components",
                "integration_id",
                "level",
                "licenses",
                "schema",
                "schema_version",
                "training",
                "upstream",
                "workflow",
            },
            context="integration manifest",
        )
        components = data["components"]
        backends = data["backends"]
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
            raise IntegrationManifestError("components must be a JSON array")
        if isinstance(backends, (str, bytes)) or not isinstance(backends, Sequence):
            raise IntegrationManifestError("backends must be a JSON array")
        licenses = data["licenses"]
        training = data["training"]
        return cls(
            schema=data["schema"],  # type: ignore[arg-type]
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            integration_id=data["integration_id"],  # type: ignore[arg-type]
            level=data["level"],  # type: ignore[arg-type]
            upstream=UpstreamSource3D.from_dict(data["upstream"]),
            components=tuple(ComponentIntegration3D.from_dict(item) for item in components),
            workflow=TaskWorkflow3D.from_dict(data["workflow"]),
            backends=tuple(BackendRequirement3D.from_dict(item) for item in backends),
            licenses=None if licenses is None else LicenseDeclarations3D.from_dict(licenses),
            training=(None if training is None else TrainingRecipeQualification3D.from_dict(training)),
        )

    @classmethod
    def loads(cls, payload: str) -> IntegrationManifest3D:
        if not isinstance(payload, str):
            raise TypeError("payload must be a string")
        try:
            data = json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_fields,
                parse_constant=_reject_non_finite,
            )
        except IntegrationManifestError:
            raise
        except json.JSONDecodeError as error:
            raise IntegrationManifestError(f"invalid manifest JSON: {error.msg}") from error
        return cls.from_dict(data)

    def dumps(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> IntegrationManifest3D:
        manifest_path = Path(path)
        try:
            payload = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise IntegrationManifestError(f"could not read integration manifest from {manifest_path}") from error
        return cls.loads(payload)

    def save(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise IntegrationManifestError(
                f"could not create integration manifest directory {destination.parent}"
            ) from error
        payload = self.dumps()
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise IntegrationManifestError(
                f"could not atomically save integration manifest to {destination}"
            ) from error
        return destination


__all__ = [
    "INTEGRATION_MANIFEST_NAME",
    "INTEGRATION_MANIFEST_SCHEMA",
    "INTEGRATION_MANIFEST_VERSION",
    "ArtifactLicense3D",
    "BackendRequirement3D",
    "CheckpointConversion3D",
    "ComponentIntegration3D",
    "ContributionLevel",
    "FineTuneStrategy",
    "IntegrationManifest3D",
    "IntegrationManifestError",
    "LicenseDeclarations3D",
    "LicenseRecord3D",
    "ParityEvidence3D",
    "ParityKind",
    "TaskWorkflow3D",
    "TrainingRecipeQualification3D",
    "UpstreamSource3D",
]
