from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ..objects import Object3DKind
from .exceptions import Object3DLoadingError, Object3DMetadataError, Object3DSchemaError

OBJECT3D_API_VERSION = "1.0"
OBJECT3D_SCHEMA_VERSION = 1
OBJECT3D_MODEL_INDEX_NAME = "object3d_model_index.json"

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_QUALIFIED_CLASS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


class ContributionStatus(str, Enum):
    """Maturity level of an object-3D contribution."""

    EXPERIMENTAL_HUB = "experimental_hub"
    REVIEWED_PACKAGE = "reviewed_package"
    UPSTREAM_DIFFUSERS = "upstream_diffusers"


class ReviewStatus(str, Enum):
    """Whether an integration passed package review."""

    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"


def _normalize_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise Object3DMetadataError(
            f"{field_name} must start with a lowercase letter and contain only lowercase letters, "
            "digits, '.', '_', or '-'"
        )
    return value


def _normalize_identifiers(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Object3DMetadataError(f"{field_name} must be a JSON array or typed tuple of strings")
    items = tuple(value)
    if not items and not allow_empty:
        raise Object3DMetadataError(f"{field_name} must not be empty")
    normalized = tuple(_normalize_identifier(item, field_name) for item in items)
    if len(set(normalized)) != len(normalized):
        raise Object3DMetadataError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _normalize_qualified_classes(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Object3DMetadataError(f"{field_name} must be a JSON array or typed tuple")
    items = tuple(value)
    if not items:
        raise Object3DMetadataError(f"{field_name} must not be empty")
    for item in items:
        if not isinstance(item, str) or not _QUALIFIED_CLASS_PATTERN.fullmatch(item):
            raise Object3DMetadataError(f"{field_name} must contain fully-qualified Python class names")
    if len(set(items)) != len(items):
        raise Object3DMetadataError(f"{field_name} must not contain duplicates")
    return tuple(sorted(items))


def _normalize_qualified_class(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _QUALIFIED_CLASS_PATTERN.fullmatch(value):
        raise Object3DMetadataError(f"{field_name} must be a fully-qualified Python class name")
    return value


def _normalize_object_kinds(value: object, field_name: str) -> tuple[Object3DKind, ...]:
    if not isinstance(value, (list, tuple)):
        raise Object3DMetadataError(f"{field_name} must be a JSON array or typed tuple of Object3DKind values")
    try:
        items = tuple(Object3DKind(item) for item in value)
    except (TypeError, ValueError) as error:
        raise Object3DMetadataError(f"{field_name} must contain valid Object3DKind values") from error
    if not items:
        raise Object3DMetadataError(f"{field_name} must not be empty")
    if len(set(items)) != len(items):
        raise Object3DMetadataError(f"{field_name} must not contain duplicates")
    return tuple(sorted(items, key=lambda item: item.value))


def _normalize_schema_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Object3DSchemaError("schema_version must be an integer")
    if value != OBJECT3D_SCHEMA_VERSION:
        raise Object3DSchemaError(
            f"Unsupported object-3D schema version {value!r}; expected {OBJECT3D_SCHEMA_VERSION}"
        )
    return value


def _normalize_subfolder(subfolder: str | os.PathLike[str] | None) -> str | None:
    if subfolder is None or str(subfolder) == "":
        return None
    path = Path(subfolder)
    if path.is_absolute() or ".." in path.parts:
        raise Object3DLoadingError("subfolder must be a relative path that does not contain '..'")
    return str(path)


def fully_qualified_class_name(class_type: type[Any]) -> str:
    """Return the stable module-qualified name used for exact registrations."""

    if not isinstance(class_type, type):
        raise Object3DMetadataError("class declaration must contain Python class types")
    class_name = f"{class_type.__module__}.{class_type.__qualname__}"
    return _normalize_qualified_class(class_name, "class name")


@dataclass(frozen=True, slots=True)
class Object3DModelMetadata:
    """Immutable reviewed-registration metadata for an object-3D model."""

    schema_version: int
    family_id: str
    component_role: str
    model_class: str
    supported_object_kinds: tuple[Object3DKind, ...]
    required_backends: tuple[str, ...]
    contribution_status: ContributionStatus
    review_status: ReviewStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _normalize_schema_version(self.schema_version))
        object.__setattr__(self, "family_id", _normalize_identifier(self.family_id, "family_id"))
        object.__setattr__(
            self,
            "component_role",
            _normalize_identifier(self.component_role, "component_role"),
        )
        object.__setattr__(
            self,
            "model_class",
            _normalize_qualified_class(self.model_class, "model_class"),
        )
        object.__setattr__(
            self,
            "supported_object_kinds",
            _normalize_object_kinds(self.supported_object_kinds, "supported_object_kinds"),
        )
        object.__setattr__(
            self,
            "required_backends",
            _normalize_identifiers(self.required_backends, "required_backends", allow_empty=True),
        )
        try:
            contribution_status = ContributionStatus(self.contribution_status)
        except (TypeError, ValueError) as error:
            raise Object3DMetadataError("contribution_status must be a valid ContributionStatus") from error
        object.__setattr__(self, "contribution_status", contribution_status)
        try:
            review_status = ReviewStatus(self.review_status)
        except (TypeError, ValueError) as error:
            raise Object3DMetadataError("review_status must be a valid ReviewStatus") from error
        object.__setattr__(self, "review_status", review_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "component_role": self.component_role,
            "contribution_status": self.contribution_status.value,
            "family_id": self.family_id,
            "model_class": self.model_class,
            "required_backends": list(self.required_backends),
            "review_status": self.review_status.value,
            "schema_version": self.schema_version,
            "supported_object_kinds": [kind.value for kind in self.supported_object_kinds],
        }


@dataclass(frozen=True, slots=True)
class Object3DModelIndex:
    """Validated contents of ``object3d_model_index.json`` for a pipeline."""

    schema_version: int
    family_id: str
    task_ids: tuple[str, ...]
    pipeline_class: str
    output_object_types: tuple[str, ...]
    output_representations: tuple[str, ...]
    object_kinds: tuple[Object3DKind, ...]
    required_backends: tuple[str, ...]
    contribution_status: ContributionStatus
    review_status: ReviewStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _normalize_schema_version(self.schema_version))
        object.__setattr__(self, "family_id", _normalize_identifier(self.family_id, "family_id"))
        object.__setattr__(self, "task_ids", _normalize_identifiers(self.task_ids, "task_ids"))
        object.__setattr__(
            self,
            "pipeline_class",
            _normalize_qualified_class(self.pipeline_class, "pipeline_class"),
        )
        object.__setattr__(
            self,
            "output_object_types",
            _normalize_qualified_classes(self.output_object_types, "output_object_types"),
        )
        object.__setattr__(
            self,
            "output_representations",
            _normalize_identifiers(self.output_representations, "output_representations"),
        )
        object.__setattr__(
            self,
            "object_kinds",
            _normalize_object_kinds(self.object_kinds, "object_kinds"),
        )
        object.__setattr__(
            self,
            "required_backends",
            _normalize_identifiers(self.required_backends, "required_backends", allow_empty=True),
        )
        try:
            contribution_status = ContributionStatus(self.contribution_status)
        except (TypeError, ValueError) as error:
            raise Object3DMetadataError("contribution_status must be a valid ContributionStatus") from error
        object.__setattr__(self, "contribution_status", contribution_status)
        try:
            review_status = ReviewStatus(self.review_status)
        except (TypeError, ValueError) as error:
            raise Object3DMetadataError("review_status must be a valid ReviewStatus") from error
        object.__setattr__(self, "review_status", review_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "contribution_status": self.contribution_status.value,
            "family_id": self.family_id,
            "object_kinds": [kind.value for kind in self.object_kinds],
            "output_object_types": list(self.output_object_types),
            "output_representations": list(self.output_representations),
            "pipeline_class": self.pipeline_class,
            "required_backends": list(self.required_backends),
            "review_status": self.review_status.value,
            "schema_version": self.schema_version,
            "task_ids": list(self.task_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Object3DModelIndex:
        if not isinstance(data, Mapping):
            raise Object3DMetadataError("Object-3D model index must contain a JSON object")
        expected_fields = {
            "contribution_status",
            "family_id",
            "object_kinds",
            "output_object_types",
            "output_representations",
            "pipeline_class",
            "required_backends",
            "review_status",
            "schema_version",
            "task_ids",
        }
        missing = expected_fields.difference(data)
        unexpected = set(data).difference(expected_fields)
        if missing:
            raise Object3DMetadataError(
                f"Object-3D model index is missing required fields: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise Object3DMetadataError(
                f"Object-3D model index contains unknown fields: {', '.join(sorted(unexpected))}"
            )
        return cls(**{name: data[name] for name in expected_fields})  # type: ignore[arg-type]

    @classmethod
    def from_json_file(cls, path: str | os.PathLike[str]) -> Object3DModelIndex:
        metadata_path = Path(path)
        try:
            contents = metadata_path.read_text(encoding="utf-8")
        except OSError as error:
            raise Object3DLoadingError(f"Could not read object-3D metadata from {metadata_path}") from error
        try:
            data = json.loads(contents)
        except json.JSONDecodeError as error:
            raise Object3DMetadataError(f"Invalid JSON in {metadata_path}: {error.msg}") from error
        return cls.from_dict(data)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        revision: str | None = None,
        subfolder: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> Object3DModelIndex:
        normalized_subfolder = _normalize_subfolder(subfolder)
        reference_path = Path(pretrained_model_name_or_path)
        is_pathlike = isinstance(pretrained_model_name_or_path, os.PathLike)
        is_explicit_local = (
            is_pathlike
            or reference_path.exists()
            or reference_path.is_absolute()
            or str(pretrained_model_name_or_path).startswith(".")
        )

        if is_explicit_local:
            if reference_path.is_file():
                if normalized_subfolder is not None:
                    raise Object3DLoadingError("subfolder cannot be used when loading a metadata file directly")
                metadata_path = reference_path
            else:
                metadata_path = reference_path
                if normalized_subfolder is not None:
                    metadata_path = metadata_path / normalized_subfolder
                metadata_path = metadata_path / OBJECT3D_MODEL_INDEX_NAME
            if not metadata_path.is_file():
                raise Object3DLoadingError(
                    f"Object-3D metadata file {metadata_path} does not exist. "
                    f"Expected {OBJECT3D_MODEL_INDEX_NAME} before loading pipeline components."
                )
            return cls.from_json_file(metadata_path)

        try:
            metadata_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=OBJECT3D_MODEL_INDEX_NAME,
                revision=revision,
                subfolder=normalized_subfolder,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
        except Exception as error:
            raise Object3DLoadingError(
                f"Could not load {OBJECT3D_MODEL_INDEX_NAME} for "
                f"{str(pretrained_model_name_or_path)!r} without loading pipeline components"
            ) from error
        return cls.from_json_file(metadata_path)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
    ) -> Path:
        directory = Path(save_directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise Object3DLoadingError(f"Could not create metadata directory {directory}") from error
        if not directory.is_dir():
            raise Object3DLoadingError(f"Metadata destination {directory} is not a directory")

        destination = directory / OBJECT3D_MODEL_INDEX_NAME
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=directory,
                prefix=f".{OBJECT3D_MODEL_INDEX_NAME}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            os.chmod(destination, 0o644)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise Object3DLoadingError(f"Could not atomically save object-3D metadata to {destination}") from error
        return destination


__all__ = [
    "OBJECT3D_API_VERSION",
    "OBJECT3D_MODEL_INDEX_NAME",
    "OBJECT3D_SCHEMA_VERSION",
    "ContributionStatus",
    "Object3DModelIndex",
    "Object3DModelMetadata",
    "ReviewStatus",
]
