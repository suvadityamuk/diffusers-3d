from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from diffusers import __version__ as diffusers_version

from .._version import __version__ as package_version
from ..execution.metadata import fully_qualified_class_name
from .exceptions import TrainingManifestError, TrainingManifestMismatchError
from .types import FineTuneStrategy3D, LoRAFineTune

TRAINING_MANIFEST_NAME = "diffusers_3d_training.json"
TRAINING_MANIFEST_SCHEMA = "diffusers-3d-training"
TRAINING_MANIFEST_VERSION = 3

StrategyConfigValue = int | float | str
ConfigValue = bool | int | float | str | None
_QUALIFIED_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def trainable_parameter_hash(names: tuple[str, ...]) -> str:
    """Hash a canonical sorted list of exact trainable parameter names."""

    payload = json.dumps(sorted(names), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_config(value: Mapping[str, ConfigValue], field_name: str) -> tuple[tuple[str, ConfigValue], ...]:
    if not isinstance(value, Mapping):
        raise TrainingManifestError(f"{field_name} must contain a JSON object")
    entries = []
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise TrainingManifestError(f"{field_name} keys must be non-empty strings")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise TrainingManifestError(f"{field_name} values must be JSON-safe scalars")
        if isinstance(item, float) and not math.isfinite(item):
            raise TrainingManifestError(f"{field_name} floating-point values must be finite")
        entries.append((name, item))
    return tuple(sorted(entries))


@dataclass(frozen=True, slots=True)
class TrainingManifest3D:
    """Deterministic local checkpoint identity for exact resume validation."""

    schema: str
    schema_version: int
    target_type: str
    example_type: str
    family_id: str
    recipe_id: str
    recipe_version: str
    strategy: str
    strategy_config: tuple[tuple[str, StrategyConfigValue], ...]
    objective_config: tuple[tuple[str, ConfigValue], ...]
    training_config: tuple[tuple[str, ConfigValue], ...]
    components: tuple[str, ...]
    base_model: str
    revision: str | None
    package_version: str
    diffusers_version: str
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_hash: str

    def __post_init__(self) -> None:
        if (
            self.schema != TRAINING_MANIFEST_SCHEMA
            or not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != TRAINING_MANIFEST_VERSION
        ):
            raise TrainingManifestError(
                f"Unsupported training manifest schema {self.schema!r} version {self.schema_version!r}"
            )
        string_fields = (
            "target_type",
            "example_type",
            "family_id",
            "recipe_id",
            "recipe_version",
            "strategy",
            "base_model",
            "package_version",
            "diffusers_version",
        )
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name) for name in string_fields):
            raise TrainingManifestError("training manifest identity fields must be non-empty strings")
        if not _QUALIFIED_TYPE_PATTERN.fullmatch(self.target_type):
            raise TrainingManifestError("target_type must be a fully-qualified concrete type")
        if not _QUALIFIED_TYPE_PATTERN.fullmatch(self.example_type):
            raise TrainingManifestError("example_type must be a fully-qualified concrete type")
        if self.strategy not in ("full", "lora"):
            raise TrainingManifestError("strategy must be 'full' or 'lora'")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise TrainingManifestError("training manifest revision must be a non-empty string or None")
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or tuple(sorted(self.components)) != self.components
        ):
            raise TrainingManifestError("training manifest components must be a non-empty sorted tuple")
        if len(set(self.components)) != len(self.components) or any(
            not isinstance(component, str) or not component for component in self.components
        ):
            raise TrainingManifestError("training manifest components must contain unique non-empty strings")
        if not isinstance(self.strategy_config, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2 for entry in self.strategy_config
        ):
            raise TrainingManifestError("training manifest strategy_config must contain name/value pairs")
        for name, value in self.strategy_config:
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float, str))
            ):
                raise TrainingManifestError("training manifest strategy_config contains an invalid entry")
        if tuple(sorted(self.strategy_config, key=lambda entry: entry[0])) != self.strategy_config:
            raise TrainingManifestError("training manifest strategy_config must be a sorted tuple")
        if len({name for name, _ in self.strategy_config}) != len(self.strategy_config):
            raise TrainingManifestError("training manifest strategy_config must not contain duplicate names")
        config_names = {name for name, _ in self.strategy_config}
        if (self.strategy == "full" and config_names) or (
            self.strategy == "lora" and config_names != {"alpha", "dropout", "rank"}
        ):
            raise TrainingManifestError("strategy_config does not match the declared strategy")
        if self.strategy == "lora":
            config = dict(self.strategy_config)
            rank = config["rank"]
            alpha = config["alpha"]
            dropout = config["dropout"]
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                raise TrainingManifestError("LoRA strategy rank must be a positive integer")
            if (
                not isinstance(alpha, (int, float))
                or isinstance(alpha, bool)
                or not math.isfinite(alpha)
                or alpha <= 0
            ):
                raise TrainingManifestError("LoRA strategy alpha must be positive and finite")
            if (
                not isinstance(dropout, (int, float))
                or isinstance(dropout, bool)
                or not math.isfinite(dropout)
                or not 0 <= dropout < 1
            ):
                raise TrainingManifestError("LoRA strategy dropout must be finite and in [0, 1)")
        for field_name in ("objective_config", "training_config"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(not isinstance(entry, tuple) or len(entry) != 2 for entry in value):
                raise TrainingManifestError(f"training manifest {field_name} must contain name/value pairs")
            names = [entry[0] for entry in value]
            if len(set(names)) != len(names):
                raise TrainingManifestError(f"training manifest {field_name} must not contain duplicate names")
            if _canonical_config(dict(value), field_name) != value:
                raise TrainingManifestError(f"training manifest {field_name} must be canonical and sorted")
        if (
            not isinstance(self.trainable_parameter_names, tuple)
            or not self.trainable_parameter_names
            or tuple(sorted(self.trainable_parameter_names)) != self.trainable_parameter_names
        ):
            raise TrainingManifestError("trainable_parameter_names must be a non-empty sorted tuple")
        if len(set(self.trainable_parameter_names)) != len(self.trainable_parameter_names) or any(
            not isinstance(name, str) or not name for name in self.trainable_parameter_names
        ):
            raise TrainingManifestError("trainable_parameter_names must contain unique non-empty strings")
        if not isinstance(self.trainable_parameter_hash, str) or not _HASH_PATTERN.fullmatch(
            self.trainable_parameter_hash
        ):
            raise TrainingManifestError("trainable_parameter_hash must be a lowercase SHA-256 digest")
        expected_hash = trainable_parameter_hash(self.trainable_parameter_names)
        if self.trainable_parameter_hash != expected_hash:
            raise TrainingManifestError("trainable_parameter_hash does not match trainable_parameter_names")

    @classmethod
    def create(
        cls,
        *,
        target_type: type[object],
        example_type: type[object],
        family_id: str,
        recipe_id: str,
        recipe_version: str,
        strategy: FineTuneStrategy3D,
        base_model: str,
        revision: str | None,
        trainable_parameter_names: tuple[str, ...],
        objective_config: Mapping[str, ConfigValue],
        training_config: Mapping[str, ConfigValue],
    ) -> TrainingManifest3D:
        names = tuple(sorted(trainable_parameter_names))
        strategy_config: tuple[tuple[str, StrategyConfigValue], ...] = ()
        if type(strategy) is LoRAFineTune:
            strategy_config = (
                ("alpha", strategy.alpha),
                ("dropout", strategy.dropout),
                ("rank", strategy.rank),
            )
        return cls(
            schema=TRAINING_MANIFEST_SCHEMA,
            schema_version=TRAINING_MANIFEST_VERSION,
            target_type=fully_qualified_class_name(target_type),
            example_type=fully_qualified_class_name(example_type),
            family_id=family_id,
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            strategy=strategy.kind.value,
            strategy_config=strategy_config,
            objective_config=_canonical_config(objective_config, "objective_config"),
            training_config=_canonical_config(training_config, "training_config"),
            components=tuple(sorted(strategy.components)),
            base_model=base_model,
            revision=revision,
            package_version=package_version,
            diffusers_version=diffusers_version,
            trainable_parameter_names=names,
            trainable_parameter_hash=trainable_parameter_hash(names),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "base_model": self.base_model,
            "components": list(self.components),
            "diffusers_version": self.diffusers_version,
            "example_type": self.example_type,
            "family_id": self.family_id,
            "objective_config": dict(self.objective_config),
            "package_version": self.package_version,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "revision": self.revision,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "strategy_config": dict(self.strategy_config),
            "target_type": self.target_type,
            "training_config": dict(self.training_config),
            "trainable_parameter_hash": self.trainable_parameter_hash,
            "trainable_parameter_names": list(self.trainable_parameter_names),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TrainingManifest3D:
        if not isinstance(data, Mapping):
            raise TrainingManifestError("training manifest must contain a JSON object")
        expected = {field.name for field in fields(cls)}
        missing = expected.difference(data)
        unexpected = set(data).difference(expected)
        if missing:
            raise TrainingManifestError(f"training manifest is missing fields: {', '.join(sorted(missing))}")
        if unexpected:
            raise TrainingManifestError(f"training manifest has unknown fields: {', '.join(sorted(unexpected))}")
        strategy_config = data["strategy_config"]
        if not isinstance(strategy_config, Mapping):
            raise TrainingManifestError("strategy_config must contain a JSON object")
        objective_config = data["objective_config"]
        training_config = data["training_config"]
        if not isinstance(objective_config, Mapping) or not isinstance(training_config, Mapping):
            raise TrainingManifestError("objective_config and training_config must contain JSON objects")
        if not isinstance(data["components"], list):
            raise TrainingManifestError("components must contain a JSON array")
        if not isinstance(data["trainable_parameter_names"], list):
            raise TrainingManifestError("trainable_parameter_names must contain a JSON array")
        try:
            return cls(
                schema=data["schema"],  # type: ignore[arg-type]
                schema_version=data["schema_version"],  # type: ignore[arg-type]
                target_type=data["target_type"],  # type: ignore[arg-type]
                example_type=data["example_type"],  # type: ignore[arg-type]
                family_id=data["family_id"],  # type: ignore[arg-type]
                recipe_id=data["recipe_id"],  # type: ignore[arg-type]
                recipe_version=data["recipe_version"],  # type: ignore[arg-type]
                strategy=data["strategy"],  # type: ignore[arg-type]
                strategy_config=tuple(sorted(strategy_config.items())),  # type: ignore[arg-type]
                objective_config=_canonical_config(objective_config, "objective_config"),  # type: ignore[arg-type]
                training_config=_canonical_config(training_config, "training_config"),  # type: ignore[arg-type]
                components=tuple(data["components"]),  # type: ignore[arg-type]
                base_model=data["base_model"],  # type: ignore[arg-type]
                revision=data["revision"],  # type: ignore[arg-type]
                package_version=data["package_version"],  # type: ignore[arg-type]
                diffusers_version=data["diffusers_version"],  # type: ignore[arg-type]
                trainable_parameter_names=tuple(data["trainable_parameter_names"]),  # type: ignore[arg-type]
                trainable_parameter_hash=data["trainable_parameter_hash"],  # type: ignore[arg-type]
            )
        except TrainingManifestError:
            raise
        except (TypeError, ValueError) as error:
            raise TrainingManifestError("training manifest fields have invalid types") from error

    @classmethod
    def load(cls, checkpoint_directory: str | os.PathLike[str]) -> TrainingManifest3D:
        path = Path(checkpoint_directory) / TRAINING_MANIFEST_NAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise TrainingManifestError(f"Could not read training manifest from {path}") from error
        except json.JSONDecodeError as error:
            raise TrainingManifestError(f"Invalid JSON in {path}: {error.msg}") from error
        return cls.from_dict(data)

    def save(self, checkpoint_directory: str | os.PathLike[str]) -> Path:
        directory = Path(checkpoint_directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TrainingManifestError(f"Could not create checkpoint directory {directory}") from error
        destination = directory / TRAINING_MANIFEST_NAME
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=directory,
                prefix=f".{TRAINING_MANIFEST_NAME}.",
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
            raise TrainingManifestError(f"Could not atomically save training manifest to {destination}") from error
        return destination

    def validate_resume(self, expected: TrainingManifest3D) -> None:
        if not isinstance(expected, TrainingManifest3D):
            raise TypeError("expected must be a TrainingManifest3D")
        differences = [
            field.name for field in fields(self) if getattr(self, field.name) != getattr(expected, field.name)
        ]
        if differences:
            raise TrainingManifestMismatchError(
                f"Training resume manifest does not exactly match: {', '.join(differences)}"
            )


__all__ = [
    "TRAINING_MANIFEST_NAME",
    "TRAINING_MANIFEST_SCHEMA",
    "TRAINING_MANIFEST_VERSION",
    "TrainingManifest3D",
    "trainable_parameter_hash",
]
