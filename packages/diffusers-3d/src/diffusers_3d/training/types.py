from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import final

import torch
from diffusers.optimization import SchedulerType
from torch import nn

from .exceptions import TrainingConfigurationError, TrainingPolicyError

_COMPONENT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_ATTRIBUTE_PATH_PATTERN = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*$")
_PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")


class FineTuneKind(str, Enum):
    """Closed set of supported object-3D fine-tuning strategies."""

    FULL = "full"
    LORA = "lora"


def _normalize_component_keys(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TrainingConfigurationError("components must be a sequence of approved component keys")
    components = tuple(value)
    if not components:
        raise TrainingConfigurationError("components must contain at least one approved component key")
    if any(not isinstance(key, str) or not _COMPONENT_KEY_PATTERN.fullmatch(key) for key in components):
        raise TrainingConfigurationError(
            "component keys must start with a lowercase letter and contain only lowercase letters, digits, '_' or '-'"
        )
    if len(set(components)) != len(components):
        raise TrainingConfigurationError("components must not contain duplicate keys")
    return tuple(sorted(components))


@final
@dataclass(frozen=True, slots=True)
class LoRAFineTune:
    """Recipe-gated LoRA settings; target modules always come from the recipe."""

    components: tuple[str, ...]
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    kind: FineTuneKind = field(default=FineTuneKind.LORA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _normalize_component_keys(self.components))
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise TrainingConfigurationError("rank must be a positive integer")
        if (
            not isinstance(self.alpha, (int, float))
            or isinstance(self.alpha, bool)
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise TrainingConfigurationError("alpha must be positive")
        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise TrainingConfigurationError("dropout must be in [0, 1)")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))


@final
@dataclass(frozen=True, slots=True)
class FullFineTune:
    """Full-parameter settings restricted to recipe-approved components."""

    components: tuple[str, ...]
    kind: FineTuneKind = field(default=FineTuneKind.FULL, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _normalize_component_keys(self.components))


FineTuneStrategy3D = LoRAFineTune | FullFineTune


@final
@dataclass(frozen=True, slots=True)
class ComponentPolicy:
    """Immutable reviewed policy for one user-facing component key."""

    key: str
    component_path: str
    expected_types: tuple[type[nn.Module], ...]
    supported_strategies: tuple[FineTuneKind, ...]
    lora_target_modules: tuple[str, ...] = ()
    full_parameter_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _COMPONENT_KEY_PATTERN.fullmatch(self.key):
            raise TrainingPolicyError("policy key must be a lowercase component key, not a path")
        if not isinstance(self.component_path, str) or (
            self.component_path and not _ATTRIBUTE_PATH_PATTERN.fullmatch(self.component_path)
        ):
            raise TrainingPolicyError("component_path must be an exact dotted attribute path or the empty root path")

        if isinstance(self.expected_types, type) or not isinstance(self.expected_types, Sequence):
            raise TrainingPolicyError("expected_types must be a sequence of exact concrete torch module types")
        expected_types = tuple(self.expected_types)
        if not expected_types:
            raise TrainingPolicyError("expected_types must not be empty")
        for expected_type in expected_types:
            if (
                not isinstance(expected_type, type)
                or expected_type is nn.Module
                or not issubclass(expected_type, nn.Module)
                or inspect.isabstract(expected_type)
            ):
                raise TrainingPolicyError("expected_types must contain exact concrete torch module types")
        if len(set(expected_types)) != len(expected_types):
            raise TrainingPolicyError("expected_types must not contain duplicates")
        object.__setattr__(self, "expected_types", expected_types)

        if isinstance(self.supported_strategies, (str, FineTuneKind)):
            raise TrainingPolicyError("supported_strategies must be a sequence of FineTuneKind values")
        try:
            strategies = tuple(FineTuneKind(value) for value in self.supported_strategies)
        except (TypeError, ValueError) as error:
            raise TrainingPolicyError("supported_strategies must contain FineTuneKind values") from error
        if not strategies or len(set(strategies)) != len(strategies):
            raise TrainingPolicyError("supported_strategies must be non-empty and contain no duplicates")
        object.__setattr__(self, "supported_strategies", tuple(sorted(strategies, key=lambda item: item.value)))

        if isinstance(self.lora_target_modules, str):
            raise TrainingPolicyError("lora_target_modules must contain exact recipe-owned module paths")
        targets = tuple(self.lora_target_modules)
        if any(not isinstance(name, str) or not _ATTRIBUTE_PATH_PATTERN.fullmatch(name) for name in targets):
            raise TrainingPolicyError("lora_target_modules must contain exact dotted module paths, not regexes")
        if len(set(targets)) != len(targets):
            raise TrainingPolicyError("lora_target_modules must not contain duplicates")
        if FineTuneKind.LORA in strategies and not targets:
            raise TrainingPolicyError("a LoRA-capable policy must own at least one exact target module")
        if FineTuneKind.LORA not in strategies and targets:
            raise TrainingPolicyError("lora_target_modules require LoRA support")
        object.__setattr__(self, "lora_target_modules", tuple(sorted(targets)))

        if self.full_parameter_names is not None:
            if isinstance(self.full_parameter_names, str):
                raise TrainingPolicyError("full_parameter_names must contain exact recipe-owned parameter names")
            names = tuple(self.full_parameter_names)
            if any(not isinstance(name, str) or not _PARAMETER_NAME_PATTERN.fullmatch(name) for name in names):
                raise TrainingPolicyError("full_parameter_names must contain exact parameter names, not regexes")
            if len(set(names)) != len(names):
                raise TrainingPolicyError("full_parameter_names must not contain duplicates")
            if FineTuneKind.FULL not in strategies:
                raise TrainingPolicyError("full_parameter_names require full fine-tuning support")
            object.__setattr__(self, "full_parameter_names", tuple(sorted(names)))


@final
@dataclass(frozen=True, slots=True)
class FrozenComponentPolicy:
    """Exact frozen objective dependency that stays unwrapped and in evaluation mode."""

    component_path: str
    expected_types: tuple[type[nn.Module], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.component_path, str) or not _ATTRIBUTE_PATH_PATTERN.fullmatch(self.component_path):
            raise TrainingPolicyError("frozen component_path must be an exact non-empty dotted attribute path")
        if isinstance(self.expected_types, type) or not isinstance(self.expected_types, Sequence):
            raise TrainingPolicyError("frozen expected_types must be a sequence of exact concrete torch module types")
        expected_types = tuple(self.expected_types)
        if not expected_types:
            raise TrainingPolicyError("frozen expected_types must not be empty")
        for expected_type in expected_types:
            if (
                not isinstance(expected_type, type)
                or expected_type is nn.Module
                or not issubclass(expected_type, nn.Module)
                or inspect.isabstract(expected_type)
            ):
                raise TrainingPolicyError("frozen expected_types must contain exact concrete torch module types")
        if len(set(expected_types)) != len(expected_types):
            raise TrainingPolicyError("frozen expected_types must not contain duplicates")
        object.__setattr__(self, "expected_types", expected_types)


@final
@dataclass(frozen=True, slots=True)
class TrainingConfig3D:
    """Compact bounded training configuration."""

    base_model: str
    revision: str | None = None
    dataset_fingerprint: str | None = None
    output_dir: Path | None = None
    train_batch_size: int = 1
    max_train_steps: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler: str = SchedulerType.CONSTANT.value
    lr_warmup_steps: int = 0
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    shuffle: bool = True
    seed: int = 0
    mixed_precision: str = "no"
    cpu: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, str) or not self.base_model:
            raise TrainingConfigurationError("base_model must be a non-empty model identifier")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise TrainingConfigurationError("revision must be a non-empty string or None")
        if self.dataset_fingerprint is not None and (
            not isinstance(self.dataset_fingerprint, str) or not self.dataset_fingerprint.strip()
        ):
            raise TrainingConfigurationError("dataset_fingerprint must be a non-empty string or None")
        if self.output_dir is not None:
            try:
                output_dir = Path(self.output_dir)
            except TypeError as error:
                raise TrainingConfigurationError("output_dir must be path-like or None") from error
            object.__setattr__(self, "output_dir", output_dir)
        for name in ("train_batch_size", "max_train_steps", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TrainingConfigurationError(f"{name} must be a positive integer")
        if (
            not isinstance(self.dataloader_num_workers, int)
            or isinstance(self.dataloader_num_workers, bool)
            or self.dataloader_num_workers < 0
        ):
            raise TrainingConfigurationError("dataloader_num_workers must be a non-negative integer")
        if (
            not isinstance(self.lr_warmup_steps, int)
            or isinstance(self.lr_warmup_steps, bool)
            or self.lr_warmup_steps < 0
        ):
            raise TrainingConfigurationError("lr_warmup_steps must be a non-negative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TrainingConfigurationError("seed must be an integer")
        for name in ("learning_rate", "adam_epsilon", "max_grad_norm"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise TrainingConfigurationError(f"{name} must be positive")
        if (
            not isinstance(self.weight_decay, (int, float))
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TrainingConfigurationError("weight_decay must be non-negative")
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise TrainingConfigurationError(f"{name} must be in [0, 1)")
        try:
            scheduler = SchedulerType(self.lr_scheduler).value
        except (TypeError, ValueError) as error:
            raise TrainingConfigurationError(f"unsupported Diffusers scheduler {self.lr_scheduler!r}") from error
        object.__setattr__(self, "lr_scheduler", scheduler)
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise TrainingConfigurationError("mixed_precision must be 'no', 'fp16', or 'bf16'")
        if not isinstance(self.shuffle, bool) or not isinstance(self.cpu, bool):
            raise TrainingConfigurationError("shuffle and cpu must be booleans")

    def resume_config(self) -> dict[str, bool | float | int | str | None]:
        """Return the canonical training settings that must match on resume."""

        return {
            "adam_beta1": float(self.adam_beta1),
            "adam_beta2": float(self.adam_beta2),
            "adam_epsilon": float(self.adam_epsilon),
            "cpu": self.cpu,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataset_fingerprint": self.dataset_fingerprint,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": float(self.learning_rate),
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_steps": self.lr_warmup_steps,
            "max_grad_norm": float(self.max_grad_norm),
            "max_train_steps": self.max_train_steps,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "train_batch_size": self.train_batch_size,
            "weight_decay": float(self.weight_decay),
        }


MetricValue = float | torch.Tensor


@final
@dataclass(frozen=True, slots=True)
class TrainingStep3DOutput:
    """Validated scalar training loss and optional named scalar metrics."""

    loss: torch.Tensor
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.loss, torch.Tensor) or self.loss.ndim != 0 or not self.loss.is_floating_point():
            raise TrainingConfigurationError("loss must be a scalar floating-point torch.Tensor")
        if not bool(torch.isfinite(self.loss)):
            raise TrainingConfigurationError("loss must be finite")
        if not isinstance(self.metrics, Mapping):
            raise TrainingConfigurationError("metrics must be a mapping of names to scalar values")
        metrics: dict[str, MetricValue] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise TrainingConfigurationError("metric names must be non-empty strings")
            if isinstance(value, torch.Tensor):
                if value.ndim != 0 or not bool(torch.isfinite(value)):
                    raise TrainingConfigurationError(f"metric {name!r} must be a finite scalar tensor")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TrainingConfigurationError(f"metric {name!r} must be a scalar number or tensor")
            else:
                value = float(value)
            metrics[name] = value
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@final
@dataclass(frozen=True, slots=True)
class TrainingSummary3D:
    """Compact CPU-only summary for one bounded trainer run."""

    final_loss: float | None
    final_metrics: Mapping[str, float] = field(default_factory=dict)
    micro_steps: int = 0
    optimizer_steps: int = 0

    def __post_init__(self) -> None:
        if self.final_loss is not None and (
            not isinstance(self.final_loss, (int, float))
            or isinstance(self.final_loss, bool)
            or not math.isfinite(self.final_loss)
        ):
            raise TrainingConfigurationError("final_loss must be a finite number or None")
        if not isinstance(self.final_metrics, Mapping):
            raise TrainingConfigurationError("final_metrics must be a mapping")
        metrics = {}
        for name, value in self.final_metrics.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise TrainingConfigurationError("final_metrics must contain non-empty names and finite numbers")
            metrics[name] = float(value)
        for name in ("micro_steps", "optimizer_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TrainingConfigurationError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "final_loss", None if self.final_loss is None else float(self.final_loss))
        object.__setattr__(self, "final_metrics", MappingProxyType(metrics))


__all__ = [
    "ComponentPolicy",
    "FineTuneKind",
    "FineTuneStrategy3D",
    "FrozenComponentPolicy",
    "FullFineTune",
    "LoRAFineTune",
    "MetricValue",
    "TrainingConfig3D",
    "TrainingSummary3D",
    "TrainingStep3DOutput",
]
