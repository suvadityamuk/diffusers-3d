from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from diffusers.loaders import PeftAdapterMixin
from torch import nn

from ..execution import ModularObject3DPipeline, Object3DModel, Object3DPipeline
from .exceptions import TrainingCheckpointError
from .types import (
    ComponentPolicy,
    FineTuneStrategy3D,
    FrozenComponentPolicy,
    FullFineTune,
    LoRAFineTune,
    TrainingStep3DOutput,
)

TRAINING_ADAPTER_NAME = "object3d_training"

TargetT = TypeVar("TargetT", bound=Object3DModel | Object3DPipeline | ModularObject3DPipeline)
ExampleT = TypeVar("ExampleT")
BatchT = TypeVar("BatchT")


class TrainingRecipe3D(ABC, Generic[TargetT, ExampleT, BatchT]):
    """Reviewed model-specific objective over an exact object-3D target."""

    recipe_id: ClassVar[str]
    recipe_version: ClassVar[str]
    family_id: ClassVar[str]
    target_type: ClassVar[type[Object3DModel] | type[Object3DPipeline] | type[ModularObject3DPipeline]]
    example_type: ClassVar[type[object]]
    batch_type: ClassVar[type[object]]
    component_policies: ClassVar[tuple[ComponentPolicy, ...]]
    frozen_component_policies: ClassVar[tuple[FrozenComponentPolicy, ...]] = ()

    def __init__(self, target: TargetT) -> None:
        self._target = target

    @property
    def target(self) -> TargetT:
        return self._target

    def objective_config(self) -> Mapping[str, bool | float | int | str | None]:
        """Return canonical JSON-safe settings that define this objective."""

        return {}

    @staticmethod
    def component_config(component: nn.Module) -> object:
        """Read config from an exact component or its Accelerator wrapper."""

        wrapped = getattr(component, "module", None)
        return getattr(wrapped if isinstance(wrapped, nn.Module) else component, "config")

    @abstractmethod
    def collate(self, examples: Sequence[ExampleT]) -> BatchT:
        """Build the recipe's exact typed batch."""

    @abstractmethod
    def validate_target(self) -> None:
        """Validate model-specific target invariants without mutating the target."""

    @abstractmethod
    def compute_loss(self, batch: BatchT) -> TrainingStep3DOutput:
        """Compute the training objective directly from components, never pipeline inference."""

    def save_weights(
        self,
        save_directory: str | Path,
        strategy: FineTuneStrategy3D,
        components: Mapping[str, nn.Module],
    ) -> None:
        """Save selected weights through public component APIs."""

        directory = Path(save_directory)
        for key in strategy.components:
            component = components[key]
            component_directory = directory / key
            if type(strategy) is LoRAFineTune:
                if not isinstance(component, PeftAdapterMixin):
                    raise TrainingCheckpointError(
                        f"LoRA component {key!r} does not expose the public PeftAdapterMixin save API"
                    )
                component.save_lora_adapter(component_directory, adapter_name=TRAINING_ADAPTER_NAME)
            elif type(strategy) is FullFineTune:
                save_pretrained = getattr(component, "save_pretrained", None)
                if not callable(save_pretrained):
                    raise TrainingCheckpointError(
                        f"Full component {key!r} has no save_pretrained API; its recipe must override save_weights"
                    )
                save_pretrained(component_directory)
            else:
                raise TrainingCheckpointError(f"unsupported fine-tuning strategy {type(strategy).__name__}")


__all__ = ["TRAINING_ADAPTER_NAME", "TrainingRecipe3D"]
