from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from diffusers.utils import BaseOutput

from ._validation import (
    Object3DValidationError,
    TensorShapeError,
    normalize_metadata,
    validate_shared_device,
    validate_tensor,
    validate_transform,
)
from .base import TensorDataMixin
from .types import CoordinateSystem, Metadata, Object3D, Object3DKind


@dataclass
class Latent3DOutput(BaseOutput, TensorDataMixin):
    """Unmaterialized tensor latents produced by a 3D generation stage."""

    latents: torch.Tensor
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    def validate(self, expensive: bool = False) -> None:
        del expensive
        normalize_metadata(self.metadata)
        validate_tensor("latents", self.latents, floating=True)
        if self.latents.ndim < 2:
            raise TensorShapeError("latents must have at least batch and feature dimensions")


@dataclass
class Object3DPipelineOutput(BaseOutput, TensorDataMixin):
    """Stable pipeline result whose first value is always a non-empty object tuple."""

    objects: tuple[Object3D, ...]
    latents: Latent3DOutput | torch.Tensor | None = None
    previews: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        try:
            self.objects = tuple(self.objects)
        except TypeError as error:
            raise Object3DValidationError("objects must be an iterable of Object3D values") from error
        if self.previews is not None:
            if not isinstance(self.previews, (list, tuple)):
                raise Object3DValidationError("previews must be a sequence")
            self.previews = tuple(self.previews)
        self.validate()
        super().__post_init__()

    def validate(self, expensive: bool = False) -> None:
        if not self.objects:
            raise Object3DValidationError("objects must contain at least one Object3D value")
        for index, obj in enumerate(self.objects):
            if not isinstance(obj, Object3D):
                raise Object3DValidationError(f"objects[{index}] does not implement the Object3D protocol")
            if not isinstance(obj.kind, Object3DKind):
                raise Object3DValidationError(f"objects[{index}].kind must be an Object3DKind")
            if not isinstance(obj.coordinate_system, CoordinateSystem):
                raise Object3DValidationError(f"objects[{index}].coordinate_system must be a CoordinateSystem")
            validate_transform(f"objects[{index}].object_to_world", obj.object_to_world)
            items = obj.tensor_items()
            if not isinstance(items, tuple) or not items:
                raise Object3DValidationError(f"objects[{index}].tensor_items() must return a non-empty tuple")
            for name, tensor in items:
                if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                    raise Object3DValidationError(
                        f"objects[{index}].tensor_items() entries must be (str, torch.Tensor) pairs"
                    )
            validate_shared_device(items)
            if obj.object_to_world.device != obj.device:
                raise Object3DValidationError(f"objects[{index}].object_to_world must be on the object's device")
            obj.validate(expensive=expensive)

        if isinstance(self.latents, Latent3DOutput):
            self.latents.validate(expensive=expensive)
        elif isinstance(self.latents, torch.Tensor):
            validate_tensor("latents", self.latents, floating=True)
            if self.latents.ndim < 2:
                raise TensorShapeError("latents must have at least batch and feature dimensions")
        elif self.latents is not None:
            raise Object3DValidationError("latents must be a Latent3DOutput, torch.Tensor, or None")


__all__ = ["Latent3DOutput", "Object3DPipelineOutput"]
