from __future__ import annotations

from dataclasses import dataclass, field

import torch
from diffusers.utils import BaseOutput

from ._validation import (
    Object3DValidationError,
    TensorShapeError,
    normalize_extras,
    normalize_metadata,
    validate_extras,
    validate_shared_device,
    validate_tensor,
)
from .base import TensorDataMixin
from .types import Metadata


def _validate_color_channel(
    name: str,
    tensor: torch.Tensor,
    *,
    channels: tuple[int, ...],
    reference_prefix: tuple[int, ...],
    allow_constant: bool = True,
) -> None:
    validate_tensor(name, tensor, floating=True)
    if tensor.ndim == 0 or tensor.shape[-1] not in channels:
        expected = " or ".join(f"(..., {size})" for size in channels)
        raise TensorShapeError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    prefix = tuple(tensor.shape[:-1])
    if prefix != reference_prefix and not (allow_constant and prefix == ()):
        raise TensorShapeError(f"{name} spatial dimensions must match base_color")


def _validate_scalar_texture(name: str, tensor: torch.Tensor, reference_prefix: tuple[int, ...]) -> None:
    validate_tensor(name, tensor, floating=True)
    shape = tuple(tensor.shape)
    valid_shapes = {(), (1,), reference_prefix, reference_prefix + (1,)}
    if shape not in valid_shapes:
        raise TensorShapeError(
            f"{name} must be scalar or match base_color spatial dimensions {reference_prefix}, got {shape}"
        )


@dataclass
class PBRMaterial(BaseOutput, TensorDataMixin):
    """Tensor-native metallic/roughness material with channel-last textures."""

    base_color: torch.Tensor
    metallic: torch.Tensor | None = None
    roughness: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    emissive: torch.Tensor | None = None
    opacity: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extras = normalize_extras(self.extras)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    def validate(self, expensive: bool = False) -> None:
        del expensive
        normalize_metadata(self.metadata)
        validate_tensor("base_color", self.base_color, floating=True)
        if self.base_color.ndim == 0 or self.base_color.shape[-1] not in (3, 4):
            raise TensorShapeError(
                f"base_color must have shape (..., 3) or (..., 4), got {tuple(self.base_color.shape)}"
            )
        if bool(((self.base_color < 0) | (self.base_color > 1)).any()):
            raise Object3DValidationError("base_color values must be in [0, 1]")

        reference_prefix = tuple(self.base_color.shape[:-1])
        for name in ("metallic", "roughness", "opacity"):
            tensor = getattr(self, name)
            if tensor is not None:
                _validate_scalar_texture(name, tensor, reference_prefix)
                if bool(((tensor < 0) | (tensor > 1)).any()):
                    raise Object3DValidationError(f"{name} values must be in [0, 1]")

        if self.normal is not None:
            _validate_color_channel("normal", self.normal, channels=(3,), reference_prefix=reference_prefix)
            if bool((torch.linalg.vector_norm(self.normal.float(), dim=-1) <= 1e-8).any()):
                raise Object3DValidationError("normal vectors must be non-zero")
        if self.emissive is not None:
            _validate_color_channel("emissive", self.emissive, channels=(3,), reference_prefix=reference_prefix)
            if bool((self.emissive < 0).any()):
                raise Object3DValidationError("emissive values must be non-negative")

        validate_extras(self.extras)
        validate_shared_device(self.tensor_items())


__all__ = ["PBRMaterial"]
