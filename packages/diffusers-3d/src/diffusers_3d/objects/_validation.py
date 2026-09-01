from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .types import CoordinateSystem, JSONValue, Metadata


class Object3DValidationError(ValueError):
    """Base exception for invalid public 3D data."""


class TensorShapeError(Object3DValidationError):
    """Raised when a tensor does not follow its declared shape contract."""


class TensorDTypeError(Object3DValidationError):
    """Raised when a tensor uses an unsupported dtype."""


class TensorDeviceError(Object3DValidationError):
    """Raised when tensors belonging to one value are on different devices."""


class MetadataValidationError(Object3DValidationError):
    """Raised when metadata is not JSON-safe."""


def identity_transform() -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32)


def normalize_coordinate_system(value: CoordinateSystem | str) -> CoordinateSystem:
    try:
        return CoordinateSystem(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in CoordinateSystem)
        raise Object3DValidationError(f"coordinate_system must be one of: {choices}") from error


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetadataValidationError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MetadataValidationError(f"{path} keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise MetadataValidationError(
        f"{path} contains {type(value).__name__}; metadata values must be JSON-safe scalars, lists, or dictionaries"
    )


def normalize_metadata(value: Mapping[str, JSONValue] | None) -> Metadata:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MetadataValidationError("metadata must be a mapping")
    metadata = dict(value)
    _validate_json_value(metadata, "metadata")
    return metadata


def normalize_extras(value: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Object3DValidationError("extras must be a mapping of names to tensors")
    extras = dict(value)
    for name, tensor in extras.items():
        if not isinstance(name, str) or not name:
            raise Object3DValidationError("extras keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise Object3DValidationError(f"extras[{name!r}] must be a torch.Tensor")
    return extras


def validate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    rank: int | None = None,
    trailing_shape: tuple[int, ...] | None = None,
    floating: bool = False,
    integer: bool = False,
    finite: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise Object3DValidationError(f"{name} must be a torch.Tensor")
    if rank is not None and tensor.ndim != rank:
        raise TensorShapeError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")
    if trailing_shape is not None and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise TensorShapeError(f"{name} must end with shape {trailing_shape}, got {tuple(tensor.shape)}")
    if floating and not tensor.is_floating_point():
        raise TensorDTypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if integer and tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TensorDTypeError(f"{name} must have an integer dtype, got {tensor.dtype}")
    if finite and (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
        raise Object3DValidationError(f"{name} must contain only finite values")


def validate_transform(name: str, transform: torch.Tensor, *, batched: bool = False) -> None:
    expected_rank = 3 if batched else 2
    validate_tensor(name, transform, rank=expected_rank, trailing_shape=(4, 4), floating=True)
    expected_bottom_row = transform.new_tensor([0.0, 0.0, 0.0, 1.0]).expand_as(transform[..., 3, :])
    if not torch.allclose(transform[..., 3, :], expected_bottom_row, atol=1e-5, rtol=0.0):
        raise Object3DValidationError(f"{name} must contain affine homogeneous transforms")
    linear = transform[..., :3, :3].float()
    if bool((torch.linalg.matrix_rank(linear) < 3).any()):
        raise Object3DValidationError(f"{name} must contain invertible transforms")


def validate_shared_device(items: tuple[tuple[str, torch.Tensor], ...]) -> None:
    if not items:
        return
    expected = items[0][1].device
    for name, tensor in items[1:]:
        if tensor.device != expected:
            raise TensorDeviceError(f"{name} is on {tensor.device}, expected all tensors on {expected}")


def validate_extras(
    extras: Mapping[str, torch.Tensor],
    *,
    allowed_first_dimensions: set[int] | None = None,
) -> None:
    if not isinstance(extras, Mapping):
        raise Object3DValidationError("extras must be a mapping of names to tensors")
    normalized_extras = normalize_extras(extras)
    for name, tensor in normalized_extras.items():
        validate_tensor(f"extras[{name!r}]", tensor)
        if tensor.ndim == 0:
            raise TensorShapeError(f"extras[{name!r}] must have at least one dimension")
        if allowed_first_dimensions is not None and tensor.shape[0] not in allowed_first_dimensions:
            expected = ", ".join(str(size) for size in sorted(allowed_first_dimensions))
            raise TensorShapeError(f"extras[{name!r}] first dimension must be one of ({expected})")


def validate_scalar_channel(name: str, tensor: torch.Tensor, count: int) -> None:
    validate_tensor(name, tensor, floating=True)
    if tensor.ndim not in (1, 2) or tensor.shape[0] != count or (tensor.ndim == 2 and tensor.shape[1] != 1):
        raise TensorShapeError(f"{name} must have shape ({count},) or ({count}, 1), got {tuple(tensor.shape)}")


__all__ = [
    "MetadataValidationError",
    "Object3DValidationError",
    "TensorDTypeError",
    "TensorDeviceError",
    "TensorShapeError",
]
