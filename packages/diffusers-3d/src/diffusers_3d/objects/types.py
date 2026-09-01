from __future__ import annotations

from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

import torch


class Object3DKind(str, Enum):
    """Native object representations supported by the public data contract."""

    MESH = "mesh"
    GAUSSIAN_SPLAT = "gaussian_splat"
    SPARSE_VOXEL = "sparse_voxel"
    O_VOXEL = "o_voxel"


class CoordinateSystem(str, Enum):
    """Handedness and up-axis convention used by object-space values."""

    RIGHT_HANDED_Y_UP = "right_handed_y_up"
    RIGHT_HANDED_Z_UP = "right_handed_z_up"
    LEFT_HANDED_Y_UP = "left_handed_y_up"
    LEFT_HANDED_Z_UP = "left_handed_z_up"


JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
Metadata: TypeAlias = dict[str, JSONValue]


@runtime_checkable
class Object3D(Protocol):
    """Structural contract implemented by all object-native assets."""

    @property
    def kind(self) -> Object3DKind: ...

    @property
    def coordinate_system(self) -> CoordinateSystem: ...

    @property
    def object_to_world(self) -> torch.Tensor: ...

    @property
    def device(self) -> torch.device: ...

    def tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]: ...

    def validate(self, expensive: bool = False) -> None: ...

    def to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Object3D: ...


__all__ = [
    "CoordinateSystem",
    "JSONPrimitive",
    "JSONValue",
    "Metadata",
    "Object3D",
    "Object3DKind",
]
