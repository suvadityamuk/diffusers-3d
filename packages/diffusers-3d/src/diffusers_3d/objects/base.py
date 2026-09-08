from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, TypeVar

import torch

from .types import Object3D

TensorDataT = TypeVar("TensorDataT", bound="TensorDataMixin")


def _iter_tensors(value: Any, prefix: str) -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, TensorDataMixin):
        return [(f"{prefix}.{name}" if prefix else name, tensor) for name, tensor in value.tensor_items()]
    if isinstance(value, Object3D):
        return [(f"{prefix}.{name}" if prefix else name, tensor) for name, tensor in value.tensor_items()]
    if isinstance(value, Mapping):
        tensors = []
        for name, item in value.items():
            item_prefix = f"{prefix}.{name}" if prefix else str(name)
            tensors.extend(_iter_tensors(item, item_prefix))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for index, item in enumerate(value):
            item_prefix = f"{prefix}.{index}" if prefix else str(index)
            tensors.extend(_iter_tensors(item, item_prefix))
        return tensors
    return []


def _move_value(
    value: Any,
    *,
    device: torch.device | str | int | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if value.is_floating_point() or value.is_complex() else None
        return value.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
    if isinstance(value, TensorDataMixin):
        return value.to(device=device, dtype=dtype, non_blocking=non_blocking)
    if isinstance(value, Object3D):
        return value.to(device=device, dtype=dtype, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            name: _move_value(item, device=device, dtype=dtype, non_blocking=non_blocking)
            for name, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_value(item, device=device, dtype=dtype, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device=device, dtype=dtype, non_blocking=non_blocking) for item in value]
    return value


class TensorDataMixin:
    """Shared tensor traversal and functional device/dtype conversion."""

    @property
    def device(self) -> torch.device:
        items = self.tensor_items()
        if not items:
            raise RuntimeError(f"{type(self).__name__} does not contain a tensor")
        return items[0][1].device

    def tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        tensors = []
        for field_info in fields(self):
            tensors.extend(_iter_tensors(getattr(self, field_info.name), field_info.name))
        return tuple(tensors)

    def to(
        self: TensorDataT,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> TensorDataT:
        values = {
            field_info.name: _move_value(
                getattr(self, field_info.name),
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for field_info in fields(self)
            if field_info.init
        }
        return type(self)(**values)


__all__ = ["TensorDataMixin"]
