from __future__ import annotations

from typing import Protocol, runtime_checkable

from .conditions import Object3DExample


@runtime_checkable
class Object3DDataset(Protocol):
    """Runtime-checkable map-style dataset of typed object-3D examples."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Object3DExample: ...


__all__ = ["Object3DDataset"]
