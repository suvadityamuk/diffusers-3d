from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

ExampleT_co = TypeVar("ExampleT_co", covariant=True)


@runtime_checkable
class Object3DDataset(Protocol[ExampleT_co]):
    """Runtime-checkable map-style dataset of typed object-3D examples."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ExampleT_co: ...


__all__ = ["Object3DDataset"]
