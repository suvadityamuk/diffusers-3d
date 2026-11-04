# Portions of this file are derived from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for package-owned sparse tensor integration.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ...objects import CoordinateSystem, SparseVoxelAsset
from ...objects.base import TensorDataMixin
from .sparse_ops import sparse_batch_positions


def trellis_grid_transform(
    resolution: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Map integer TRELLIS grid coordinates to native ``[-0.5, 0.5]`` cell centers."""

    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] *= 1.0 / resolution
    transform[:3, 3] = 0.5 / resolution - 0.5
    return transform


class _CoordinateCache:
    """Structures derived from one coordinate tensor; not a mapping, so tensor traversal and ``.to`` skip it."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[Any, Any] = {}


@dataclass(frozen=True, slots=True)
class TrellisSparseTensor(TensorDataMixin):
    """Immutable coordinate/feature bridge using TRELLIS ``[batch, x, y, z]`` coordinates."""

    coordinates: torch.Tensor
    features: torch.Tensor
    source_assets: tuple[SparseVoxelAsset, ...] | None = None
    # Shared by every tensor :meth:`replace` derives from this one; see :meth:`cached`.
    _coordinate_cache: _CoordinateCache = field(
        default_factory=_CoordinateCache, init=False, repr=False, compare=False
    )
    _batch_size: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, torch.Tensor) or self.coordinates.ndim != 2:
            raise ValueError("coordinates must be a rank-two tensor")
        if self.coordinates.shape[1] != 4 or self.coordinates.is_floating_point():
            raise ValueError("coordinates must have integer shape (active_voxels, 4)")
        self._check_features(self.coordinates, self.features)
        if bool((self.coordinates < 0).any()):
            raise ValueError("TRELLIS sparse coordinates must be non-negative")
        batch_indices = self.coordinates[:, 0]
        unique_batches = torch.unique(batch_indices, sorted=True)
        batch_size = int(batch_indices.max().item()) + 1
        expected_batches = torch.arange(batch_size, device=batch_indices.device, dtype=batch_indices.dtype)
        if not torch.equal(unique_batches, expected_batches):
            raise ValueError("batch indices must be contiguous and start at zero")
        object.__setattr__(self, "_batch_size", batch_size)
        if torch.unique(self.coordinates, dim=0).shape[0] != self.coordinates.shape[0]:
            raise ValueError("coordinates must not contain duplicate active voxels")
        if self.source_assets is not None:
            assets = tuple(self.source_assets)
            if len(assets) != self.batch_size or any(type(asset) is not SparseVoxelAsset for asset in assets):
                raise ValueError("source_assets must contain one exact SparseVoxelAsset per batch item")
            expected_counts = torch.bincount(batch_indices.to(dtype=torch.int64), minlength=self.batch_size)
            if any(asset.coordinates.shape[0] != int(expected_counts[index]) for index, asset in enumerate(assets)):
                raise ValueError("source asset active-voxel counts must match coordinates")
            object.__setattr__(self, "source_assets", assets)

    @staticmethod
    def _check_features(coordinates: torch.Tensor, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a rank-two tensor")
        if not features.is_floating_point() or features.shape[1] == 0:
            raise ValueError("features must have at least one floating-point channel")
        if coordinates.shape[0] == 0 or features.shape[0] != coordinates.shape[0]:
            raise ValueError("coordinates and features must contain the same non-zero active-voxel count")
        if coordinates.device != features.device:
            raise ValueError("coordinates and features must be on the same device")

    def _derive(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        source_assets: tuple[SparseVoxelAsset, ...] | None,
        coordinate_cache: _CoordinateCache,
    ) -> TrellisSparseTensor:
        """A tensor over this one's already validated coordinates; only ``features`` is checked, without syncing."""

        self._check_features(coordinates, features)
        derived = object.__new__(TrellisSparseTensor)
        for name, value in (
            ("coordinates", coordinates),
            ("features", features),
            ("source_assets", source_assets),
            ("_coordinate_cache", coordinate_cache),
            ("_batch_size", self._batch_size),
        ):
            object.__setattr__(derived, name, value)
        return derived

    def cached(self, key: Any, build: Callable[[], Any]) -> Any:
        """``build()``, computed once per coordinate set and shared with every tensor :meth:`replace` derives.

        Use it for structures that depend only on ``coordinates`` (neighbour tables, pooling indices, batch
        positions), so a sampling loop that keeps the coordinates builds them once rather than every step.
        """

        entries = self._coordinate_cache.entries
        if key not in entries:
            entries[key] = build()
        return entries[key]

    def has_coordinates_of(self, other: TrellisSparseTensor) -> bool:
        """Whether ``other`` has exactly these coordinates; a tensor comparison only the first time per pair."""

        if self.coordinates is other.coordinates:
            return True
        key = ("has_coordinates_of", id(other.coordinates))
        entry = self._coordinate_cache.entries.get(key)
        # The entry keeps ``other.coordinates`` alive, so its id cannot be reused while the entry exists.
        if entry is None or entry[0] is not other.coordinates:
            entry = (other.coordinates, torch.equal(self.coordinates, other.coordinates))
            self._coordinate_cache.entries[key] = entry
        return entry[1]

    def batch_positions(self) -> tuple[torch.Tensor, ...]:
        """Row positions of each batch item, computed once per coordinate set."""

        return self.cached("batch_positions", lambda: sparse_batch_positions(self.coordinates[:, 0], self.batch_size))

    @property
    def coords(self) -> torch.Tensor:
        return self.coordinates

    @property
    def feats(self) -> torch.Tensor:
        return self.features

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def dtype(self) -> torch.dtype:
        return self.features.dtype

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def channels(self) -> int:
        return self.features.shape[1]

    def replace(self, features: torch.Tensor) -> TrellisSparseTensor:
        return self._derive(self.coordinates, features, self.source_assets, self._coordinate_cache)

    def to(self, *args: Any, **kwargs: Any) -> TrellisSparseTensor:
        features = self.features.to(*args, **kwargs)
        if features.device == self.coordinates.device:
            return self.replace(features)
        coordinates = self.coordinates.to(device=features.device)
        source_assets = None
        if self.source_assets is not None:
            source_assets = tuple(asset.to(device=features.device) for asset in self.source_assets)
        return self._derive(coordinates, features, source_assets, _CoordinateCache())

    def _aligned_value(self, value: torch.Tensor | float | TrellisSparseTensor) -> torch.Tensor | float:
        """Operand for elementwise arithmetic; ``(batch, 1 | channels)`` tensors broadcast per batch item."""

        if isinstance(value, TrellisSparseTensor):
            if not torch.equal(self.coordinates, value.coordinates):
                raise ValueError("sparse tensor arithmetic requires identical coordinates")
            return value.features
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[0] == self.batch_size
            and value.shape[1] in (1, self.channels)
        ):
            return value[self.coordinates[:, 0].to(dtype=torch.int64)]
        return value

    def __add__(self, value: torch.Tensor | float | TrellisSparseTensor) -> TrellisSparseTensor:
        return self.replace(self.features + self._aligned_value(value))

    def __mul__(self, value: torch.Tensor | float | TrellisSparseTensor) -> TrellisSparseTensor:
        return self.replace(self.features * self._aligned_value(value))

    @classmethod
    def from_sparse_voxel_assets(
        cls, assets: tuple[SparseVoxelAsset, ...] | list[SparseVoxelAsset]
    ) -> TrellisSparseTensor:
        assets = tuple(assets)
        if not assets or any(type(asset) is not SparseVoxelAsset for asset in assets):
            raise TypeError("assets must contain exact SparseVoxelAsset values")
        device = assets[0].device
        channels = assets[0].features.shape[1]
        if any(asset.device != device or asset.features.shape[1] != channels for asset in assets):
            raise ValueError("assets must share a device and feature-channel count")
        coordinates = []
        features = []
        for batch_index, asset in enumerate(assets):
            asset.validate(expensive=True)
            batch_column = torch.full(
                (asset.coordinates.shape[0], 1), batch_index, device=device, dtype=asset.coordinates.dtype
            )
            coordinates.append(torch.cat([batch_column, asset.coordinates], dim=1))
            features.append(asset.features)
        return cls(torch.cat(coordinates), torch.cat(features), assets)

    def to_sparse_voxel_assets(self, *, resolution: int | None = None) -> tuple[SparseVoxelAsset, ...]:
        assets = []
        for batch_index in range(self.batch_size):
            mask = self.coordinates[:, 0] == batch_index
            coordinates = self.coordinates[mask, 1:].to(dtype=torch.int64)
            features = self.features[mask]
            if self.source_assets is not None:
                source = self.source_assets[batch_index]
                assets.append(
                    SparseVoxelAsset(
                        coordinates=coordinates,
                        features=features,
                        voxel_size=source.voxel_size,
                        grid_transform=source.grid_transform,
                        transform=source.transform,
                        coordinate_system=source.coordinate_system,
                        semantic_labels=source.semantic_labels,
                        extras=source.extras,
                        metadata=source.metadata,
                    )
                )
            else:
                if resolution is None:
                    raise ValueError("resolution is required for sparse tensors without source assets")
                assets.append(
                    SparseVoxelAsset(
                        coordinates=coordinates,
                        features=features,
                        grid_transform=trellis_grid_transform(
                            resolution, device=features.device, dtype=features.dtype
                        ),
                        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                        metadata={"family": "trellis", "resolution": resolution},
                    )
                )
        return tuple(assets)

    def _channel_stats(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype).reshape(1, -1)
        std = torch.as_tensor(std, device=self.device, dtype=self.dtype).reshape(1, -1)
        if mean.shape[1] != self.channels or std.shape[1] != self.channels or bool((std <= 0).any()):
            raise ValueError("mean and positive std must contain one value per feature channel")
        return mean, std

    def normalize(self, mean: torch.Tensor, std: torch.Tensor) -> TrellisSparseTensor:
        mean, std = self._channel_stats(mean, std)
        return self.replace((self.features - mean) / std)

    def denormalize(self, mean: torch.Tensor, std: torch.Tensor) -> TrellisSparseTensor:
        mean, std = self._channel_stats(mean, std)
        return self.replace(self.features * std + mean)


__all__ = ["TrellisSparseTensor", "trellis_grid_transform"]
