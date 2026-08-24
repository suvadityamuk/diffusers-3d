# Portions of this file are derived from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for package-owned sparse tensor integration.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...objects import CoordinateSystem, SparseVoxelAsset
from ...objects.base import TensorDataMixin


def trellis_grid_transform(
    resolution: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map integer TRELLIS grid coordinates to native ``[-0.5, 0.5]`` cell centers."""

    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] *= 1.0 / resolution
    transform[:3, 3] = 0.5 / resolution - 0.5
    return transform


@dataclass(frozen=True, slots=True)
class TrellisSparseTensor(TensorDataMixin):
    """Immutable coordinate/feature bridge using TRELLIS ``[batch, x, y, z]`` coordinates."""

    coordinates: torch.Tensor
    features: torch.Tensor
    source_assets: tuple[SparseVoxelAsset, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, torch.Tensor) or self.coordinates.ndim != 2:
            raise ValueError("coordinates must be a rank-two tensor")
        if self.coordinates.shape[1] != 4 or self.coordinates.is_floating_point():
            raise ValueError("coordinates must have integer shape (active_voxels, 4)")
        if not isinstance(self.features, torch.Tensor) or self.features.ndim != 2:
            raise ValueError("features must be a rank-two tensor")
        if not self.features.is_floating_point() or self.features.shape[1] == 0:
            raise ValueError("features must have at least one floating-point channel")
        if self.coordinates.shape[0] == 0 or self.features.shape[0] != self.coordinates.shape[0]:
            raise ValueError("coordinates and features must contain the same non-zero active-voxel count")
        if self.coordinates.device != self.features.device:
            raise ValueError("coordinates and features must be on the same device")
        if bool((self.coordinates < 0).any()):
            raise ValueError("TRELLIS sparse coordinates must be non-negative")
        batch_indices = self.coordinates[:, 0]
        unique_batches = torch.unique(batch_indices, sorted=True)
        expected_batches = torch.arange(
            int(batch_indices.max().item()) + 1,
            device=batch_indices.device,
            dtype=batch_indices.dtype,
        )
        if not torch.equal(unique_batches, expected_batches):
            raise ValueError("batch indices must be contiguous and start at zero")
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
        return int(self.coordinates[:, 0].max().item()) + 1

    @property
    def channels(self) -> int:
        return self.features.shape[1]

    def replace(self, features: torch.Tensor) -> TrellisSparseTensor:
        return TrellisSparseTensor(self.coordinates, features, self.source_assets)

    def to(self, *args: Any, **kwargs: Any) -> TrellisSparseTensor:
        features = self.features.to(*args, **kwargs)
        coordinates = self.coordinates.to(device=features.device)
        source_assets = None
        if self.source_assets is not None:
            source_assets = tuple(asset.to(device=features.device) for asset in self.source_assets)
        return TrellisSparseTensor(coordinates, features, source_assets)

    def _aligned_value(self, value: torch.Tensor | float) -> torch.Tensor | float:
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim == 2 and value.shape[0] == self.batch_size and value.shape[1] in (1, self.channels):
            return value[self.coordinates[:, 0].to(dtype=torch.int64)]
        return value

    def __add__(self, value: torch.Tensor | float | TrellisSparseTensor) -> TrellisSparseTensor:
        if isinstance(value, TrellisSparseTensor):
            if not torch.equal(self.coordinates, value.coordinates):
                raise ValueError("sparse tensor arithmetic requires identical coordinates")
            value = value.features
        return self.replace(self.features + self._aligned_value(value))

    def __mul__(self, value: torch.Tensor | float | TrellisSparseTensor) -> TrellisSparseTensor:
        if isinstance(value, TrellisSparseTensor):
            if not torch.equal(self.coordinates, value.coordinates):
                raise ValueError("sparse tensor arithmetic requires identical coordinates")
            value = value.features
        return self.replace(self.features * self._aligned_value(value))

    @classmethod
    def from_sparse_voxel_assets(
        cls,
        assets: tuple[SparseVoxelAsset, ...] | list[SparseVoxelAsset],
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
                (asset.coordinates.shape[0], 1),
                batch_index,
                device=device,
                dtype=asset.coordinates.dtype,
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
                            resolution,
                            device=features.device,
                            dtype=features.dtype,
                        ),
                        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                        metadata={"family": "trellis", "resolution": resolution},
                    )
                )
        return tuple(assets)

    def normalize(self, mean: torch.Tensor, std: torch.Tensor) -> TrellisSparseTensor:
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype).reshape(1, -1)
        std = torch.as_tensor(std, device=self.device, dtype=self.dtype).reshape(1, -1)
        if mean.shape[1] != self.channels or std.shape[1] != self.channels or bool((std <= 0).any()):
            raise ValueError("mean and positive std must contain one value per feature channel")
        return self.replace((self.features - mean) / std)

    def denormalize(self, mean: torch.Tensor, std: torch.Tensor) -> TrellisSparseTensor:
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype).reshape(1, -1)
        std = torch.as_tensor(std, device=self.device, dtype=self.dtype).reshape(1, -1)
        if mean.shape[1] != self.channels or std.shape[1] != self.channels or bool((std <= 0).any()):
            raise ValueError("mean and positive std must contain one value per feature channel")
        return self.replace(self.features * std + mean)


__all__ = ["TrellisSparseTensor", "trellis_grid_transform"]
