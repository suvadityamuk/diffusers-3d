"""Row lookup for integer voxel coordinates by sorting their linear keys and binary searching them."""

from __future__ import annotations

import torch

_CUBE_CORNERS = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])


class VoxelIndex:
    """Map unique ``(x, y, z)`` voxel coordinates on a ``grid_size`` grid to their row.

    :meth:`lookup` returns ``-1`` for voxels that are absent or outside the grid.
    """

    def __init__(self, coordinates: torch.Tensor, grid_size: tuple[int, int, int]) -> None:
        self.grid_size = tuple(int(size) for size in grid_size)
        keys = self._keys(coordinates)
        if bool((keys < 0).any()):
            raise ValueError("indexed coordinates must lie inside the grid")
        self._sorted_keys, self._rows = torch.sort(keys)

    def _keys(self, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = coordinates.to(torch.int64)
        inside = ((coordinates >= 0) & (coordinates < coordinates.new_tensor(self.grid_size))).all(-1)
        _, height, depth = self.grid_size
        keys = (coordinates[..., 0] * height + coordinates[..., 1]) * depth + coordinates[..., 2]
        return torch.where(inside, keys, -1)

    def lookup(self, coordinates: torch.Tensor) -> torch.Tensor:
        keys = self._keys(coordinates)
        if not self._sorted_keys.numel():
            return torch.full_like(keys, -1)
        position = torch.searchsorted(self._sorted_keys, keys).clamp(max=self._sorted_keys.numel() - 1)
        found = (keys >= 0) & (self._sorted_keys[position] == keys)
        return torch.where(found, self._rows[position], -1)


def trilinear_sample(index: VoxelIndex, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Interpolate per-voxel ``features`` at ``points`` in voxel units, with voxel centres at ``+0.5``.

    Absent corners are skipped and the remaining weights renormalised, as FlexGEMM's ``grid_sample_3d`` does.
    """

    base = torch.floor(points - 0.5).to(torch.int64)
    samples = torch.zeros(points.shape[0], features.shape[1], device=features.device, dtype=features.dtype)
    total = torch.zeros(points.shape[0], 1, device=points.device, dtype=points.dtype)
    for offset in _CUBE_CORNERS.to(points.device):
        corner = base + offset
        rows = index.lookup(corner)
        weight = (1 - (points - corner - 0.5).abs()).prod(-1, keepdim=True) * (rows >= 0).unsqueeze(-1)
        samples += weight.to(features.dtype) * features[rows.clamp(min=0)]
        total += weight
    return samples / total.clamp(min=1e-12).to(features.dtype)


__all__ = ["VoxelIndex", "trilinear_sample"]
