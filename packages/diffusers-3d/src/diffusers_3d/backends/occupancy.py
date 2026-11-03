# Portions of this file are derived from Tencent Hunyuan3D-2.1 (`hy3dshape/models/autoencoders/volume_decoders.py`):
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: one shape at a time, an explicit query-point dtype, plain PyTorch neighbour shifts.

"""Hierarchical occupancy decoding: a coarse dense grid, refined only near the surface, as Hunyuan3D-2.1 does."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Hunyuan3D marks cells it never evaluated with this value before replacing them with ``empty_value``.
_UNEVALUATED = -10000.0


def dense_grid_points(bounds: Sequence[float], resolution: int) -> torch.Tensor:
    """The ``(resolution + 1) ** 3`` float32 points of ``(x_min, y_min, z_min, x_max, y_max, z_max)``, ``ij`` order.

    Built with ``numpy.linspace`` in float32, as upstream, so coordinates match it bit for bit.
    """

    axes = [np.linspace(bounds[axis], bounds[axis + 3], resolution + 1, dtype=np.float32) for axis in range(3)]
    return torch.from_numpy(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))


def octree_resolutions(octree_resolution: int, min_resolution: int = 63) -> list[int]:
    """Halve ``octree_resolution`` while it is at least ``min_resolution``; coarsest first."""

    resolutions = [octree_resolution] if octree_resolution < min_resolution else []
    while octree_resolution >= min_resolution:
        resolutions.append(octree_resolution)
        octree_resolution //= 2
    return resolutions[::-1]


def near_surface_mask(grid: torch.Tensor, level: float) -> torch.Tensor:
    """``int32`` mask of evaluated samples whose sign differs from one of their six (edge-clamped) neighbours."""

    values = grid + level
    valid = values > -9000
    sign = torch.sign(values.float())
    differs = torch.zeros_like(valid)
    for dim in range(3):
        size = values.shape[dim]
        for step in (1, -1):
            index = (torch.arange(size, device=values.device) + step).clamp(0, size - 1)
            neighbour = values.index_select(dim, index)
            neighbour = torch.where(neighbour > -9000, neighbour, values)
            differs |= torch.sign(neighbour.float()) != sign
    return differs.to(torch.int32) * valid.to(torch.int32)


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    """Upstream's all-ones ``3x3x3`` convolution: counts set neighbours, so any positive value is "set"."""

    weight = torch.ones(1, 1, 3, 3, 3, dtype=mask.dtype, device=mask.device)
    return F.conv3d(mask[None, None], weight, padding=1)[0, 0]


def refinement_indices(
    grid: torch.Tensor,
    resolution: int,
    *,
    level: float = 0.0,
    finest: bool,
    candidate_dilations: int = 1,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices of the ``(resolution + 1) ** 3`` grid to evaluate after the coarser ``grid``.

    Coarse samples that change sign against a neighbour or have ``|logit| < 0.95`` are candidates; unless this is
    the ``finest`` level they are dilated ``candidate_dilations`` times (Hunyuan3D-2.1 once, TripoSG twice). Their
    doubled positions are dilated once, or twice on the finest level. Masks are held in ``dtype`` (default
    ``grid.dtype``).
    """

    dtype = grid.dtype if dtype is None else dtype
    candidates = near_surface_mask(grid, level)
    candidates += grid.abs() < 0.95
    if not finest:
        for _ in range(candidate_dilations):
            candidates = _dilate(candidates.to(dtype))
    coarse = torch.nonzero(candidates > 0, as_tuple=True)
    selected = torch.zeros((resolution + 1,) * 3, dtype=dtype, device=grid.device)
    selected[coarse[0] * 2, coarse[1] * 2, coarse[2] * 2] = 1
    for _ in range(2 if finest else 1):
        selected = _dilate(selected)
    return torch.nonzero(selected > 0, as_tuple=True)


@torch.no_grad()
def hierarchical_occupancy(
    query: Callable[[torch.Tensor], torch.Tensor],
    bounds: Sequence[float],
    resolutions: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    level: float = 0.0,
    num_chunks: int = 65536,
    point_dtype: torch.dtype | None = None,
    empty_value: float = float("nan"),
) -> torch.Tensor:
    """Occupancy logits of one shape on the finest grid of ``resolutions``, evaluated only near the surface.

    ``query`` maps ``(1, points, 3)`` positions to ``(1, points)`` logits. The coarsest grid is evaluated densely.
    Each finer level doubles the grid and evaluates the neighbourhood of every coarse sample that changes sign
    against a neighbour or has ``|logit| < 0.95``, dilated once (twice on the finest level's input side).
    Unevaluated samples hold ``empty_value``. The grid is held in ``dtype``; refined points are computed in
    ``point_dtype`` (default ``dtype``) as ``index * spacing + minimum``.
    """

    warnings.warn(
        "Hierarchical occupancy decoding is derived from Tencent Hunyuan3D-2.1, licensed under the Tencent Hunyuan "
        "3D 2.1 Community License, which does not apply in the European Union, the United Kingdom, or South Korea. "
        "You are responsible for making sure your use of it complies with that license.",
        UserWarning,
        stacklevel=3,
    )
    point_dtype = dtype if point_dtype is None else point_dtype
    bbox_min = np.array(bounds[:3], dtype=np.float64)
    bbox_size = np.array(bounds[3:], dtype=np.float64) - bbox_min

    def evaluate(points: torch.Tensor) -> torch.Tensor:
        return torch.cat([query(chunk[None].to(dtype)) for chunk in points.split(num_chunks)], dim=1)[0]

    points = dense_grid_points(bounds, resolutions[0]).to(device=device, dtype=dtype)
    grid = evaluate(points).view(*(resolutions[0] + 1,) * 3)
    for index, resolution in enumerate(resolutions[1:], start=1):
        fine = refinement_indices(grid, resolution, level=level, finest=index == len(resolutions) - 1, dtype=dtype)
        points = torch.stack(fine, dim=1) * torch.tensor(bbox_size / resolution, dtype=point_dtype, device=device)
        points = points + torch.tensor(bbox_min, dtype=point_dtype, device=device)
        grid = torch.full((resolution + 1,) * 3, _UNEVALUATED, dtype=dtype, device=device)
        grid[fine] = evaluate(points).to(dtype)
    grid[grid == _UNEVALUATED] = empty_value
    return grid


__all__ = [
    "dense_grid_points",
    "hierarchical_occupancy",
    "near_surface_mask",
    "octree_resolutions",
    "refinement_indices",
]
