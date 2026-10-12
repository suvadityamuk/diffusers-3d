# Portions of this file reproduce operator semantics from Microsoft TRELLIS and TRELLIS.2:
# https://github.com/microsoft/TRELLIS (revision 442aa1e1afb9014e80681d3bf604e8d728a86ee7)
# https://github.com/microsoft/TRELLIS.2 (revision 75fbf0183001ed9876c8dbb35de6b68552ee08bd)
#
# MIT License. Copyright (c) Microsoft Corporation.
# Reimplemented with plain tensor operations; no compiled sparse kernels are vendored.

"""Pure-PyTorch sparse voxel primitives shared by the TRELLIS families.

Coordinates are integer ``(N, 4)`` tensors in ``[batch, x, y, z]`` order and features are ``(N, C)``. The
functions here reproduce what the released checkpoints were trained with (spconv / FlexGEMM submanifold
convolution, TRELLIS average-pool down/upsampling, TRELLIS.2 channel-to-spatial subdivision, and windowed
self-attention) using gather/scatter and dense attention, so every stage runs on any device. They trade
speed for portability: the compiled backends in ``diffusers_3d.backends`` remain the fast path on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _check_coordinates(coordinates: torch.Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or coordinates.is_floating_point():
        raise ValueError("coordinates must be an integer tensor of shape (N, 4) in [batch, x, y, z] order")


def _linear_codes(coordinates: torch.Tensor, extents: torch.Tensor) -> torch.Tensor:
    """Row-major integer code for each ``[batch, x, y, z]`` row given per-column extents."""

    strides = torch.ones_like(extents)
    strides[:-1] = torch.flip(torch.cumprod(torch.flip(extents[1:], dims=(0,)), dim=0), dims=(0,))
    return (coordinates.to(torch.int64) * strides).sum(dim=1)


# ---------------------------------------------------------------------------
# Submanifold convolution
# ---------------------------------------------------------------------------


def submanifold_neighbors(
    coordinates: torch.Tensor,
    kernel_size: int = 3,
    dilation: int = 1,
) -> torch.Tensor:
    """Index of the voxel at ``p + offset_k`` for each active voxel ``p`` and kernel tap ``k``.

    Returns an ``(N, kernel_size**3)`` int64 tensor with ``-1`` where the neighbour is not active. Taps are
    ordered like the kernel axes of the weight tensor: ``meshgrid(dx, dy, dz, indexing="ij")`` with each
    offset running from ``-(kernel_size // 2) * dilation`` to ``+(kernel_size // 2) * dilation``. This is the
    convention of both spconv (``SubMConv3d``) and FlexGEMM (``sparse_submanifold_conv3d``).
    """

    _check_coordinates(coordinates)
    if kernel_size <= 0 or kernel_size % 2 == 0 or dilation <= 0:
        raise ValueError("kernel_size must be a positive odd integer and dilation positive")
    half = kernel_size // 2
    axis = torch.arange(-half * dilation, half * dilation + 1, dilation, device=coordinates.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)

    coordinates = coordinates.to(torch.int64)
    minimum = coordinates.amin(dim=0)
    shifted = coordinates - minimum
    extents = shifted.amax(dim=0) + 1
    codes = _linear_codes(shifted, extents)
    sorted_codes, sorted_index = torch.sort(codes)

    neighbours = shifted[:, None, :].repeat(1, offsets.shape[0], 1)
    neighbours[:, :, 1:] += offsets[None]
    inside = ((neighbours >= 0) & (neighbours < extents)).all(dim=-1)
    neighbour_codes = _linear_codes(neighbours.clamp(min=0).reshape(-1, 4), extents).reshape(neighbours.shape[:2])
    position = torch.searchsorted(sorted_codes, neighbour_codes).clamp(max=sorted_codes.shape[0] - 1)
    found = inside & (sorted_codes[position] == neighbour_codes)
    return torch.where(found, sorted_index[position], torch.full_like(position, -1))


def sparse_submanifold_conv3d(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    dilation: int = 1,
    neighbors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Submanifold 3D convolution: the output is defined on the input's active voxels only.

    ``weight`` has the spconv/FlexGEMM layout ``(out_channels, k, k, k, in_channels)``. The result equals a
    dense ``Conv3d`` (cross-correlation, ``padding=k // 2``, zeros elsewhere) evaluated at the active voxels.
    Pass a precomputed ``neighbors`` table from :func:`submanifold_neighbors` to reuse it across layers.
    """

    if weight.ndim != 5 or weight.shape[1] != weight.shape[2] or weight.shape[1] != weight.shape[3]:
        raise ValueError("weight must have shape (out_channels, k, k, k, in_channels)")
    out_channels, kernel_size = weight.shape[0], weight.shape[1]
    if features.ndim != 2 or features.shape[1] != weight.shape[4]:
        raise ValueError(f"features must have shape (N, {weight.shape[4]})")
    if neighbors is None:
        neighbors = submanifold_neighbors(coordinates, kernel_size, dilation)
    elif neighbors.shape != (features.shape[0], kernel_size**3):
        raise ValueError("neighbors table does not match the features and kernel size")

    taps = weight.reshape(out_channels, kernel_size**3, weight.shape[4])
    # Sum the taps in float32 so half-precision runs do not lose bits on every one of the 27 additions.
    accumulate_dtype = torch.float32 if features.dtype in (torch.float16, torch.bfloat16) else features.dtype
    output = features.new_zeros(features.shape[0], out_channels, dtype=accumulate_dtype)
    if bias is not None:
        output += bias.to(dtype=output.dtype)
    center = kernel_size**3 // 2
    for tap in range(kernel_size**3):
        if tap == center:
            output += features @ taps[:, tap].T
            continue
        index = neighbors[:, tap]
        rows = torch.nonzero(index >= 0, as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue
        output.index_add_(0, rows, (features[index[rows]] @ taps[:, tap].T).to(dtype=accumulate_dtype))
    return output.to(dtype=features.dtype)


class SparseConv3d(nn.Module):
    """Submanifold convolution module with the released ``(out, k, k, k, in)`` weight layout.

    State-dict keys are ``weight`` and ``bias``, matching FlexGEMM-trained TRELLIS.2 checkpoints. TRELLIS
    checkpoints wrap the same tensors one level deeper (``conv.weight``); see the TRELLIS models module.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, *, bias: bool = True) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size) <= 0 or kernel_size % 2 == 0:
            raise ValueError("channels must be positive and kernel_size a positive odd integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        *,
        neighbors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sparse_submanifold_conv3d(coordinates, features, self.weight, self.bias, neighbors=neighbors)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def sparse_downsample(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    factor: int = 2,
    *,
    count_zero_buffer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average-pool active voxels into ``factor``-sized cells (TRELLIS ``SparseDownsample``).

    Returns ``(coarse_coordinates, pooled_features, inverse)`` where ``inverse[i]`` is the coarse row that
    fine voxel ``i`` fell into; feed it to :func:`sparse_upsample` to go back.

    TRELLIS (v1) pools with ``scatter_reduce(reduce="mean")`` into a zero buffer and leaves the default
    ``include_self=True``, so every cell divides by ``n + 1`` rather than ``n``. The released SLAT flow
    weights were trained that way; pass ``count_zero_buffer=True`` to reproduce it.
    """

    _check_coordinates(coordinates)
    if factor <= 0:
        raise ValueError("factor must be positive")
    coarse = coordinates.clone()
    coarse[:, 1:] = torch.div(coarse[:, 1:], factor, rounding_mode="floor")
    coarse, inverse = torch.unique(coarse, dim=0, return_inverse=True)
    counts = torch.bincount(inverse, minlength=coarse.shape[0]).to(dtype=features.dtype)
    if count_zero_buffer:
        counts = counts + 1
    pooled = features.new_zeros(coarse.shape[0], features.shape[1]).index_add_(0, inverse, features)
    return coarse, pooled / counts[:, None], inverse


def sparse_upsample(features: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour unpooling back onto the fine voxels recorded by :func:`sparse_downsample`."""

    return features[inverse]


def sparse_subdivide(
    coordinates: torch.Tensor,
    subdivision: torch.Tensor,
    factor: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand each voxel into the children selected by ``subdivision`` (TRELLIS.2 ``SparseChannel2Spatial``).

    ``subdivision`` is a boolean ``(N, factor**3)`` mask; child ``s`` sits at offset
    ``(s % f, s // f % f, s // f**2 % f)``. Returns ``(child_coordinates, parent_index, child_index)``.
    """

    _check_coordinates(coordinates)
    if subdivision.shape != (coordinates.shape[0], factor**3) or subdivision.dtype != torch.bool:
        raise ValueError(f"subdivision must be a boolean tensor of shape (N, {factor**3})")
    parent_index, child_index = torch.nonzero(subdivision, as_tuple=True)
    children = coordinates[parent_index].to(torch.int64)
    children[:, 1:] *= factor
    children[:, 1] += child_index % factor
    children[:, 2] += torch.div(child_index, factor, rounding_mode="floor") % factor
    children[:, 3] += torch.div(child_index, factor**2, rounding_mode="floor") % factor
    return children.to(dtype=coordinates.dtype), parent_index, child_index


def channel_to_spatial(
    features: torch.Tensor,
    parent_index: torch.Tensor,
    child_index: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Hand child ``s`` the ``s``-th channel block of its parent: ``(N, f**3 * C) -> (M, C)``."""

    if features.shape[1] % factor**3:
        raise ValueError(f"feature channels must be divisible by {factor**3}")
    blocks = features.reshape(features.shape[0] * factor**3, features.shape[1] // factor**3)
    return blocks[parent_index * factor**3 + child_index]


# ---------------------------------------------------------------------------
# Windowed attention
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseWindowPartition:
    """Grouping of sparse tokens into dense, zero-padded attention windows."""

    order: torch.Tensor
    window: torch.Tensor
    slot: torch.Tensor
    num_windows: int
    max_tokens: int

    @property
    def key_mask(self) -> torch.Tensor:
        """``(num_windows, 1, 1, max_tokens)`` boolean mask that is ``True`` on real keys."""

        mask = torch.zeros(self.num_windows, self.max_tokens, dtype=torch.bool, device=self.order.device)
        mask[self.window, self.slot] = True
        return mask[:, None, None, :]

    def pad(self, features: torch.Tensor) -> torch.Tensor:
        padded = features.new_zeros(self.num_windows, self.max_tokens, *features.shape[1:])
        padded[self.window, self.slot] = features[self.order]
        return padded

    def unpad(self, padded: torch.Tensor) -> torch.Tensor:
        output = padded.new_empty(self.order.shape[0], *padded.shape[2:])
        output[self.order] = padded[self.window, self.slot]
        return output


def sparse_window_partition(
    coordinates: torch.Tensor,
    window_size: int,
    shift: int = 0,
) -> SparseWindowPartition:
    """Partition tokens into ``window_size`` cubes per batch item, optionally shifted (TRELLIS ``swin``)."""

    _check_coordinates(coordinates)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    cells = coordinates.to(torch.int64).clone()
    cells[:, 1:] = torch.div(cells[:, 1:] + shift, window_size, rounding_mode="floor")
    _, window = torch.unique(cells, dim=0, return_inverse=True)
    window, order = torch.sort(window, stable=True)
    counts = torch.bincount(window)
    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(order.shape[0], device=order.device) - starts[window]
    return SparseWindowPartition(
        order=order,
        window=window,
        slot=slot,
        num_windows=int(counts.shape[0]),
        max_tokens=int(counts.max().item()),
    )


__all__ = [
    "SparseConv3d",
    "SparseWindowPartition",
    "channel_to_spatial",
    "sparse_downsample",
    "sparse_subdivide",
    "sparse_submanifold_conv3d",
    "sparse_upsample",
    "sparse_window_partition",
    "submanifold_neighbors",
]
