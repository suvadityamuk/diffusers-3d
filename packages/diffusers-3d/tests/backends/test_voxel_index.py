from __future__ import annotations

import math

import pytest
import torch

from diffusers_3d.backends import VoxelIndex, trilinear_sample


def _random_voxels(count: int, grid_size: tuple[int, int, int], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, math.prod(grid_size), (2 * count,), generator=generator).unique()
    keys = keys[torch.randperm(keys.numel(), generator=generator)[:count]]
    _, height, depth = grid_size
    return torch.stack([keys // (height * depth), keys // depth % height, keys % depth], dim=1).int()


def _brute_force_lookup(coordinates: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    rows = {tuple(coordinate): row for row, coordinate in enumerate(coordinates.tolist())}
    return torch.tensor([rows.get(tuple(query), -1) for query in queries.reshape(-1, 3).tolist()]).reshape(
        queries.shape[:-1]
    )


def test_lookup_matches_brute_force_for_present_absent_and_out_of_grid_voxels():
    grid_size = (6, 9, 7)
    coordinates = _random_voxels(150, grid_size)
    index = VoxelIndex(coordinates, grid_size)
    queries = torch.randint(-2, 11, (40, 8, 3), generator=torch.Generator().manual_seed(1))
    queries[:5, 0] = coordinates[:5]

    rows = index.lookup(queries)

    assert rows.dtype is torch.int64 and rows.shape == (40, 8)
    assert torch.equal(rows, _brute_force_lookup(coordinates, queries))
    assert torch.equal(index.lookup(coordinates), torch.arange(150))


def test_empty_index_and_out_of_grid_insertion():
    empty = VoxelIndex(torch.zeros(0, 3, dtype=torch.int32), (4, 4, 4))
    assert torch.equal(empty.lookup(torch.tensor([[0, 0, 0], [1, 2, 3]])), torch.tensor([-1, -1]))
    with pytest.raises(ValueError, match="inside the grid"):
        VoxelIndex(torch.tensor([[0, 0, 4]]), (4, 4, 4))


def _flex_gemm_trilinear(coordinates, features, grid_size, points):
    """Line-by-line transcription of FlexGEMM's ``trilinear_neighbor_map_weight`` kernel plus its weighted sum."""

    rows = {tuple(coordinate): row for row, coordinate in enumerate(coordinates.tolist())}
    width, height, depth = grid_size
    output = torch.zeros(points.shape[0], features.shape[1])
    for point_index, (qx, qy, qz) in enumerate(points.tolist()):
        base = (math.floor(qx - 0.5), math.floor(qy - 0.5), math.floor(qz - 0.5))
        neighbors, weights = [], []
        for i in range(8):
            x, y, z = base[0] + (i & 1), base[1] + ((i >> 1) & 1), base[2] + ((i >> 2) & 1)
            if 0 <= x < width and 0 <= y < height and 0 <= z < depth and (x, y, z) in rows:
                neighbors.append(rows[(x, y, z)])
                weights.append((1 - abs(qx - x - 0.5)) * (1 - abs(qy - y - 0.5)) * (1 - abs(qz - z - 0.5)))
        total = max(sum(weights), 1e-12)
        for row, weight in zip(neighbors, weights):
            output[point_index] += weight / total * features[row]
    return output


def test_trilinear_sample_matches_flex_gemm_kernel():
    grid_size = (5, 6, 4)
    coordinates = _random_voxels(60, grid_size, seed=2)
    features = torch.rand(60, 6, generator=torch.Generator().manual_seed(3))
    points = torch.rand(300, 3, generator=torch.Generator().manual_seed(4)) * torch.tensor([6.0, 7.0, 5.0]) - 0.5
    points[0] = torch.tensor([40.0, 40.0, 40.0])  # no corner present: zero, as upstream

    sampled = trilinear_sample(VoxelIndex(coordinates, grid_size), features, points)

    torch.testing.assert_close(sampled, _flex_gemm_trilinear(coordinates, features, grid_size, points))
    assert torch.equal(sampled[0], torch.zeros(6))
