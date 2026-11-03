from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from diffusers_3d.backends.occupancy import (
    dense_grid_points,
    hierarchical_occupancy,
    near_surface_mask,
    octree_resolutions,
)

BOUNDS = (-1.05, -0.9, -1.0, 1.05, 1.1, 0.95)


def _sphere(points: torch.Tensor) -> torch.Tensor:
    return 40 * (0.55 - points.float().norm(dim=-1)).to(points.dtype)


def test_dense_grid_points_match_numpy_linspace_bitwise():
    axes = [np.linspace(BOUNDS[axis], BOUNDS[axis + 3], 17, dtype=np.float32) for axis in range(3)]
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    actual = dense_grid_points(BOUNDS, 16)
    assert actual.dtype == torch.float32
    assert np.array_equal(actual.numpy(), expected)


def test_octree_resolutions_halve_down_to_the_minimum():
    assert octree_resolutions(384) == [96, 192, 384]
    assert octree_resolutions(252) == [63, 126, 252]
    assert octree_resolutions(63) == [63]
    assert octree_resolutions(16) == [16]


def test_near_surface_mask_flags_sign_changes_against_clamped_neighbours():
    generator = torch.Generator().manual_seed(0)
    grid = torch.randn(5, 6, 4, generator=generator)
    grid[0, 0, 0] = -10000.0
    mask = near_surface_mask(grid, 0.0)
    expected = torch.zeros_like(mask)
    for index in itertools.product(*map(range, grid.shape)):
        value = grid[index]
        if value <= -9000:
            continue
        for dim, step in itertools.product(range(3), (1, -1)):
            neighbour = list(index)
            neighbour[dim] = min(max(neighbour[dim] + step, 0), grid.shape[dim] - 1)
            other = grid[tuple(neighbour)]
            other = value if other <= -9000 else other
            if torch.sign(other) != torch.sign(value):
                expected[index] = 1
    assert mask.dtype == torch.int32
    assert torch.equal(mask, expected)


def test_single_level_is_the_dense_grid():
    with pytest.warns(UserWarning, match="Tencent Hunyuan 3D 2.1 Community License"):
        grid = hierarchical_occupancy(
            lambda points: _sphere(points[0])[None], BOUNDS, [16], dtype=torch.float32, device="cpu", num_chunks=100
        )
    expected = _sphere(dense_grid_points(BOUNDS, 16)).view(17, 17, 17)
    assert torch.equal(grid, expected)


@pytest.mark.filterwarnings("ignore:Hierarchical occupancy decoding")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_refined_levels_evaluate_the_surface_band_at_index_times_spacing(dtype):
    def query(points: torch.Tensor) -> torch.Tensor:
        assert points.dtype == dtype and points.shape[0] == 1
        return _sphere(points[0])[None]

    grid = hierarchical_occupancy(query, BOUNDS, [31, 62, 124], dtype=dtype, device="cpu", num_chunks=5000)
    assert grid.shape == (125, 125, 125) and grid.dtype == dtype
    evaluated = torch.isfinite(grid)
    assert 0.01 < evaluated.float().mean() < 0.5

    # Refined points are ``index * spacing + minimum`` in the grid dtype.
    indices = torch.nonzero(evaluated)
    spacing = torch.tensor((np.array(BOUNDS[3:]) - np.array(BOUNDS[:3])) / 124, dtype=dtype)
    points = indices * spacing + torch.tensor(BOUNDS[:3], dtype=dtype)
    assert torch.equal(grid[evaluated], _sphere(points))

    # Every cell the surface crosses has all eight corners evaluated.
    dense = _sphere(dense_grid_points(BOUNDS, 124).to(dtype)).view(125, 125, 125)
    inside = dense > 0
    corners = [inside[x : x + 124, y : y + 124, z : z + 124] for x, y, z in itertools.product((0, 1), repeat=3)]
    crossing = torch.stack(corners).any(0) & ~torch.stack(corners).all(0)
    for x, y, z in itertools.product((0, 1), repeat=3):
        assert evaluated[x : x + 124, y : y + 124, z : z + 124][crossing].all()


def test_point_dtype_and_empty_value_are_configurable():
    with pytest.warns(UserWarning):
        grid = hierarchical_occupancy(
            lambda points: _sphere(points[0].float())[None].half(),
            BOUNDS,
            [31, 62],
            dtype=torch.float16,
            device="cpu",
            point_dtype=torch.float32,
            empty_value=-1.0,
        )
    assert not torch.isnan(grid).any() and (grid == -1.0).any()
