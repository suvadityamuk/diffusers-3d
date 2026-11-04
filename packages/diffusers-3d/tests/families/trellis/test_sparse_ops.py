"""Pure-PyTorch sparse primitives against dense references."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffusers_3d.families.trellis import sparse_ops
from diffusers_3d.families.trellis.sparse_ops import (
    SparseConv3d,
    SubmanifoldNeighbors,
    channel_to_spatial,
    set_sparse_conv_backend,
    sparse_downsample,
    sparse_subdivide,
    sparse_submanifold_conv3d,
    sparse_upsample,
    sparse_window_partition,
    submanifold_neighbors,
)


def _random_coordinates(batch_size: int, resolution: int, count: int, generator: torch.Generator) -> torch.Tensor:
    dense = torch.zeros(batch_size, resolution, resolution, resolution, dtype=torch.bool)
    flat = torch.randperm(dense.numel(), generator=generator)[:count]
    dense.view(-1)[flat] = True
    return torch.nonzero(dense)


def _densify(coordinates: torch.Tensor, features: torch.Tensor, batch_size: int, resolution: int) -> torch.Tensor:
    dense = features.new_zeros(batch_size, features.shape[1], resolution, resolution, resolution)
    dense[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = features
    return dense


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_submanifold_conv_matches_dense_conv3d_at_active_voxels(kernel_size):
    generator = torch.Generator().manual_seed(0)
    batch_size, resolution = 2, 6
    coordinates = _random_coordinates(batch_size, resolution, 60, generator)
    features = torch.randn(coordinates.shape[0], 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(7, kernel_size, kernel_size, kernel_size, 5, generator=generator, dtype=torch.float64)
    bias = torch.randn(7, generator=generator, dtype=torch.float64)

    sparse = sparse_submanifold_conv3d(coordinates, features, weight, bias)

    dense = F.conv3d(
        _densify(coordinates, features, batch_size, resolution),
        weight.permute(0, 4, 1, 2, 3),
        bias,
        padding=kernel_size // 2,
    )
    expected = dense[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    torch.testing.assert_close(sparse, expected)


def test_submanifold_neighbors_are_reusable_and_marked_missing():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    neighbors = submanifold_neighbors(coordinates, kernel_size=3)
    assert neighbors.shape == (3, 27)
    center = 13
    assert neighbors[:, center].tolist() == [0, 1, 2]
    # Tap (+1, 0, 0) from voxel 0 hits voxel 1 in batch 0; batch 1 has no such neighbour.
    plus_x = center + 9
    assert neighbors[0, plus_x] == 1 and neighbors[2, plus_x] == -1

    module = SparseConv3d(4, 3, kernel_size=3).double()
    features = torch.randn(3, 4, dtype=torch.float64)
    torch.testing.assert_close(
        module(coordinates, features, neighbors=neighbors), module(coordinates, features), rtol=0, atol=0
    )
    assert set(module.state_dict()) == {"weight", "bias"}
    assert module.weight.shape == (3, 3, 3, 3, 4)


def test_downsample_upsample_round_trip_and_zero_buffer_mean():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 1]])
    features = torch.tensor([[2.0], [4.0], [8.0], [16.0]])

    coarse, pooled, inverse = sparse_downsample(coordinates, features)
    assert coarse.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    torch.testing.assert_close(pooled, torch.tensor([[3.0], [8.0], [16.0]]))
    torch.testing.assert_close(sparse_upsample(pooled, inverse), torch.tensor([[3.0], [3.0], [8.0], [16.0]]))

    _, upstream_v1, _ = sparse_downsample(coordinates, features, count_zero_buffer=True)
    torch.testing.assert_close(upstream_v1, torch.tensor([[2.0], [4.0], [8.0]]))


def test_subdivide_and_channel_to_spatial_follow_the_child_layout():
    coordinates = torch.tensor([[0, 1, 2, 3], [1, 0, 0, 0]])
    subdivision = torch.zeros(2, 8, dtype=torch.bool)
    subdivision[0, [0, 7]] = True  # children (0,0,0) and (1,1,1)
    subdivision[1, 1] = True  # child (1,0,0)

    children, parent_index, child_index = sparse_subdivide(coordinates, subdivision)
    assert children.tolist() == [[0, 2, 4, 6], [0, 3, 5, 7], [1, 1, 0, 0]]
    assert parent_index.tolist() == [0, 0, 1] and child_index.tolist() == [0, 7, 1]

    features = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 24)
    scattered = channel_to_spatial(features, parent_index, child_index)
    expected = torch.stack([features[0, 0:3], features[0, 21:24], features[1, 3:6]])
    torch.testing.assert_close(scattered, expected)

    with pytest.raises(ValueError):
        sparse_subdivide(coordinates, subdivision.float())


def test_window_partition_groups_per_batch_and_pads_losslessly():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 0, 0], [0, 3, 3, 3], [1, 0, 0, 0], [1, 1, 0, 0]])
    partition = sparse_window_partition(coordinates, window_size=2)
    # Batch 0 splits into (0..1)^3 -> {0, 1}, (2..3, 0..1, 0..1) -> {2}, (2..3)^3 -> {3}; batch 1 is one window.
    assert partition.num_windows == 4 and partition.max_tokens == 2
    windows = [sorted(partition.order[partition.window == w].tolist()) for w in range(4)]
    assert sorted(windows) == [[0, 1], [2], [3], [4, 5]]

    features = torch.randn(6, 3)
    padded = partition.pad(features)
    assert padded.shape == (4, 2, 3)
    torch.testing.assert_close(partition.unpad(padded), features)
    assert partition.key_mask.shape == (4, 1, 1, 2)
    assert int(partition.key_mask.sum()) == 6

    shifted = sparse_window_partition(coordinates, window_size=2, shift=1)
    # Shifting by one cell moves (1,1,1) out of the origin window.
    assert shifted.num_windows == 6


@pytest.fixture
def sparse_conv_backend():
    """Restore the process-wide sparse convolution backend after the test."""

    previous = sparse_ops.get_sparse_conv_backend()
    yield set_sparse_conv_backend
    set_sparse_conv_backend(previous)


def test_shared_neighbors_reproduce_a_fresh_convolution_exactly():
    generator = torch.Generator().manual_seed(1)
    coordinates = _random_coordinates(2, 6, 80, generator)
    neighbors = SubmanifoldNeighbors(coordinates)
    for _ in range(2):  # the second pass reuses the extracted taps
        features = torch.randn(coordinates.shape[0], 5, generator=generator)
        weight = torch.randn(4, 3, 3, 3, 5, generator=generator)
        torch.testing.assert_close(
            sparse_submanifold_conv3d(coordinates, features, weight, neighbors=neighbors),
            sparse_submanifold_conv3d(coordinates, features, weight),
            rtol=0,
            atol=0,
        )


def test_sparse_conv_backend_is_validated(sparse_conv_backend):
    with pytest.raises(ValueError, match="backend must be one of"):
        sparse_conv_backend("spconv")
    sparse_conv_backend("flex_gemm")
    coordinates = torch.tensor([[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="needs CUDA tensors"):
        sparse_submanifold_conv3d(coordinates, torch.ones(1, 2), torch.ones(3, 3, 3, 3, 2))


@pytest.mark.parametrize(
    ("backend", "upstream", "expected"),
    [
        ("auto", "flex_gemm", True),
        ("auto", "torch", False),
        ("flex_gemm", "torch", True),
        ("torch", "flex_gemm", False),
    ],
)
def test_auto_backend_follows_the_release(sparse_conv_backend, monkeypatch, backend, upstream, expected):
    """``auto`` uses FlexGEMM only where the release does; an explicit choice applies to every family."""

    monkeypatch.setattr(sparse_ops, "_flex_gemm_spconv", lambda: object())
    monkeypatch.setattr(sparse_ops, "require", lambda *args: None)
    sparse_conv_backend(backend)
    cuda_features = type("CudaFeatures", (), {"is_cuda": True})()
    assert sparse_ops._uses_flex_gemm(cuda_features, upstream) is expected


@pytest.mark.accelerated
@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(torch.float32, 2e-3), (torch.float16, 1e-2), (torch.bfloat16, 5e-2)]
)
def test_flex_gemm_backend_matches_the_dense_reference(sparse_conv_backend, dtype, tolerance):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("flex_gemm.ops.spconv")
    generator = torch.Generator().manual_seed(2)
    batch_size, resolution, channels = 2, 24, 64
    coordinates = _random_coordinates(batch_size, resolution, 6000, generator)
    features = torch.randn(coordinates.shape[0], channels, generator=generator, dtype=torch.float64)
    weight = torch.randn(96, 3, 3, 3, channels, generator=generator, dtype=torch.float64) / (27 * channels) ** 0.5
    bias = torch.randn(96, generator=generator, dtype=torch.float64)
    dense = F.conv3d(
        _densify(coordinates, features, batch_size, resolution), weight.permute(0, 4, 1, 2, 3), bias, padding=1
    )
    expected = dense[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]

    sparse_conv_backend("flex_gemm")
    coordinates = coordinates.cuda()
    neighbors = SubmanifoldNeighbors(coordinates)
    cast = {"device": "cuda", "dtype": dtype}
    for _ in range(2):  # the second call reuses FlexGEMM's neighbour cache
        actual = sparse_submanifold_conv3d(
            coordinates, features.to(**cast), weight.to(**cast), bias.to(**cast), neighbors=neighbors
        )
        torch.testing.assert_close(actual.double().cpu(), expected, rtol=tolerance, atol=tolerance)


def test_downsample_indices_are_reusable_exactly():
    generator = torch.Generator().manual_seed(3)
    coordinates = _random_coordinates(2, 6, 70, generator)
    features = torch.randn(coordinates.shape[0], 4, generator=generator)
    indices = sparse_ops.sparse_downsample_indices(coordinates)
    for expected, actual in zip(
        sparse_downsample(coordinates, features, count_zero_buffer=True),
        sparse_downsample(coordinates, features, count_zero_buffer=True, indices=indices),
    ):
        assert torch.equal(expected, actual)
