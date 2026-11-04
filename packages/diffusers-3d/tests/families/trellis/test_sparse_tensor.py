"""``TrellisSparseTensor``'s trusted derivation path and coordinate cache."""

from __future__ import annotations

import pytest
import torch

from diffusers_3d import TrellisSparseTensor


def _tensor() -> TrellisSparseTensor:
    coordinates = torch.tensor([[0, 0, 0, 0], [1, 1, 2, 3], [0, 2, 2, 2], [1, 4, 0, 1], [1, 0, 0, 0]])
    return TrellisSparseTensor(coordinates, torch.randn(5, 3, generator=torch.Generator().manual_seed(0)))


def test_replace_shares_coordinates_batch_size_and_cache():
    tensor = _tensor()
    calls = []
    built = tensor.cached("key", lambda: calls.append(1) or object())
    replaced = tensor.replace(tensor.features * 2)
    assert replaced.coordinates is tensor.coordinates
    assert replaced.batch_size == tensor.batch_size == 2
    assert replaced.cached("key", lambda: calls.append(1) or object()) is built
    assert tensor.to(dtype=torch.float64).cached("key", lambda: None) is built
    assert calls == [1]


def test_replace_still_checks_the_features():
    tensor = _tensor()
    with pytest.raises(ValueError, match="same non-zero active-voxel count"):
        tensor.replace(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="floating-point"):
        tensor.replace(torch.zeros(5, 3, dtype=torch.int64))


def test_coordinate_comparison_and_batch_positions():
    tensor = _tensor()
    assert tensor.has_coordinates_of(tensor.replace(tensor.features))
    assert tensor.has_coordinates_of(TrellisSparseTensor(tensor.coordinates.clone(), tensor.features))
    shifted = tensor.coordinates.clone()
    shifted[0, 1] = 5
    assert not tensor.has_coordinates_of(TrellisSparseTensor(shifted, tensor.features))
    first, second = tensor.batch_positions()
    assert first.tolist() == [0, 2] and second.tolist() == [1, 3, 4]
    assert tensor.batch_positions() is tensor.batch_positions()


def test_cache_is_skipped_by_tensor_traversal():
    tensor = _tensor()
    tensor.cached("neighbours", lambda: torch.ones(3))
    assert [name for name, _ in tensor.tensor_items()] == ["coordinates", "features"]
