from __future__ import annotations

import pytest
import torch

from diffusers_3d import Object3D


@pytest.mark.parametrize("fixture_name", ("mesh", "gaussian", "sparse_voxel", "o_voxel"))
def test_every_object_asset_is_a_lossless_pytree_and_preserves_index_dtypes(request, fixture_name):
    asset = request.getfixturevalue(fixture_name)
    leaves, tree_spec = torch.utils._pytree.tree_flatten(asset)

    assert isinstance(asset, Object3D)
    assert not torch.utils._pytree.tree_is_leaf(asset)
    assert any(isinstance(leaf, torch.Tensor) for leaf in leaves)

    rebuilt = torch.utils._pytree.tree_unflatten(leaves, tree_spec)
    assert type(rebuilt) is type(asset)
    rebuilt.validate(expensive=True)
    for (original_name, original), (rebuilt_name, restored) in zip(
        asset.tensor_items(),
        rebuilt.tensor_items(),
        strict=True,
    ):
        assert rebuilt_name == original_name
        assert torch.equal(restored, original)

    moved = rebuilt.to(device="cpu", dtype=torch.float64)
    assert moved.device == torch.device("cpu")
    for _, tensor in moved.tensor_items():
        if tensor.is_floating_point():
            assert tensor.dtype is torch.float64
        else:
            assert not tensor.is_floating_point()
