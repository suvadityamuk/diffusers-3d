from __future__ import annotations

import math

import pytest
import torch

from diffusers_3d import (
    BackendUnavailableError,
    CoordinateSystem,
    GaussianSplatAsset,
    SparseVoxelAsset,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseTensor,
    trellis_grid_transform,
)


def test_sparse_tensor_asset_roundtrip_and_normalization_are_lossless():
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    first = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.int64),
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        grid_transform=trellis_grid_transform(8),
        transform=transform,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        semantic_labels=torch.tensor([5, 6], dtype=torch.int64),
        extras={"confidence": torch.tensor([0.7, 0.8])},
        metadata={"name": "first"},
    )
    second = SparseVoxelAsset(
        coordinates=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        features=torch.tensor([[5.0, 6.0]]),
        grid_transform=trellis_grid_transform(8),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        metadata={"name": "second"},
    )
    sparse = TrellisSparseTensor.from_sparse_voxel_assets((first, second))
    assert torch.equal(sparse.coordinates[:, 0], torch.tensor([0, 0, 1]))
    restored = sparse.normalize(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])).denormalize(
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    outputs = restored.to_sparse_voxel_assets()
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].features, first.features)
    assert torch.equal(outputs[0].coordinates, first.coordinates)
    assert torch.equal(outputs[0].transform, first.transform)
    assert torch.equal(outputs[0].semantic_labels, first.semantic_labels)
    assert torch.equal(outputs[0].extras["confidence"], first.extras["confidence"])
    assert outputs[0].metadata == first.metadata
    assert outputs[1].metadata == second.metadata


def test_sparse_structure_decoder_threshold_and_native_grid_transform():
    decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    hidden_states = torch.zeros(1, 2, 4, 4, 4)
    asset = decoder.decode_to_sparse_voxels(hidden_states)[0]
    assert type(asset) is SparseVoxelAsset
    assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert asset.coordinates.shape == (8**3, 3)
    assert torch.equal(asset.coordinates[0], torch.tensor([0, 0, 0]))
    assert torch.equal(asset.coordinates[-1], torch.tensor([7, 7, 7]))
    torch.testing.assert_close(asset.features, torch.ones(8**3, 1))
    torch.testing.assert_close(asset.grid_transform, trellis_grid_transform(8))


def test_gaussian_decoder_maps_released_parameterization_to_canonical_asset():
    decoder = TrellisSLatGaussianDecoder(**TrellisSLatGaussianDecoder.tiny_config()).eval()
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 7, 7, 7]], dtype=torch.int64)
    sparse = TrellisSparseTensor(coordinates, torch.zeros(2, decoder.config.latent_channels))
    with torch.no_grad():
        output = decoder(sparse)
    asset = output.assets[0]
    assert type(asset) is GaussianSplatAsset
    assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert asset.means.shape == (4, 3)
    torch.testing.assert_close(asset.means[:2], torch.full((2, 3), 0.5 / 8 - 0.5))
    torch.testing.assert_close(asset.means[2:], torch.full((2, 3), 7.5 / 8 - 0.5))
    expected_scale = math.sqrt(float(decoder.rep_config["scaling_bias"]) ** 2 + 9e-4**2)
    torch.testing.assert_close(asset.log_scales.exp(), torch.full((4, 3), expected_scale))
    torch.testing.assert_close(asset.quaternions_wxyz, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, -1))
    torch.testing.assert_close(asset.opacity_logits, torch.full((4, 1), math.log(0.1 / 0.9)))
    torch.testing.assert_close(asset.sh_coefficients, torch.zeros(4, 1, 3))
    assert set(asset.extras) == {
        "trellis_raw_opacity",
        "trellis_raw_rotation",
        "trellis_raw_scaling",
        "trellis_raw_xyz",
    }


def test_unported_mesh_and_radiance_decoders_fail_explicitly():
    sparse = TrellisSparseTensor(torch.tensor([[0, 0, 0, 0]]), torch.zeros(1, 8))
    mesh = TrellisSLatMeshDecoder()
    with pytest.raises((BackendUnavailableError, NotImplementedError), match="Kaolin|kaolin|mesh field"):
        mesh(sparse)
    radiance = TrellisSLatRadianceFieldDecoder()
    with pytest.raises(NotImplementedError, match="package-native Object3D"):
        radiance(sparse)
