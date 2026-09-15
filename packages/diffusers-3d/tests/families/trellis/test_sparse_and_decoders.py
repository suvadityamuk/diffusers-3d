from __future__ import annotations

import math

import pytest
import torch

from diffusers_3d import (
    CoordinateSystem,
    GaussianSplatAsset,
    MeshAsset,
    SparseVoxelAsset,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseTensor,
    trellis_grid_transform,
)
from diffusers_3d.families.trellis.flexicubes import sparse_cubes_to_mesh


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


def test_flexicubes_closes_a_single_voxel_with_outward_faces():
    resolution = 8
    coordinates = torch.tensor([[3, 4, 5]])
    # Zero features: the voxel's corners sit at ``sdf = -1 / res`` (inside) and every other grid vertex outside.
    result = sparse_cubes_to_mesh(coordinates, torch.zeros(1, 53), resolution, use_color=False)
    vertices, faces = result.vertices, result.faces
    assert result.vertex_colors is None
    edges = torch.unique(torch.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), dim=1).values, dim=0)
    # Closed genus-0 surface: V - E + F == 2.
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 2
    center = (coordinates[0].float() + 0.5) / resolution - 0.5
    assert bool(((vertices - center).abs() <= 1.5 / resolution + 1e-6).all())
    triangles = vertices[faces]
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    assert bool(((normals * (triangles.mean(1) - center)).sum(1) > 0).all())


def test_mesh_decoder_returns_zup_mesh_assets_with_colors_and_normal_map():
    decoder = TrellisSLatMeshDecoder(**TrellisSLatMeshDecoder.tiny_config()).eval()
    assert decoder.out_channels == 8 + 24 + 21 + 48
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 7, 7, 7]], dtype=torch.int64)
    sparse = TrellisSparseTensor(coordinates, torch.zeros(3, decoder.config.latent_channels))
    with torch.no_grad():
        output = decoder(sparse)
    assert len(output.assets) == 2
    for asset in output.assets:
        assert type(asset) is MeshAsset
        assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
        assert asset.colors.shape == (asset.vertices.shape[0], 3)
        assert asset.extras["normal_map"].shape == (asset.vertices.shape[0], 3)
        assert asset.metadata["resolution"] == 32 and asset.metadata["extraction"] == "flexicubes"
        assert bool((asset.vertices.abs() <= 0.5).all())
        y_up = asset.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)
        torch.testing.assert_close(y_up.vertices[:, 1], asset.vertices[:, 2])
    # Two voxels span a larger surface than one.
    assert output.assets[0].faces.shape[0] > output.assets[1].faces.shape[0]

    # Half-precision weights: group norms compute in float32 like upstream, and the mesh is extracted in float32.
    with torch.no_grad():
        half = decoder.to(torch.bfloat16)(sparse.to(dtype=torch.bfloat16))
    assert all(asset.vertices.dtype is torch.float32 and asset.faces.shape[0] > 0 for asset in half.assets)
    decoder.to(torch.float32)

    plain = TrellisSLatMeshDecoder(**{**TrellisSLatMeshDecoder.tiny_config(), "representation_config": {}})
    assert plain.out_channels == 53 and plain.use_color is False
    with torch.no_grad():
        # Push every SDF corner outside the surface: nothing to extract.
        decoder.out_layer.bias[:8].fill_(2.0)
        with pytest.raises(ValueError, match="empty surface"):
            decoder(sparse)


def test_unported_radiance_decoder_fails_explicitly():
    sparse = TrellisSparseTensor(torch.tensor([[0, 0, 0, 0]]), torch.zeros(1, 8))
    radiance = TrellisSLatRadianceFieldDecoder()
    with pytest.raises(NotImplementedError, match="package-native Object3D"):
        radiance(sparse)
