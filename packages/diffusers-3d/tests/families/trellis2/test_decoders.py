from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    BackendUnavailableError,
    CoordinateSystem,
    OVoxelAsset,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SparseStructureDecoder,
    TrellisSparseTensor,
)


def test_sparse_structure_decoder_exact_reuse_metadata_and_shape():
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    assets = decoder.decode_to_sparse_voxels(torch.zeros(1, 2, 2, 2, 2))
    assert len(assets) == 1
    assert assets[0].coordinates.shape == (4**3, 3)
    assert assets[0].coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert assets[0].metadata["decoder_checkpoint_semantics"] == "trellis-image-large-exact-reuse"


def test_tiny_shape_and_pbr_decoders_preserve_dual_grid_and_all_material_channels():
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(12)
    shape_slat = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    texture_slat = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    shape_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
    pbr_decoder = Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.tiny_config())

    shape_assets = shape_decoder(shape_slat).assets
    assert len(shape_assets) == 2
    assert all(type(asset) is OVoxelAsset for asset in shape_assets)
    assert all(asset.dual_grid_vertex_offsets.shape[1] == 3 for asset in shape_assets)
    assert all(asset.intersection_data.dtype is torch.bool for asset in shape_assets)
    assert all(asset.split_weights.shape[1] == 1 for asset in shape_assets)
    assert all(asset.metadata["official_checkpoint_parity"] is False for asset in shape_assets)

    pbr_assets = pbr_decoder(texture_slat, shape_assets).assets
    for shape, pbr in zip(shape_assets, pbr_assets):
        assert torch.equal(pbr.active_coordinates, shape.active_coordinates)
        assert torch.equal(pbr.dual_grid_vertex_offsets, shape.dual_grid_vertex_offsets)
        assert torch.equal(pbr.intersection_data, shape.intersection_data)
        assert torch.equal(pbr.split_weights, shape.split_weights)
        assert pbr.base_color.shape[1] == pbr.normals.shape[1] == pbr.emissive.shape[1] == 3
        assert pbr.metallic.shape[1] == pbr.roughness.shape[1] == pbr.opacity.shape[1] == 1
        torch.testing.assert_close(torch.linalg.vector_norm(pbr.normals, dim=1), torch.ones(pbr.normals.shape[0]))


def test_production_shape_and_pbr_decoders_are_explicit_backend_gates():
    with pytest.raises((BackendUnavailableError, NotImplementedError), match="flex_gemm|FlexGEMM"):
        Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.production_config())
    with pytest.raises((BackendUnavailableError, NotImplementedError), match="flex_gemm|FlexGEMM"):
        Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.production_config())
