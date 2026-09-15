from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffusers_3d import (
    CoordinateSystem,
    OVoxelAsset,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SparseStructureDecoder,
    TrellisSparseTensor,
)
from diffusers_3d.families.trellis.decoders import TrellisSparseStructureDecoderOutput
from diffusers_3d.families.trellis.sparse import trellis_grid_transform


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


def test_sparse_structure_decoder_max_pools_occupancy_before_argwhere(monkeypatch):
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    logits = torch.full((1, 1, 64, 64, 64), -1.0)
    logits[0, 0, 2, 4, 6] = 0.25
    logits[0, 0, 3, 5, 7] = 2.0
    logits[0, 0, 63, 63, 63] = 1.0
    monkeypatch.setattr(decoder, "forward", lambda hidden_states: TrellisSparseStructureDecoderOutput(sample=logits))

    asset = decoder.decode_to_sparse_voxels(torch.empty(1), target_resolution=32)[0]
    expected_occupancy = F.max_pool3d((logits > 0).float(), 2, 2, 0) > 0.5
    expected_coordinates = torch.argwhere(expected_occupancy[0, 0]).to(torch.int64)

    assert torch.equal(asset.coordinates, expected_coordinates)
    assert torch.equal(asset.features, torch.ones_like(asset.features))
    assert asset.metadata["resolution"] == 32
    assert asset.metadata["decoded_resolution"] == 64
    torch.testing.assert_close(asset.grid_transform, trellis_grid_transform(32))


def test_shape_and_pbr_decoders_share_subdivision_and_expose_all_material_channels():
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(12)
    shape_slat = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    texture_slat = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    torch.manual_seed(0)
    shape_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
    pbr_decoder = Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.tiny_config())
    with torch.no_grad():
        # Zero-initialised subdivision heads would pick no children; bias them so every block selects some.
        shape_decoder.blocks[0][-1].to_subdiv.bias.fill_(1.0)

    output = shape_decoder(shape_slat, resolution=16)
    shape_assets = output.assets
    assert len(shape_assets) == 2
    assert len(output.subdivisions) == shape_decoder.num_upsamples == 1
    assert output.subdivisions[0].shape == (4, 8) and output.subdivisions[0].dtype is torch.bool
    total = sum(asset.active_coordinates.shape[0] for asset in shape_assets)
    assert total == int(output.subdivisions[0].sum())
    assert all(type(asset) is OVoxelAsset for asset in shape_assets)
    for asset in shape_assets:
        assert asset.dual_grid_vertex_offsets.shape == (asset.active_coordinates.shape[0], 3)
        assert bool((asset.dual_grid_vertex_offsets >= -0.5).all()) and bool(
            (asset.dual_grid_vertex_offsets <= 1.5).all()
        )
        assert asset.intersection_data.dtype is torch.bool and asset.intersection_data.shape[1] == 3
        assert asset.split_weights.shape[1] == 1 and bool((asset.split_weights >= 0).all())
        assert asset.metadata["resolution"] == [16, 16, 16]
        assert bool((asset.active_coordinates < 16).all())
        # Children live inside their parent voxel: parent = child // 2.
        parents = torch.unique(asset.active_coordinates // 2, dim=0)
        assert parents.shape[0] <= 2

    pbr_assets = pbr_decoder(texture_slat, shape_assets, output.subdivisions).assets
    for shape, pbr in zip(shape_assets, pbr_assets):
        assert torch.equal(pbr.active_coordinates, shape.active_coordinates)
        assert torch.equal(pbr.dual_grid_vertex_offsets, shape.dual_grid_vertex_offsets)
        assert torch.equal(pbr.intersection_data, shape.intersection_data)
        assert torch.equal(pbr.split_weights, shape.split_weights)
        assert pbr.base_color.shape[1] == pbr.normals.shape[1] == pbr.emissive.shape[1] == 3
        assert pbr.metallic.shape[1] == pbr.roughness.shape[1] == pbr.opacity.shape[1] == 1
        assert bool((pbr.base_color >= 0).all()) and bool((pbr.base_color <= 1).all())
        torch.testing.assert_close(torch.linalg.vector_norm(pbr.normals, dim=1), torch.ones(pbr.normals.shape[0]))
        assert pbr.metadata["stage"] == "pbr_decoder"

    # Texture decoding must replay the shape decoder's masks; a mismatching mask is rejected.
    flipped = (~output.subdivisions[0],)
    with pytest.raises(ValueError, match="line up|no child"):
        pbr_decoder(texture_slat, shape_assets, flipped)


def test_upsample_coordinates_matches_the_forward_grid():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 3, 3, 3]], dtype=torch.int64)
    slat = TrellisSparseTensor(coordinates, torch.randn(3, 4, generator=torch.Generator().manual_seed(3)))
    torch.manual_seed(1)
    decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
    with torch.no_grad():
        decoder.blocks[0][-1].to_subdiv.bias.fill_(1.0)
    grown = decoder.upsample_coordinates(slat, 1)
    decoded = decoder(slat, resolution=8).assets[0]
    assert torch.equal(grown[:, 1:].to(torch.int64), decoded.active_coordinates)
    assert torch.equal(decoder.upsample_coordinates(slat, 0), coordinates)
    with pytest.raises(ValueError, match="upsample_times"):
        decoder.upsample_coordinates(slat, 2)


def test_production_decoder_configs_build_the_released_layout():
    with torch.device("meta"):
        shape = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.production_config())
        pbr = Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.production_config())
    shape_keys = set(shape.state_dict())
    pbr_keys = set(pbr.state_dict())
    # Released shape_dec_next_dc_f16c32_fp16 / tex_dec_next_dc_f16c32_fp16 safetensors headers.
    assert len(shape_keys) == 292 and len(pbr_keys) == 284
    assert shape_keys - pbr_keys == {
        f"blocks.{stage}.{index}.to_subdiv.{name}"
        for stage, index in ((0, 4), (1, 16), (2, 8), (3, 4))
        for name in ("weight", "bias")
    }
    assert shape.blocks[0][4].conv1.weight.shape == (512 * 8, 3, 3, 3, 1024)
    assert shape.output_layer.weight.shape == (7, 64) and pbr.output_layer.weight.shape == (6, 64)
    assert shape.blocks[1][0].conv.weight.dtype is torch.float16 and shape.from_latent.weight.dtype is torch.float32
    with pytest.raises(ValueError, match="pred_subdiv=False"):
        Trellis2PBRSparseDecoder(**{**Trellis2PBRSparseDecoder.tiny_config(), "pred_subdiv": True})
    with pytest.raises(ValueError, match="unsupported block types"):
        Trellis2ShapeDualGridDecoder(**{**Trellis2ShapeDualGridDecoder.tiny_config(), "up_block_type": ["Other"]})
