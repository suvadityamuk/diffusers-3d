from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
    Trellis2ImageTo3DPipeline,
)


def test_reviewed_sparse_structure_pipeline_is_deterministic_and_uses_released_defaults(tiny_trellis2_pipeline):
    pipeline = tiny_trellis2_pipeline
    defaults = pipeline.config.sparse_structure_sampler_defaults
    assert defaults == {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "guidance_interval": (0.6, 1.0),
        "rescale_t": 5.0,
    }
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8)
    latents = torch.linspace(-1.0, 1.0, 2 * 2 * 2 * 2).reshape(1, 2, 2, 2, 2)
    first = pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_sampler_params={"steps": 2},
    )
    second = pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_sampler_params={"steps": 2},
    )
    assert type(first) is Object3DPipelineOutput
    assert len(first.objects) == 1
    assert type(first.objects[0]) is SparseVoxelAsset
    assert first.objects[0].metadata["representation"] == "sparse_structure"
    assert first.objects[0].metadata["decoder_checkpoint_semantics"] == "trellis-image-large-exact-reuse"
    torch.testing.assert_close(first.objects[0].features, second.objects[0].features)
    torch.testing.assert_close(first.latents.latents, second.latents.latents)


def test_pipeline_save_load_auto_and_serialized_capability_limitations(tmp_path, tiny_trellis2_pipeline):
    tiny_trellis2_pipeline.save_pretrained(tmp_path)
    loaded = Trellis2ImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    automatic = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is Trellis2ImageTo3DPipeline
    assert type(automatic) is Trellis2ImageTo3DPipeline
    assert loaded.shape_slat_flow_model is None
    assert loaded.texture_slat_flow_model is None
    assert loaded.pbr_decoder is None
    assert loaded.config.capability_limitations["reviewed_formats"] == ["sparse_structure"]
    assert (
        automatic.config.capability_limitations["production_1024_cascade"]
        == "unsupported_until_flex_gemm_ovoxel_gpu_parity"
    )


def test_tiny_experimental_shape_texture_and_ovoxel_stages_return_native_assets(tiny_trellis2_full_pipeline):
    pipeline = tiny_trellis2_full_pipeline
    image = torch.zeros(3, 8, 8)
    output = pipeline(
        image,
        formats=("sparse_structure", "shape_slat", "texture_slat", "o_voxel"),
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
        generator=torch.Generator().manual_seed(5),
    )
    assert [type(value) for value in output.objects] == [
        SparseVoxelAsset,
        SparseVoxelAsset,
        SparseVoxelAsset,
        OVoxelAsset,
    ]
    shape_slat, texture_slat, ovoxel = output.objects[1:]
    assert torch.equal(shape_slat.coordinates, texture_slat.coordinates)
    assert ovoxel.metadata["stage"] == "pbr_decoder_tiny"
    assert ovoxel.metadata["full_pbr_asset_channels"]
    assert ovoxel.base_color.shape[1] == 3
    assert ovoxel.metallic.shape[1] == ovoxel.roughness.shape[1] == ovoxel.opacity.shape[1] == 1
    assert ovoxel.normals.shape[1] == ovoxel.emissive.shape[1] == 3
    assert ovoxel.split_weights.shape[1] == 1


def test_production_cascade_and_missing_experimental_components_fail_explicitly(tiny_trellis2_pipeline):
    with pytest.raises(NotImplementedError, match="1024 cascade"):
        tiny_trellis2_pipeline(
            torch.zeros(3, 8, 8),
            formats=("shape_slat",),
            pipeline_type="1024_cascade",
        )
    with pytest.raises(RuntimeError, match="shape SLAT"):
        tiny_trellis2_pipeline(
            torch.zeros(3, 8, 8),
            formats=("shape_slat",),
            pipeline_type="tiny",
        )
