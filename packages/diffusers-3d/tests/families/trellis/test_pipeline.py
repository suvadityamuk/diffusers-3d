from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    GaussianSplatAsset,
    Object3DPipelineOutput,
    SparseVoxelAsset,
    TrellisImageTo3DPipeline,
)

pytestmark = pytest.mark.integration


def test_portable_sparse_structure_pipeline_is_deterministic(tiny_trellis_pipeline):
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8)
    latents = torch.linspace(-1.0, 1.0, 2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4)
    first = tiny_trellis_pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_num_inference_steps=2,
    )
    second = tiny_trellis_pipeline(
        image,
        sparse_structure_latents=latents.clone(),
        sparse_structure_num_inference_steps=2,
    )
    assert type(first) is Object3DPipelineOutput
    assert len(first.objects) == 1
    assert type(first.objects[0]) is SparseVoxelAsset
    assert first.objects[0].metadata["representation"] == "sparse_structure"
    torch.testing.assert_close(first.objects[0].features, second.objects[0].features)
    torch.testing.assert_close(first.latents.latents, second.latents.latents)


def test_pipeline_save_load_auto_and_optional_components(tmp_path, tiny_trellis_pipeline):
    tiny_trellis_pipeline.save_pretrained(tmp_path)
    loaded = TrellisImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    automatic = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is TrellisImageTo3DPipeline
    assert type(automatic) is TrellisImageTo3DPipeline
    assert loaded.slat_flow_model is None
    assert loaded.slat_scheduler is None
    assert loaded.gaussian_decoder is None
    assert loaded.mesh_decoder is None
    assert loaded.config.slat_mean is None
    assert automatic.config.slat_std is None


def test_portable_full_attention_slat_and_gaussian_paths(tiny_trellis_full_pipeline):
    image = torch.zeros(3, 8, 8)
    sparse_structure_latents = torch.zeros(1, 2, 4, 4, 4)
    output = tiny_trellis_full_pipeline(
        image,
        formats=("sparse_structure", "slat", "gaussian"),
        sparse_structure_num_inference_steps=2,
        slat_num_inference_steps=2,
        sparse_structure_latents=sparse_structure_latents,
        generator=torch.Generator().manual_seed(5),
    )
    assert [type(value) for value in output.objects] == [
        SparseVoxelAsset,
        SparseVoxelAsset,
        GaussianSplatAsset,
    ]
    assert output.objects[1].features.shape == (8**3, 4)
    assert output.objects[2].means.shape == (8**3 * 2, 3)


def test_pipeline_rejects_unavailable_requested_formats(tiny_trellis_pipeline):
    with pytest.raises(RuntimeError, match="SLAT"):
        tiny_trellis_pipeline(torch.zeros(3, 8, 8), formats=("slat",))
    with pytest.raises(ValueError, match="unique"):
        tiny_trellis_pipeline(torch.zeros(3, 8, 8), formats=("sparse_structure", "sparse_structure"))
