from __future__ import annotations

import types

import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    Hunyuan3DImageToShapePipeline,
    MeshAsset,
    Object3DPipelineOutput,
)

pytestmark = pytest.mark.integration


def _triangle_decode(self, latents, **kwargs):
    del kwargs
    values = latents.mean(dim=(1, 2))
    meshes = []
    for value in values:
        vertices = torch.stack(
            [
                torch.stack([value, value.new_tensor(0.0), value.new_tensor(0.0)]),
                torch.stack([value + 1.0, value.new_tensor(0.0), value.new_tensor(0.0)]),
                torch.stack([value, value.new_tensor(1.0), value.new_tensor(0.0)]),
            ]
        )
        meshes.append(
            MeshAsset(
                vertices=vertices,
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            )
        )
    return tuple(meshes)


def test_tiny_pipeline_is_deterministic(tiny_hunyuan_pipeline):
    pipeline = tiny_hunyuan_pipeline
    pipeline.vae.decode_to_meshes = types.MethodType(_triangle_decode, pipeline.vae)
    image = torch.zeros(3, 8, 8)
    latents = torch.linspace(-1, 1, 32).reshape(1, 4, 8)
    first = pipeline(
        image,
        latents=latents.clone(),
        num_inference_steps=2,
        guidance_scale=5.0,
    )
    second = pipeline(
        image,
        latents=latents.clone(),
        num_inference_steps=2,
        guidance_scale=5.0,
    )
    assert type(first) is Object3DPipelineOutput
    assert type(first.objects[0]) is MeshAsset
    torch.testing.assert_close(first.objects[0].vertices, second.objects[0].vertices)
    torch.testing.assert_close(first.latents.latents, second.latents.latents)


def test_pipeline_save_load_and_auto_pipeline(tmp_path, tiny_hunyuan_pipeline):
    tiny_hunyuan_pipeline.save_pretrained(tmp_path)
    loaded = Hunyuan3DImageToShapePipeline.from_pretrained(tmp_path, local_files_only=True)
    automatic = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is Hunyuan3DImageToShapePipeline
    assert type(automatic) is Hunyuan3DImageToShapePipeline
    assert loaded.config.image_processor_size == 8
    assert automatic.config.image_processor_border_ratio == 0.0
