from __future__ import annotations

import warnings

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from diffusers_3d import (
    Hunyuan3DDiTModel,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DImageTo3DPipeline,
    Hunyuan3DShapeVAE,
)


def make_tiny_hunyuan3d_pipeline() -> Hunyuan3DImageTo3DPipeline:
    torch.manual_seed(0)
    components = {
        "image_encoder": Dinov2Model(
            Dinov2Config(
                hidden_size=24, num_hidden_layers=1, num_attention_heads=4, mlp_ratio=2, image_size=28, patch_size=14
            )
        ),
        "transformer": Hunyuan3DDiTModel(**Hunyuan3DDiTModel.tiny_config()),
        "vae": Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config()),
        "scheduler": Hunyuan3DFlowMatchEulerDiscreteScheduler(),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        return Hunyuan3DImageTo3DPipeline(**components, image_size=16)


def sample_latents(pipeline: Hunyuan3DImageTo3DPipeline, image: torch.Tensor | None = None) -> torch.Tensor:
    """Run the sampling stages (no surface extraction, so no scikit-image) on a fixed image and seed."""

    image = torch.linspace(0, 1, 3 * 12 * 10).reshape(3, 12, 10) if image is None else image
    with torch.no_grad():
        context = pipeline.encode_image(pipeline.preprocess(image), do_classifier_free_guidance=True)
        latents = pipeline.prepare_latents(context.shape[0] // 2, torch.Generator().manual_seed(0))
        return pipeline.denoise(latents, context, num_inference_steps=2, guidance_scale=5.0)


@pytest.fixture
def tiny_hunyuan3d_pipeline() -> Hunyuan3DImageTo3DPipeline:
    return make_tiny_hunyuan3d_pipeline()
