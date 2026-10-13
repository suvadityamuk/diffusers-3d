from __future__ import annotations

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

from diffusers_3d import ImageCondition, TripoSGImageTo3DPipeline, TripoSGTransformerModel, TripoSGVAEDecoder


def make_tiny_vae_decoder() -> TripoSGVAEDecoder:
    vae = TripoSGVAEDecoder(**TripoSGVAEDecoder.tiny_config())
    with torch.no_grad():
        # Spread the untrained field across its zero level so marching cubes finds a surface.
        vae.proj_out.weight.mul_(50)
    return vae


@pytest.fixture
def tiny_triposg_components():
    torch.manual_seed(0)
    image_encoder = Dinov2Model(
        Dinov2Config(
            image_size=28,
            patch_size=14,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            mlp_ratio=2,
        )
    ).eval()
    torch.manual_seed(1)
    transformer = TripoSGTransformerModel(**TripoSGTransformerModel.tiny_config())
    torch.manual_seed(2)
    vae = make_tiny_vae_decoder()
    return {
        "feature_extractor": BitImageProcessorPil(
            size={"shortest_edge": 32},
            crop_size={"height": 28, "width": 28},
            resample=3,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        ),
        "image_encoder": image_encoder,
        "scheduler": FlowMatchEulerDiscreteScheduler(),
        "transformer": transformer,
        "vae": vae,
    }


@pytest.fixture
def tiny_triposg_pipeline(tiny_triposg_components):
    return TripoSGImageTo3DPipeline(**tiny_triposg_components)


@pytest.fixture
def rgba_condition():
    """A 40x30 RGBA image whose opaque foreground spans rows 10-29 and columns 5-19."""

    image = torch.zeros(4, 40, 30)
    image[:3] = torch.rand(3, 40, 30, generator=torch.Generator().manual_seed(3))
    image[3, 10:30, 5:20] = 1.0
    return ImageCondition(image=image)
