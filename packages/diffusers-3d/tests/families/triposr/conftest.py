from __future__ import annotations

import math

import pytest
import torch
from _configs import tiny_config
from transformers import ViTConfig, ViTModel

from diffusers_3d import ImageCondition
from diffusers_3d.families.triposr.models import TripoSRTransformerModel, TripoSRTriplaneDecoder
from diffusers_3d.families.triposr.pipeline import TripoSRImageTo3DPipeline

from .configs import tiny_vit_config


def make_tiny_triposr_components() -> dict:
    torch.manual_seed(0)
    image_encoder = ViTModel(ViTConfig(**tiny_vit_config())).eval()
    transformer = TripoSRTransformerModel(**tiny_config(TripoSRTransformerModel)).eval()
    decoder = TripoSRTriplaneDecoder(**tiny_config(TripoSRTriplaneDecoder)).eval()
    with torch.no_grad():
        # Spread the untrained density and centre it on the default threshold of 25 for a grey image, so marching
        # cubes finds a surface for any small test image.
        decoder.layers[-1].weight.mul_(30)
        tokens = image_encoder(torch.zeros(1, 3, 32, 32)).last_hidden_state
        grid = torch.rand(1, 512, 3, generator=torch.Generator().manual_seed(0)) * 1.6 - 0.8
        density = decoder(transformer(tokens).sample, grid).density
        decoder.layers[-1].bias[0] += math.log(25) - density.log().median()
    return {"image_encoder": image_encoder, "transformer": transformer, "decoder": decoder}


def make_tiny_triposr_pipeline() -> TripoSRImageTo3DPipeline:
    pipeline = TripoSRImageTo3DPipeline(**make_tiny_triposr_components())
    pipeline.disable_background_removal()  # upstream's default rembg would download u2net
    return pipeline


@pytest.fixture
def tiny_triposr_pipeline() -> TripoSRImageTo3DPipeline:
    return make_tiny_triposr_pipeline()


@pytest.fixture
def rgba_condition() -> ImageCondition:
    """A 40x30 RGBA image whose opaque foreground spans rows 10-29 and columns 5-19."""

    image = torch.zeros(4, 40, 30)
    image[:3] = torch.rand(3, 40, 30, generator=torch.Generator().manual_seed(3))
    image[3, 10:30, 5:20] = 1.0
    return ImageCondition(image=image)
