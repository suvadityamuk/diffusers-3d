from __future__ import annotations

import pytest
import torch
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from diffusers_3d import (
    Step1X3DFlowMatchEulerScheduler,
    Step1X3DImageTo3DPipeline,
    Step1X3DShapeDecoder,
    Step1X3DTransformer1DModel,
)


@pytest.fixture
def tiny_step1x3d_components():
    torch.manual_seed(0)
    image_encoder = Dinov2WithRegistersModel(
        Dinov2WithRegistersConfig(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            mlp_ratio=2,
            image_size=8,
            patch_size=4,
            num_register_tokens=2,
        )
    )
    torch.manual_seed(1)
    transformer = Step1X3DTransformer1DModel(**Step1X3DTransformer1DModel.tiny_config())
    torch.manual_seed(2)
    vae = Step1X3DShapeDecoder(**Step1X3DShapeDecoder.tiny_config())
    return {
        "image_encoder": image_encoder.eval(),
        "transformer": transformer.eval(),
        "vae": vae.eval(),
        "scheduler": Step1X3DFlowMatchEulerScheduler(),
    }


@pytest.fixture
def tiny_step1x3d_pipeline(tiny_step1x3d_components):
    return Step1X3DImageTo3DPipeline(**tiny_step1x3d_components)


@pytest.fixture
def rgba_condition():
    """A 4x20x16 RGBA image whose opaque foreground is an off-centre 11x9 rectangle."""

    from diffusers_3d import ImageCondition

    image = torch.linspace(0.0, 1.0, 4 * 20 * 16).reshape(4, 20, 16)
    image[3] = 0
    image[3, 4:15, 3:12] = 1
    return ImageCondition(image)


@pytest.fixture
def sphere_field():
    """Build an occupancy-style field (positive inside a sphere) on the released grid."""

    from diffusers_3d.families.step1x3d.vae import step1x3d_grid_points

    def make(resolution: int, *, radius: float = 0.6, bounds: float = 1.05) -> torch.Tensor:
        points = step1x3d_grid_points(bounds, resolution)
        return (radius - points.norm(dim=-1)).view(1, *([resolution + 1] * 3))

    return make
