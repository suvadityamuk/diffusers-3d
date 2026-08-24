from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    Hunyuan3DDinov2Conditioner,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DImageToShapePipeline,
    Hunyuan3DShapeDiTModel,
    Hunyuan3DShapeExample,
    Hunyuan3DShapeVAE,
    ImageCondition,
)


class TinyHunyuanPrecomputedLatentDataset:
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Hunyuan3DShapeExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        offset = float(index) / self.length
        image = (torch.linspace(-1.0, 1.0, 3 * 8 * 8) * (1.0 - 0.1 * offset)).reshape(3, 8, 8)
        shape_latents = torch.linspace(-0.75 + offset, 0.75 + offset, 4 * 8).reshape(4, 8)
        return Hunyuan3DShapeExample(
            condition=ImageCondition(image=image),
            shape_latents=shape_latents,
            example_id=f"tiny-hunyuan-{index}",
        )


@pytest.fixture
def tiny_hunyuan_components():
    def make():
        denoiser = Hunyuan3DShapeDiTModel(**Hunyuan3DShapeDiTModel.tiny_config())
        vae = Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config())
        conditioner = Hunyuan3DDinov2Conditioner(**Hunyuan3DDinov2Conditioner.tiny_config())
        scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        return conditioner, denoiser, vae, scheduler

    return make


@pytest.fixture
def tiny_hunyuan_pipeline(tiny_hunyuan_components):
    conditioner, denoiser, vae, scheduler = tiny_hunyuan_components()
    return Hunyuan3DImageToShapePipeline(
        conditioner=conditioner,
        denoiser=denoiser,
        vae=vae,
        scheduler=scheduler,
        image_processor_size=8,
        image_processor_border_ratio=0.0,
    )


@pytest.fixture
def tiny_hunyuan_latent_dataset():
    return TinyHunyuanPrecomputedLatentDataset()
