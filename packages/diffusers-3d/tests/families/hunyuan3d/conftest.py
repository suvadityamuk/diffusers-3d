from __future__ import annotations

import pytest

from diffusers_3d import (
    Hunyuan3DDinov2Conditioner,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DImageToShapePipeline,
    Hunyuan3DShapeDiTModel,
    Hunyuan3DShapeVAE,
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
