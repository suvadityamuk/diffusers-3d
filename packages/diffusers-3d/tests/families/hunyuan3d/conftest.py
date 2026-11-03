from __future__ import annotations

import copy
import dataclasses
import warnings

import pytest
import torch
from _configs import tiny_config
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import (
    Hunyuan3DDiTModel,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DImageTo3DPipeline,
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintRRDBNet,
    Hunyuan3DPaintUNet,
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
        "transformer": Hunyuan3DDiTModel(**tiny_config(Hunyuan3DDiTModel)),
        "vae": Hunyuan3DShapeVAE(**tiny_config(Hunyuan3DShapeVAE)),
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


def make_tiny_paint_pipeline(upscaler: bool = True) -> Hunyuan3DPaintPipeline:
    """Tiny random Paint components: 16 px views (8 px latents) and a 10-dimensional DINOv2."""

    torch.manual_seed(0)
    unet = Hunyuan3DPaintUNet(**tiny_config(Hunyuan3DPaintUNet))
    with torch.no_grad():
        for parameter in unet.parameters():
            parameter.normal_(std=0.05)
    components = {
        "feature_extractor": BitImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28}),
        "image_encoder": Dinov2Model(
            Dinov2Config(hidden_size=10, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14)
        ),
        "unet": unet,
        "vae": AutoencoderKL(
            block_out_channels=(8, 16),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            norm_num_groups=8,
        ),
        "scheduler": UniPCMultistepScheduler(
            prediction_type="v_prediction", rescale_betas_zero_snr=True, timestep_spacing="trailing"
        ),
        "upscaler": Hunyuan3DPaintRRDBNet(**tiny_config(Hunyuan3DPaintRRDBNet)) if upscaler else None,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        return Hunyuan3DPaintPipeline(**components)


# Small sizes for the tiny components; the released defaults are 512, 2048, and 2048.
TINY_PAINT_SIZES = {"view_size": 16, "render_size": 32, "texture_size": 16}


@pytest.fixture
def local_external_components(monkeypatch):
    """Load a pipeline class's external components (Meta's DINOv2) from local objects instead of their repository.

    ``use(pipeline_class, image_encoder=module, ...)`` returns the list of ``(name, repository, revision, dtype)``
    loads; each load returns a copy of the object, cast to the requested dtype when it is a module.
    """

    def use(pipeline_class, **components):
        calls = []

        def loader(name):
            def load(repository, *, revision, dtype, **hub_kwargs):
                calls.append((name, repository, revision, dtype))
                component = copy.deepcopy(components[name])
                return (
                    component.to(dtype) if dtype is not None and isinstance(component, torch.nn.Module) else component
                )

            return load

        external = {
            name: dataclasses.replace(component, load=loader(name))
            for name, component in pipeline_class.external_components.items()
        }
        monkeypatch.setattr(pipeline_class, "external_components", external)
        monkeypatch.setattr(pipelines.huggingface_hub, "auth_check", lambda *args, **kwargs: None)
        return calls

    return use
