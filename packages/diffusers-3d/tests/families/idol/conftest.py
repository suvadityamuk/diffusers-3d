from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import ImageCondition
from diffusers_3d.families.idol import IdolImageTo3DPipeline
from diffusers_3d.families.triposplat import (
    TripoSplatGaussianDecoder,
    TripoSplatImageVAE,
    TripoSplatTransformerModel,
)

from .configs import tiny_decoder_config, tiny_dinov3_config, tiny_transformer_config, tiny_vae_config

# A 32-pixel canvas gives 2x2 DINOv3 patches and 2x2 patchified VAE latents.
SMALL = {"num_inference_steps": 2, "num_gaussians": 24, "image_size": 32}


@pytest.fixture
def tiny_idol_components():
    torch.manual_seed(0)
    components = {
        "image_encoder": DINOv3ViTModel(DINOv3ViTConfig(**tiny_dinov3_config())).eval(),
        "vae": TripoSplatImageVAE(**tiny_vae_config()).eval(),
        "transformer": TripoSplatTransformerModel(**tiny_transformer_config()).eval(),
        "decoder": TripoSplatGaussianDecoder(**tiny_decoder_config()).eval(),
        "scheduler": FlowMatchEulerDiscreteScheduler(shift=3.0),
    }
    with torch.no_grad():
        components["vae"].bn.running_mean.normal_()
        components["vae"].bn.running_var.uniform_(0.5, 2.0)
    return components


@pytest.fixture
def tiny_idol_pipeline(tiny_idol_components):
    pipeline = IdolImageTo3DPipeline(**tiny_idol_components)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.disable_background_removal()  # upstream's default BiRefNet would download its weights
    return pipeline


@pytest.fixture
def rgba_condition():
    """A 40x30 RGBA image whose opaque foreground spans rows 10-29 and columns 5-19."""

    image = torch.zeros(4, 40, 30)
    image[:3] = torch.rand(3, 40, 30, generator=torch.Generator().manual_seed(3))
    image[3, 10:30, 5:20] = 1.0
    return ImageCondition(image=image)


@pytest.fixture
def local_external_components(monkeypatch):
    """Load a pipeline class's gated external components from local modules instead of their repositories.

    ``use(pipeline_class, image_encoder=module, ...)`` returns the list of ``(name, repository, revision, dtype)``
    loads; each load returns a copy of the module in the requested dtype.
    """

    def use(pipeline_class, **modules):
        calls = []

        def loader(name):
            def load(repository, *, revision, dtype, **hub_kwargs):
                calls.append((name, repository, revision, dtype))
                module = copy.deepcopy(modules[name])
                return module if dtype is None else module.to(dtype)

            return load

        external = {
            name: dataclasses.replace(component, load=loader(name))
            for name, component in pipeline_class.external_components.items()
        }
        monkeypatch.setattr(pipeline_class, "external_components", external)
        monkeypatch.setattr(pipelines.huggingface_hub, "auth_check", lambda *args, **kwargs: None)
        return calls

    return use
