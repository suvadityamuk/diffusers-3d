from __future__ import annotations

import copy
import dataclasses
import warnings

import pytest
import torch
from _configs import tiny_config
from transformers import Dinov2Config, Dinov2Model

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d.families.hunyuan3d.models import Hunyuan3DDiTModel
from diffusers_3d.families.hunyuan3d.scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from diffusers_3d.families.hunyuan3d.vae import Hunyuan3DShapeVAE
from diffusers_3d.families.hunyuan3d_omni.models import Hunyuan3DOmniControlEncoder
from diffusers_3d.families.hunyuan3d_omni.pipeline import Hunyuan3DOmniImageTo3DPipeline


def make_tiny_omni_components() -> dict:
    torch.manual_seed(0)
    return {
        "image_encoder": Dinov2Model(
            Dinov2Config(
                hidden_size=24, num_hidden_layers=1, num_attention_heads=4, mlp_ratio=2, image_size=28, patch_size=14
            )
        ),
        "control_encoder": Hunyuan3DOmniControlEncoder(**tiny_config(Hunyuan3DOmniControlEncoder)),
        "transformer": Hunyuan3DDiTModel(**tiny_config(Hunyuan3DDiTModel)),
        "vae": Hunyuan3DShapeVAE(**tiny_config(Hunyuan3DShapeVAE)),
        "scheduler": Hunyuan3DFlowMatchEulerDiscreteScheduler(),
    }


def make_tiny_omni_pipeline() -> Hunyuan3DOmniImageTo3DPipeline:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-Omni weights", category=UserWarning)
        pipeline = Hunyuan3DOmniImageTo3DPipeline(**make_tiny_omni_components(), image_size=16)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def rgba_condition() -> torch.Tensor:
    rgba = torch.zeros(4, 20, 24)
    rgba[:3] = torch.linspace(0, 1, 3 * 20 * 24).reshape(3, 20, 24)
    rgba[3, 4:15, 6:20] = 1.0
    return rgba


def point_cloud() -> torch.Tensor:
    return torch.linspace(-0.9, 0.9, 7 * 3).reshape(7, 3)


def sample_latents(pipeline: Hunyuan3DOmniImageTo3DPipeline, **control) -> torch.Tensor:
    """Run the sampling stages (no surface extraction, so no scikit-image) on a fixed image, control, and seed."""

    control = control or {"point_cloud": point_cloud()}
    ((control_type, value),) = control.items()
    with torch.no_grad():
        images = pipeline.preprocess(rgba_condition())
        context = pipeline.encode_conditions(images, value, control_type, do_classifier_free_guidance=True)
        latents = pipeline.prepare_latents(1, torch.Generator().manual_seed(0))
        return pipeline.denoise(latents, context, num_inference_steps=2, guidance_scale=4.5)


@pytest.fixture
def tiny_omni_pipeline() -> Hunyuan3DOmniImageTo3DPipeline:
    return make_tiny_omni_pipeline()


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
