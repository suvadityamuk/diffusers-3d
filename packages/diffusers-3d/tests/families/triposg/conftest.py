from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from _configs import tiny_config
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import ImageCondition, TripoSGImageTo3DPipeline, TripoSGTransformerModel, TripoSGVAEDecoder


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
    transformer = TripoSGTransformerModel(**tiny_config(TripoSGTransformerModel))
    torch.manual_seed(2)
    vae = TripoSGVAEDecoder(**tiny_config(TripoSGVAEDecoder))
    with torch.no_grad():
        # Spread the untrained field across its zero level so marching cubes finds a surface.
        vae.proj_out.weight.mul_(50)
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
    pipeline = TripoSGImageTo3DPipeline(**tiny_triposg_components)
    pipeline.disable_background_removal()  # the tiny images have no object for RMBG-1.4 to cut out
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
