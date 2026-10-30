from __future__ import annotations

import copy
import dataclasses
import warnings

import pytest
import torch
from _configs import tiny_config
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import (
    ImageCondition,
    Sam3DConditionEmbedder,
    Sam3DObjectsPipeline,
    Sam3DSparseStructureFlowModel,
    TrellisFlowEulerScheduler,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSparseStructureDecoder,
)


def tiny_dinov2_config() -> Dinov2WithRegistersConfig:
    return Dinov2WithRegistersConfig(
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        mlp_ratio=2,
        patch_size=4,
        image_size=8,
        num_register_tokens=2,
    )


def tiny_sam3d_components(*, include_slat: bool = True) -> dict:
    torch.manual_seed(0)
    decoder = TrellisSparseStructureDecoder(**tiny_config(TrellisSparseStructureDecoder))
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    components = {
        "image_encoder": Dinov2WithRegistersModel(tiny_dinov2_config()).eval(),
        "sparse_structure_conditioner": Sam3DConditionEmbedder(**tiny_config(Sam3DConditionEmbedder)),
        "sparse_structure_flow_model": Sam3DSparseStructureFlowModel(**tiny_config(Sam3DSparseStructureFlowModel)),
        "sparse_structure_decoder": decoder,
        "sparse_structure_scheduler": TrellisFlowEulerScheduler(sigma_min=0.0),
    }
    if include_slat:
        slat_flow = TrellisSLatFlowModel(**tiny_config(TrellisSLatFlowModel))
        components |= {
            "slat_conditioner": Sam3DConditionEmbedder(
                **{**tiny_config(Sam3DConditionEmbedder), "prenorm_features": True, "pointmap_config": None}
            ),
            "slat_flow_model": slat_flow,
            "slat_scheduler": TrellisFlowEulerScheduler(sigma_min=0.0),
            "gaussian_decoder": TrellisSLatGaussianDecoder(**tiny_config(TrellisSLatGaussianDecoder)),
            "mesh_decoder": TrellisSLatMeshDecoder(**tiny_config(TrellisSLatMeshDecoder)),
            "slat_mean": [0.5] * slat_flow.config.out_channels,
            "slat_std": [2.0] * slat_flow.config.out_channels,
        }
    return components


def tiny_sam3d_pipeline(*, include_slat: bool = True) -> Sam3DObjectsPipeline:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Sam3DObjectsPipeline(**tiny_sam3d_components(include_slat=include_slat))


def tiny_condition(seed: int = 0, *, height: int = 12, width: int = 10) -> tuple[ImageCondition, torch.Tensor]:
    """An image with a rectangular object mask and an OpenCV-convention pointmap with a few invalid points."""

    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, height, width)
    mask[:, 3:9, 2:8] = 1.0
    pointmap = torch.randn(3, height, width, generator=generator)
    pointmap[2] = pointmap[2].abs() + 2.0
    pointmap[:, 0, :3] = float("nan")
    return ImageCondition(image=torch.rand(3, height, width, generator=generator), mask=mask), pointmap


@pytest.fixture
def tiny_pipeline():
    return tiny_sam3d_pipeline()


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
