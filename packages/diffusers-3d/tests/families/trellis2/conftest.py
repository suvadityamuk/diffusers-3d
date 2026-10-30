from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from _configs import tiny_config
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import (
    Trellis2FlowEulerScheduler,
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
)


def tiny_image_encoder() -> DINOv3ViTModel:
    # 4-pixel patches and 2 registers: 7 tokens on an 8-pixel side. Eval mode turns off DINOv3's RoPE augmentation.
    return DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=8,
            patch_size=4,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_register_tokens=2,
        )
    ).eval()


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


@pytest.fixture
def tiny_trellis2_components():
    def make(*, include_slat: bool = False):
        torch.manual_seed(0)
        image_encoder = tiny_image_encoder()
        torch.manual_seed(1)
        sparse_structure_flow_model = Trellis2SparseStructureFlowModel(**tiny_config(Trellis2SparseStructureFlowModel))
        torch.manual_seed(2)
        sparse_structure_decoder = Trellis2SparseStructureDecoder(**tiny_config(Trellis2SparseStructureDecoder))
        with torch.no_grad():
            sparse_structure_decoder.out_layer[-1].weight.zero_()
            sparse_structure_decoder.out_layer[-1].bias.fill_(1.0)
        sparse_structure_scheduler = Trellis2FlowEulerScheduler()
        if not include_slat:
            return (image_encoder, sparse_structure_flow_model, sparse_structure_decoder, sparse_structure_scheduler)

        torch.manual_seed(3)
        shape_slat_flow_model = Trellis2SLatFlowModel(**tiny_config(Trellis2SLatFlowModel))
        shape_slat_scheduler = Trellis2FlowEulerScheduler()
        torch.manual_seed(4)
        shape_slat_decoder = Trellis2ShapeDualGridDecoder(**tiny_config(Trellis2ShapeDualGridDecoder))
        torch.manual_seed(5)
        texture_slat_flow_model = Trellis2SLatFlowModel(**tiny_config(Trellis2SLatFlowModel, texture=True))
        texture_slat_scheduler = Trellis2FlowEulerScheduler()
        torch.manual_seed(6)
        pbr_decoder = Trellis2PBRSparseDecoder(**tiny_config(Trellis2PBRSparseDecoder))
        return (
            image_encoder,
            sparse_structure_flow_model,
            sparse_structure_decoder,
            sparse_structure_scheduler,
            shape_slat_flow_model,
            shape_slat_scheduler,
            shape_slat_decoder,
            texture_slat_flow_model,
            texture_slat_scheduler,
            pbr_decoder,
        )

    return make


@pytest.fixture
def tiny_trellis2_pipeline(tiny_trellis2_components):
    image_encoder, flow, decoder, scheduler = tiny_trellis2_components()
    pipeline = Trellis2ImageTo3DPipeline(
        image_encoder=image_encoder,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        default_pipeline_type="512",
        image_size=8,
    )
    pipeline.disable_background_removal()  # upstream's default RMBG-2.0 weights are gated and would download
    return pipeline


@pytest.fixture
def tiny_trellis2_full_pipeline(tiny_trellis2_components):
    (
        image_encoder,
        flow,
        decoder,
        scheduler,
        shape_flow,
        shape_scheduler,
        shape_decoder,
        texture_flow,
        texture_scheduler,
        pbr_decoder,
    ) = tiny_trellis2_components(include_slat=True)
    torch.manual_seed(7)
    shape_flow_1024 = Trellis2SLatFlowModel(**tiny_config(Trellis2SLatFlowModel))
    torch.manual_seed(8)
    texture_flow_1024 = Trellis2SLatFlowModel(**tiny_config(Trellis2SLatFlowModel, texture=True))
    pipeline = Trellis2ImageTo3DPipeline(
        image_encoder=image_encoder,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        shape_slat_flow_model=shape_flow,
        shape_slat_flow_model_1024=shape_flow_1024,
        shape_slat_scheduler=shape_scheduler,
        shape_slat_decoder=shape_decoder,
        texture_slat_flow_model=texture_flow,
        texture_slat_flow_model_1024=texture_flow_1024,
        texture_slat_scheduler=texture_scheduler,
        pbr_decoder=pbr_decoder,
        shape_slat_mean=[0.0] * shape_flow.config.out_channels,
        shape_slat_std=[1.0] * shape_flow.config.out_channels,
        texture_slat_mean=[0.0] * texture_flow.config.out_channels,
        texture_slat_std=[1.0] * texture_flow.config.out_channels,
        default_pipeline_type="512",
        image_size=8,
    )
    pipeline.disable_background_removal()
    return pipeline
