from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from _configs import tiny_config
from transformers import CLIPTextConfig, CLIPTextModel, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import (
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    TrellisTextTo3DPipeline,
)


def tiny_image_encoder() -> Dinov2WithRegistersModel:
    config = Dinov2WithRegistersConfig(
        image_size=8,
        patch_size=4,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        mlp_ratio=2,
        num_register_tokens=2,
    )
    return Dinov2WithRegistersModel(config).eval()


@pytest.fixture
def local_external_components(monkeypatch):
    """Load a pipeline class's external encoders from local modules instead of their repositories.

    ``use(pipeline_class, image_encoder=module, ...)`` returns the list of ``(name, repository, revision)`` loads.
    """

    def use(pipeline_class, **modules):
        calls = []

        def loader(name):
            def load(repository, *, revision, dtype, **hub_kwargs):
                calls.append((name, repository, revision))
                return copy.deepcopy(modules[name])

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
def tiny_trellis_components():
    def make(*, include_slat: bool = False):
        torch.manual_seed(0)
        image_encoder = tiny_image_encoder()
        torch.manual_seed(1)
        sparse_structure_flow_model = TrellisSparseStructureFlowModel(**tiny_config(TrellisSparseStructureFlowModel))
        torch.manual_seed(2)
        sparse_structure_decoder = TrellisSparseStructureDecoder(**tiny_config(TrellisSparseStructureDecoder))
        with torch.no_grad():
            sparse_structure_decoder.out_layer[-1].weight.zero_()
            sparse_structure_decoder.out_layer[-1].bias.fill_(1.0)
        sparse_structure_scheduler = TrellisFlowEulerScheduler()
        if not include_slat:
            return (image_encoder, sparse_structure_flow_model, sparse_structure_decoder, sparse_structure_scheduler)
        torch.manual_seed(3)
        slat_flow_model = TrellisSLatFlowModel(**tiny_config(TrellisSLatFlowModel))
        slat_scheduler = TrellisFlowEulerScheduler()
        torch.manual_seed(4)
        gaussian_decoder = TrellisSLatGaussianDecoder(**tiny_config(TrellisSLatGaussianDecoder))
        torch.manual_seed(5)
        mesh_decoder = TrellisSLatMeshDecoder(**tiny_config(TrellisSLatMeshDecoder))
        torch.manual_seed(6)
        radiance_field_decoder = TrellisSLatRadianceFieldDecoder(**tiny_config(TrellisSLatRadianceFieldDecoder))
        return (
            image_encoder,
            sparse_structure_flow_model,
            sparse_structure_decoder,
            sparse_structure_scheduler,
            slat_flow_model,
            slat_scheduler,
            gaussian_decoder,
            mesh_decoder,
            radiance_field_decoder,
        )

    return make


@pytest.fixture
def tiny_trellis_pipeline(tiny_trellis_components):
    image_encoder, flow, decoder, scheduler = tiny_trellis_components()
    pipeline = TrellisImageTo3DPipeline(
        image_encoder=image_encoder,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
    )
    pipeline.disable_background_removal()  # upstream's default rembg would download u2net
    return pipeline


@pytest.fixture
def tiny_trellis_full_pipeline(tiny_trellis_components):
    image_encoder, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder, mesh_decoder, rf_decoder = (
        tiny_trellis_components(include_slat=True)
    )
    pipeline = TrellisImageTo3DPipeline(
        image_encoder=image_encoder,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        slat_flow_model=slat_flow,
        slat_scheduler=slat_scheduler,
        gaussian_decoder=gaussian_decoder,
        mesh_decoder=mesh_decoder,
        radiance_field_decoder=rf_decoder,
        slat_mean=[0.0] * slat_flow.config.out_channels,
        slat_std=[1.0] * slat_flow.config.out_channels,
    )
    pipeline.disable_background_removal()  # upstream's default rembg would download u2net
    return pipeline


def tiny_text_encoder() -> CLIPTextModel:
    config = CLIPTextConfig(
        vocab_size=64,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return CLIPTextModel(config).eval()


@pytest.fixture
def tiny_trellis_text_pipeline(tiny_trellis_components, tiny_clip_tokenizer):
    _, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder, mesh_decoder, rf_decoder = (
        tiny_trellis_components(include_slat=True)
    )
    return TrellisTextTo3DPipeline(
        text_encoder=tiny_text_encoder(),
        tokenizer=tiny_clip_tokenizer,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        slat_flow_model=slat_flow,
        slat_scheduler=slat_scheduler,
        gaussian_decoder=gaussian_decoder,
        mesh_decoder=mesh_decoder,
        radiance_field_decoder=rf_decoder,
        slat_mean=[0.0] * slat_flow.config.out_channels,
        slat_std=[1.0] * slat_flow.config.out_channels,
    )
