from __future__ import annotations

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

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
def tiny_trellis_components():
    def make(*, include_slat: bool = False):
        torch.manual_seed(0)
        image_encoder = tiny_image_encoder()
        torch.manual_seed(1)
        sparse_structure_flow_model = TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config())
        torch.manual_seed(2)
        sparse_structure_decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
        with torch.no_grad():
            sparse_structure_decoder.out_layer[-1].weight.zero_()
            sparse_structure_decoder.out_layer[-1].bias.fill_(1.0)
        sparse_structure_scheduler = TrellisFlowEulerScheduler()
        if not include_slat:
            return (image_encoder, sparse_structure_flow_model, sparse_structure_decoder, sparse_structure_scheduler)
        torch.manual_seed(3)
        slat_flow_model = TrellisSLatFlowModel(**TrellisSLatFlowModel.tiny_config())
        slat_scheduler = TrellisFlowEulerScheduler()
        torch.manual_seed(4)
        gaussian_decoder = TrellisSLatGaussianDecoder(**TrellisSLatGaussianDecoder.tiny_config())
        torch.manual_seed(5)
        mesh_decoder = TrellisSLatMeshDecoder(**TrellisSLatMeshDecoder.tiny_config())
        torch.manual_seed(6)
        radiance_field_decoder = TrellisSLatRadianceFieldDecoder(**TrellisSLatRadianceFieldDecoder.tiny_config())
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
    return TrellisImageTo3DPipeline(
        image_encoder=image_encoder,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
    )


@pytest.fixture
def tiny_trellis_full_pipeline(tiny_trellis_components):
    image_encoder, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder, mesh_decoder, rf_decoder = (
        tiny_trellis_components(include_slat=True)
    )
    return TrellisImageTo3DPipeline(
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
