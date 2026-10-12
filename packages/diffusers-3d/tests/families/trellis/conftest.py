from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    TrellisClipTextConditioner,
    TrellisDinov2Conditioner,
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


@pytest.fixture
def tiny_trellis_components():
    def make(*, include_slat: bool = False):
        torch.manual_seed(0)
        conditioner = TrellisDinov2Conditioner(**TrellisDinov2Conditioner.tiny_config())
        torch.manual_seed(1)
        sparse_structure_flow_model = TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config())
        torch.manual_seed(2)
        sparse_structure_decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
        with torch.no_grad():
            sparse_structure_decoder.out_layer[-1].weight.zero_()
            sparse_structure_decoder.out_layer[-1].bias.fill_(1.0)
        sparse_structure_scheduler = TrellisFlowEulerScheduler()
        if not include_slat:
            return (
                conditioner,
                sparse_structure_flow_model,
                sparse_structure_decoder,
                sparse_structure_scheduler,
            )
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
            conditioner,
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
    conditioner, flow, decoder, scheduler = tiny_trellis_components()
    return TrellisImageTo3DPipeline(
        conditioner=conditioner,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
    )


@pytest.fixture
def tiny_trellis_full_pipeline(tiny_trellis_components):
    conditioner, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder, mesh_decoder, rf_decoder = (
        tiny_trellis_components(include_slat=True)
    )
    return TrellisImageTo3DPipeline(
        conditioner=conditioner,
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


@pytest.fixture
def tiny_text_conditioner(tiny_clip_tokenizer) -> TrellisClipTextConditioner:
    torch.manual_seed(0)
    conditioner = TrellisClipTextConditioner(**TrellisClipTextConditioner.tiny_config())
    conditioner.tokenizer = tiny_clip_tokenizer
    return conditioner


@pytest.fixture
def tiny_trellis_text_pipeline(tiny_trellis_components, tiny_text_conditioner):
    _, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder, mesh_decoder, rf_decoder = (
        tiny_trellis_components(include_slat=True)
    )
    return TrellisTextTo3DPipeline(
        conditioner=tiny_text_conditioner,
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
