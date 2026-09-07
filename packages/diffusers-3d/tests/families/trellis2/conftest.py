from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    Trellis2Dinov3Conditioner,
    Trellis2FlowEulerScheduler,
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
)


@pytest.fixture
def tiny_trellis2_components():
    def make(*, include_experimental: bool = False):
        torch.manual_seed(0)
        conditioner = Trellis2Dinov3Conditioner(**Trellis2Dinov3Conditioner.tiny_config())
        torch.manual_seed(1)
        sparse_structure_flow_model = Trellis2SparseStructureFlowModel(
            **Trellis2SparseStructureFlowModel.tiny_config()
        )
        torch.manual_seed(2)
        sparse_structure_decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
        with torch.no_grad():
            sparse_structure_decoder.out_layer[-1].weight.zero_()
            sparse_structure_decoder.out_layer[-1].bias.fill_(1.0)
        sparse_structure_scheduler = Trellis2FlowEulerScheduler()
        if not include_experimental:
            return (
                conditioner,
                sparse_structure_flow_model,
                sparse_structure_decoder,
                sparse_structure_scheduler,
            )

        torch.manual_seed(3)
        shape_slat_flow_model = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config())
        shape_slat_scheduler = Trellis2FlowEulerScheduler()
        torch.manual_seed(4)
        shape_slat_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
        torch.manual_seed(5)
        texture_slat_flow_model = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config(texture=True))
        texture_slat_scheduler = Trellis2FlowEulerScheduler()
        torch.manual_seed(6)
        pbr_decoder = Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.tiny_config())
        return (
            conditioner,
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
    conditioner, flow, decoder, scheduler = tiny_trellis2_components()
    return Trellis2ImageTo3DPipeline(
        conditioner=conditioner,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        default_pipeline_type="tiny",
    )


@pytest.fixture
def tiny_trellis2_full_pipeline(tiny_trellis2_components):
    (
        conditioner,
        flow,
        decoder,
        scheduler,
        shape_flow,
        shape_scheduler,
        shape_decoder,
        texture_flow,
        texture_scheduler,
        pbr_decoder,
    ) = tiny_trellis2_components(include_experimental=True)
    return Trellis2ImageTo3DPipeline(
        conditioner=conditioner,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        shape_slat_flow_model=shape_flow,
        shape_slat_scheduler=shape_scheduler,
        shape_slat_decoder=shape_decoder,
        texture_slat_flow_model=texture_flow,
        texture_slat_scheduler=texture_scheduler,
        pbr_decoder=pbr_decoder,
        shape_slat_mean=[0.0] * shape_flow.config.out_channels,
        shape_slat_std=[1.0] * shape_flow.config.out_channels,
        texture_slat_mean=[0.0] * texture_flow.config.out_channels,
        texture_slat_std=[1.0] * texture_flow.config.out_channels,
        default_pipeline_type="tiny",
    )
