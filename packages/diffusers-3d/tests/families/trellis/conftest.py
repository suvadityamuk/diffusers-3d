from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    TrellisDinov2Conditioner,
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
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
        return (
            conditioner,
            sparse_structure_flow_model,
            sparse_structure_decoder,
            sparse_structure_scheduler,
            slat_flow_model,
            slat_scheduler,
            gaussian_decoder,
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
    conditioner, flow, decoder, scheduler, slat_flow, slat_scheduler, gaussian_decoder = tiny_trellis_components(
        include_slat=True
    )
    return TrellisImageTo3DPipeline(
        conditioner=conditioner,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=scheduler,
        slat_flow_model=slat_flow,
        slat_scheduler=slat_scheduler,
        gaussian_decoder=gaussian_decoder,
        slat_mean=[0.0] * slat_flow.config.out_channels,
        slat_std=[1.0] * slat_flow.config.out_channels,
    )
