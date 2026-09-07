from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    ImageCondition,
    TrellisDinov2Conditioner,
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureExample,
    TrellisSparseStructureFlowModel,
)


class TinyTrellisPrecomputedLatentDataset:
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> TrellisSparseStructureExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        offset = float(index) / self.length
        image = torch.linspace(0.0, 1.0 - 0.1 * offset, 3 * 8 * 8).reshape(3, 8, 8)
        latents = torch.linspace(-0.75 + offset, 0.75 + offset, 2 * 4 * 4 * 4).reshape(2, 4, 4, 4)
        return TrellisSparseStructureExample(
            condition=ImageCondition(image=image),
            sparse_structure_latents=latents,
            example_id=f"tiny-trellis-{index}",
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


@pytest.fixture
def tiny_trellis_latent_dataset():
    return TinyTrellisPrecomputedLatentDataset()
