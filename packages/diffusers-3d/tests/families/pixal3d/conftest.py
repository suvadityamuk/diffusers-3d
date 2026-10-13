from __future__ import annotations

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers_3d import (
    Pixal3DImageTo3DPipeline,
    Pixal3DNAFUpsampler,
    Pixal3DSLatFlowModel,
    Pixal3DSparseStructureFlowModel,
    Trellis2FlowEulerScheduler,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SparseStructureDecoder,
)

# The tiny DINOv3 has 4-pixel patches; a 16-pixel side keeps every NAF window (3 cells) inside the patch grid.
TINY_IMAGE_SIZE = 16


def tiny_pipeline(*, multiview: bool = False, texture: bool = True) -> Pixal3DImageTo3DPipeline:
    torch.manual_seed(0)
    image_encoder = DINOv3ViTModel(
        DINOv3ViTConfig(
            patch_size=4,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_register_tokens=2,
        )
    ).eval()
    torch.manual_seed(1)
    feature_upsampler = Pixal3DNAFUpsampler(**Pixal3DNAFUpsampler.tiny_config())
    torch.manual_seed(2)
    flow = Pixal3DSparseStructureFlowModel(**Pixal3DSparseStructureFlowModel.tiny_config())
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    with torch.no_grad():
        # Occupy every cell so the SLAT stages always have voxels.
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    torch.manual_seed(3)
    shape_flow = Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.tiny_config())
    shape_flow_1024 = Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.tiny_config())
    shape_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
    texture_components = {}
    if texture:
        torch.manual_seed(4)
        texture_components = {
            "texture_slat_flow_model_1024": Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.tiny_config(texture=True)),
            "texture_slat_scheduler": Trellis2FlowEulerScheduler(),
            "pbr_decoder": Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.tiny_config()),
            "texture_slat_mean": [0.0] * 4,
            "texture_slat_std": [1.0] * 4,
        }
    for model in (flow, shape_flow, shape_flow_1024, *texture_components.values()):
        if isinstance(model, torch.nn.Module) and hasattr(model, "out_layer"):
            with torch.no_grad():
                model.out_layer.weight.normal_(std=0.02)
    return Pixal3DImageTo3DPipeline(
        image_encoder=image_encoder,
        feature_upsampler=feature_upsampler,
        sparse_structure_flow_model=flow,
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=Trellis2FlowEulerScheduler(),
        shape_slat_flow_model=shape_flow,
        shape_slat_flow_model_1024=shape_flow_1024,
        shape_slat_scheduler=Trellis2FlowEulerScheduler(),
        shape_slat_decoder=shape_decoder,
        shape_slat_mean=[0.0] * 4,
        shape_slat_std=[1.0] * 4,
        multiview=multiview,
        image_size=TINY_IMAGE_SIZE,
        **texture_components,
    )


@pytest.fixture
def tiny_pixal3d_pipeline():
    return tiny_pipeline()
