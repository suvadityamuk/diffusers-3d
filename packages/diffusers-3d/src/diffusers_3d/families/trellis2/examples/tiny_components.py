"""Randomly initialised tiny TRELLIS.2 components so the examples run on CPU in seconds without a checkpoint.

Real use loads a converted checkpoint with ``Trellis2ImageTo3DPipeline.from_pretrained(path)`` instead.
"""

from __future__ import annotations

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers_3d import (
    Trellis2FlowEulerScheduler,
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
)


def build_tiny_pipeline(*, include_slat: bool = False) -> Trellis2ImageTo3DPipeline:
    """A TRELLIS.2 pipeline of random ``tiny_config()`` components; ``include_slat`` adds the SLAT stages."""

    torch.manual_seed(0)
    components: dict[str, object] = {
        "image_encoder": DINOv3ViTModel(
            DINOv3ViTConfig(
                image_size=8,
                patch_size=4,
                hidden_size=12,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=3,
                num_register_tokens=2,
            )
        ).eval(),
        "sparse_structure_flow_model": Trellis2SparseStructureFlowModel(
            **Trellis2SparseStructureFlowModel.tiny_config()
        ),
        "sparse_structure_decoder": Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config()),
        "sparse_structure_scheduler": Trellis2FlowEulerScheduler(),
    }
    if include_slat:
        shape_flow = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config())
        texture_flow = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config(texture=True))
        shape_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config())
        with torch.no_grad():
            # A trained decoder learns which children to keep; random weights would keep none.
            for stage in shape_decoder.blocks[:-1]:
                stage[-1].to_subdiv.bias.fill_(1.0)
        components.update(
            shape_slat_flow_model=shape_flow,
            shape_slat_scheduler=Trellis2FlowEulerScheduler(),
            shape_slat_decoder=shape_decoder,
            texture_slat_flow_model=texture_flow,
            texture_slat_scheduler=Trellis2FlowEulerScheduler(),
            pbr_decoder=Trellis2PBRSparseDecoder(**Trellis2PBRSparseDecoder.tiny_config()),
            shape_slat_mean=[0.0] * shape_flow.config.out_channels,
            shape_slat_std=[1.0] * shape_flow.config.out_channels,
            texture_slat_mean=[0.0] * texture_flow.config.out_channels,
            texture_slat_std=[1.0] * texture_flow.config.out_channels,
        )
    # Converted checkpoints ship "1024_cascade"; the single-stage "512" preset is enough for tiny weights.
    return Trellis2ImageTo3DPipeline(**components, default_pipeline_type="512", image_size=8)


__all__ = ["build_tiny_pipeline"]
