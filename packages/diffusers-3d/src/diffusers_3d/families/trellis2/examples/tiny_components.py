"""Randomly initialised tiny TRELLIS.2 components for running the examples without a checkpoint.

This is test scaffolding, not part of the normal user path. With a converted checkpoint you load the
pipeline with ``Trellis2ImageTo3DPipeline.from_pretrained(path)`` and never construct components by
hand. The tiny pipeline exists so the examples and docs can be executed on CPU in seconds.
"""

from __future__ import annotations

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


def build_tiny_pipeline(*, include_slat: bool = False) -> Trellis2ImageTo3DPipeline:
    """Assemble a TRELLIS.2 pipeline from ``tiny_config()`` components with random weights.

    Component keyword names are the pipeline's constructor arguments and match the subfolders of a
    saved checkpoint. ``include_slat=True`` adds the shape SLAT, texture SLAT, and O-Voxel decoder
    stages so the full pipeline runs end to end.
    """

    torch.manual_seed(0)
    components: dict[str, object] = {
        "conditioner": Trellis2Dinov3Conditioner(**Trellis2Dinov3Conditioner.tiny_config()),
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
    return Trellis2ImageTo3DPipeline(**components, default_pipeline_type="512")


__all__ = ["build_tiny_pipeline"]
