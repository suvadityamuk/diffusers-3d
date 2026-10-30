from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from _configs import tiny_config
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import diffusers_3d.execution.pipelines as pipelines
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
    feature_upsampler = Pixal3DNAFUpsampler(**tiny_config(Pixal3DNAFUpsampler))
    torch.manual_seed(2)
    flow = Pixal3DSparseStructureFlowModel(**tiny_config(Pixal3DSparseStructureFlowModel))
    decoder = Trellis2SparseStructureDecoder(**tiny_config(Trellis2SparseStructureDecoder))
    with torch.no_grad():
        # Occupy every cell so the SLAT stages always have voxels.
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    torch.manual_seed(3)
    shape_flow = Pixal3DSLatFlowModel(**tiny_config(Pixal3DSLatFlowModel))
    shape_flow_1024 = Pixal3DSLatFlowModel(**tiny_config(Pixal3DSLatFlowModel))
    shape_decoder = Trellis2ShapeDualGridDecoder(**tiny_config(Trellis2ShapeDualGridDecoder))
    texture_components = {}
    if texture:
        torch.manual_seed(4)
        texture_components = {
            "texture_slat_flow_model_1024": Pixal3DSLatFlowModel(**tiny_config(Pixal3DSLatFlowModel, texture=True)),
            "texture_slat_scheduler": Trellis2FlowEulerScheduler(),
            "pbr_decoder": Trellis2PBRSparseDecoder(**tiny_config(Trellis2PBRSparseDecoder)),
            "texture_slat_mean": [0.0] * 4,
            "texture_slat_std": [1.0] * 4,
        }
    for model in (flow, shape_flow, shape_flow_1024, *texture_components.values()):
        if isinstance(model, torch.nn.Module) and hasattr(model, "out_layer"):
            with torch.no_grad():
                model.out_layer.weight.normal_(std=0.02)
    pipeline = Pixal3DImageTo3DPipeline(
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
    pipeline.disable_background_removal()  # upstream's default RMBG-2.0 weights are gated and would download
    return pipeline


@pytest.fixture
def tiny_pixal3d_pipeline():
    return tiny_pipeline()


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
