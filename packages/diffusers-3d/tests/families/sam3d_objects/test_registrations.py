from __future__ import annotations

from transformers import Dinov2WithRegistersModel

from diffusers_3d import (
    Sam3DConditionEmbedder,
    Sam3DObjectsPipeline,
    Sam3DSparseStructureFlowModel,
    TrellisSLatFlowModel,
    TrellisSparseStructureDecoder,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_sam3d_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.sam3d_objects.")


def test_registries_contain_exact_sam3d_classes_and_reuse_trellis_components():
    assert _MODEL_REGISTRY.frozen and _PIPELINE_REGISTRY.frozen
    assert {
        registration.model_class for registration in _MODEL_REGISTRY if _is_sam3d_type(registration.model_class)
    } == {Sam3DConditionEmbedder, Sam3DSparseStructureFlowModel}
    pipelines = [registration for registration in _PIPELINE_REGISTRY if _is_sam3d_type(registration.pipeline_class)]
    assert [registration.pipeline_class for registration in pipelines] == [Sam3DObjectsPipeline]
    assert set(pipelines[0].metadata.output_representations) == {"sparse-structure", "gaussian-splat", "mesh"}

    index = Sam3DObjectsPipeline.object3d_model_index()
    assert _PIPELINE_REGISTRY.resolve(index, "image-to-3d") is Sam3DObjectsPipeline
    expected = {component.name: component.expected_class for component in index.components}
    assert expected["image_encoder"].endswith(Dinov2WithRegistersModel.__name__)
    assert expected["sparse_structure_decoder"].endswith(TrellisSparseStructureDecoder.__name__)
    assert expected["slat_flow_model"].endswith(TrellisSLatFlowModel.__name__)
