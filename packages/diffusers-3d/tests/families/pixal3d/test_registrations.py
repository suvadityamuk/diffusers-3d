from __future__ import annotations

from diffusers_3d import (
    Pixal3DImageTo3DPipeline,
    Pixal3DNAFUpsampler,
    Pixal3DSLatFlowModel,
    Pixal3DSparseStructureFlowModel,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_pixal3d_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.pixal3d.")


def test_registries_contain_exact_pixal3d_classes():
    assert {r.model_class for r in _MODEL_REGISTRY if _is_pixal3d_type(r.model_class)} == {
        Pixal3DSparseStructureFlowModel,
        Pixal3DSLatFlowModel,
        Pixal3DNAFUpsampler,
    }
    assert {r.pipeline_class for r in _PIPELINE_REGISTRY if _is_pixal3d_type(r.pipeline_class)} == {
        Pixal3DImageTo3DPipeline
    }
    metadata = Pixal3DImageTo3DPipeline.object3d_model_index()
    assert _PIPELINE_REGISTRY.resolve(metadata, "image-to-3d") is Pixal3DImageTo3DPipeline
    # The decoders and the scheduler are the registered TRELLIS.2 classes, reused as-is; DINOv3 is Transformers'.
    expected = {component.name: component.expected_class for component in metadata.components}
    assert expected["image_encoder"] == "transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTModel"
    assert expected["pbr_decoder"].startswith("diffusers_3d.families.trellis2.")
    assert expected["feature_upsampler"].endswith(Pixal3DNAFUpsampler.__name__)
