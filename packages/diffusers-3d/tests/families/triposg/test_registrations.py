from __future__ import annotations

from diffusers_3d import TripoSGImageTo3DPipeline, TripoSGTransformerModel, TripoSGVAEDecoder
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_triposg_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.triposg.")


def test_registries_contain_exact_triposg_classes():
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen
    models = {
        registration.model_class for registration in _MODEL_REGISTRY if _is_triposg_type(registration.model_class)
    }
    pipelines = [registration for registration in _PIPELINE_REGISTRY if _is_triposg_type(registration.pipeline_class)]
    assert models == {TripoSGTransformerModel, TripoSGVAEDecoder}
    assert [registration.pipeline_class for registration in pipelines] == [TripoSGImageTo3DPipeline]
    metadata = pipelines[0].metadata
    assert metadata.output_representations == ("mesh",)
    assert metadata.required_backends == ("scikit-image",)
    assert {component.name: component.expected_class for component in metadata.components} == {
        "feature_extractor": "transformers.models.bit.image_processing_pil_bit.BitImageProcessorPil",
        "image_encoder": "transformers.models.dinov2.modeling_dinov2.Dinov2Model",
        "scheduler": "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
        "transformer": "diffusers_3d.families.triposg.models.TripoSGTransformerModel",
        "vae": "diffusers_3d.families.triposg.vae.TripoSGVAEDecoder",
    }
    assert _PIPELINE_REGISTRY.resolve(metadata, "image-to-3d") is TripoSGImageTo3DPipeline
