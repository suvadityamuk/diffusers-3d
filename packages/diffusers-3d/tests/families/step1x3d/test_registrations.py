from __future__ import annotations

from diffusers_3d import (
    Step1X3DImageTo3DPipeline,
    Step1X3DMeshTexturePipeline,
    Step1X3DMultiViewAdapter,
    Step1X3DShapeDecoder,
    Step1X3DTransformer1DModel,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_step1x3d_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.step1x3d.")


def test_registries_contain_exact_step1x3d_classes():
    assert _MODEL_REGISTRY.frozen
    assert _PIPELINE_REGISTRY.frozen
    models = {
        registration.model_class for registration in _MODEL_REGISTRY if _is_step1x3d_type(registration.model_class)
    }
    pipelines = [registration for registration in _PIPELINE_REGISTRY if _is_step1x3d_type(registration.pipeline_class)]

    assert models == {Step1X3DTransformer1DModel, Step1X3DShapeDecoder, Step1X3DMultiViewAdapter}
    assert {registration.pipeline_class for registration in pipelines} == {
        Step1X3DImageTo3DPipeline,
        Step1X3DMeshTexturePipeline,
    }
    assert all(registration.metadata.output_representations == ("mesh",) for registration in pipelines)
    for pipeline, task in (
        (Step1X3DImageTo3DPipeline, "image-to-3d"),
        (Step1X3DMeshTexturePipeline, "mesh-texturing"),
    ):
        assert _PIPELINE_REGISTRY.resolve(pipeline.object3d_model_index(), task) is pipeline
