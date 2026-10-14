from __future__ import annotations

from diffusers_3d import (
    Hunyuan3DDiTModel,
    Hunyuan3DImageTo3DPipeline,
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintRRDBNet,
    Hunyuan3DPaintUNet,
    Hunyuan3DShapeVAE,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY


def _is_hunyuan3d_type(value: type[object]) -> bool:
    return value.__module__.startswith("diffusers_3d.families.hunyuan3d.")


def test_registries_contain_exact_hunyuan3d_classes():
    models = {
        registration.model_class for registration in _MODEL_REGISTRY if _is_hunyuan3d_type(registration.model_class)
    }
    pipelines = [
        registration for registration in _PIPELINE_REGISTRY if _is_hunyuan3d_type(registration.pipeline_class)
    ]
    assert models == {Hunyuan3DDiTModel, Hunyuan3DShapeVAE, Hunyuan3DPaintUNet, Hunyuan3DPaintRRDBNet}
    assert {registration.pipeline_class for registration in pipelines} == {
        Hunyuan3DImageTo3DPipeline,
        Hunyuan3DPaintPipeline,
    }
    assert all(registration.metadata.output_representations == ("mesh",) for registration in pipelines)
    assert (
        _PIPELINE_REGISTRY.resolve(Hunyuan3DImageTo3DPipeline.object3d_model_index(), "image-to-3d")
        is Hunyuan3DImageTo3DPipeline
    )
    paint = Hunyuan3DPaintPipeline.object3d_model_index()
    assert paint.task_ids == ("mesh-texturing",)
    assert _PIPELINE_REGISTRY.resolve(paint, "mesh-texturing") is Hunyuan3DPaintPipeline
    assert [component.name for component in paint.components if component.optional] == ["upscaler"]
