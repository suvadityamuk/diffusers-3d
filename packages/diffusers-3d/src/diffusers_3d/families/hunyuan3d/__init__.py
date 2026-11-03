"""Tencent Hunyuan3D-2.1: the shape stage (flow DiT, ShapeVAE, scheduler, pipeline) and the Paint PBR texture stage."""

from importlib import import_module

_EXPORT_MODULES = {
    "HUNYUAN3D_REFERENCE_REVISION": ".conversion",
    "Hunyuan3DDiTModel": ".models",
    "Hunyuan3DDiTOutput": ".models",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler": ".scheduler",
    "Hunyuan3DImageTo3DPipeline": ".pipeline",
    "Hunyuan3DPaintPipeline": ".paint_pipeline",
    "Hunyuan3DPaintRRDBNet": ".paint_models",
    "Hunyuan3DPaintUNet": ".paint_models",
    "Hunyuan3DShapeVAE": ".vae",
    "Hunyuan3DShapeVAEOutput": ".vae",
    "convert_hunyuan3d_checkpoint": ".conversion",
    "convert_hunyuan3d_paint_checkpoint": ".paint_conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
