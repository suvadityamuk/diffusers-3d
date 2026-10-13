"""Tencent Hunyuan3D-2.1 shape stage: flow DiT, ShapeVAE decoder, scheduler, and pipeline."""

from importlib import import_module

_EXPORT_MODULES = {
    "HUNYUAN3D_REFERENCE_REVISION": ".conversion",
    "Hunyuan3DDiTModel": ".models",
    "Hunyuan3DDiTOutput": ".models",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler": ".scheduler",
    "Hunyuan3DImageTo3DPipeline": ".pipeline",
    "Hunyuan3DShapeVAE": ".vae",
    "Hunyuan3DShapeVAEOutput": ".vae",
    "convert_hunyuan3d_checkpoint": ".conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
