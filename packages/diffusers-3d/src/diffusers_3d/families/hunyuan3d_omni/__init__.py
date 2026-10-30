"""Tencent Hunyuan3D-Omni: Hunyuan3D-2.1 image-to-shape with point-cloud, voxel, bounding-box, or skeleton control."""

from importlib import import_module

_EXPORT_MODULES = {
    "HUNYUAN3D_OMNI_REFERENCE_REVISION": ".conversion",
    "Hunyuan3DOmniControlEncoder": ".models",
    "Hunyuan3DOmniControlEncoderOutput": ".models",
    "Hunyuan3DOmniImageTo3DPipeline": ".pipeline",
    "convert_hunyuan3d_omni_checkpoint": ".conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
