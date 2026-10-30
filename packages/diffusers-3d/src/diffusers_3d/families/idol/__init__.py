"""Speridlabs IDOL image-to-Gaussians: TripoSplat's flow transformer and octree Gaussian decoder with IDOL weights."""

from importlib import import_module

_EXPORT_MODULES = {
    "IDOL_REFERENCE_REVISION": ".conversion",
    "IDOL_WEIGHTS_REVISION": ".conversion",
    "IdolImageTo3DPipeline": ".pipeline",
    "convert_idol_checkpoint": ".conversion",
    "decode_idol_camera": ".pipeline",
    "prepare_idol_image": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
