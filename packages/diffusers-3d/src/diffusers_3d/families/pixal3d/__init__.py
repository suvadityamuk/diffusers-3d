"""Tencent ARC Pixal3D: TRELLIS.2 cascade with pixel-aligned projection conditioning and NAF feature upsampling."""

from importlib import import_module

_EXPORT_MODULES = {
    "PIXAL3D_REFERENCE_REVISION": ".conversion",
    "Pixal3DImageTo3DPipeline": ".pipeline",
    "Pixal3DNAFUpsampler": ".conditioner",
    "Pixal3DNAFUpsamplerOutput": ".conditioner",
    "Pixal3DSLatFlowModel": ".models",
    "Pixal3DSparseStructureFlowModel": ".models",
    "convert_pixal3d_checkpoint": ".conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
