"""Meta SAM 3D Objects: image + mask (+ pointmap) to layout-placed Gaussian splats and meshes on TRELLIS stages."""

from importlib import import_module

_EXPORT_MODULES = {
    "SAM3D_REFERENCE_REVISION": ".conversion",
    "Sam3DConditionEmbedder": ".conditioner",
    "Sam3DConditionerOutput": ".conditioner",
    "Sam3DObjectsPipeline": ".pipeline",
    "Sam3DSparseStructureFlowModel": ".models",
    "Sam3DSparseStructureFlowOutput": ".models",
    "convert_sam3d_checkpoint": ".conversion",
    "preprocess_sam3d_condition": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
