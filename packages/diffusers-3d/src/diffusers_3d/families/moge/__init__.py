"""Microsoft MoGe and MoGe-2 monocular geometry: point maps, depth, masks and camera intrinsics from one image."""

from importlib import import_module

_EXPORT_MODULES = {
    "MOGE2_REFERENCE_REVISION": ".conversion",
    "MOGE2_WEIGHTS_REPOSITORY": ".conversion",
    "MOGE2_WEIGHTS_REVISION": ".conversion",
    "MOGE_REFERENCE_REVISION": ".conversion",
    "MOGE_WEIGHTS_REPOSITORY": ".conversion",
    "MOGE_WEIGHTS_REVISION": ".conversion",
    "MoGe2Model": ".models",
    "MoGeModel": ".models",
    "convert_moge_checkpoint": ".conversion",
    "recover_focal_shift": ".geometry",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
