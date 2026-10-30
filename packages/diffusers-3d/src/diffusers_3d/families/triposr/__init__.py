"""Stability AI / Tripo AI TripoSR feed-forward image-to-3D: triplane transformer, NeRF decoder, and pipeline."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRIPOSR_REFERENCE_REVISION": ".conversion",
    "TRIPOSR_WEIGHTS_REVISION": ".conversion",
    "TripoSRImageTo3DPipeline": ".pipeline",
    "TripoSRTransformerModel": ".models",
    "TripoSRTriplaneDecoder": ".models",
    "TripoSRTriplaneDecoderOutput": ".models",
    "convert_triposr_checkpoint": ".conversion",
    "prepare_triposr_image": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
