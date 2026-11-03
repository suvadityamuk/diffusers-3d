"""VAST-AI TripoSG image-to-shape: rectified-flow transformer, shape VAE decoder, and pipeline."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRIPOSG_REFERENCE_REVISION": ".conversion",
    "TRIPOSG_WEIGHTS_REVISION": ".conversion",
    "TripoSGImageTo3DPipeline": ".pipeline",
    "TripoSGTransformerModel": ".models",
    "TripoSGVAEDecoder": ".vae",
    "convert_triposg_checkpoint": ".conversion",
    "prepare_triposg_image": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
