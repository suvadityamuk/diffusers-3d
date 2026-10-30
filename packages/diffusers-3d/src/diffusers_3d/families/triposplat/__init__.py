"""VAST-AI TripoSplat image-to-Gaussians: rectified-flow transformer, octree Gaussian decoder, and pipeline."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRIPOSPLAT_REFERENCE_REVISION": ".conversion",
    "TRIPOSPLAT_WEIGHTS_REVISION": ".conversion",
    "TripoSplatGaussianDecoder": ".decoder",
    "TripoSplatGaussianDecoderOutput": ".decoder",
    "TripoSplatImageTo3DPipeline": ".pipeline",
    "TripoSplatImageVAE": ".models",
    "TripoSplatTransformerModel": ".models",
    "TripoSplatTransformerOutput": ".models",
    "convert_triposplat_checkpoint": ".conversion",
    "prepare_triposplat_image": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
