"""Reviewed portable and experimental capability-gated Microsoft TRELLIS components."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRELLIS_REFERENCE_REVISION": ".conversion",
    "TrellisConditionerOutput": ".conditioner",
    "TrellisDinov2Conditioner": ".conditioner",
    "TrellisFlowEulerScheduler": ".scheduler",
    "TrellisFlowEulerSchedulerOutput": ".scheduler",
    "TrellisGaussianDecoderOutput": ".decoders",
    "TrellisImageTo3DPipeline": ".pipeline",
    "TrellisSLatFlowModel": ".models",
    "TrellisSLatFlowOutput": ".models",
    "TrellisSLatGaussianDecoder": ".decoders",
    "TrellisSLatRadianceFieldDecoder": ".decoders",
    "TrellisSparseStructureDecoder": ".decoders",
    "TrellisSparseStructureDecoderOutput": ".decoders",
    "TrellisSparseStructureFlowModel": ".models",
    "TrellisSparseStructureFlowOutput": ".models",
    "TrellisSparseTensor": ".sparse",
    "convert_trellis_checkpoint": ".conversion",
    "trellis_grid_transform": ".sparse",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
