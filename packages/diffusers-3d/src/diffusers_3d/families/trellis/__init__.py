"""Microsoft TRELLIS components: sparse-structure stage, SLAT flow, Gaussian decoder, image and text pipelines."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRELLIS_REFERENCE_REVISION": ".conversion",
    "TrellisClipTextConditioner": ".text_conditioner",
    "TrellisConditionerOutput": ".conditioner",
    "TrellisDinov2Conditioner": ".conditioner",
    "TrellisFlowEulerScheduler": ".scheduler",
    "TrellisFlowEulerSchedulerOutput": ".scheduler",
    "TrellisGaussianDecoderOutput": ".decoders",
    "TrellisImageTo3DPipeline": ".pipeline",
    "TrellisMeshDecoderOutput": ".decoders",
    "TrellisRadianceFieldDecoderOutput": ".decoders",
    "TrellisSLatFlowModel": ".models",
    "TrellisSLatFlowOutput": ".models",
    "TrellisSLatGaussianDecoder": ".decoders",
    "TrellisSLatMeshDecoder": ".decoders",
    "TrellisSLatRadianceFieldDecoder": ".decoders",
    "TrellisSparseStructureDecoder": ".decoders",
    "TrellisSparseStructureDecoderOutput": ".decoders",
    "TrellisSparseStructureFlowModel": ".models",
    "TrellisSparseStructureFlowOutput": ".models",
    "TrellisSparseTensor": ".sparse",
    "TrellisTextConditionerOutput": ".text_conditioner",
    "TrellisTextTo3DPipeline": ".pipeline",
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
