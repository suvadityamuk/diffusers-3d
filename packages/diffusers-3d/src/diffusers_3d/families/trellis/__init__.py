"""Reviewed portable and experimental capability-gated Microsoft TRELLIS components."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRELLIS_REFERENCE_REVISION": ".conversion",
    "TRELLIS_SLAT_FLOW_POLICY": ".training",
    "TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY": ".training",
    "TrellisConditionerOutput": ".conditioner",
    "TrellisDinov2Conditioner": ".conditioner",
    "TrellisFlowEulerScheduler": ".scheduler",
    "TrellisFlowEulerSchedulerOutput": ".scheduler",
    "TrellisGaussianDecoderOutput": ".decoders",
    "TrellisImageTo3DPipeline": ".pipeline",
    "TrellisSLatBatch": ".training",
    "TrellisSLatExample": ".training",
    "TrellisSLatFlowModel": ".models",
    "TrellisSLatFlowOutput": ".models",
    "TrellisSLatFlowRecipe": ".training",
    "TrellisSLatGaussianDecoder": ".decoders",
    "TrellisSLatMeshDecoder": ".decoders",
    "TrellisSLatRadianceFieldDecoder": ".decoders",
    "TrellisSparseStructureBatch": ".training",
    "TrellisSparseStructureDecoder": ".decoders",
    "TrellisSparseStructureDecoderOutput": ".decoders",
    "TrellisSparseStructureExample": ".training",
    "TrellisSparseStructureFlowModel": ".models",
    "TrellisSparseStructureFlowOutput": ".models",
    "TrellisSparseStructureFlowRecipe": ".training",
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
