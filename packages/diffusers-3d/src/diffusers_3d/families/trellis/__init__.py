"""Microsoft TRELLIS components: sparse-structure stage, SLAT flow, Gaussian decoder, image and text pipelines."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRELLIS_REFERENCE_REVISION": ".conversion",
    "TRELLIS_SLAT_FLOW_POLICY": ".training",
    "TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY": ".training",
    "TrellisClipTextConditioner": ".text_conditioner",
    "TrellisConditionerOutput": ".conditioner",
    "TrellisDinov2Conditioner": ".conditioner",
    "TrellisFlowEulerScheduler": ".scheduler",
    "TrellisFlowEulerSchedulerOutput": ".scheduler",
    "TrellisGaussianDecoderOutput": ".decoders",
    "TrellisImageTo3DPipeline": ".pipeline",
    "TrellisMeshDecoderOutput": ".decoders",
    "TrellisRadianceFieldDecoderOutput": ".decoders",
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
