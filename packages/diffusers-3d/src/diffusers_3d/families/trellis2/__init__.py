"""Reviewed portable and experimental capability-gated Microsoft TRELLIS.2 components."""

from importlib import import_module

_EXPORT_MODULES = {
    "TRELLIS2_REFERENCE_REVISION": ".conversion",
    "TRELLIS2_SHAPE_SLAT_FLOW_POLICY": ".training",
    "TRELLIS2_SPARSE_STRUCTURE_FLOW_POLICY": ".training",
    "TRELLIS2_TEXTURE_SLAT_FLOW_POLICY": ".training",
    "Trellis2ConditionerOutput": ".conditioner",
    "Trellis2Dinov3Conditioner": ".conditioner",
    "Trellis2FlowEulerScheduler": ".scheduler",
    "Trellis2FlowEulerSchedulerOutput": ".scheduler",
    "Trellis2ImageTo3DPipeline": ".pipeline",
    "Trellis2PBRDecoderOutput": ".decoders",
    "Trellis2PBRSparseDecoder": ".decoders",
    "Trellis2SLatBatch": ".training",
    "Trellis2SLatExample": ".training",
    "Trellis2SLatFlowModel": ".models",
    "Trellis2SLatFlowOutput": ".models",
    "Trellis2ShapeDecoderOutput": ".decoders",
    "Trellis2ShapeDualGridDecoder": ".decoders",
    "Trellis2ShapeSLatFlowRecipe": ".training",
    "Trellis2SparseStructureBatch": ".training",
    "Trellis2SparseStructureDecoder": ".decoders",
    "Trellis2SparseStructureExample": ".training",
    "Trellis2SparseStructureFlowModel": ".models",
    "Trellis2SparseStructureFlowOutput": ".models",
    "Trellis2SparseStructureFlowRecipe": ".training",
    "Trellis2TextureSLatBatch": ".training",
    "Trellis2TextureSLatExample": ".training",
    "Trellis2TextureSLatFlowRecipe": ".training",
    "convert_trellis2_checkpoint": ".conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
