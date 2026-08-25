"""Reviewed production model families."""

from importlib import import_module

_HUNYUAN3D_EXPORTS = {
    "HUNYUAN3D_DENOISER_POLICY",
    "Hunyuan3DConditionerOutput",
    "Hunyuan3DDinov2Conditioner",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler",
    "Hunyuan3DImageToShapePipeline",
    "Hunyuan3DShapeBatch",
    "Hunyuan3DShapeExample",
    "Hunyuan3DShapeDiTModel",
    "Hunyuan3DShapeDiTOutput",
    "Hunyuan3DShapeFieldOutput",
    "Hunyuan3DShapeFlowMatchingRecipe",
    "Hunyuan3DShapeVAE",
    "Hunyuan3DShapeVAEOutput",
}

_TRELLIS_EXPORTS = {
    "TRELLIS_REFERENCE_REVISION",
    "TRELLIS_SLAT_FLOW_POLICY",
    "TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY",
    "TrellisConditionerOutput",
    "TrellisDinov2Conditioner",
    "TrellisFlowEulerScheduler",
    "TrellisFlowEulerSchedulerOutput",
    "TrellisGaussianDecoderOutput",
    "TrellisImageTo3DPipeline",
    "TrellisSLatBatch",
    "TrellisSLatExample",
    "TrellisSLatFlowModel",
    "TrellisSLatFlowOutput",
    "TrellisSLatFlowRecipe",
    "TrellisSLatGaussianDecoder",
    "TrellisSLatRadianceFieldDecoder",
    "TrellisSparseStructureBatch",
    "TrellisSparseStructureDecoder",
    "TrellisSparseStructureDecoderOutput",
    "TrellisSparseStructureExample",
    "TrellisSparseStructureFlowModel",
    "TrellisSparseStructureFlowOutput",
    "TrellisSparseStructureFlowRecipe",
    "TrellisSparseTensor",
    "convert_trellis_checkpoint",
    "trellis_grid_transform",
}

_TRELLIS2_EXPORTS = {
    "TRELLIS2_REFERENCE_REVISION",
    "TRELLIS2_SHAPE_SLAT_FLOW_POLICY",
    "TRELLIS2_SPARSE_STRUCTURE_FLOW_POLICY",
    "TRELLIS2_TEXTURE_SLAT_FLOW_POLICY",
    "Trellis2ConditionerOutput",
    "Trellis2Dinov3Conditioner",
    "Trellis2FlowEulerScheduler",
    "Trellis2FlowEulerSchedulerOutput",
    "Trellis2ImageTo3DPipeline",
    "Trellis2PBRDecoderOutput",
    "Trellis2PBRSparseDecoder",
    "Trellis2SLatBatch",
    "Trellis2SLatExample",
    "Trellis2SLatFlowModel",
    "Trellis2SLatFlowOutput",
    "Trellis2ShapeDecoderOutput",
    "Trellis2ShapeDualGridDecoder",
    "Trellis2ShapeSLatFlowRecipe",
    "Trellis2SparseStructureBatch",
    "Trellis2SparseStructureDecoder",
    "Trellis2SparseStructureExample",
    "Trellis2SparseStructureFlowModel",
    "Trellis2SparseStructureFlowOutput",
    "Trellis2SparseStructureFlowRecipe",
    "Trellis2TextureSLatBatch",
    "Trellis2TextureSLatExample",
    "Trellis2TextureSLatFlowRecipe",
    "convert_trellis2_checkpoint",
}


def __getattr__(name: str):
    if name in _HUNYUAN3D_EXPORTS:
        module_name = ".hunyuan3d"
    elif name in _TRELLIS_EXPORTS:
        module_name = ".trellis"
    elif name in _TRELLIS2_EXPORTS:
        module_name = ".trellis2"
    else:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_HUNYUAN3D_EXPORTS | _TRELLIS_EXPORTS | _TRELLIS2_EXPORTS)
