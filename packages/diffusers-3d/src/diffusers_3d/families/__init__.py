"""Reviewed production model families."""

from importlib import import_module

_TRELLIS_EXPORTS = {
    "TRELLIS_REFERENCE_REVISION",
    "TrellisFlowEulerScheduler",
    "TrellisFlowEulerSchedulerOutput",
    "TrellisGaussianDecoderOutput",
    "TrellisImageTo3DPipeline",
    "TrellisMeshDecoderOutput",
    "TrellisRadianceFieldDecoderOutput",
    "TrellisSLatFlowModel",
    "TrellisSLatFlowOutput",
    "TrellisSLatGaussianDecoder",
    "TrellisSLatMeshDecoder",
    "TrellisSLatRadianceFieldDecoder",
    "TrellisSparseStructureDecoder",
    "TrellisSparseStructureDecoderOutput",
    "TrellisSparseStructureFlowModel",
    "TrellisSparseStructureFlowOutput",
    "TrellisSparseTensor",
    "TrellisTextTo3DPipeline",
    "convert_trellis_checkpoint",
    "trellis_grid_transform",
}

_TRELLIS2_EXPORTS = {
    "TRELLIS2_REFERENCE_REVISION",
    "Trellis2FlowEulerScheduler",
    "Trellis2FlowEulerSchedulerOutput",
    "Trellis2ImageTo3DPipeline",
    "Trellis2PBRDecoderOutput",
    "Trellis2PBRSparseDecoder",
    "Trellis2SLatFlowModel",
    "Trellis2SLatFlowOutput",
    "Trellis2ShapeDecoderOutput",
    "Trellis2ShapeDualGridDecoder",
    "Trellis2SparseStructureDecoder",
    "Trellis2SparseStructureFlowModel",
    "Trellis2SparseStructureFlowOutput",
    "convert_trellis2_checkpoint",
}

_TRIPOSG_EXPORTS = {
    "TRIPOSG_REFERENCE_REVISION",
    "TRIPOSG_WEIGHTS_REVISION",
    "TripoSGImageTo3DPipeline",
    "TripoSGTransformerModel",
    "TripoSGVAEDecoder",
    "convert_triposg_checkpoint",
    "prepare_triposg_image",
}

_SAM3D_OBJECTS_EXPORTS = {
    "SAM3D_REFERENCE_REVISION",
    "Sam3DConditionEmbedder",
    "Sam3DConditionerOutput",
    "Sam3DObjectsPipeline",
    "Sam3DSparseStructureFlowModel",
    "Sam3DSparseStructureFlowOutput",
    "convert_sam3d_checkpoint",
    "preprocess_sam3d_condition",
}

_STEP1X3D_EXPORTS = {
    "STEP1X3D_CHECKPOINT_REVISION",
    "STEP1X3D_REFERENCE_REVISION",
    "Step1X3DFlowMatchEulerScheduler",
    "Step1X3DImageTo3DPipeline",
    "Step1X3DMeshTexturePipeline",
    "Step1X3DMultiViewAdapter",
    "Step1X3DMultiViewAttnProcessor",
    "Step1X3DShapeDecoder",
    "Step1X3DShapeDecoderOutput",
    "Step1X3DTransformer1DModel",
    "Step1X3DTransformerOutput",
    "convert_step1x3d_checkpoint",
    "convert_step1x3d_texture_checkpoint",
    "preprocess_step1x3d_image",
    "preprocess_step1x3d_reference_image",
}

_PIXAL3D_EXPORTS = {
    "PIXAL3D_REFERENCE_REVISION",
    "Pixal3DImageTo3DPipeline",
    "Pixal3DNAFUpsampler",
    "Pixal3DNAFUpsamplerOutput",
    "Pixal3DSLatFlowModel",
    "Pixal3DSparseStructureFlowModel",
    "convert_pixal3d_checkpoint",
}

_HUNYUAN3D_EXPORTS = {
    "HUNYUAN3D_REFERENCE_REVISION",
    "Hunyuan3DDiTModel",
    "Hunyuan3DDiTOutput",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler",
    "Hunyuan3DImageTo3DPipeline",
    "Hunyuan3DPaintPipeline",
    "Hunyuan3DPaintRRDBNet",
    "Hunyuan3DPaintUNet",
    "Hunyuan3DShapeVAE",
    "Hunyuan3DShapeVAEOutput",
    "convert_hunyuan3d_checkpoint",
    "convert_hunyuan3d_paint_checkpoint",
}

_HUNYUAN3D_OMNI_EXPORTS = {
    "HUNYUAN3D_OMNI_REFERENCE_REVISION",
    "Hunyuan3DOmniControlEncoder",
    "Hunyuan3DOmniControlEncoderOutput",
    "Hunyuan3DOmniImageTo3DPipeline",
    "convert_hunyuan3d_omni_checkpoint",
}

_TRIPOSR_EXPORTS = {
    "TRIPOSR_REFERENCE_REVISION",
    "TRIPOSR_WEIGHTS_REVISION",
    "TripoSRImageTo3DPipeline",
    "TripoSRTransformerModel",
    "TripoSRTriplaneDecoder",
    "TripoSRTriplaneDecoderOutput",
    "convert_triposr_checkpoint",
    "prepare_triposr_image",
}

_TRIPOSPLAT_EXPORTS = {
    "TRIPOSPLAT_REFERENCE_REVISION",
    "TRIPOSPLAT_WEIGHTS_REVISION",
    "TripoSplatGaussianDecoder",
    "TripoSplatGaussianDecoderOutput",
    "TripoSplatImageTo3DPipeline",
    "TripoSplatImageVAE",
    "TripoSplatTransformerModel",
    "TripoSplatTransformerOutput",
    "convert_triposplat_checkpoint",
    "prepare_triposplat_image",
}

_IDOL_EXPORTS = {
    "IDOL_REFERENCE_REVISION",
    "IDOL_WEIGHTS_REVISION",
    "IdolImageTo3DPipeline",
    "convert_idol_checkpoint",
    "decode_idol_camera",
    "prepare_idol_image",
}

_MOGE_EXPORTS = {
    "MOGE2_REFERENCE_REVISION",
    "MOGE2_WEIGHTS_REPOSITORY",
    "MOGE2_WEIGHTS_REVISION",
    "MOGE_REFERENCE_REVISION",
    "MOGE_WEIGHTS_REPOSITORY",
    "MOGE_WEIGHTS_REVISION",
    "MoGe2Model",
    "MoGeModel",
    "convert_moge_checkpoint",
}

_MODULES = {
    ".trellis": _TRELLIS_EXPORTS,
    ".trellis2": _TRELLIS2_EXPORTS,
    ".triposg": _TRIPOSG_EXPORTS,
    ".sam3d_objects": _SAM3D_OBJECTS_EXPORTS,
    ".step1x3d": _STEP1X3D_EXPORTS,
    ".pixal3d": _PIXAL3D_EXPORTS,
    ".hunyuan3d": _HUNYUAN3D_EXPORTS,
    ".moge": _MOGE_EXPORTS,
    ".idol": _IDOL_EXPORTS,
    ".triposplat": _TRIPOSPLAT_EXPORTS,
    ".triposr": _TRIPOSR_EXPORTS,
    ".hunyuan3d_omni": _HUNYUAN3D_OMNI_EXPORTS,
}
_EXPORT_MODULES = {name: module for module, names in _MODULES.items() for name in names}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
