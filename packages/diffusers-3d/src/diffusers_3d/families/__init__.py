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
    "Step1X3DShapeDecoder",
    "Step1X3DShapeDecoderOutput",
    "Step1X3DTransformer1DModel",
    "Step1X3DTransformerOutput",
    "convert_step1x3d_checkpoint",
    "preprocess_step1x3d_image",
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
    "Hunyuan3DShapeVAE",
    "Hunyuan3DShapeVAEOutput",
    "convert_hunyuan3d_checkpoint",
}


def __getattr__(name: str):
    if name in _TRELLIS_EXPORTS:
        module_name = ".trellis"
    elif name in _TRELLIS2_EXPORTS:
        module_name = ".trellis2"
    elif name in _TRIPOSG_EXPORTS:
        module_name = ".triposg"
    elif name in _SAM3D_OBJECTS_EXPORTS:
        module_name = ".sam3d_objects"
    elif name in _STEP1X3D_EXPORTS:
        module_name = ".step1x3d"
    elif name in _PIXAL3D_EXPORTS:
        module_name = ".pixal3d"
    elif name in _HUNYUAN3D_EXPORTS:
        module_name = ".hunyuan3d"
    else:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(
    _TRELLIS_EXPORTS
    | _TRELLIS2_EXPORTS
    | _TRIPOSG_EXPORTS
    | _SAM3D_OBJECTS_EXPORTS
    | _STEP1X3D_EXPORTS
    | _PIXAL3D_EXPORTS
    | _HUNYUAN3D_EXPORTS
)
