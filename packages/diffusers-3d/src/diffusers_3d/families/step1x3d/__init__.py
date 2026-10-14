"""StepFun Step1X-3D: image-to-geometry and mesh-texturing components and pipelines."""

from importlib import import_module

_EXPORT_MODULES = {
    "STEP1X3D_CHECKPOINT_REVISION": ".conversion",
    "STEP1X3D_REFERENCE_REVISION": ".conversion",
    "Step1X3DFlowMatchEulerScheduler": ".scheduler",
    "Step1X3DImageTo3DPipeline": ".pipeline",
    "Step1X3DMeshTexturePipeline": ".texture_pipeline",
    "Step1X3DMultiViewAdapter": ".texture_attention",
    "Step1X3DMultiViewAttnProcessor": ".texture_attention",
    "Step1X3DShapeDecoder": ".vae",
    "Step1X3DShapeDecoderOutput": ".vae",
    "Step1X3DTransformer1DModel": ".models",
    "Step1X3DTransformerOutput": ".models",
    "convert_step1x3d_checkpoint": ".conversion",
    "convert_step1x3d_texture_checkpoint": ".texture_conversion",
    "preprocess_step1x3d_image": ".pipeline",
    "preprocess_step1x3d_reference_image": ".texture_pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
