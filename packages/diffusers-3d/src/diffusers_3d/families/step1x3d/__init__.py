"""StepFun Step1X-3D geometry components: flow transformer, shape decoder, scheduler, and pipeline."""

from importlib import import_module

_EXPORT_MODULES = {
    "STEP1X3D_CHECKPOINT_REVISION": ".conversion",
    "STEP1X3D_REFERENCE_REVISION": ".conversion",
    "Step1X3DFlowMatchEulerScheduler": ".scheduler",
    "Step1X3DImageTo3DPipeline": ".pipeline",
    "Step1X3DShapeDecoder": ".vae",
    "Step1X3DShapeDecoderOutput": ".vae",
    "Step1X3DTransformer1DModel": ".models",
    "Step1X3DTransformerOutput": ".models",
    "convert_step1x3d_checkpoint": ".conversion",
    "preprocess_step1x3d_image": ".pipeline",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORT_MODULES)
