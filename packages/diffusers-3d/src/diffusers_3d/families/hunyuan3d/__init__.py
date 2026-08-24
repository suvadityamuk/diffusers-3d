"""Reviewed Hunyuan3D-2.1 image-to-shape integration."""

from importlib import import_module

_EXPORT_MODULES = {
    "HUNYUAN3D_REFERENCE_REVISION": ".conversion",
    "HUNYUAN3D_DENOISER_POLICY": ".training",
    "Hunyuan3DConditionerOutput": ".conditioner",
    "Hunyuan3DDinov2Conditioner": ".conditioner",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler": ".scheduler",
    "Hunyuan3DImageToShapePipeline": ".pipeline",
    "Hunyuan3DShapeBatch": ".training",
    "Hunyuan3DShapeExample": ".training",
    "Hunyuan3DShapeDiTModel": ".models",
    "Hunyuan3DShapeDiTOutput": ".models",
    "Hunyuan3DShapeFieldOutput": ".vae",
    "Hunyuan3DShapeFlowMatchingRecipe": ".training",
    "Hunyuan3DShapeVAE": ".vae",
    "Hunyuan3DShapeVAEOutput": ".vae",
    "convert_hunyuan3d_checkpoint": ".conversion",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "HUNYUAN3D_REFERENCE_REVISION",
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
    "convert_hunyuan3d_checkpoint",
]
