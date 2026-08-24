"""Reviewed production model families."""

from importlib import import_module


def __getattr__(name: str):
    module = import_module(".hunyuan3d", __name__)
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(name) from None
    globals()[name] = value
    return value


__all__ = [
    "HUNYUAN3D_DENOISER_POLICY",
    "Hunyuan3DConditionerOutput",
    "Hunyuan3DDinov2Conditioner",
    "Hunyuan3DFlowMatchEulerDiscreteScheduler",
    "Hunyuan3DImageToShapePipeline",
    "Hunyuan3DShapeBatch",
    "Hunyuan3DShapeDiTModel",
    "Hunyuan3DShapeDiTOutput",
    "Hunyuan3DShapeFieldOutput",
    "Hunyuan3DShapeFlowMatchingRecipe",
    "Hunyuan3DShapeVAE",
    "Hunyuan3DShapeVAEOutput",
]
