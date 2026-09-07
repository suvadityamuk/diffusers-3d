"""Reviewed production model families."""

from importlib import import_module

_TRELLIS_EXPORTS = {
    "TRELLIS_REFERENCE_REVISION",
    "TrellisConditionerOutput",
    "TrellisDinov2Conditioner",
    "TrellisFlowEulerScheduler",
    "TrellisFlowEulerSchedulerOutput",
    "TrellisGaussianDecoderOutput",
    "TrellisImageTo3DPipeline",
    "TrellisSLatFlowModel",
    "TrellisSLatFlowOutput",
    "TrellisSLatGaussianDecoder",
    "TrellisSLatRadianceFieldDecoder",
    "TrellisSparseStructureDecoder",
    "TrellisSparseStructureDecoderOutput",
    "TrellisSparseStructureFlowModel",
    "TrellisSparseStructureFlowOutput",
    "TrellisSparseTensor",
    "convert_trellis_checkpoint",
    "trellis_grid_transform",
}


def __getattr__(name: str):
    if name in _TRELLIS_EXPORTS:
        module_name = ".trellis"
    else:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_TRELLIS_EXPORTS)
