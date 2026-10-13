from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_torch_available


_import_structure = {}

if is_torch_available():
    _import_structure["lora_base"] = ["LoraBaseMixin"]
    _import_structure["single_file_model"] = ["FromOriginalModelMixin"]
    _import_structure["transformer_flux"] = ["FluxTransformer2DLoadersMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]

_import_structure["peft"] = ["PeftAdapterMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .lora_base import LoraBaseMixin
        from .single_file_model import FromOriginalModelMixin
        from .transformer_flux import FluxTransformer2DLoadersMixin
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
