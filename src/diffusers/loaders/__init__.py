from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_torch_available


_import_structure = {}

if is_torch_available():
    _import_structure["lora_base"] = ["LoraBaseMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]

_import_structure["peft"] = ["PeftAdapterMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .lora_base import LoraBaseMixin
        from .utils import AttnProcsLayers

    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
