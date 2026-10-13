from __future__ import annotations

import os
from typing import Any, ClassVar

import torch
from diffusers.loaders import LoraBaseMixin, PeftAdapterMixin
from diffusers.loaders.lora_base import _fetch_state_dict
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.constants import MIN_PEFT_VERSION


def require_peft() -> None:
    """Raise an actionable error when the optional PEFT backend is not installed."""

    if not USE_PEFT_BACKEND:
        raise ImportError(
            f'Loading LoRA weights requires `peft>={MIN_PEFT_VERSION}`. Install it with `pip install "peft>={MIN_PEFT_VERSION}"`.'
        )


class Object3DPeftAdapterMixin(PeftAdapterMixin):
    """Diffusers `PeftAdapterMixin` that fails early with an install hint when `peft` is missing."""

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict: str | os.PathLike[str] | dict[str, torch.Tensor],
        prefix: str | None = "transformer",
        hotswap: bool = False,
        **kwargs: Any,
    ) -> None:
        require_peft()
        super().load_lora_adapter(pretrained_model_name_or_path_or_dict, prefix=prefix, hotswap=hotswap, **kwargs)


class Object3DLoraLoaderMixin(LoraBaseMixin):
    """Load Diffusers-format LoRA weights into a pipeline's flow models.

    A Diffusers-format checkpoint is a safetensors file whose keys start with the name of the pipeline component they
    adapt, for example ``sparse_structure_flow_model.blocks.0.self_attn.to_qkv.lora_A.weight``. Only components listed
    in ``_lora_loadable_modules`` accept weights.
    """

    _lora_loadable_modules: ClassVar[list[str]] = []

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: str | os.PathLike[str] | dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
        """Fetch a LoRA state dict and its adapter metadata from a Hub repository, directory, file, or dict."""

        require_peft()
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=kwargs.pop("weight_name", None),
            use_safetensors=True,
            local_files_only=kwargs.pop("local_files_only", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            subfolder=kwargs.pop("subfolder", None),
            user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
            allow_pickle=False,
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(kwargs))}")
        return state_dict, metadata

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | os.PathLike[str] | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a Diffusers-format LoRA into every flow model it adapts, under one adapter name."""

        require_peft()
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        if not state_dict or not all("lora" in key for key in state_dict):
            raise ValueError("Invalid LoRA checkpoint: every key must be a LoRA weight.")

        loadable = self._lora_loadable_modules
        prefixes = {key.split(".", 1)[0] for key in state_dict}
        unknown = prefixes.difference(loadable)
        if unknown:
            raise ValueError(
                f"Only Diffusers-format LoRA checkpoints are supported: keys must start with one of {loadable}, "
                f"but found {sorted(unknown)}."
            )
        components = {name: getattr(self, name, None) for name in loadable if name in prefixes}
        missing = sorted(name for name, component in components.items() if component is None)
        if missing:
            raise ValueError(f"The LoRA adapts {missing}, which this pipeline was loaded without.")

        if adapter_name is None:
            existing = {name for component in components.values() for name in getattr(component, "peft_config", {})}
            adapter_name = f"default_{len(existing)}"
        for name, component in components.items():
            component.load_lora_adapter(
                state_dict,
                prefix=name,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )

    def fuse_lora(self, components: list[str] | None = None, **kwargs: Any) -> None:
        """Fuse loaded LoRA weights into the flow models; ``components`` defaults to every loadable flow model."""

        super().fuse_lora(components=components or list(self._lora_loadable_modules), **kwargs)

    def unfuse_lora(self, components: list[str] | None = None, **kwargs: Any) -> None:
        """Reverse :meth:`fuse_lora`; ``components`` defaults to every loadable flow model."""

        super().unfuse_lora(components=components or list(self._lora_loadable_modules), **kwargs)


__all__ = ["Object3DLoraLoaderMixin", "Object3DPeftAdapterMixin", "require_peft"]
