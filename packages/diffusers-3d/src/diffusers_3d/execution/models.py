from __future__ import annotations

import os
from typing import Any

from diffusers import ConfigMixin, ModelMixin


class Object3DModel(ModelMixin, ConfigMixin):
    """Base class for diffusers-3d model components."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str] | None, **kwargs: Any) -> Any:
        """``ModelMixin.from_pretrained`` that raises, instead of warning, when the checkpoint lacks weights."""

        output_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path, output_loading_info=True, **kwargs
        )
        missing = sorted(loading_info["missing_keys"])
        if missing:
            raise ValueError(
                f"{pretrained_model_name_or_path} has no weights for {cls.__name__} parameters {missing[:8]} "
                f"({len(missing)} in total)"
            )
        return (model, loading_info) if output_loading_info else model


__all__ = ["Object3DModel"]
