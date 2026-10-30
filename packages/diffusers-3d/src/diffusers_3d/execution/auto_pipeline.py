from __future__ import annotations

import os
from typing import Any, ClassVar

from diffusers import DiffusionPipeline

from .. import families

_CONFIG_KWARGS = ("cache_dir", "force_download", "proxies", "token", "local_files_only", "revision", "subfolder")
PIPELINES = tuple(name for name in families.__all__ if name.endswith("Pipeline"))


class AutoPipelineFor3D:
    """Load any diffusers-3d pipeline from the ``_class_name`` in its ``model_index.json``.

    Keyword arguments go to the concrete pipeline's ``from_pretrained``. Pass a component as an object (for example
    ``image_encoder=...`` for a gated encoder) and its subfolder is neither downloaded nor required.
    """

    pipelines: ClassVar[tuple[str, ...]] = PIPELINES

    @classmethod
    def pipeline_class(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any) -> type:
        """Return the pipeline class a checkpoint names, without loading any weights."""

        config_kwargs = {key: kwargs[key] for key in _CONFIG_KWARGS if key in kwargs}
        name = DiffusionPipeline.load_config(pretrained_model_name_or_path, **config_kwargs)["_class_name"]
        if name not in cls.pipelines:
            raise ValueError(f"{cls.__name__} cannot load {name!r}; it supports {', '.join(cls.pipelines)}")
        return getattr(families, name)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any
    ) -> DiffusionPipeline:
        pipeline_class = cls.pipeline_class(pretrained_model_name_or_path, **kwargs)
        return pipeline_class.from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoPipelineForImageTo3D(AutoPipelineFor3D):
    """Load an image-to-3D pipeline."""

    pipelines = tuple(name for name in PIPELINES if "ImageTo3D" in name or name == "Sam3DObjectsPipeline")


class AutoPipelineForTextTo3D(AutoPipelineFor3D):
    """Load a text-to-3D pipeline."""

    pipelines = tuple(name for name in PIPELINES if "TextTo3D" in name)


class AutoPipelineForMeshTexturing(AutoPipelineFor3D):
    """Load a pipeline that paints a texture onto an existing mesh."""

    pipelines = ("Hunyuan3DPaintPipeline", "Step1X3DMeshTexturePipeline")


__all__ = [
    "AutoPipelineFor3D",
    "AutoPipelineForImageTo3D",
    "AutoPipelineForMeshTexturing",
    "AutoPipelineForTextTo3D",
]
