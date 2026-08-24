from __future__ import annotations

import os
from typing import Any, ClassVar

from .exceptions import Object3DLoadingError, Object3DTaskError
from .metadata import Object3DModelIndex
from .pipelines import ModularObject3DPipeline, Object3DPipeline
from .registry import _PIPELINE_REGISTRY, Object3DPipelineRegistry

IMAGE_TO_3D_TASK = "image-to-3d"
TEXT_TO_3D_TASK = "text-to-3d"


class AutoPipelineFor3D:
    """Config-first loader for exact, locally registered object-3D pipelines."""

    _registry: ClassVar[Object3DPipelineRegistry] = _PIPELINE_REGISTRY
    _task_id: ClassVar[str | None] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        task: str | None = None,
        revision: str | None = None,
        subfolder: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Object3DPipeline | ModularObject3DPipeline:
        """Load sidecar metadata, resolve an exact reviewed class, then delegate."""

        if trust_remote_code:
            raise Object3DLoadingError(
                "Object-3D auto pipelines do not support trust_remote_code=True; "
                "the concrete pipeline class must be installed and reviewed"
            )

        metadata = Object3DModelIndex.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            subfolder=subfolder,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

        constrained_task = cls._task_id
        if constrained_task is not None:
            if task is not None and task != constrained_task:
                raise Object3DTaskError(f"{cls.__name__} only supports task {constrained_task!r}, not {task!r}")
            selected_task = constrained_task
        elif task is not None:
            selected_task = task
        elif len(metadata.task_ids) == 1:
            selected_task = metadata.task_ids[0]
        else:
            choices = ", ".join(metadata.task_ids)
            raise Object3DTaskError(f"Pipeline metadata declares multiple tasks ({choices}); pass task=... explicitly")

        if selected_task not in metadata.task_ids:
            choices = ", ".join(metadata.task_ids)
            raise Object3DTaskError(
                f"Task {selected_task!r} is not declared by this pipeline. Declared tasks: {choices}"
            )

        pipeline_class = cls._registry.resolve(metadata, selected_task)
        delegate_kwargs = dict(kwargs)
        delegate_kwargs["local_files_only"] = local_files_only
        delegate_kwargs["trust_remote_code"] = False
        if revision is not None:
            delegate_kwargs["revision"] = revision
        if subfolder is not None:
            delegate_kwargs["subfolder"] = subfolder
        if cache_dir is not None:
            delegate_kwargs["cache_dir"] = cache_dir
        if token is not None:
            delegate_kwargs["token"] = token

        try:
            return pipeline_class.from_pretrained(
                pretrained_model_name_or_path,
                **delegate_kwargs,
            )
        except Exception as error:
            raise Object3DLoadingError(
                f"Reviewed pipeline {metadata.pipeline_class!r} failed to load for task {selected_task!r}"
            ) from error


class AutoPipelineForImageTo3D(AutoPipelineFor3D):
    """Exact reviewed auto-loader constrained to image-to-3D pipelines."""

    _task_id = IMAGE_TO_3D_TASK


class AutoPipelineForTextTo3D(AutoPipelineFor3D):
    """Exact reviewed auto-loader constrained to text-to-3D pipelines."""

    _task_id = TEXT_TO_3D_TASK


__all__ = [
    "IMAGE_TO_3D_TASK",
    "TEXT_TO_3D_TASK",
    "AutoPipelineFor3D",
    "AutoPipelineForImageTo3D",
    "AutoPipelineForTextTo3D",
]
