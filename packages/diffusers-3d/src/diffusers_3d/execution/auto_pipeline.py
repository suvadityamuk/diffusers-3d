from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import huggingface_hub
from diffusers import DiffusionPipeline

from .exceptions import Object3DLoadingError, Object3DTaskError
from .metadata import (
    OBJECT3D_MODEL_INDEX_NAME,
    Object3DComponentSpec,
    Object3DModelIndex,
    _normalize_subfolder,
    fully_qualified_class_name,
)
from .pipelines import ModularObject3DPipeline, Object3DPipeline
from .registry import _PIPELINE_REGISTRY, Object3DPipelineRegistry

IMAGE_TO_3D_TASK = "image-to-3d"
TEXT_TO_3D_TASK = "text-to-3d"


def _is_explicit_local_reference(reference: str | os.PathLike[str]) -> bool:
    path = Path(reference)
    return isinstance(reference, os.PathLike) or path.exists() or path.is_absolute() or str(reference).startswith(".")


def _pipeline_directory(
    reference: str | os.PathLike[str],
    subfolder: str | None,
) -> Path:
    path = Path(reference)
    if path.is_file():
        raise Object3DLoadingError("AutoPipelineFor3D requires a pipeline directory, not a metadata file")
    if subfolder is not None:
        path = path / subfolder
    if not path.is_dir():
        raise Object3DLoadingError(f"Object-3D pipeline directory {path} does not exist")
    return path


def _snapshot_allow_patterns(metadata: Object3DModelIndex, subfolder: str | None) -> list[str]:
    prefix = "" if subfolder is None else f"{subfolder}/"
    patterns = [
        f"{prefix}{DiffusionPipeline.config_name}",
        f"{prefix}{OBJECT3D_MODEL_INDEX_NAME}",
    ]
    for component in metadata.components:
        if component.loading_eligible:
            folder = f"{prefix}{component.subfolder}"
            patterns.extend(
                (
                    f"{folder}/*.json",
                    f"{folder}/*.safetensors",
                    f"{folder}/*.bin",
                    f"{folder}/*.flashpack",
                )
            )
    return patterns


def _expected_component_type(component: Object3DComponentSpec) -> type[Any]:
    module_name, _, class_name = component.expected_class.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        expected_type = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise Object3DLoadingError(
            f"Reviewed component class {component.expected_class!r} is not available from the installed package"
        ) from error
    if not isinstance(expected_type, type) or fully_qualified_class_name(expected_type) != component.expected_class:
        raise Object3DLoadingError(
            f"Installed component binding for {component.expected_class!r} does not resolve to that exact class"
        )
    return expected_type


def _validate_loaded_pipeline(
    pipeline: Object3DPipeline | ModularObject3DPipeline,
    pipeline_class: type[Object3DPipeline] | type[ModularObject3DPipeline],
    metadata: Object3DModelIndex,
) -> None:
    if type(pipeline) is not pipeline_class:
        raise Object3DLoadingError(
            f"Concrete loader returned {fully_qualified_class_name(type(pipeline))!r}; "
            f"expected exact reviewed pipeline {metadata.pipeline_class!r}"
        )
    for component in metadata.components:
        value = getattr(pipeline, component.name, None)
        if value is None:
            if component.optional:
                continue
            raise Object3DLoadingError(f"Loaded pipeline is missing required component {component.name!r}")
        if not component.loading_eligible:
            raise Object3DLoadingError(
                f"Loaded pipeline contains unexpected experimental component {component.name!r}"
            )
        expected_type = _expected_component_type(component)
        actual_class = fully_qualified_class_name(type(value))
        if type(value) is not expected_type or actual_class != component.expected_class:
            raise Object3DLoadingError(
                f"Loaded component {component.name!r} has exact type {actual_class!r}; "
                f"expected {component.expected_class!r}"
            )

    try:
        loaded_components = pipeline.components
    except Exception as error:
        raise Object3DLoadingError("Loaded pipeline does not expose a valid Diffusers component mapping") from error
    if not isinstance(loaded_components, Mapping):
        raise Object3DLoadingError("Loaded pipeline components must be a mapping")

    declared_names = {component.name for component in metadata.components}
    unexpected = sorted(
        name for name, value in loaded_components.items() if name not in declared_names and value is not None
    )
    if unexpected:
        raise Object3DLoadingError(f"Loaded pipeline contains undeclared non-None components: {unexpected}")


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
        """Load only exact reviewed installed classes from a validated local snapshot."""

        if trust_remote_code:
            raise Object3DLoadingError(
                "Object-3D auto pipelines do not support trust_remote_code=True; "
                "the concrete pipeline class must be installed and reviewed"
            )

        normalized_subfolder = _normalize_subfolder(subfolder)
        metadata = Object3DModelIndex.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            subfolder=normalized_subfolder,
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
        forbidden_delegate_options = {"custom_pipeline", "custom_revision", "dduf_file", "load_connected_pipeline"}
        supplied_forbidden_options = sorted(forbidden_delegate_options.intersection(kwargs))
        if supplied_forbidden_options:
            raise Object3DLoadingError(
                f"Object-3D auto pipelines do not accept code-loading options: {supplied_forbidden_options}"
            )

        if _is_explicit_local_reference(pretrained_model_name_or_path):
            local_pipeline_directory = _pipeline_directory(
                pretrained_model_name_or_path,
                normalized_subfolder,
            )
        else:
            try:
                snapshot_path = huggingface_hub.snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=_snapshot_allow_patterns(metadata, normalized_subfolder),
                )
            except Exception as error:
                raise Object3DLoadingError(
                    f"Could not download reviewed object-3D snapshot for {str(pretrained_model_name_or_path)!r}"
                ) from error
            local_pipeline_directory = _pipeline_directory(snapshot_path, normalized_subfolder)
            snapshot_metadata = Object3DModelIndex.from_pretrained(local_pipeline_directory)
            if snapshot_metadata != metadata:
                raise Object3DLoadingError(
                    "Object-3D sidecar changed between initial validation and immutable snapshot download"
                )

        metadata.validate_diffusers_model_index(
            local_pipeline_directory / DiffusionPipeline.config_name,
            pipeline_class_name=pipeline_class.__name__,
        )
        delegate_kwargs = dict(kwargs)
        delegate_kwargs["local_files_only"] = True
        delegate_kwargs["trust_remote_code"] = False

        try:
            pipeline = pipeline_class.from_pretrained(
                local_pipeline_directory,
                **delegate_kwargs,
            )
        except Exception as error:
            raise Object3DLoadingError(
                f"Reviewed pipeline {metadata.pipeline_class!r} failed to load for task {selected_task!r}"
            ) from error
        _validate_loaded_pipeline(pipeline, pipeline_class, metadata)
        return pipeline


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
