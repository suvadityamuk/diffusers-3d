from __future__ import annotations

import os
from typing import Any, ClassVar

from diffusers import DiffusionPipeline, ModularPipeline

from ..objects import Object3DKind
from .exceptions import Object3DSchemaError
from .metadata import (
    OBJECT3D_API_VERSION,
    OBJECT3D_SCHEMA_VERSION,
    ContributionStatus,
    Object3DModelIndex,
    ReviewStatus,
    fully_qualified_class_name,
)


class _Object3DPipelineMetadataMixin:
    api_version: ClassVar[str] = OBJECT3D_API_VERSION
    schema_version: ClassVar[int] = OBJECT3D_SCHEMA_VERSION
    family_id: ClassVar[str | None] = None
    task_ids: ClassVar[tuple[str, ...]] = ()
    output_object_types: ClassVar[tuple[type[object], ...]] = ()
    output_representations: ClassVar[tuple[str, ...]] = ()
    object_kinds: ClassVar[tuple[Object3DKind, ...]] = ()
    required_backends: ClassVar[tuple[str, ...]] = ()
    contribution_status: ClassVar[ContributionStatus] = ContributionStatus.EXPERIMENTAL_HUB
    review_status: ClassVar[ReviewStatus] = ReviewStatus.UNREVIEWED

    @classmethod
    def object3d_model_index(cls) -> Object3DModelIndex:
        """Build and validate the sidecar metadata declared by this exact class."""

        if cls.api_version != OBJECT3D_API_VERSION:
            raise Object3DSchemaError(
                f"{fully_qualified_class_name(cls)} declares API version {cls.api_version!r}; "
                f"expected {OBJECT3D_API_VERSION!r}"
            )
        return Object3DModelIndex(
            schema_version=cls.schema_version,
            family_id=cls.family_id,  # type: ignore[arg-type]
            task_ids=cls.task_ids,
            pipeline_class=fully_qualified_class_name(cls),
            output_object_types=tuple(
                fully_qualified_class_name(output_type) for output_type in cls.output_object_types
            ),
            output_representations=cls.output_representations,
            object_kinds=cls.object_kinds,
            required_backends=cls.required_backends,
            contribution_status=cls.contribution_status,
            review_status=cls.review_status,
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        safe_serialization: bool = True,
        variant: str | None = None,
        max_shard_size: int | str | None = None,
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Save normal Diffusers artifacts plus deterministic object-3D metadata."""

        type(self).object3d_model_index().save_pretrained(save_directory)
        return super().save_pretrained(  # type: ignore[misc]
            save_directory,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )


class Object3DPipeline(_Object3DPipelineMetadataMixin, DiffusionPipeline):
    """Standard Diffusers pipeline base for reviewed object-native 3D families."""


class ModularObject3DPipeline(_Object3DPipelineMetadataMixin, ModularPipeline):
    """Modular Diffusers pipeline base for reviewed object-native 3D families."""


__all__ = ["ModularObject3DPipeline", "Object3DPipeline"]
