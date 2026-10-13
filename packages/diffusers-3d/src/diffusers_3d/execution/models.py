from __future__ import annotations

import os
from typing import Any, ClassVar

from diffusers import ConfigMixin, ModelMixin

from ..objects import Object3DKind
from .exceptions import Object3DLoadingError, Object3DSchemaError
from .metadata import (
    OBJECT3D_API_VERSION,
    OBJECT3D_SCHEMA_VERSION,
    ContributionStatus,
    Object3DModelMetadata,
    ReviewStatus,
    fully_qualified_class_name,
)


class Object3DModel(ModelMixin, ConfigMixin):
    """Nominal base for model components with an object-native 3D contract.

    Subclassing this marker does not register or review a model. Concrete
    integrations must also appear in an exact ``Object3DModelRegistry``.
    """

    api_version: ClassVar[str] = OBJECT3D_API_VERSION
    schema_version: ClassVar[int] = OBJECT3D_SCHEMA_VERSION
    family_id: ClassVar[str | None] = None
    component_role: ClassVar[str | None] = None
    supported_object_kinds: ClassVar[tuple[Object3DKind, ...]] = ()
    required_backends: ClassVar[tuple[str, ...]] = ()
    contribution_status: ClassVar[ContributionStatus] = ContributionStatus.EXPERIMENTAL_HUB
    review_status: ClassVar[ReviewStatus] = ReviewStatus.UNREVIEWED

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str] | None, **kwargs: Any) -> Any:
        """``ModelMixin.from_pretrained`` that raises, instead of warning, when the checkpoint lacks weights."""

        output_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path, output_loading_info=True, **kwargs
        )
        if loading_info["missing_keys"]:
            raise Object3DLoadingError(
                f"{pretrained_model_name_or_path} has no weights for {cls.__name__} parameters "
                f"{sorted(loading_info['missing_keys'])[:8]} ({len(loading_info['missing_keys'])} in total)"
            )
        return (model, loading_info) if output_loading_info else model

    @classmethod
    def object3d_metadata(cls) -> Object3DModelMetadata:
        """Build and validate the class's immutable object-3D metadata."""

        if cls.api_version != OBJECT3D_API_VERSION:
            raise Object3DSchemaError(
                f"{fully_qualified_class_name(cls)} declares API version {cls.api_version!r}; "
                f"expected {OBJECT3D_API_VERSION!r}"
            )
        return Object3DModelMetadata(
            schema_version=cls.schema_version,
            family_id=cls.family_id,  # type: ignore[arg-type]
            component_role=cls.component_role,  # type: ignore[arg-type]
            model_class=fully_qualified_class_name(cls),
            supported_object_kinds=cls.supported_object_kinds,
            required_backends=cls.required_backends,
            contribution_status=cls.contribution_status,
            review_status=cls.review_status,
        )


__all__ = ["Object3DModel"]
