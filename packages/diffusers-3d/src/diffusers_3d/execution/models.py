from __future__ import annotations

from typing import ClassVar

from diffusers import ConfigMixin, ModelMixin

from ..objects import Object3DKind
from .exceptions import Object3DSchemaError
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
