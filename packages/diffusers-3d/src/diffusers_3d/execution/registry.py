from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields

from ..families.registrations import production_execution_registrations
from .exceptions import Object3DRegistrationError
from .metadata import Object3DModelIndex, Object3DModelMetadata, ReviewStatus
from .models import Object3DModel
from .pipelines import ModularObject3DPipeline, Object3DPipeline


def _require_same_metadata(expected: object, actual: object, message: str) -> None:
    if expected != actual:
        differences = ", ".join(
            field.name
            for field in fields(expected)  # type: ignore[arg-type]
            if getattr(expected, field.name) != getattr(actual, field.name)
        )
        raise Object3DRegistrationError(f"{message}; differing fields: {differences}")


@dataclass(frozen=True, slots=True)
class Object3DModelRegistration:
    """Exact model class and the metadata approved for it."""

    model_class: type[Object3DModel]
    metadata: Object3DModelMetadata


@dataclass(frozen=True, slots=True)
class Object3DPipelineRegistration:
    """Exact pipeline class and the sidecar metadata approved for it."""

    pipeline_class: type[Object3DPipeline] | type[ModularObject3DPipeline]
    metadata: Object3DModelIndex


class _Registry:
    """Exact registrations keyed by class name, with a secondary index; can be frozen read-only."""

    _kind = ""

    def __init__(self, registrations: Iterable[object] = ()) -> None:
        self._by_class: dict[str, object] = {}
        self._by_key: dict[tuple[str, str], object] = {}
        self._frozen = False
        for registration in registrations:
            self.register(registration)

    def __len__(self) -> int:
        return len(self._by_class)

    def __iter__(self) -> Iterator[object]:
        return iter(self.list())

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        self._frozen = True
        return self

    def list(self) -> tuple[object, ...]:
        return tuple(self._by_class[name] for name in sorted(self._by_class))

    def _check_writable(self) -> None:
        if self._frozen:
            raise Object3DRegistrationError(f"This object-3D {self._kind} registry is read-only")


class Object3DModelRegistry(_Registry):
    """Mutable factory for an exact model registry that can be frozen."""

    _kind = "model"

    def register(self, registration: Object3DModelRegistration) -> Object3DModelRegistration:
        self._check_writable()
        if not isinstance(registration, Object3DModelRegistration):
            raise TypeError("registration must be an Object3DModelRegistration")
        model_class = registration.model_class
        if (
            not isinstance(model_class, type)
            or model_class is Object3DModel
            or not issubclass(model_class, Object3DModel)
            or inspect.isabstract(model_class)
        ):
            raise Object3DRegistrationError("model_class must be a concrete nominal Object3DModel subclass")
        if not isinstance(registration.metadata, Object3DModelMetadata):
            raise TypeError("registration metadata must be Object3DModelMetadata")
        if registration.metadata.review_status is not ReviewStatus.REVIEWED:
            raise Object3DRegistrationError("Only explicitly reviewed model metadata can be registered")
        try:
            declared_metadata = model_class.object3d_metadata()
        except Exception as error:
            raise Object3DRegistrationError(
                f"Model class {model_class!r} has an invalid object-3D contract"
            ) from error
        _require_same_metadata(
            registration.metadata, declared_metadata, "Model registration does not match exact class metadata"
        )

        class_key = registration.metadata.model_class
        if class_key in self._by_class:
            raise Object3DRegistrationError(f"Model class {class_key!r} is already registered")
        family_role_key = (registration.metadata.family_id, registration.metadata.component_role)
        if family_role_key in self._by_key:
            existing = self._by_key[family_role_key]
            raise Object3DRegistrationError(
                f"Ambiguous model registration for family {family_role_key[0]!r} and component role "
                f"{family_role_key[1]!r}: {existing.metadata.model_class!r} and {class_key!r}"
            )
        self._by_class[class_key] = registration
        self._by_key[family_role_key] = registration
        return registration

    def resolve(self, metadata: Object3DModelMetadata) -> type[Object3DModel]:
        if not isinstance(metadata, Object3DModelMetadata):
            raise TypeError("metadata must be Object3DModelMetadata")
        registration = self._by_class.get(metadata.model_class)
        if registration is None:
            raise Object3DRegistrationError(f"Model class {metadata.model_class!r} has no exact reviewed registration")
        _require_same_metadata(
            registration.metadata, metadata, "Model metadata does not match its reviewed registration"
        )
        try:
            current_metadata = registration.model_class.object3d_metadata()
        except Exception as error:
            raise Object3DRegistrationError("Registered model class metadata is no longer valid") from error
        _require_same_metadata(
            registration.metadata, current_metadata, "Registered model class drifted from reviewed metadata"
        )
        return registration.model_class


class Object3DPipelineRegistry(_Registry):
    """Mutable factory for an exact pipeline registry that can be frozen."""

    _kind = "pipeline"

    def register(self, registration: Object3DPipelineRegistration) -> Object3DPipelineRegistration:
        self._check_writable()
        if not isinstance(registration, Object3DPipelineRegistration):
            raise TypeError("registration must be an Object3DPipelineRegistration")
        pipeline_class = registration.pipeline_class
        pipeline_bases = (Object3DPipeline, ModularObject3DPipeline)
        if (
            not isinstance(pipeline_class, type)
            or pipeline_class in pipeline_bases
            or not issubclass(pipeline_class, pipeline_bases)
            or inspect.isabstract(pipeline_class)
        ):
            raise Object3DRegistrationError(
                "pipeline_class must be a concrete nominal Object3DPipeline or ModularObject3DPipeline subclass"
            )
        if not isinstance(registration.metadata, Object3DModelIndex):
            raise TypeError("registration metadata must be Object3DModelIndex")
        if registration.metadata.review_status is not ReviewStatus.REVIEWED:
            raise Object3DRegistrationError("Only explicitly reviewed pipeline metadata can be registered")
        try:
            declared_metadata = pipeline_class.object3d_model_index()
        except Exception as error:
            raise Object3DRegistrationError(
                f"Pipeline class {pipeline_class!r} has an invalid object-3D contract"
            ) from error
        _require_same_metadata(
            registration.metadata, declared_metadata, "Pipeline registration does not match exact class metadata"
        )

        class_key = registration.metadata.pipeline_class
        if class_key in self._by_class:
            raise Object3DRegistrationError(f"Pipeline class {class_key!r} is already registered")
        for task_id in registration.metadata.task_ids:
            family_task_key = (registration.metadata.family_id, task_id)
            if family_task_key in self._by_key:
                existing = self._by_key[family_task_key]
                raise Object3DRegistrationError(
                    f"Ambiguous pipeline registration for family {family_task_key[0]!r} and task "
                    f"{family_task_key[1]!r}: {existing.metadata.pipeline_class!r} and {class_key!r}"
                )
        self._by_class[class_key] = registration
        for task_id in registration.metadata.task_ids:
            self._by_key[(registration.metadata.family_id, task_id)] = registration
        return registration

    def resolve(
        self, metadata: Object3DModelIndex, task_id: str
    ) -> type[Object3DPipeline] | type[ModularObject3DPipeline]:
        if not isinstance(metadata, Object3DModelIndex):
            raise TypeError("metadata must be Object3DModelIndex")
        if not isinstance(task_id, str) or task_id not in metadata.task_ids:
            raise Object3DRegistrationError(
                f"Task {task_id!r} is not declared by pipeline metadata for {metadata.pipeline_class!r}"
            )
        registration = self._by_class.get(metadata.pipeline_class)
        if registration is None:
            raise Object3DRegistrationError(
                f"Pipeline class {metadata.pipeline_class!r} has no exact reviewed registration"
            )
        _require_same_metadata(
            registration.metadata, metadata, "Pipeline metadata does not match its reviewed registration"
        )
        indexed_registration = self._by_key.get((metadata.family_id, task_id))
        if indexed_registration is not registration:
            raise Object3DRegistrationError(
                f"Pipeline registration is not exact for family {metadata.family_id!r} and task {task_id!r}"
            )
        try:
            current_metadata = registration.pipeline_class.object3d_model_index()
        except Exception as error:
            raise Object3DRegistrationError("Registered pipeline class metadata is no longer valid") from error
        _require_same_metadata(
            registration.metadata, current_metadata, "Registered pipeline class drifted from reviewed metadata"
        )
        return registration.pipeline_class


_INTERNAL_MODEL_REGISTRATIONS, _INTERNAL_PIPELINE_REGISTRATIONS = production_execution_registrations(
    Object3DModelRegistration, Object3DPipelineRegistration
)
_MODEL_REGISTRY = Object3DModelRegistry(_INTERNAL_MODEL_REGISTRATIONS).freeze()
_PIPELINE_REGISTRY = Object3DPipelineRegistry(_INTERNAL_PIPELINE_REGISTRATIONS).freeze()

__all__ = [
    "Object3DModelRegistration",
    "Object3DModelRegistry",
    "Object3DPipelineRegistration",
    "Object3DPipelineRegistry",
]
