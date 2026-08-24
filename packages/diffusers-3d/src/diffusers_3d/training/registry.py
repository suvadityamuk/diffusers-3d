from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from ..execution import ModularObject3DPipeline, Object3DModel, Object3DPipeline, ReviewStatus
from .exceptions import TrainingRegistrationError, TrainingTargetError
from .recipe import TrainingRecipe3D
from .types import ComponentPolicy

_RECIPE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


@dataclass(frozen=True, slots=True)
class TrainingRecipeRegistration:
    """Exact recipe declaration and its separately reviewed training metadata."""

    recipe_type: type[TrainingRecipe3D]
    target_type: type[Object3DModel] | type[Object3DPipeline] | type[ModularObject3DPipeline]
    batch_type: type[object]
    recipe_id: str
    recipe_version: str
    family_id: str
    component_policies: tuple[ComponentPolicy, ...]
    review_status: ReviewStatus


class TrainingRecipeRegistry:
    """Mutable factory registry with exact, non-polymorphic recipe resolution."""

    def __init__(self, registrations: Iterable[TrainingRecipeRegistration] = ()) -> None:
        self._registrations: dict[
            tuple[
                type[TrainingRecipe3D],
                type[Object3DModel] | type[Object3DPipeline] | type[ModularObject3DPipeline],
                str,
            ],
            TrainingRecipeRegistration,
        ] = {}
        self._frozen = False
        for registration in registrations:
            self.register(registration)

    def __len__(self) -> int:
        return len(self._registrations)

    def __iter__(self) -> Iterator[TrainingRecipeRegistration]:
        return iter(self.list())

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> TrainingRecipeRegistry:
        self._frozen = True
        return self

    def register(self, registration: TrainingRecipeRegistration) -> TrainingRecipeRegistration:
        if self._frozen:
            raise TrainingRegistrationError("This object-3D training recipe registry is read-only")
        if not isinstance(registration, TrainingRecipeRegistration):
            raise TypeError("registration must be a TrainingRecipeRegistration")
        recipe_type = registration.recipe_type
        if (
            not isinstance(recipe_type, type)
            or recipe_type is TrainingRecipe3D
            or not issubclass(recipe_type, TrainingRecipe3D)
            or inspect.isabstract(recipe_type)
        ):
            raise TrainingRegistrationError("recipe_type must be a concrete nominal TrainingRecipe3D subclass")
        target_type = registration.target_type
        target_bases = (Object3DModel, Object3DPipeline, ModularObject3DPipeline)
        if (
            not isinstance(target_type, type)
            or target_type in target_bases
            or not issubclass(target_type, target_bases)
            or inspect.isabstract(target_type)
        ):
            raise TrainingRegistrationError(
                "target_type must be a concrete nominal Object3DModel, Object3DPipeline, or "
                "ModularObject3DPipeline subclass"
            )
        if (
            not isinstance(registration.batch_type, type)
            or registration.batch_type is object
            or issubclass(registration.batch_type, Mapping)
        ):
            raise TrainingRegistrationError("batch_type must be an exact typed batch class, not a mapping")
        if not callable(getattr(registration.batch_type, "validate", None)) or not callable(
            getattr(registration.batch_type, "to", None)
        ):
            raise TrainingRegistrationError("batch_type must expose typed validate() and functional to() methods")
        if not isinstance(registration.recipe_id, str) or not _RECIPE_ID_PATTERN.fullmatch(registration.recipe_id):
            raise TrainingRegistrationError("recipe_id must be a stable lowercase identifier")
        if not isinstance(registration.recipe_version, str) or not _VERSION_PATTERN.fullmatch(
            registration.recipe_version
        ):
            raise TrainingRegistrationError("recipe_version must be a stable non-empty version")
        if not isinstance(registration.family_id, str) or not _RECIPE_ID_PATTERN.fullmatch(registration.family_id):
            raise TrainingRegistrationError("family_id must be a stable lowercase identifier")
        if registration.review_status is not ReviewStatus.REVIEWED:
            raise TrainingRegistrationError("Only explicitly reviewed training recipes can be registered")
        if not isinstance(registration.component_policies, tuple) or not registration.component_policies:
            raise TrainingRegistrationError("component_policies must be a non-empty tuple")
        if any(type(policy) is not ComponentPolicy for policy in registration.component_policies):
            raise TrainingRegistrationError("component_policies must contain exact ComponentPolicy values")
        keys = [policy.key for policy in registration.component_policies]
        paths = [policy.component_path for policy in registration.component_policies]
        if len(set(keys)) != len(keys):
            raise TrainingRegistrationError("component policy keys must be unique")
        if len(set(paths)) != len(paths):
            raise TrainingRegistrationError("component policy paths must be unique")
        for index, path in enumerate(paths):
            for other_path in paths[index + 1 :]:
                if (
                    not path
                    or not other_path
                    or path.startswith(f"{other_path}.")
                    or other_path.startswith(f"{path}.")
                ):
                    raise TrainingRegistrationError("component policy paths must not overlap")

        declarations = (
            ("recipe_id", registration.recipe_id),
            ("recipe_version", registration.recipe_version),
            ("family_id", registration.family_id),
            ("target_type", registration.target_type),
            ("batch_type", registration.batch_type),
            ("component_policies", registration.component_policies),
        )
        for name, expected in declarations:
            if getattr(recipe_type, name, None) != expected:
                raise TrainingRegistrationError(
                    f"registration {name} does not match the exact {recipe_type.__name__} declaration"
                )
        if getattr(target_type, "family_id", None) != registration.family_id:
            raise TrainingRegistrationError("training family_id does not match the exact target declaration")

        key = (recipe_type, target_type, registration.recipe_id)
        if key in self._registrations:
            raise TrainingRegistrationError(
                f"Recipe {recipe_type.__module__}.{recipe_type.__qualname__} is already registered for "
                f"{target_type.__module__}.{target_type.__qualname__} and {registration.recipe_id!r}"
            )
        self._registrations[key] = registration
        return registration

    def resolve(
        self,
        recipe_type: type[TrainingRecipe3D],
        target_type: type[Object3DModel] | type[Object3DPipeline] | type[ModularObject3DPipeline],
        recipe_id: str,
    ) -> TrainingRecipeRegistration:
        key = (recipe_type, target_type, recipe_id)
        registration = self._registrations.get(key)
        if registration is None:
            raise TrainingRegistrationError(
                "No exact reviewed object-3D training registration exists for "
                f"({getattr(recipe_type, '__name__', recipe_type)!r}, "
                f"{getattr(target_type, '__name__', target_type)!r}, {recipe_id!r})"
            )
        declarations = (
            ("recipe_id", registration.recipe_id),
            ("recipe_version", registration.recipe_version),
            ("family_id", registration.family_id),
            ("target_type", registration.target_type),
            ("batch_type", registration.batch_type),
            ("component_policies", registration.component_policies),
        )
        for name, expected in declarations:
            if getattr(recipe_type, name, None) != expected:
                raise TrainingRegistrationError(f"Registered recipe class drifted from reviewed metadata at {name!r}")
        if getattr(target_type, "family_id", None) != registration.family_id:
            raise TrainingRegistrationError("Registered target class drifted from its reviewed training family")
        return registration

    def validate(self, recipe: TrainingRecipe3D) -> TrainingRecipeRegistration:
        if not isinstance(recipe, TrainingRecipe3D) or type(recipe) is TrainingRecipe3D:
            raise TrainingRegistrationError("recipe must be a concrete TrainingRecipe3D instance")
        recipe_type = type(recipe)
        target = recipe.target
        registration = self.resolve(recipe_type, type(target), getattr(recipe_type, "recipe_id", None))
        if type(target) is not registration.target_type:
            raise TrainingTargetError(
                f"Recipe target must be exact {registration.target_type.__name__}, got {type(target).__name__}"
            )
        return registration

    def list(self) -> tuple[TrainingRecipeRegistration, ...]:
        return tuple(
            sorted(
                self._registrations.values(),
                key=lambda item: (
                    f"{item.recipe_type.__module__}.{item.recipe_type.__qualname__}",
                    f"{item.target_type.__module__}.{item.target_type.__qualname__}",
                    item.recipe_id,
                ),
            )
        )


def create_training_recipe_registry(
    registrations: Iterable[TrainingRecipeRegistration] = (),
) -> TrainingRecipeRegistry:
    """Create a mutable registry for package integrations or isolated tests."""

    return TrainingRecipeRegistry(registrations)


_TRAINING_RECIPE_REGISTRY = create_training_recipe_registry().freeze()


__all__ = [
    "TrainingRecipeRegistration",
    "TrainingRecipeRegistry",
    "create_training_recipe_registry",
]
