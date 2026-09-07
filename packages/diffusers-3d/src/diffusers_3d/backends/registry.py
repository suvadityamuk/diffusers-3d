from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .discovery import DistributionGetter, ModuleFinder, VersionGetter, discover_backend
from .exceptions import (
    BackendIncompatibleError,
    BackendNotFoundError,
    BackendPolicyError,
    BackendUnavailableError,
)
from .types import (
    BackendCapability,
    BackendDiscoveryReport,
    BackendLicenseClass,
    BackendSpec,
    BackendStatus,
    BackendSupportLevel,
)

_SUPPORT_ORDER = {
    BackendSupportLevel.PORTABLE: 0,
    BackendSupportLevel.ACCELERATED: 1,
    BackendSupportLevel.RESEARCH_ONLY: 2,
}


def _normalize_capability(capability: BackendCapability | str | None) -> BackendCapability | None:
    if capability is None:
        return None
    try:
        return BackendCapability(capability)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown backend capability: {capability!r}") from error


def _normalize_device(device: Any | None) -> str | None:
    if device is None:
        return None
    if not isinstance(device, str):
        device = str(device)
    normalized = device.lower().split(":", maxsplit=1)[0]
    if not normalized:
        raise ValueError("device must not be empty")
    return normalized


def _normalize_dtype(dtype: Any | None) -> str | None:
    if dtype is None:
        return None
    if not isinstance(dtype, str):
        dtype = str(dtype)
    normalized = dtype.removeprefix("torch.").lower()
    if not normalized:
        raise ValueError("dtype must not be empty")
    return normalized


def _matches(
    spec: BackendSpec,
    *,
    capability: BackendCapability | None,
    device: str | None,
    dtype: str | None,
    differentiable: bool | None,
) -> bool:
    return (
        (capability is None or capability in spec.capabilities)
        and (device is None or device in spec.devices)
        and (dtype is None or dtype in spec.dtypes)
        and (differentiable is None or differentiable is spec.differentiable)
    )


class BackendRegistry:
    """Registry and deterministic selector for optional 3D backends."""

    def __init__(
        self,
        specs: Iterable[BackendSpec] = (),
        *,
        module_finder: ModuleFinder | None = None,
        version_getter: VersionGetter | None = None,
        distribution_getter: DistributionGetter | None = None,
    ) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._module_finder = module_finder
        self._version_getter = version_getter
        self._distribution_getter = distribution_getter
        self._frozen = False
        for spec in specs:
            self.register(spec)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[BackendSpec]:
        return iter(self.list())

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> BackendRegistry:
        """Prevent further registration and return this registry."""

        self._frozen = True
        return self

    def register(self, spec: BackendSpec) -> BackendSpec:
        if self._frozen:
            raise RuntimeError("This backend registry is read-only")
        if not isinstance(spec, BackendSpec):
            raise TypeError("spec must be a BackendSpec")
        if spec.name in self._specs:
            raise ValueError(f"Backend {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> BackendSpec:
        if not isinstance(name, str):
            raise TypeError("backend name must be a string")
        try:
            return self._specs[name]
        except KeyError as error:
            raise BackendNotFoundError(name, sorted(self._specs)) from error

    def status(self, name: str) -> BackendStatus:
        return discover_backend(
            self.get(name),
            module_finder=self._module_finder,
            version_getter=self._version_getter,
            distribution_getter=self._distribution_getter,
        )

    def report(self) -> BackendDiscoveryReport:
        return BackendDiscoveryReport(tuple(self.status(spec.name) for spec in self.list()))

    def list(
        self,
        *,
        capability: BackendCapability | str | None = None,
        support_level: BackendSupportLevel | str | None = None,
        license_class: BackendLicenseClass | str | None = None,
    ) -> tuple[BackendSpec, ...]:
        normalized_capability = _normalize_capability(capability)
        normalized_support = BackendSupportLevel(support_level) if support_level is not None else None
        normalized_license = BackendLicenseClass(license_class) if license_class is not None else None
        return tuple(
            spec
            for spec in sorted(self._specs.values(), key=lambda item: item.name)
            if (normalized_capability is None or normalized_capability in spec.capabilities)
            and (normalized_support is None or spec.support_level is normalized_support)
            and (normalized_license is None or spec.license_class is normalized_license)
        )

    def available(
        self,
        *,
        capability: BackendCapability | str | None = None,
        device: Any | None = None,
        dtype: Any | None = None,
        differentiable: bool | None = None,
        include_research_only: bool = True,
    ) -> tuple[BackendSpec, ...]:
        return self.candidates(
            capability=capability,
            device=device,
            dtype=dtype,
            differentiable=differentiable,
            include_research_only=include_research_only,
            available_only=True,
        )

    def candidates(
        self,
        *,
        capability: BackendCapability | str | None = None,
        device: Any | None = None,
        dtype: Any | None = None,
        differentiable: bool | None = None,
        include_research_only: bool = False,
        available_only: bool = True,
    ) -> tuple[BackendSpec, ...]:
        normalized_capability = _normalize_capability(capability)
        normalized_device = _normalize_device(device)
        normalized_dtype = _normalize_dtype(dtype)
        if differentiable is not None and not isinstance(differentiable, bool):
            raise TypeError("differentiable must be a bool or None")

        matches = [
            spec
            for spec in self._specs.values()
            if (include_research_only or spec.support_level is not BackendSupportLevel.RESEARCH_ONLY)
            and _matches(
                spec,
                capability=normalized_capability,
                device=normalized_device,
                dtype=normalized_dtype,
                differentiable=differentiable,
            )
        ]
        matches.sort(key=lambda spec: (_SUPPORT_ORDER[spec.support_level], spec.name))
        if available_only:
            matches = [spec for spec in matches if self.status(spec.name).available]
        return tuple(matches)

    def select(
        self,
        capability: BackendCapability | str,
        name: str | None = None,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
        differentiable: bool | None = None,
        allow_research_only: bool = False,
    ) -> BackendSpec:
        normalized_capability = _normalize_capability(capability)
        assert normalized_capability is not None
        normalized_device = _normalize_device(device)
        normalized_dtype = _normalize_dtype(dtype)
        if differentiable is not None and not isinstance(differentiable, bool):
            raise TypeError("differentiable must be a bool or None")

        if name is not None:
            spec = self.get(name)
            if spec.support_level is BackendSupportLevel.RESEARCH_ONLY and not allow_research_only:
                raise BackendPolicyError(spec)
            if not _matches(
                spec,
                capability=normalized_capability,
                device=normalized_device,
                dtype=normalized_dtype,
                differentiable=differentiable,
            ):
                raise BackendIncompatibleError(
                    requested_name=name,
                    capability=normalized_capability,
                    device=normalized_device,
                    dtype=normalized_dtype,
                    differentiable=differentiable,
                    specs=(spec,),
                )
            status = self.status(name)
            if not status.available:
                raise BackendUnavailableError(status, requested_name=name)
            return spec

        compatible = self.candidates(
            capability=normalized_capability,
            device=normalized_device,
            dtype=normalized_dtype,
            differentiable=differentiable,
            include_research_only=False,
            available_only=False,
        )
        if not compatible:
            raise BackendIncompatibleError(
                requested_name=None,
                capability=normalized_capability,
                device=normalized_device,
                dtype=normalized_dtype,
                differentiable=differentiable,
                specs=self.list(capability=normalized_capability),
            )

        statuses = tuple(self.status(spec.name) for spec in compatible)
        for status in statuses:
            if status.available:
                return status.spec
        raise BackendUnavailableError(statuses)


__all__ = ["BackendRegistry"]
