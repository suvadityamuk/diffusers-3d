from __future__ import annotations

from collections.abc import Iterable

from .types import BackendCapability, BackendSpec, BackendStatus


class BackendError(RuntimeError):
    """Base class for actionable backend errors."""


class BackendNotFoundError(BackendError):
    """Raised when an exact backend name is not registered."""

    def __init__(self, name: str, registered_names: Iterable[str]) -> None:
        self.name = name
        self.registered_names = tuple(registered_names)
        choices = ", ".join(self.registered_names) or "<none>"
        super().__init__(f"Backend {name!r} is not registered. Registered backend names: {choices}.")


class BackendUnavailableError(BackendError):
    """Raised when a compatible backend is not installed or importable."""

    def __init__(
        self, statuses: BackendStatus | Iterable[BackendStatus], *, requested_name: str | None = None
    ) -> None:
        if isinstance(statuses, BackendStatus):
            normalized_statuses = (statuses,)
        else:
            normalized_statuses = tuple(statuses)
        self.statuses = normalized_statuses
        self.requested_name = requested_name

        if requested_name is not None and len(normalized_statuses) == 1:
            status = normalized_statuses[0]
            detail = status.reason or "the backend did not pass discovery"
            message = f"Backend {requested_name!r} is unavailable: {detail}"
        elif normalized_statuses:
            detail = " ".join(f"{status.name}: {status.reason}" for status in normalized_statuses)
            message = f"No compatible installed backend is available. {detail}"
        else:
            message = "No compatible installed backend is available."
        super().__init__(message)


class BackendIncompatibleError(BackendError):
    """Raised when a backend does not satisfy requested runtime constraints."""

    def __init__(
        self,
        *,
        requested_name: str | None,
        capability: BackendCapability,
        device: str | None,
        dtype: str | None,
        differentiable: bool | None,
        specs: Iterable[BackendSpec],
    ) -> None:
        self.requested_name = requested_name
        self.capability = capability
        self.device = device
        self.dtype = dtype
        self.differentiable = differentiable
        self.specs = tuple(specs)

        constraints = [f"capability={capability.value!r}"]
        if device is not None:
            constraints.append(f"device={device!r}")
        if dtype is not None:
            constraints.append(f"dtype={dtype!r}")
        if differentiable is not None:
            constraints.append(f"differentiable={differentiable!r}")
        subject = f"Backend {requested_name!r}" if requested_name is not None else "No registered backend"
        available_details = "; ".join(
            f"{spec.name} (capabilities={sorted(item.value for item in spec.capabilities)}, "
            f"devices={sorted(spec.devices)}, dtypes={sorted(spec.dtypes)}, "
            f"differentiable={spec.differentiable})"
            for spec in self.specs
        )
        if requested_name is not None:
            message = f"{subject} does not satisfy the requested constraints ({', '.join(constraints)})."
        else:
            message = f"{subject} satisfies the requested constraints ({', '.join(constraints)})."
        if available_details:
            message += f" Registered choices: {available_details}."
        super().__init__(message)


class BackendPolicyError(BackendError):
    """Raised when selection violates backend support policy."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        super().__init__(
            f"Backend {spec.name!r} is research-only and cannot be selected without "
            f"allow_research_only=True. License class: {spec.license_class.value}. "
            f"Review its license and build requirements before enabling it. {spec.install_hint.rstrip('.')}."
        )


__all__ = [
    "BackendError",
    "BackendIncompatibleError",
    "BackendNotFoundError",
    "BackendPolicyError",
    "BackendUnavailableError",
]
