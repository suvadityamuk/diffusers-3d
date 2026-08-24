from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class BackendSupportLevel(str, Enum):
    """The maintenance and portability policy for a backend."""

    PORTABLE = "portable"
    ACCELERATED = "accelerated"
    RESEARCH_ONLY = "research_only"


class BackendLicenseClass(str, Enum):
    """Coarse license classification used by backend selection policy."""

    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class BackendCapability(str, Enum):
    """Operations that can be supplied by an optional backend."""

    SPARSE_COMPUTE = "sparse_compute"
    MESH_RASTERIZATION = "mesh_rasterization"
    GAUSSIAN_RASTERIZATION = "gaussian_rasterization"
    SURFACE_EXTRACTION = "surface_extraction"
    GEOMETRY_PROCESSING = "geometry_processing"
    NATIVE_REPRESENTATION = "native_representation"
    PBR_BAKING = "pbr_baking"
    FIELD_RENDERING = "field_rendering"
    SERIALIZATION = "serialization"
    CONVERSION = "conversion"


_BACKEND_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_IMPORT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_DISTRIBUTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")
_DEVICE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DTYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _as_nonempty_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a string")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be a sequence of strings") from error
    if not items:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items


def _as_optional_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a string")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be a sequence of strings") from error
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items


def _as_string_frozenset(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a collection of strings, not a string")
    try:
        items = frozenset(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be a collection of strings") from error
    if not items or any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{field_name} must contain at least one non-empty string")
    return items


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Immutable discovery, compatibility, and policy metadata for a backend.

    Import names and distribution names are intentionally separate. For example,
    ``scikit-image`` provides ``skimage``. Source-only packages may leave
    ``distribution_names`` empty when their provenance cannot be established
    safely from Python distribution metadata.
    """

    name: str
    import_names: tuple[str, ...]
    distribution_names: tuple[str, ...]
    capabilities: frozenset[BackendCapability]
    support_level: BackendSupportLevel
    license_class: BackendLicenseClass
    devices: frozenset[str]
    dtypes: frozenset[str]
    differentiable: bool
    install_hint: str
    tested_version: str | None = None
    tested_build: str | None = None
    source_url: str | None = None
    source_revision: str | None = None
    requires_source_provenance: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _BACKEND_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "name must start with a lowercase letter and contain only lowercase letters, digits, '.', '_', or '-'"
            )

        import_names = _as_nonempty_tuple(self.import_names, "import_names")
        if any(not _IMPORT_NAME_PATTERN.fullmatch(name) for name in import_names):
            raise ValueError("import_names must contain valid absolute Python module names")
        object.__setattr__(self, "import_names", import_names)

        distribution_names = _as_optional_tuple(self.distribution_names, "distribution_names")
        if any(not _DISTRIBUTION_NAME_PATTERN.fullmatch(name) for name in distribution_names):
            raise ValueError("distribution_names must contain valid Python distribution names")
        object.__setattr__(self, "distribution_names", distribution_names)

        if isinstance(self.capabilities, (str, BackendCapability)):
            raise TypeError("capabilities must be a collection of BackendCapability values")
        try:
            capabilities = frozenset(BackendCapability(capability) for capability in self.capabilities)
        except (TypeError, ValueError) as error:
            raise ValueError("capabilities must contain valid BackendCapability values") from error
        if not capabilities:
            raise ValueError("capabilities must not be empty")
        object.__setattr__(self, "capabilities", capabilities)

        try:
            support_level = BackendSupportLevel(self.support_level)
        except (TypeError, ValueError) as error:
            raise ValueError("support_level must be a valid BackendSupportLevel") from error
        object.__setattr__(self, "support_level", support_level)

        try:
            license_class = BackendLicenseClass(self.license_class)
        except (TypeError, ValueError) as error:
            raise ValueError("license_class must be a valid BackendLicenseClass") from error
        object.__setattr__(self, "license_class", license_class)

        devices = frozenset(device.lower() for device in _as_string_frozenset(self.devices, "devices"))
        if any(not _DEVICE_PATTERN.fullmatch(device) for device in devices):
            raise ValueError("devices must contain simple device types such as 'cpu', 'cuda', or 'mps'")
        object.__setattr__(self, "devices", devices)

        dtypes = frozenset(
            dtype.removeprefix("torch.").lower() for dtype in _as_string_frozenset(self.dtypes, "dtypes")
        )
        if any(not _DTYPE_PATTERN.fullmatch(dtype) for dtype in dtypes):
            raise ValueError("dtypes must contain simple dtype names such as 'float16' or 'float32'")
        object.__setattr__(self, "dtypes", dtypes)

        if not isinstance(self.differentiable, bool):
            raise TypeError("differentiable must be a bool")
        if not isinstance(self.install_hint, str) or not self.install_hint.strip():
            raise ValueError("install_hint must be a non-empty actionable installation instruction")

        for field_name in ("tested_version", "tested_build", "source_url", "source_revision"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or a non-empty string")
        if self.source_revision is not None and self.source_url is None:
            raise ValueError("source_revision requires source_url")
        if not isinstance(self.requires_source_provenance, bool):
            raise TypeError("requires_source_provenance must be a bool")
        if self.requires_source_provenance and (self.source_url is None or self.source_revision is None):
            raise ValueError("requires_source_provenance requires both source_url and source_revision")
        if not self.distribution_names and self.source_url is None:
            raise ValueError("a backend without distribution_names must provide source_url")


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Result of a side-effect-free backend discovery check."""

    spec: BackendSpec
    installed: bool
    importable: bool
    version: str | None
    distribution_name: str | None
    reason: str | None
    provenance_verified: bool = True
    missing_import_names: tuple[str, ...] = ()
    missing_distribution_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BackendSpec):
            raise TypeError("spec must be a BackendSpec")
        for field_name in ("installed", "importable", "provenance_verified"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in ("version", "distribution_name", "reason"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        object.__setattr__(
            self,
            "missing_import_names",
            _as_optional_tuple(self.missing_import_names, "missing_import_names"),
        )
        object.__setattr__(
            self,
            "missing_distribution_names",
            _as_optional_tuple(self.missing_distribution_names, "missing_distribution_names"),
        )

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def available(self) -> bool:
        return self.installed and self.importable and self.provenance_verified


@dataclass(frozen=True, slots=True)
class BackendDiscoveryReport:
    """Immutable point-in-time discovery results for a registry."""

    statuses: tuple[BackendStatus, ...]

    def __post_init__(self) -> None:
        if isinstance(self.statuses, BackendStatus):
            raise TypeError("statuses must be a sequence of BackendStatus values")
        try:
            statuses = tuple(self.statuses)
        except TypeError as error:
            raise TypeError("statuses must be a sequence of BackendStatus values") from error
        if any(not isinstance(status, BackendStatus) for status in statuses):
            raise TypeError("statuses must contain only BackendStatus values")
        object.__setattr__(self, "statuses", statuses)

    @property
    def available(self) -> tuple[BackendStatus, ...]:
        return tuple(status for status in self.statuses if status.available)

    @property
    def unavailable(self) -> tuple[BackendStatus, ...]:
        return tuple(status for status in self.statuses if not status.available)


__all__ = [
    "BackendCapability",
    "BackendDiscoveryReport",
    "BackendLicenseClass",
    "BackendSpec",
    "BackendStatus",
    "BackendSupportLevel",
]
