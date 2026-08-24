"""Optional backend contracts, discovery, policy, and selection.

Importing this module never imports an optional third-party backend.
"""

from .defaults import BACKEND_REGISTRY, DEFAULT_BACKEND_SPECS, create_default_backend_registry
from .discovery import discover_backend
from .exceptions import (
    BackendError,
    BackendIncompatibleError,
    BackendNotFoundError,
    BackendPolicyError,
    BackendUnavailableError,
)
from .protocols import (
    FieldRenderingBackend,
    GaussianRasterizerBackend,
    GeometryProcessingBackend,
    MeshRasterizerBackend,
    NativeRepresentationBackend,
    PBRBakingBackend,
    SparseComputeBackend,
    SurfaceExtractionBackend,
    TensorMap,
)
from .registry import BackendRegistry
from .types import (
    BackendCapability,
    BackendDiscoveryReport,
    BackendLicenseClass,
    BackendSpec,
    BackendStatus,
    BackendSupportLevel,
)

__all__ = [
    "BACKEND_REGISTRY",
    "DEFAULT_BACKEND_SPECS",
    "BackendCapability",
    "BackendDiscoveryReport",
    "BackendError",
    "BackendIncompatibleError",
    "BackendLicenseClass",
    "BackendNotFoundError",
    "BackendPolicyError",
    "BackendRegistry",
    "BackendSpec",
    "BackendStatus",
    "BackendSupportLevel",
    "BackendUnavailableError",
    "FieldRenderingBackend",
    "GaussianRasterizerBackend",
    "GeometryProcessingBackend",
    "MeshRasterizerBackend",
    "NativeRepresentationBackend",
    "PBRBakingBackend",
    "SparseComputeBackend",
    "SurfaceExtractionBackend",
    "TensorMap",
    "create_default_backend_registry",
    "discover_backend",
]
