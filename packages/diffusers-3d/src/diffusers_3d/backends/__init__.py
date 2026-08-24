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
from .gsplat import GsplatBackend
from .kaolin_flexicubes import KaolinFlexiCubesBackend
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
from .research import (
    DiffoctreerastBackendFacade,
    MipGaussianBackendFacade,
    NvdiffrastBackendFacade,
    ResearchOnlyBackendFacade,
)
from .scikit_image import ScikitImageBackend
from .spconv import SPCONV_BATCH_INDICES, SpconvBackend
from .trimesh import TrimeshBackend
from .types import (
    BackendCapability,
    BackendDiscoveryReport,
    BackendLicenseClass,
    BackendSpec,
    BackendStatus,
    BackendSupportLevel,
)
from .xatlas import XAtlasBackend

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
    "DiffoctreerastBackendFacade",
    "FieldRenderingBackend",
    "GaussianRasterizerBackend",
    "GsplatBackend",
    "GeometryProcessingBackend",
    "MeshRasterizerBackend",
    "MipGaussianBackendFacade",
    "NativeRepresentationBackend",
    "NvdiffrastBackendFacade",
    "PBRBakingBackend",
    "ResearchOnlyBackendFacade",
    "SPCONV_BATCH_INDICES",
    "ScikitImageBackend",
    "SparseComputeBackend",
    "SpconvBackend",
    "SurfaceExtractionBackend",
    "TensorMap",
    "TrimeshBackend",
    "KaolinFlexiCubesBackend",
    "XAtlasBackend",
    "create_default_backend_registry",
    "discover_backend",
]
