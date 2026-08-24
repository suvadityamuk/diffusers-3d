"""Optional backend contracts, discovery, policy, and selection.

Importing this module never imports an optional third-party backend.
"""

from .cumesh import CUMESH_SOURCE_URL, CuMeshBackend
from .defaults import BACKEND_REGISTRY, DEFAULT_BACKEND_SPECS, create_default_backend_registry
from .discovery import discover_backend
from .exceptions import (
    BackendError,
    BackendIncompatibleError,
    BackendNotFoundError,
    BackendPolicyError,
    BackendUnavailableError,
)
from .flex_gemm import FLEX_GEMM_BATCH_INDICES, FLEX_GEMM_SOURCE_URL, FlexGemmBackend
from .gsplat import GsplatBackend
from .kaolin_flexicubes import KaolinFlexiCubesBackend
from .o_voxel import (
    OVOXEL_METADATA_PREFIX,
    OVOXEL_REFERENCE_REVISION,
    OVoxelBackend,
    OVoxelCapability,
    OVoxelRuntimeUnavailableError,
    morton_decode_3d,
    morton_encode_3d,
    official_tensors_from_ovoxel_asset,
    ovoxel_asset_from_official,
    ovoxel_grid_transform,
    read_ovoxel_npz,
    write_ovoxel_npz,
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
from .research import (
    DiffoctreerastBackendFacade,
    MipGaussianBackendFacade,
    NvdiffrastBackendFacade,
    ResearchOnlyBackendFacade,
)
from .scikit_image import ScikitImageBackend
from .spconv import SPCONV_BATCH_INDICES, SpconvBackend
from .trellis2_pbr import OVoxelPBRPostprocessFacade, Trellis2PBRPostprocessFacade
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
    "CUMESH_SOURCE_URL",
    "DEFAULT_BACKEND_SPECS",
    "FLEX_GEMM_BATCH_INDICES",
    "FLEX_GEMM_SOURCE_URL",
    "OVOXEL_METADATA_PREFIX",
    "OVOXEL_REFERENCE_REVISION",
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
    "CuMeshBackend",
    "FieldRenderingBackend",
    "GaussianRasterizerBackend",
    "GsplatBackend",
    "FlexGemmBackend",
    "GeometryProcessingBackend",
    "MeshRasterizerBackend",
    "MipGaussianBackendFacade",
    "NativeRepresentationBackend",
    "NvdiffrastBackendFacade",
    "OVoxelBackend",
    "OVoxelCapability",
    "OVoxelPBRPostprocessFacade",
    "OVoxelRuntimeUnavailableError",
    "PBRBakingBackend",
    "ResearchOnlyBackendFacade",
    "SPCONV_BATCH_INDICES",
    "ScikitImageBackend",
    "SparseComputeBackend",
    "SpconvBackend",
    "SurfaceExtractionBackend",
    "TensorMap",
    "TrimeshBackend",
    "Trellis2PBRPostprocessFacade",
    "KaolinFlexiCubesBackend",
    "XAtlasBackend",
    "create_default_backend_registry",
    "discover_backend",
    "morton_decode_3d",
    "morton_encode_3d",
    "official_tensors_from_ovoxel_asset",
    "ovoxel_asset_from_official",
    "ovoxel_grid_transform",
    "read_ovoxel_npz",
    "write_ovoxel_npz",
]
