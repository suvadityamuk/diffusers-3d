"""Optional backend contracts, discovery, policy, and selection.

Importing this module never imports an optional third-party backend.
"""

from .cumesh import CUMESH_SOURCE_REVISION, CUMESH_SOURCE_URL, CuMeshBackend
from .defaults import BACKEND_REGISTRY, DEFAULT_BACKEND_SPECS, create_default_backend_registry
from .discovery import discover_backend
from .exceptions import (
    BackendError,
    BackendIncompatibleError,
    BackendNotFoundError,
    BackendPolicyError,
    BackendUnavailableError,
)
from .flex_gemm import (
    FLEX_GEMM_BATCH_INDICES,
    FLEX_GEMM_SOURCE_REVISION,
    FLEX_GEMM_SOURCE_URL,
    FlexGemmBackend,
)
from .gsplat import GsplatBackend
from .kaolin_flexicubes import KaolinFlexiCubesBackend
from .mesh_rasterizer import Rasterization, interpolate, rasterize, rasterize_mesh
from .multiview_baking import (
    canny,
    inpaint_texture,
    look_at_view,
    merge_views,
    orthographic_projection,
    project_view,
    rasterize_texels,
    splat_view,
)
from .o_voxel import (
    OVOXEL_REFERENCE_REVISION,
    OVoxelBackend,
    OVoxelCapability,
    OVoxelRuntimeUnavailableError,
    morton_decode_3d,
    morton_encode_3d,
    official_tensors_from_ovoxel_asset,
    ovoxel_asset_from_official,
    ovoxel_grid_transform,
    ovoxel_to_mesh,
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
from .trellis2_pbr import Trellis2PBRPostprocessFacade
from .trellis_glb import TrellisGlbPostprocessFacade
from .trimesh import TrimeshBackend
from .types import (
    BackendCapability,
    BackendDiscoveryReport,
    BackendLicenseClass,
    BackendSpec,
    BackendStatus,
    BackendSupportLevel,
)
from .voxel_index import VoxelIndex, trilinear_sample
from .xatlas import XAtlasBackend

__all__ = [
    "BACKEND_REGISTRY",
    "CUMESH_SOURCE_REVISION",
    "CUMESH_SOURCE_URL",
    "DEFAULT_BACKEND_SPECS",
    "FLEX_GEMM_BATCH_INDICES",
    "FLEX_GEMM_SOURCE_REVISION",
    "FLEX_GEMM_SOURCE_URL",
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
    "OVoxelRuntimeUnavailableError",
    "PBRBakingBackend",
    "Rasterization",
    "ResearchOnlyBackendFacade",
    "SPCONV_BATCH_INDICES",
    "ScikitImageBackend",
    "SparseComputeBackend",
    "SpconvBackend",
    "SurfaceExtractionBackend",
    "TensorMap",
    "TrimeshBackend",
    "Trellis2PBRPostprocessFacade",
    "TrellisGlbPostprocessFacade",
    "KaolinFlexiCubesBackend",
    "VoxelIndex",
    "XAtlasBackend",
    "canny",
    "create_default_backend_registry",
    "discover_backend",
    "inpaint_texture",
    "interpolate",
    "look_at_view",
    "merge_views",
    "morton_decode_3d",
    "morton_encode_3d",
    "official_tensors_from_ovoxel_asset",
    "ovoxel_to_mesh",
    "trilinear_sample",
    "orthographic_projection",
    "ovoxel_asset_from_official",
    "ovoxel_grid_transform",
    "project_view",
    "rasterize",
    "rasterize_mesh",
    "rasterize_texels",
    "read_ovoxel_npz",
    "splat_view",
    "write_ovoxel_npz",
]
