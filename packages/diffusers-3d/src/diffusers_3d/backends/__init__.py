"""Mesh, splat, and voxel tools: plain-PyTorch rasterizers and bakers, plus adapters for optional libraries.

Importing this module never imports an optional library; each adapter imports its library when constructed.
"""

from .cumesh import CuMeshBackend
from .gsplat import GsplatBackend
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
    official_tensors_from_ovoxel_asset,
    ovoxel_asset_from_official,
    ovoxel_grid_transform,
    ovoxel_to_mesh,
    read_ovoxel_npz,
    write_ovoxel_npz,
)
from .scikit_image import ScikitImageBackend
from .trellis2_pbr import textured_mesh_from_ovoxel
from .trellis_glb import textured_mesh_from_splats
from .trimesh import TrimeshBackend
from .voxel_index import VoxelIndex, trilinear_sample
from .xatlas import XAtlasBackend

__all__ = [
    "OVOXEL_REFERENCE_REVISION",
    "CuMeshBackend",
    "GsplatBackend",
    "Rasterization",
    "ScikitImageBackend",
    "textured_mesh_from_ovoxel",
    "textured_mesh_from_splats",
    "TrimeshBackend",
    "VoxelIndex",
    "XAtlasBackend",
    "canny",
    "inpaint_texture",
    "interpolate",
    "look_at_view",
    "merge_views",
    "official_tensors_from_ovoxel_asset",
    "orthographic_projection",
    "ovoxel_asset_from_official",
    "ovoxel_grid_transform",
    "ovoxel_to_mesh",
    "project_view",
    "rasterize",
    "rasterize_mesh",
    "rasterize_texels",
    "read_ovoxel_npz",
    "splat_view",
    "trilinear_sample",
    "write_ovoxel_npz",
]
