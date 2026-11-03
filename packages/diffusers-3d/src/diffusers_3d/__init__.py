"""Generative 3D on Diffusers: pipelines that return typed meshes, Gaussian splats, and voxels."""

from . import families as _families
from ._version import __version__
from .backends import (
    CuMeshBackend,
    GsplatBackend,
    ScikitImageBackend,
    TrimeshBackend,
    XAtlasBackend,
    ovoxel_to_mesh,
    read_ovoxel_npz,
    textured_mesh_from_ovoxel,
    textured_mesh_from_splats,
    write_ovoxel_npz,
)
from .data import (
    BackgroundRemoval,
    BiRefNetBackgroundRemover,
    ImageCondition,
    ImageInput,
    MultiViewCondition,
    RembgBackgroundRemover,
    RMBG14BackgroundRemover,
    TextCondition,
    as_image_conditions,
)
from .execution import (
    AutoPipelineFor3D,
    AutoPipelineForImageTo3D,
    AutoPipelineForMeshTexturing,
    AutoPipelineForTextTo3D,
    Object3DLoraLoaderMixin,
    Object3DModel,
    Object3DPipeline,
)
from .objects import (
    CameraRig,
    CoordinateSystem,
    GaussianSplatAsset,
    Latent3DOutput,
    MeshAsset,
    Object3DKind,
    Object3DPipelineOutput,
    Object3DValidationError,
    OVoxelAsset,
    PBRMaterial,
    RadianceFieldAsset,
    SparseVoxelAsset,
)


def __getattr__(name: str):
    if name not in _families.__all__:
        raise AttributeError(name)
    value = getattr(_families, name)
    globals()[name] = value
    return value


__all__ = [
    "AutoPipelineFor3D",
    "AutoPipelineForImageTo3D",
    "AutoPipelineForMeshTexturing",
    "AutoPipelineForTextTo3D",
    "CameraRig",
    "CoordinateSystem",
    "CuMeshBackend",
    "GaussianSplatAsset",
    "GsplatBackend",
    "ImageCondition",
    "ImageInput",
    "Latent3DOutput",
    "MeshAsset",
    "MultiViewCondition",
    "OVoxelAsset",
    "Object3DKind",
    "Object3DLoraLoaderMixin",
    "Object3DModel",
    "Object3DPipeline",
    "Object3DPipelineOutput",
    "Object3DValidationError",
    "PBRMaterial",
    "BackgroundRemoval",
    "BiRefNetBackgroundRemover",
    "RMBG14BackgroundRemover",
    "RembgBackgroundRemover",
    "RadianceFieldAsset",
    "ScikitImageBackend",
    "SparseVoxelAsset",
    "TextCondition",
    "textured_mesh_from_ovoxel",
    "textured_mesh_from_splats",
    "TrimeshBackend",
    "XAtlasBackend",
    "__version__",
    "as_image_conditions",
    "ovoxel_to_mesh",
    "read_ovoxel_npz",
    "write_ovoxel_npz",
]

__all__ = sorted(set(__all__) | set(_families.__all__))
