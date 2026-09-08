from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from ..objects import (
    CameraRig,
    GaussianSplatAsset,
    MeshAsset,
    Object3D,
    Object3DKind,
    PBRMaterial,
    SparseVoxelAsset,
)

TensorMap = Mapping[str, torch.Tensor]


@runtime_checkable
class SparseComputeBackend(Protocol):
    """Sparse feature processing over package-owned voxel assets."""

    def sparse_compute(
        self,
        voxels: SparseVoxelAsset,
        *,
        operation: str,
        parameters: TensorMap | None = None,
    ) -> SparseVoxelAsset: ...


@runtime_checkable
class MeshRasterizerBackend(Protocol):
    """Rasterization of triangle meshes into named tensor buffers."""

    def rasterize_mesh(
        self,
        mesh: MeshAsset,
        cameras: CameraRig,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> TensorMap: ...


@runtime_checkable
class GaussianRasterizerBackend(Protocol):
    """Rasterization of Gaussian splats into named tensor buffers."""

    def rasterize_gaussians(
        self,
        gaussians: GaussianSplatAsset,
        cameras: CameraRig,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> TensorMap: ...


@runtime_checkable
class SurfaceExtractionBackend(Protocol):
    """Extraction of a triangle mesh from a scalar field."""

    def extract_surface(
        self,
        field: torch.Tensor,
        *,
        level: float = 0.0,
        spacing: Sequence[float] | None = None,
    ) -> MeshAsset: ...


@runtime_checkable
class GeometryProcessingBackend(Protocol):
    """Named mesh transformations such as cleanup, simplification, or UV unwrapping."""

    def process_geometry(
        self,
        mesh: MeshAsset,
        *,
        operation: str,
        parameters: Mapping[str, object] | None = None,
    ) -> MeshAsset: ...


@runtime_checkable
class NativeRepresentationBackend(Protocol):
    """Conversion between Object3D contracts and backend-native tensors."""

    def encode_native(self, object_3d: Object3D) -> TensorMap: ...

    def decode_native(self, tensors: TensorMap, *, kind: Object3DKind) -> Object3D: ...


@runtime_checkable
class PBRBakingBackend(Protocol):
    """Bake view observations into a package-owned PBR material."""

    def bake_pbr(
        self,
        mesh: MeshAsset,
        cameras: CameraRig,
        images: torch.Tensor,
        *,
        texture_size: tuple[int, int],
    ) -> PBRMaterial: ...


@runtime_checkable
class FieldRenderingBackend(Protocol):
    """Render tensor fields for a camera rig."""

    def render_field(
        self,
        field: torch.Tensor,
        cameras: CameraRig,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> TensorMap: ...


__all__ = [
    "FieldRenderingBackend",
    "GaussianRasterizerBackend",
    "GeometryProcessingBackend",
    "MeshRasterizerBackend",
    "NativeRepresentationBackend",
    "PBRBakingBackend",
    "SparseComputeBackend",
    "SurfaceExtractionBackend",
    "TensorMap",
]
