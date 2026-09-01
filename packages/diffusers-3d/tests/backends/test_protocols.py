from __future__ import annotations

from inspect import signature
from typing import get_type_hints

import torch

from diffusers_3d.backends import (
    FieldRenderingBackend,
    GaussianRasterizerBackend,
    GeometryProcessingBackend,
    KaolinFlexiCubesBackend,
    MeshRasterizerBackend,
    NativeRepresentationBackend,
    PBRBakingBackend,
    SparseComputeBackend,
    SurfaceExtractionBackend,
)
from diffusers_3d.objects import GaussianSplatAsset, MeshAsset, Object3D, PBRMaterial, SparseVoxelAsset


class CompleteStructuralBackend:
    def sparse_compute(self, voxels, *, operation, parameters=None):
        return voxels

    def rasterize_mesh(self, mesh, cameras, *, image_size=None):
        return {}

    def rasterize_gaussians(self, gaussians, cameras, *, image_size=None):
        return {}

    def extract_surface(self, field, *, level=0.0, spacing=None):
        return None

    def process_geometry(self, mesh, *, operation, parameters=None):
        return mesh

    def encode_native(self, object_3d):
        return {}

    def decode_native(self, tensors, *, kind):
        return None

    def bake_pbr(self, mesh, cameras, images, *, texture_size):
        return None

    def render_field(self, field, cameras, *, image_size=None):
        return {}


class NotABackend:
    pass


def test_runtime_protocols_accept_structural_implementations():
    implementation = CompleteStructuralBackend()

    assert isinstance(implementation, SparseComputeBackend)
    assert isinstance(implementation, MeshRasterizerBackend)
    assert isinstance(implementation, GaussianRasterizerBackend)
    assert isinstance(implementation, SurfaceExtractionBackend)
    assert isinstance(implementation, GeometryProcessingBackend)
    assert isinstance(implementation, NativeRepresentationBackend)
    assert isinstance(implementation, PBRBakingBackend)
    assert isinstance(implementation, FieldRenderingBackend)


def test_runtime_protocols_reject_missing_methods():
    implementation = NotABackend()

    assert not isinstance(implementation, SparseComputeBackend)
    assert not isinstance(implementation, MeshRasterizerBackend)
    assert not isinstance(implementation, GaussianRasterizerBackend)
    assert not isinstance(implementation, SurfaceExtractionBackend)
    assert not isinstance(implementation, GeometryProcessingBackend)
    assert not isinstance(implementation, NativeRepresentationBackend)
    assert not isinstance(implementation, PBRBakingBackend)
    assert not isinstance(implementation, FieldRenderingBackend)


def test_surface_extraction_protocol_signature_is_compatible_with_kaolin_adapter():
    protocol_parameters = signature(SurfaceExtractionBackend.extract_surface).parameters
    kaolin_parameters = signature(KaolinFlexiCubesBackend.extract_surface).parameters

    assert tuple(protocol_parameters) == tuple(kaolin_parameters) == ("self", "field", "level", "spacing")


def test_protocol_signatures_use_tensor_and_object_contracts():
    sparse_hints = get_type_hints(SparseComputeBackend.sparse_compute)
    mesh_hints = get_type_hints(MeshRasterizerBackend.rasterize_mesh)
    gaussian_hints = get_type_hints(GaussianRasterizerBackend.rasterize_gaussians)
    surface_hints = get_type_hints(SurfaceExtractionBackend.extract_surface)
    native_hints = get_type_hints(NativeRepresentationBackend.encode_native)
    pbr_hints = get_type_hints(PBRBakingBackend.bake_pbr)
    field_hints = get_type_hints(FieldRenderingBackend.render_field)

    assert sparse_hints["voxels"] is SparseVoxelAsset
    assert sparse_hints["return"] is SparseVoxelAsset
    assert mesh_hints["mesh"] is MeshAsset
    assert gaussian_hints["gaussians"] is GaussianSplatAsset
    assert surface_hints["field"] is torch.Tensor
    assert surface_hints["return"] is MeshAsset
    assert native_hints["object_3d"] is Object3D
    assert pbr_hints["images"] is torch.Tensor
    assert pbr_hints["return"] is PBRMaterial
    assert field_hints["field"] is torch.Tensor
