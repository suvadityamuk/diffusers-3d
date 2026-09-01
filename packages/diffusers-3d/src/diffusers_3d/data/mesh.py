from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..objects import CoordinateSystem, MeshAsset, Metadata, PBRMaterial
from ..objects._validation import (
    Object3DValidationError,
    TensorDTypeError,
    TensorShapeError,
    validate_shared_device,
    validate_tensor,
    validate_transform,
)
from ..objects.base import TensorDataMixin


@dataclass(frozen=True, slots=True)
class PackedMeshBatch(TensorDataMixin):
    """Losslessly packed mesh topology with per-mesh representation channels."""

    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_offsets: torch.Tensor
    face_offsets: torch.Tensor
    transforms: torch.Tensor
    coordinate_systems: tuple[CoordinateSystem, ...]
    normals: tuple[torch.Tensor | None, ...]
    colors: tuple[torch.Tensor | None, ...]
    uvs: tuple[torch.Tensor | None, ...]
    face_material_ids: tuple[torch.Tensor | None, ...]
    materials: tuple[tuple[PBRMaterial, ...], ...]
    extras: tuple[dict[str, torch.Tensor], ...]
    metadata: tuple[Metadata, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return len(self.coordinate_systems)

    @classmethod
    def pack(cls, meshes: Sequence[MeshAsset]) -> PackedMeshBatch:
        if isinstance(meshes, (str, bytes)) or not isinstance(meshes, Sequence):
            raise Object3DValidationError("meshes must be a sequence of MeshAsset values")
        meshes = tuple(meshes)
        if not meshes:
            raise Object3DValidationError("meshes must contain at least one MeshAsset")
        if any(type(mesh) is not MeshAsset for mesh in meshes):
            raise Object3DValidationError("meshes must contain exact MeshAsset values")

        for mesh in meshes:
            mesh.validate(expensive=True)
        first = meshes[0]
        for index, mesh in enumerate(meshes[1:], start=1):
            if mesh.device != first.device:
                raise Object3DValidationError(f"meshes[{index}] is on {mesh.device}, expected {first.device}")
            if mesh.vertices.dtype != first.vertices.dtype:
                raise TensorDTypeError("all packed mesh vertices must have the same dtype")
            if mesh.faces.dtype != first.faces.dtype:
                raise TensorDTypeError("all packed mesh faces must have the same dtype")
            if mesh.transform.dtype != first.transform.dtype:
                raise TensorDTypeError("all packed mesh transforms must have the same dtype")

        vertex_counts = [mesh.vertices.shape[0] for mesh in meshes]
        face_counts = [mesh.faces.shape[0] for mesh in meshes]
        vertex_offsets = torch.tensor(
            [0, *torch.tensor(vertex_counts, dtype=torch.int64).cumsum(0).tolist()],
            dtype=torch.int64,
            device=first.device,
        )
        face_offsets = torch.tensor(
            [0, *torch.tensor(face_counts, dtype=torch.int64).cumsum(0).tolist()],
            dtype=torch.int64,
            device=first.device,
        )
        packed_faces = torch.cat(
            [mesh.faces + vertex_offsets[index].to(dtype=mesh.faces.dtype) for index, mesh in enumerate(meshes)]
        )

        return cls(
            vertices=torch.cat([mesh.vertices for mesh in meshes]),
            faces=packed_faces,
            vertex_offsets=vertex_offsets,
            face_offsets=face_offsets,
            transforms=torch.stack([mesh.transform for mesh in meshes]),
            coordinate_systems=tuple(mesh.coordinate_system for mesh in meshes),
            normals=tuple(mesh.normals for mesh in meshes),
            colors=tuple(mesh.colors for mesh in meshes),
            uvs=tuple(mesh.uvs for mesh in meshes),
            face_material_ids=tuple(mesh.face_material_ids for mesh in meshes),
            materials=tuple(mesh.materials for mesh in meshes),
            extras=tuple(dict(mesh.extras) for mesh in meshes),
            metadata=tuple(dict(mesh.metadata) for mesh in meshes),
        )

    def unpack(self) -> tuple[MeshAsset, ...]:
        meshes = []
        for index in range(self.batch_size):
            vertex_start = int(self.vertex_offsets[index])
            vertex_end = int(self.vertex_offsets[index + 1])
            face_start = int(self.face_offsets[index])
            face_end = int(self.face_offsets[index + 1])
            meshes.append(
                MeshAsset(
                    vertices=self.vertices[vertex_start:vertex_end],
                    faces=self.faces[face_start:face_end] - vertex_start,
                    transform=self.transforms[index],
                    coordinate_system=self.coordinate_systems[index],
                    normals=self.normals[index],
                    colors=self.colors[index],
                    uvs=self.uvs[index],
                    face_material_ids=self.face_material_ids[index],
                    materials=self.materials[index],
                    extras=self.extras[index],
                    metadata=self.metadata[index],
                )
            )
        return tuple(meshes)

    def validate(self, expensive: bool = False) -> None:
        validate_tensor("vertices", self.vertices, rank=2, trailing_shape=(3,), floating=True)
        validate_tensor("faces", self.faces, rank=2, trailing_shape=(3,), integer=True, finite=False)
        validate_tensor("vertex_offsets", self.vertex_offsets, rank=1, integer=True, finite=False)
        validate_tensor("face_offsets", self.face_offsets, rank=1, integer=True, finite=False)
        if self.vertex_offsets.dtype is not torch.int64 or self.face_offsets.dtype is not torch.int64:
            raise TensorDTypeError("vertex_offsets and face_offsets must use torch.int64")
        validate_transform("transforms", self.transforms, batched=True)

        batch_size = len(self.coordinate_systems)
        if batch_size == 0:
            raise TensorShapeError("PackedMeshBatch must contain at least one mesh")
        if self.transforms.shape[0] != batch_size:
            raise TensorShapeError("transforms must contain one transform per mesh")
        tuple_fields = (
            self.normals,
            self.colors,
            self.uvs,
            self.face_material_ids,
            self.materials,
            self.extras,
            self.metadata,
        )
        if any(not isinstance(value, tuple) or len(value) != batch_size for value in tuple_fields):
            raise TensorShapeError("all per-mesh fields must be tuples with batch_size entries")
        if any(type(system) is not CoordinateSystem for system in self.coordinate_systems):
            raise Object3DValidationError("coordinate_systems must contain exact CoordinateSystem values")

        for name, offsets, total in (
            ("vertex_offsets", self.vertex_offsets, self.vertices.shape[0]),
            ("face_offsets", self.face_offsets, self.faces.shape[0]),
        ):
            if offsets.shape[0] != batch_size + 1:
                raise TensorShapeError(f"{name} must have batch_size + 1 entries")
            if int(offsets[0]) != 0 or int(offsets[-1]) != total:
                raise TensorShapeError(f"{name} must start at zero and end at the packed element count")
            if bool((offsets[1:] <= offsets[:-1]).any()):
                raise TensorShapeError(f"{name} must be strictly increasing")

        validate_shared_device(self.tensor_items())
        for mesh in self.unpack():
            mesh.validate(expensive=True if expensive else False)


__all__ = ["PackedMeshBatch"]
