from __future__ import annotations

from dataclasses import dataclass, field

import torch
from diffusers.utils import BaseOutput

from ._validation import (
    Object3DValidationError,
    TensorShapeError,
    identity_transform,
    normalize_coordinate_system,
    normalize_extras,
    normalize_metadata,
    validate_extras,
    validate_shared_device,
    validate_tensor,
    validate_transform,
)
from .base import TensorDataMixin
from .material import PBRMaterial
from .types import CoordinateSystem, Metadata, Object3DKind


@dataclass
class MeshAsset(BaseOutput, TensorDataMixin):
    """Triangle mesh preserving vertex, face, material, and custom channels."""

    vertices: torch.Tensor
    faces: torch.Tensor
    transform: torch.Tensor = field(default_factory=identity_transform)
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Y_UP
    normals: torch.Tensor | None = None
    colors: torch.Tensor | None = None
    uvs: torch.Tensor | None = None
    face_material_ids: torch.Tensor | None = None
    materials: tuple[PBRMaterial, ...] = ()
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinate_system = normalize_coordinate_system(self.coordinate_system)
        self.materials = tuple(self.materials)
        self.extras = normalize_extras(self.extras)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    @property
    def kind(self) -> Object3DKind:
        return Object3DKind.MESH

    @property
    def object_to_world(self) -> torch.Tensor:
        return self.transform

    def validate(self, expensive: bool = False) -> None:
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        validate_tensor("vertices", self.vertices, rank=2, trailing_shape=(3,), floating=True)
        validate_tensor("faces", self.faces, rank=2, trailing_shape=(3,), integer=True, finite=False)
        if self.vertices.shape[0] == 0:
            raise TensorShapeError("vertices must contain at least one vertex")
        if self.faces.shape[0] == 0:
            raise TensorShapeError("faces must contain at least one triangle")
        validate_transform("transform", self.transform)

        vertex_count = self.vertices.shape[0]
        face_count = self.faces.shape[0]
        if self.normals is not None:
            validate_tensor("normals", self.normals, rank=2, trailing_shape=(3,), floating=True)
            if self.normals.shape[0] != vertex_count:
                raise TensorShapeError("normals must have one row per vertex")
        if self.colors is not None:
            validate_tensor("colors", self.colors, rank=2, floating=True)
            if self.colors.shape[0] != vertex_count or self.colors.shape[1] not in (3, 4):
                raise TensorShapeError("colors must have shape (num_vertices, 3) or (num_vertices, 4)")
            if bool(((self.colors < 0) | (self.colors > 1)).any()):
                raise Object3DValidationError("colors values must be in [0, 1]")
        if self.uvs is not None:
            validate_tensor("uvs", self.uvs, rank=2, trailing_shape=(2,), floating=True)
            if self.uvs.shape[0] != vertex_count:
                raise TensorShapeError("uvs must have one row per vertex")
        if self.face_material_ids is not None:
            validate_tensor("face_material_ids", self.face_material_ids, rank=1, integer=True, finite=False)
            if self.face_material_ids.shape[0] != face_count:
                raise TensorShapeError("face_material_ids must have one value per face")
            if not self.materials:
                raise Object3DValidationError("face_material_ids requires at least one material")

        for index, material in enumerate(self.materials):
            if not isinstance(material, PBRMaterial):
                raise Object3DValidationError(f"materials[{index}] must be a PBRMaterial")
            material.validate(expensive=expensive)

        validate_extras(self.extras, allowed_first_dimensions={vertex_count, face_count})
        validate_shared_device(self.tensor_items())

        if expensive:
            if bool((self.faces < 0).any()) or bool((self.faces >= vertex_count).any()):
                raise Object3DValidationError("faces contains a vertex index outside the valid range")
            if self.face_material_ids is not None and (
                bool((self.face_material_ids < 0).any()) or bool((self.face_material_ids >= len(self.materials)).any())
            ):
                raise Object3DValidationError("face_material_ids contains a material index outside the valid range")


__all__ = ["MeshAsset"]
