from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

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
    validate_scalar_channel,
    validate_shared_device,
    validate_tensor,
    validate_transform,
)
from .base import TensorDataMixin
from .types import CoordinateSystem, Metadata, Object3DKind


@dataclass
class SparseVoxelAsset(BaseOutput, TensorDataMixin):
    """Sparse integer grid coordinates and aligned feature channels."""

    coordinates: torch.Tensor
    features: torch.Tensor
    voxel_size: float | torch.Tensor | None = None
    grid_transform: torch.Tensor | None = None
    transform: torch.Tensor = field(default_factory=identity_transform)
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Y_UP
    semantic_labels: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinate_system = normalize_coordinate_system(self.coordinate_system)
        self.extras = normalize_extras(self.extras)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    @property
    def kind(self) -> Object3DKind:
        return Object3DKind.SPARSE_VOXEL

    @property
    def object_to_world(self) -> torch.Tensor:
        return self.transform

    def validate(self, expensive: bool = False) -> None:
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        validate_tensor("coordinates", self.coordinates, rank=2, trailing_shape=(3,), integer=True, finite=False)
        validate_tensor("features", self.features, rank=2, floating=True)
        count = self.coordinates.shape[0]
        if count == 0:
            raise TensorShapeError("coordinates must contain at least one active voxel")
        if self.features.shape[0] != count or self.features.shape[1] == 0:
            raise TensorShapeError("features must have shape (num_voxels, num_channels) with at least one channel")

        if (self.voxel_size is None) == (self.grid_transform is None):
            raise Object3DValidationError("exactly one of voxel_size or grid_transform must be provided")
        if isinstance(self.voxel_size, torch.Tensor):
            validate_tensor("voxel_size", self.voxel_size, floating=True)
            if tuple(self.voxel_size.shape) not in ((), (1,), (3,)):
                raise TensorShapeError("voxel_size tensor must be scalar or have shape (1,) or (3,)")
            if bool((self.voxel_size <= 0).any()):
                raise Object3DValidationError("voxel_size must be positive")
        elif self.voxel_size is not None:
            if isinstance(self.voxel_size, bool) or not isinstance(self.voxel_size, Real):
                raise Object3DValidationError("voxel_size must be a positive real number or tensor")
            if not math.isfinite(float(self.voxel_size)) or self.voxel_size <= 0:
                raise Object3DValidationError("voxel_size must be finite and positive")
        if self.grid_transform is not None:
            validate_transform("grid_transform", self.grid_transform)

        validate_transform("transform", self.transform)
        if self.semantic_labels is not None:
            validate_tensor("semantic_labels", self.semantic_labels, rank=1, integer=True, finite=False)
            if self.semantic_labels.shape[0] != count:
                raise TensorShapeError("semantic_labels must have one value per active voxel")
        validate_extras(self.extras, allowed_first_dimensions={count})
        validate_shared_device(self.tensor_items())

        if expensive and torch.unique(self.coordinates, dim=0).shape[0] != count:
            raise Object3DValidationError("coordinates must not contain duplicate active voxels")


@dataclass
class OVoxelAsset(BaseOutput, TensorDataMixin):
    """O-Voxel active grid with either dual-grid topology or intersection data."""

    active_coordinates: torch.Tensor
    base_color: torch.Tensor
    metallic: torch.Tensor
    roughness: torch.Tensor
    dual_grid_vertex_offsets: torch.Tensor | None = None
    dual_grid_topology: torch.Tensor | None = None
    intersection_data: torch.Tensor | None = None
    opacity: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    transform: torch.Tensor = field(default_factory=identity_transform)
    grid_transform: torch.Tensor = field(default_factory=identity_transform)
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Y_UP
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinate_system = normalize_coordinate_system(self.coordinate_system)
        self.extras = normalize_extras(self.extras)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    @property
    def kind(self) -> Object3DKind:
        return Object3DKind.O_VOXEL

    @property
    def object_to_world(self) -> torch.Tensor:
        return self.transform

    def validate(self, expensive: bool = False) -> None:
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        validate_tensor(
            "active_coordinates", self.active_coordinates, rank=2, trailing_shape=(3,), integer=True, finite=False
        )
        count = self.active_coordinates.shape[0]
        if count == 0:
            raise TensorShapeError("active_coordinates must contain at least one active voxel")

        validate_tensor("base_color", self.base_color, rank=2, floating=True)
        if self.base_color.shape[0] != count or self.base_color.shape[1] not in (3, 4):
            raise TensorShapeError("base_color must have shape (num_voxels, 3) or (num_voxels, 4)")
        if bool(((self.base_color < 0) | (self.base_color > 1)).any()):
            raise Object3DValidationError("base_color values must be in [0, 1]")
        for name in ("metallic", "roughness", "opacity"):
            tensor = getattr(self, name)
            if tensor is not None:
                validate_scalar_channel(name, tensor, count)
                if bool(((tensor < 0) | (tensor > 1)).any()):
                    raise Object3DValidationError(f"{name} values must be in [0, 1]")
        if self.normals is not None:
            validate_tensor("normals", self.normals, rank=2, trailing_shape=(3,), floating=True)
            if self.normals.shape[0] != count:
                raise TensorShapeError("normals must have one row per active voxel")
            if bool((torch.linalg.vector_norm(self.normals.float(), dim=1) <= 1e-8).any()):
                raise Object3DValidationError("normals must be non-zero")

        has_offsets = self.dual_grid_vertex_offsets is not None
        has_topology = self.dual_grid_topology is not None
        if has_offsets != has_topology:
            raise Object3DValidationError("dual_grid_vertex_offsets and dual_grid_topology must be provided together")
        if not has_offsets and self.intersection_data is None:
            raise Object3DValidationError("dual-grid vertex offsets/topology or intersection_data must be provided")

        geometry_counts = {count}
        vertex_count = 0
        if self.dual_grid_vertex_offsets is not None:
            validate_tensor("dual_grid_vertex_offsets", self.dual_grid_vertex_offsets, floating=True)
            if self.dual_grid_vertex_offsets.ndim not in (2, 3) or self.dual_grid_vertex_offsets.shape[-1] != 3:
                raise TensorShapeError(
                    "dual_grid_vertex_offsets must have shape (num_vertices, 3) or (num_voxels, K, 3)"
                )
            if self.dual_grid_vertex_offsets.ndim == 3:
                if self.dual_grid_vertex_offsets.shape[0] != count:
                    raise TensorShapeError("dual_grid_vertex_offsets must align with active_coordinates")
                vertex_count = count * self.dual_grid_vertex_offsets.shape[1]
            else:
                vertex_count = self.dual_grid_vertex_offsets.shape[0]
            if vertex_count == 0:
                raise TensorShapeError("dual_grid_vertex_offsets must contain at least one vertex")
            geometry_counts.add(vertex_count)

        if self.dual_grid_topology is not None:
            validate_tensor("dual_grid_topology", self.dual_grid_topology, rank=2, integer=True, finite=False)
            if self.dual_grid_topology.shape[0] == 0 or self.dual_grid_topology.shape[1] not in (2, 3, 4):
                raise TensorShapeError(
                    "dual_grid_topology must have shape (num_elements, 2), (num_elements, 3), or (num_elements, 4)"
                )
            geometry_counts.add(self.dual_grid_topology.shape[0])
            if expensive and (
                bool((self.dual_grid_topology < 0).any()) or bool((self.dual_grid_topology >= vertex_count).any())
            ):
                raise Object3DValidationError("dual_grid_topology contains a vertex index outside the valid range")

        if self.intersection_data is not None:
            validate_tensor("intersection_data", self.intersection_data, floating=True)
            if self.intersection_data.ndim < 2 or self.intersection_data.shape[0] != count:
                raise TensorShapeError("intersection_data must have shape (num_voxels, ...)")

        validate_transform("transform", self.transform)
        validate_transform("grid_transform", self.grid_transform)
        validate_extras(self.extras, allowed_first_dimensions=geometry_counts)
        validate_shared_device(self.tensor_items())

        if expensive and torch.unique(self.active_coordinates, dim=0).shape[0] != count:
            raise Object3DValidationError("active_coordinates must not contain duplicates")


__all__ = ["OVoxelAsset", "SparseVoxelAsset"]
