from __future__ import annotations

from dataclasses import dataclass, field

import torch
from diffusers.utils import BaseOutput

from ._validation import (
    Object3DValidationError,
    TensorShapeError,
    follow_device,
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
from .types import CoordinateSystem, Metadata, Object3DKind


@dataclass
class RadianceFieldAsset(BaseOutput, TensorDataMixin):
    """Sparse voxel radiance field with a rank-decomposed (tri-vector) field inside every active voxel.

    Each active voxel ``n`` holds ``rank`` components. Component ``r`` has one ``dim``-sample vector per axis
    (``trivec[n, r, axis]``); its value at local coordinates ``(x, y, z)`` in ``[0, 1]^3`` is the product of the
    three linearly interpolated vectors. Density is ``softplus(sum_r density[n, r] * component_r)`` and colour
    ``sigmoid(sum_r color_coefficients[n, r] * component_r)`` with the coefficients being degree-0 spherical
    harmonics (scaled by ``0.2820948``). This is the TRELLIS ``Strivec`` layout, stored without any renderer.

    ``grid_transform`` maps integer grid coordinates to voxel centres in object space, the same convention as the
    TRELLIS :class:`SparseVoxelAsset` (each voxel spans half a grid unit around its coordinate); ``transform``
    maps object space to world space.
    """

    coordinates: torch.Tensor
    trivec: torch.Tensor
    density: torch.Tensor
    color_coefficients: torch.Tensor
    resolution: int
    grid_transform: torch.Tensor
    transform: torch.Tensor = field(default_factory=identity_transform)
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Y_UP
    density_shift: float = 0.0
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinate_system = normalize_coordinate_system(self.coordinate_system)
        self.transform = follow_device(self.coordinates, self.transform)
        self.grid_transform = follow_device(self.coordinates, self.grid_transform)
        self.extras = normalize_extras(self.extras)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    @property
    def kind(self) -> Object3DKind:
        return Object3DKind.RADIANCE_FIELD

    @property
    def object_to_world(self) -> torch.Tensor:
        return self.transform

    @property
    def rank(self) -> int:
        return int(self.trivec.shape[1])

    @property
    def dim(self) -> int:
        return int(self.trivec.shape[3])

    def validate(self, expensive: bool = False) -> None:
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, int) or self.resolution <= 0:
            raise Object3DValidationError("resolution must be a positive integer")
        if not isinstance(self.density_shift, (int, float)) or isinstance(self.density_shift, bool):
            raise Object3DValidationError("density_shift must be a number")
        validate_tensor("coordinates", self.coordinates, rank=2, trailing_shape=(3,), integer=True, finite=False)
        count = self.coordinates.shape[0]
        if count == 0:
            raise TensorShapeError("coordinates must contain at least one active voxel")
        if bool((self.coordinates < 0).any()) or bool((self.coordinates >= self.resolution).any()):
            raise Object3DValidationError("coordinates must lie inside the resolution^3 grid")
        validate_tensor("trivec", self.trivec, rank=4, floating=True)
        if self.trivec.shape[0] != count or self.trivec.shape[2] != 3 or self.trivec.shape[3] < 2:
            raise TensorShapeError("trivec must have shape (num_voxels, rank, 3, dim) with dim >= 2")
        rank = self.trivec.shape[1]
        validate_tensor("density", self.density, rank=2, floating=True)
        if tuple(self.density.shape) != (count, rank):
            raise TensorShapeError("density must have shape (num_voxels, rank)")
        validate_tensor("color_coefficients", self.color_coefficients, rank=3, floating=True)
        if tuple(self.color_coefficients.shape) != (count, rank, 3):
            raise TensorShapeError("color_coefficients must have shape (num_voxels, rank, 3)")
        validate_transform("grid_transform", self.grid_transform)
        validate_transform("transform", self.transform)
        validate_extras(self.extras, allowed_first_dimensions={count})
        validate_shared_device(self.tensor_items())
        if expensive and torch.unique(self.coordinates, dim=0).shape[0] != count:
            raise Object3DValidationError("coordinates must not contain duplicate active voxels")


__all__ = ["RadianceFieldAsset"]
