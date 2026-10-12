from __future__ import annotations

from dataclasses import dataclass, field

import torch
from diffusers.utils import BaseOutput

from ._validation import (
    Object3DValidationError,
    TensorShapeError,
    normalize_coordinate_system,
    normalize_metadata,
    validate_shared_device,
    validate_tensor,
    validate_transform,
)
from .base import TensorDataMixin
from .types import CoordinateSystem, Metadata


@dataclass
class CameraRig(BaseOutput, TensorDataMixin):
    """A batch of pinhole cameras using world-to-camera extrinsics."""

    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    image_sizes: torch.Tensor
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Y_UP
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinate_system = normalize_coordinate_system(self.coordinate_system)
        self.metadata = normalize_metadata(self.metadata)
        self.validate()
        super().__post_init__()

    def validate(self, expensive: bool = False) -> None:
        del expensive
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        validate_transform("world_to_camera", self.world_to_camera, batched=True)
        validate_tensor("intrinsics", self.intrinsics, rank=3, trailing_shape=(3, 3), floating=True)
        validate_tensor("image_sizes", self.image_sizes, rank=2, trailing_shape=(2,), integer=True, finite=False)
        camera_count = self.world_to_camera.shape[0]
        if camera_count == 0:
            raise TensorShapeError("world_to_camera must contain at least one camera")
        if self.intrinsics.shape[0] != camera_count or self.image_sizes.shape[0] != camera_count:
            raise TensorShapeError("world_to_camera, intrinsics, and image_sizes must have the same camera count")
        if bool((self.intrinsics[:, (0, 1), (0, 1)] <= 0).any()):
            raise Object3DValidationError("intrinsics focal lengths must be positive")
        expected_last_row = self.intrinsics.new_tensor([0.0, 0.0, 1.0]).expand_as(self.intrinsics[:, 2, :])
        if not torch.allclose(self.intrinsics[:, 2, :], expected_last_row, atol=1e-5, rtol=0.0):
            raise Object3DValidationError("intrinsics must be homogeneous pinhole matrices")
        if bool((self.image_sizes <= 0).any()):
            raise Object3DValidationError("image_sizes must be positive (height, width) pairs")
        validate_shared_device(self.tensor_items())


__all__ = ["CameraRig"]
