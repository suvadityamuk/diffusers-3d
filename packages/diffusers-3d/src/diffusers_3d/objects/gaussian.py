from __future__ import annotations

import math
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
from .types import CoordinateSystem, Metadata, Object3DKind

QUATERNION_NORM_TOLERANCE = 1e-3


@dataclass
class GaussianSplatAsset(BaseOutput, TensorDataMixin):
    """Anisotropic 3D Gaussians with opacity logits and RGB spherical harmonics."""

    means: torch.Tensor
    log_scales: torch.Tensor
    quaternions_wxyz: torch.Tensor
    opacity_logits: torch.Tensor
    sh_coefficients: torch.Tensor
    active_sh_degree: int
    transform: torch.Tensor = field(default_factory=identity_transform)
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
        return Object3DKind.GAUSSIAN_SPLAT

    @property
    def object_to_world(self) -> torch.Tensor:
        return self.transform

    def validate(self, expensive: bool = False) -> None:
        del expensive
        if not isinstance(self.coordinate_system, CoordinateSystem):
            raise Object3DValidationError("coordinate_system must be a CoordinateSystem")
        normalize_metadata(self.metadata)
        validate_tensor("means", self.means, rank=2, trailing_shape=(3,), floating=True)
        count = self.means.shape[0]
        if count == 0:
            raise TensorShapeError("means must contain at least one Gaussian")

        validate_tensor("log_scales", self.log_scales, rank=2, trailing_shape=(3,), floating=True)
        validate_tensor("quaternions_wxyz", self.quaternions_wxyz, rank=2, trailing_shape=(4,), floating=True)
        validate_tensor("opacity_logits", self.opacity_logits, floating=True)
        validate_tensor("sh_coefficients", self.sh_coefficients, rank=3, floating=True)
        if self.log_scales.shape[0] != count:
            raise TensorShapeError("log_scales must have one row per Gaussian")
        if self.quaternions_wxyz.shape[0] != count:
            raise TensorShapeError("quaternions_wxyz must have one row per Gaussian")
        if tuple(self.opacity_logits.shape) not in ((count,), (count, 1)):
            raise TensorShapeError(
                f"opacity_logits must have shape ({count},) or ({count}, 1), got {tuple(self.opacity_logits.shape)}"
            )
        if self.sh_coefficients.shape[0] != count or self.sh_coefficients.shape[2] != 3:
            raise TensorShapeError("sh_coefficients must have shape (num_gaussians, num_coefficients, 3)")

        quaternion_norms = torch.linalg.vector_norm(self.quaternions_wxyz.float(), dim=1)
        if not bool(
            torch.allclose(
                quaternion_norms, torch.ones_like(quaternion_norms), atol=QUATERNION_NORM_TOLERANCE, rtol=0.0
            )
        ):
            raise Object3DValidationError(f"quaternions_wxyz must be unit length within {QUATERNION_NORM_TOLERANCE}")
        if not bool(torch.isfinite(torch.exp(self.log_scales)).all()):
            raise Object3DValidationError("log_scales must represent finite positive scales")

        coefficient_count = self.sh_coefficients.shape[1]
        maximum_degree = math.isqrt(coefficient_count) - 1
        if (maximum_degree + 1) ** 2 != coefficient_count:
            raise TensorShapeError("sh_coefficients coefficient count must be a squared SH basis size")
        if isinstance(self.active_sh_degree, bool) or not isinstance(self.active_sh_degree, int):
            raise Object3DValidationError("active_sh_degree must be an integer")
        if not 0 <= self.active_sh_degree <= maximum_degree:
            raise Object3DValidationError(
                f"active_sh_degree must be between 0 and {maximum_degree} for {coefficient_count} coefficients"
            )

        validate_transform("transform", self.transform)
        validate_extras(self.extras, allowed_first_dimensions={count})
        validate_shared_device(self.tensor_items())


__all__ = ["GaussianSplatAsset", "QUATERNION_NORM_TOLERANCE"]
