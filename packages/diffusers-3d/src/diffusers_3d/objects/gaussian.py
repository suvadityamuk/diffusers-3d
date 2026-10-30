from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import Any

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
from .coordinates import changes_handedness, coordinate_change_matrix
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
        self.transform = follow_device(self.means, self.transform)
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

    def to_coordinate_system(self, target: CoordinateSystem | str) -> GaussianSplatAsset:
        """Rotate the Gaussians into another right-handed convention (for example TRELLIS Z-up to Y-up).

        Only degree-0 colour is rotation invariant, so assets with non-zero higher-order SH coefficients raise.
        """

        target = normalize_coordinate_system(target)
        if target is self.coordinate_system:
            return self
        if changes_handedness(self.coordinate_system, target):
            raise ValueError("Gaussian splats can only be rotated between conventions of the same handedness")
        if self.sh_coefficients.shape[1] > 1 and bool(self.sh_coefficients[:, 1:].ne(0).any()):
            raise ValueError("rotating higher-order spherical harmonics is not supported")
        matrix = coordinate_change_matrix(self.coordinate_system, target, device=self.means.device)
        rotation = _matrix_to_quaternion(matrix).to(self.quaternions_wxyz)
        matrix4 = torch.eye(4, device=self.transform.device, dtype=self.transform.dtype)
        matrix4[:3, :3] = matrix.to(dtype=self.transform.dtype)
        values = {info.name: getattr(self, info.name) for info in fields(self) if info.init}
        values.update(
            means=self.means @ matrix.to(self.means.dtype).T,
            quaternions_wxyz=_quaternion_multiply(rotation.expand_as(self.quaternions_wxyz), self.quaternions_wxyz),
            transform=matrix4 @ self.transform @ matrix4.T,
            coordinate_system=target,
        )
        return GaussianSplatAsset(**values)

    def save(self, file: str | os.PathLike[str] | Any) -> None:
        """Write a 3D Gaussian Splatting PLY, oriented the way the model's upstream writes it.

        When ``metadata["export_rotation"]`` holds a proper 3x3 rotation ``R``, means become ``means @ R.T`` and the
        orientations rotate with them (higher-order SH must be zero); otherwise the file is right-handed Y-up.
        """

        from ..backends.splat_ply import write_splat_ply

        rotation = self.metadata.get("export_rotation")
        if rotation is None:
            write_splat_ply(file, self.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP))
            return
        matrix = torch.tensor(rotation, dtype=torch.float32, device=self.means.device)
        if self.sh_coefficients.shape[1] > 1 and bool(self.sh_coefficients[:, 1:].ne(0).any()):
            raise ValueError("rotating higher-order spherical harmonics is not supported")
        values = {info.name: getattr(self, info.name) for info in fields(self) if info.init}
        values.update(
            means=self.means @ matrix.to(self.means.dtype).T,
            quaternions_wxyz=_quaternion_multiply(
                _matrix_to_quaternion(matrix).to(self.quaternions_wxyz).expand_as(self.quaternions_wxyz),
                self.quaternions_wxyz,
            ),
        )
        write_splat_ply(file, GaussianSplatAsset(**values))

    @classmethod
    def load(cls, file: str | os.PathLike[str] | Any) -> GaussianSplatAsset:
        """Read a 3D Gaussian Splatting PLY as right-handed Y-up CPU tensors."""

        from ..backends.splat_ply import read_splat_ply

        return read_splat_ply(file)

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


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """``(w, x, y, z)`` of a proper rotation matrix (Shepperd's method, stable for every rotation)."""

    m = matrix.double()
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    candidates = torch.stack(
        [
            torch.stack([1 + trace, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]),
            torch.stack([m[2, 1] - m[1, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]]),
            torch.stack([m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], 1 - m[0, 0] + m[1, 1] - m[2, 2], m[1, 2] + m[2, 1]]),
            torch.stack([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], 1 - m[0, 0] - m[1, 1] + m[2, 2]]),
        ]
    )
    best = candidates[torch.argmax(torch.stack([trace, m[0, 0], m[1, 1], m[2, 2]]))]
    return (best / best.norm()).float()


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


__all__ = ["GaussianSplatAsset", "QUATERNION_NORM_TOLERANCE"]
