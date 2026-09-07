from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..objects import MeshAsset
from ._optional import load_selected_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


class ScikitImageBackend:
    """Portable, non-differentiable marching cubes on detached CPU fields."""

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        module = load_selected_backend(
            "scikit-image",
            "skimage.measure",
            (BackendCapability.SURFACE_EXTRACTION,),
            registry=registry,
        )
        self._marching_cubes = module.marching_cubes

    def extract_surface(
        self,
        field: torch.Tensor,
        *,
        level: float = 0.0,
        spacing: Sequence[float] | None = None,
        gradient_direction: str = "ascent",
        allow_degenerate: bool = False,
    ) -> MeshAsset:
        """Extract ``field >= level`` with outward right-handed face winding.

        Coordinates follow the three tensor axes and are scaled by ``spacing``.
        The first sample is at world coordinate ``(0, 0, 0)``.
        Generic calls default to outward ``"ascent"`` winding and reject
        degenerate faces. Integrations may explicitly select upstream settings.
        """

        if not isinstance(field, torch.Tensor):
            raise TypeError("field must be a torch.Tensor")
        if field.ndim != 3:
            raise ValueError(f"field must be a dense rank-3 scalar tensor, got shape {tuple(field.shape)}")
        if not field.is_floating_point():
            raise TypeError(f"field must have a floating-point dtype, got {field.dtype}")
        if any(size < 2 for size in field.shape):
            raise ValueError("every field dimension must contain at least two samples")
        if not isinstance(level, (int, float)) or not math.isfinite(float(level)):
            raise ValueError("level must be a finite number")

        resolved_spacing = (1.0, 1.0, 1.0) if spacing is None else tuple(float(value) for value in spacing)
        if len(resolved_spacing) != 3:
            raise ValueError("spacing must contain exactly three values")
        if any(not math.isfinite(value) or value <= 0.0 for value in resolved_spacing):
            raise ValueError("spacing values must be finite and positive")
        if gradient_direction not in ("ascent", "descent"):
            raise ValueError("gradient_direction must be 'ascent' or 'descent'")
        if type(allow_degenerate) is not bool:
            raise TypeError("allow_degenerate must be a bool")

        cpu_field = field.detach().cpu().float()
        minimum = float(cpu_field.min())
        maximum = float(cpu_field.max())
        if not minimum <= float(level) <= maximum:
            raise ValueError(f"level {float(level)} must lie within the field range [{minimum}, {maximum}]")
        if minimum == maximum:
            raise ValueError("surface extraction requires a non-constant scalar field")

        vertices, faces, normals, _ = self._marching_cubes(
            cpu_field.numpy(),
            level=float(level),
            spacing=resolved_spacing,
            gradient_direction=gradient_direction,
            allow_degenerate=allow_degenerate,
            method="lewiner",
        )
        return MeshAsset(
            vertices=torch.from_numpy(vertices.copy()).to(torch.float32),
            faces=torch.from_numpy(faces.copy()).to(torch.int64),
            normals=torch.from_numpy(normals.copy()).to(torch.float32),
        )


__all__ = ["ScikitImageBackend"]
