from __future__ import annotations

from collections.abc import Sequence

import torch

from ..objects import CoordinateSystem, MeshAsset
from ._optional import load_explicit_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


class KaolinFlexiCubesBackend:
    """Explicit surface adapter limited to Apache-2.0 ``kaolin.ops`` FlexiCubes."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        kaolin = load_explicit_backend(
            "kaolin",
            "kaolin",
            (BackendCapability.SURFACE_EXTRACTION,),
            device=device,
            dtype=dtype,
            differentiable=True,
            registry=registry,
        )
        conversions = getattr(getattr(kaolin, "ops", None), "conversions", None)
        flexicubes_type = getattr(conversions, "FlexiCubes", None)
        if not callable(flexicubes_type):
            raise RuntimeError("the selected Kaolin build does not expose kaolin.ops.conversions.FlexiCubes")
        module_name = getattr(flexicubes_type, "__module__", "")
        if "non_commercial" in module_name or not module_name.startswith("kaolin.ops"):
            raise RuntimeError("KaolinFlexiCubesBackend only permits the Apache-2.0 kaolin.ops implementation")
        self._extractor = flexicubes_type(device=str(self.device))

    def extract_surface(
        self,
        field: torch.Tensor,
        *,
        level: float = 0.0,
        spacing: Sequence[float] | None = None,
    ) -> MeshAsset:
        """Extract a differentiable triangle mesh from one cubic scalar grid."""

        if not isinstance(field, torch.Tensor) or field.ndim != 3:
            raise ValueError("field must be a rank-three tensor")
        if len(set(field.shape)) != 1 or field.shape[0] < 2:
            raise ValueError("Kaolin FlexiCubes requires a cubic field with at least two samples per axis")
        if not field.is_floating_point():
            raise ValueError("field must have a floating dtype")
        if field.device.type != self.device.type:
            raise ValueError(f"KaolinFlexiCubesBackend was configured for {self.device.type}, got {field.device.type}")
        if field.dtype is not self.dtype:
            raise ValueError(f"KaolinFlexiCubesBackend was configured for {self.dtype}, got {field.dtype}")
        if spacing is None:
            resolved_spacing = (1.0, 1.0, 1.0)
        else:
            resolved_spacing = tuple(float(value) for value in spacing)
            if len(resolved_spacing) != 3 or any(value <= 0 for value in resolved_spacing):
                raise ValueError("spacing must contain three positive values")

        resolution = field.shape[0] - 1
        voxel_vertices, cube_indices = self._extractor.construct_voxel_grid(resolution)
        result = self._extractor(
            voxel_vertices,
            (field - float(level)).reshape(-1),
            cube_indices,
            resolution,
            training=torch.is_grad_enabled(),
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("Kaolin FlexiCubes returned an unsupported result")
        vertices, faces = result[:2]
        spacing_tensor = vertices.new_tensor(resolved_spacing)
        vertices = (vertices + 0.5) * resolution * spacing_tensor
        return MeshAsset(
            vertices=vertices,
            faces=faces.to(dtype=torch.int64),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
            metadata={
                "surface_backend": "kaolin-flexicubes",
                "level": float(level),
            },
        )


__all__ = ["KaolinFlexiCubesBackend"]
