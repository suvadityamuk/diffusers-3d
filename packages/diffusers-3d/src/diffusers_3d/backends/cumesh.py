from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import torch

from ..objects import MeshAsset
from ._optional import require


def _call(native: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(native, name, None)
    if not callable(function):
        raise RuntimeError(f"the selected CuMesh build does not expose {name}")
    return function(*args, **kwargs)


class CuMeshBackend:
    """CuMesh repair, simplify, remesh, UV-unwrap, and BVH adapter."""

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self._module = require("cumesh", "cumesh")
        warnings.warn(
            "CuMesh bundles cubvh, which NVIDIA licenses for non-commercial research and evaluation use only. "
            "You are responsible for making sure your use of it complies with that license.",
            UserWarning,
            stacklevel=2,
        )

    def _native(self, mesh: MeshAsset) -> Any:
        if type(mesh) is not MeshAsset:
            raise TypeError("mesh must be an exact MeshAsset")
        mesh.validate(expensive=True)
        if mesh.device.type != self.device.type or (
            self.device.index is not None and mesh.device.index != self.device.index
        ):
            raise ValueError(f"CuMeshBackend was configured for {self.device}, got {mesh.device}")
        if mesh.vertices.dtype is not torch.float32:
            raise ValueError("CuMeshBackend requires float32 vertices")
        native = self._module.CuMesh()
        native.init(mesh.vertices, mesh.faces.to(dtype=torch.int32))
        return native

    def _asset(
        self,
        template: MeshAsset,
        native: Any,
        *,
        uvs: torch.Tensor | None = None,
        extras: Mapping[str, torch.Tensor] | None = None,
        operation: str,
    ) -> MeshAsset:
        values = native.read()
        if not isinstance(values, tuple) or len(values) < 2:
            raise RuntimeError("CuMesh.read() must return vertices and faces")
        vertices, faces = values[:2]
        metadata = {**template.metadata, "cumesh_operation": operation}
        return MeshAsset(
            vertices=vertices,
            faces=faces.to(dtype=torch.int64),
            transform=template.transform,
            coordinate_system=template.coordinate_system,
            uvs=uvs,
            extras={} if extras is None else dict(extras),
            metadata=metadata,
        )

    def repair(
        self,
        mesh: MeshAsset,
        *,
        max_hole_perimeter: float = 3e-2,
        minimum_component_area: float = 1e-5,
        unify_face_orientations: bool = True,
    ) -> MeshAsset:
        """Remove duplicate faces, non-manifold edges, and small components, fill holes, and unify winding."""

        native = self._native(mesh)
        _call(native, "remove_duplicate_faces")
        _call(native, "repair_non_manifold_edges")
        _call(native, "remove_small_connected_components", float(minimum_component_area))
        _call(native, "fill_holes", max_hole_perimeter=float(max_hole_perimeter))
        if unify_face_orientations:
            _call(native, "unify_face_orientations")
        return self._asset(mesh, native, operation="repair")

    def simplify(self, mesh: MeshAsset, target_faces: int, *, verbose: bool = False) -> MeshAsset:
        if not isinstance(target_faces, int) or isinstance(target_faces, bool) or target_faces <= 0:
            raise ValueError("target_faces must be a positive integer")
        native = self._native(mesh)
        native.simplify(target_faces, verbose=verbose)
        return self._asset(mesh, native, operation="simplify")

    def remesh(
        self,
        mesh: MeshAsset,
        resolution: int,
        *,
        center: torch.Tensor | None = None,
        scale: float | None = None,
        band: float = 1.0,
        project_back: float = 0.9,
        verbose: bool = False,
    ) -> MeshAsset:
        """Narrow-band dual-contouring remesh at ``resolution`` around ``center`` (default: the vertex mean)."""

        native = self._native(mesh)
        center = mesh.vertices.mean(dim=0) if center is None else center
        scale = float((mesh.vertices.amax(dim=0) - mesh.vertices.amin(dim=0)).max()) if scale is None else scale
        remeshed = self._module.remeshing.remesh_narrow_band_dc(
            mesh.vertices,
            mesh.faces,
            center=torch.as_tensor(center, device=mesh.device, dtype=torch.float32),
            scale=float(scale),
            resolution=int(resolution),
            band=float(band),
            project_back=float(project_back),
            verbose=verbose,
            bvh=self.build_bvh(mesh),
        )
        native.init(*remeshed)
        return self._asset(mesh, native, operation="remesh")

    def unwrap_uv(
        self, mesh: MeshAsset, *, compute_charts_kwargs: Mapping[str, Any] | None = None, verbose: bool = False
    ) -> MeshAsset:
        """UV-unwrap ``mesh``; ``extras["cumesh_vertex_map"]`` maps each output vertex to its input vertex."""

        native = self._native(mesh)
        result = native.uv_unwrap(
            compute_charts_kwargs=dict(compute_charts_kwargs or {}), return_vmaps=True, verbose=verbose
        )
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("CuMesh.uv_unwrap() must return vertices, faces, uvs, and vertex maps")
        vertices, faces, uvs, vertex_maps = result
        native.init(vertices, faces)
        return self._asset(
            mesh,
            native,
            uvs=uvs,
            extras={"cumesh_vertex_map": vertex_maps.to(dtype=torch.int64)},
            operation="uv_unwrap",
        )

    def build_bvh(self, mesh: MeshAsset) -> Any:
        self._native(mesh)  # validates the mesh
        return self._module.cuBVH(mesh.vertices, mesh.faces.to(dtype=torch.int32))

    def unsigned_distance(
        self,
        mesh: MeshAsset,
        points: torch.Tensor,
        *,
        return_uvw: bool = False,
    ) -> Any:
        if not isinstance(points, torch.Tensor) or points.shape[-1] != 3 or not points.is_floating_point():
            raise ValueError("points must be a floating-point tensor ending in XYZ")
        if points.device != mesh.device:
            raise ValueError("points and mesh must be on the same device")
        bvh = self.build_bvh(mesh)
        function = getattr(bvh, "unsigned_distance", None)
        if not callable(function):
            raise RuntimeError("the selected CuMesh BVH does not expose unsigned_distance")
        return function(points, return_uvw=return_uvw)


__all__ = ["CuMeshBackend"]
