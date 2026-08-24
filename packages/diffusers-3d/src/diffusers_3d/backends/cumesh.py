from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..objects import MeshAsset
from ._optional import load_explicit_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability

CUMESH_SOURCE_URL = "https://github.com/JeffreyXiang/CuMesh.git"
CUMESH_SOURCE_REVISION = "12289e1062f0603f2f0d0771b02e1395d247f26f"


class CuMeshBackend:
    """Explicit source-attested CuMesh repair, simplify, remesh, UV, and BVH adapter."""

    def __init__(
        self,
        *,
        source_revision: str = CUMESH_SOURCE_REVISION,
        build_id: str,
        source_url: str = CUMESH_SOURCE_URL,
        device: str | torch.device = "cuda",
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        if source_url != CUMESH_SOURCE_URL:
            raise ValueError(f"CuMesh source_url must be {CUMESH_SOURCE_URL!r}")
        if source_revision != CUMESH_SOURCE_REVISION:
            raise ValueError(f"CuMesh source_revision must be {CUMESH_SOURCE_REVISION!r}")
        if not isinstance(build_id, str) or not build_id.strip():
            raise ValueError("build_id must record the PyTorch/CUDA/compiler build")
        self.source_url = source_url
        self.source_revision = source_revision
        self.build_id = build_id
        self.device = torch.device(device)
        self._module = load_explicit_backend(
            "cumesh",
            "cumesh",
            (BackendCapability.GEOMETRY_PROCESSING, BackendCapability.CONVERSION),
            device=device,
            dtype=torch.float32,
            differentiable=False,
            registry=registry,
        )
        declared_revision = getattr(self._module, "__source_revision__", None)
        declared_build = getattr(self._module, "__build_id__", None)
        if declared_revision is None or declared_build is None:
            raise RuntimeError(
                "the selected CuMesh wrapper must expose __source_revision__ and __build_id__ attestations"
            )
        if declared_revision != source_revision:
            raise RuntimeError(f"loaded CuMesh declares revision {declared_revision!r}, expected {source_revision!r}")
        if declared_build != build_id:
            raise RuntimeError(f"loaded CuMesh declares build {declared_build!r}, expected {build_id!r}")
        if not callable(getattr(self._module, "CuMesh", None)):
            raise RuntimeError("the selected CuMesh build does not expose CuMesh")

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
        return MeshAsset(
            vertices=vertices,
            faces=faces.to(dtype=torch.int64),
            transform=template.transform,
            coordinate_system=template.coordinate_system,
            uvs=uvs,
            extras={} if extras is None else dict(extras),
            metadata={
                **template.metadata,
                "cumesh_operation": operation,
                "cumesh_source_revision": self.source_revision,
                "cumesh_build_id": self.build_id,
            },
        )

    def process_geometry(
        self,
        mesh: MeshAsset,
        *,
        operation: str,
        parameters: Mapping[str, object] | None = None,
    ) -> MeshAsset:
        values = {} if parameters is None else dict(parameters)
        native = self._native(mesh)
        if operation == "repair":
            unknown = set(values).difference(
                {"max_hole_perimeter", "minimum_component_area", "unify_face_orientations"}
            )
            if unknown:
                raise ValueError(f"unknown repair parameters: {sorted(unknown)}")
            for name in ("remove_duplicate_faces", "repair_non_manifold_edges"):
                function = getattr(native, name, None)
                if not callable(function):
                    raise RuntimeError(f"the selected CuMesh build does not expose {name}")
                function()
            function = getattr(native, "remove_small_connected_components", None)
            if not callable(function):
                raise RuntimeError("the selected CuMesh build does not expose remove_small_connected_components")
            function(float(values.get("minimum_component_area", 1e-5)))
            function = getattr(native, "fill_holes", None)
            if not callable(function):
                raise RuntimeError("the selected CuMesh build does not expose fill_holes")
            function(max_hole_perimeter=float(values.get("max_hole_perimeter", 3e-2)))
            if bool(values.get("unify_face_orientations", True)):
                function = getattr(native, "unify_face_orientations", None)
                if not callable(function):
                    raise RuntimeError("the selected CuMesh build does not expose unify_face_orientations")
                function()
            return self._asset(mesh, native, operation=operation)
        if operation == "simplify":
            if set(values).difference({"target_faces", "verbose"}) or "target_faces" not in values:
                raise ValueError("simplify requires target_faces and accepts verbose")
            target = values["target_faces"]
            if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
                raise ValueError("target_faces must be a positive integer")
            native.simplify(target, verbose=bool(values.get("verbose", False)))
            return self._asset(mesh, native, operation=operation)
        if operation == "remesh":
            unknown = set(values).difference({"center", "scale", "resolution", "band", "project_back", "verbose"})
            if unknown or "resolution" not in values:
                raise ValueError(
                    "remesh requires resolution and accepts center, scale, band, project_back, and verbose"
                )
            remesh = getattr(getattr(self._module, "remeshing", None), "remesh_narrow_band_dc", None)
            if not callable(remesh):
                raise RuntimeError("the selected CuMesh build does not expose remeshing.remesh_narrow_band_dc")
            center = values.get("center", mesh.vertices.mean(dim=0))
            scale = float(values.get("scale", (mesh.vertices.amax(dim=0) - mesh.vertices.amin(dim=0)).max()))
            remeshed = remesh(
                mesh.vertices,
                mesh.faces,
                center=torch.as_tensor(center, device=mesh.device, dtype=torch.float32),
                scale=scale,
                resolution=int(values["resolution"]),
                band=float(values.get("band", 1.0)),
                project_back=float(values.get("project_back", 0.9)),
                verbose=bool(values.get("verbose", False)),
                bvh=self.build_bvh(mesh),
            )
            native.init(*remeshed)
            return self._asset(mesh, native, operation=operation)
        if operation == "uv_unwrap":
            unknown = set(values).difference({"compute_charts_kwargs", "verbose"})
            if unknown:
                raise ValueError(f"unknown uv_unwrap parameters: {sorted(unknown)}")
            result = native.uv_unwrap(
                compute_charts_kwargs=dict(values.get("compute_charts_kwargs", {})),
                return_vmaps=True,
                verbose=bool(values.get("verbose", False)),
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
                operation=operation,
            )
        raise ValueError("CuMeshBackend supports 'repair', 'simplify', 'remesh', and 'uv_unwrap'")

    def build_bvh(self, mesh: MeshAsset) -> Any:
        self._native(mesh)
        bvh_type = getattr(self._module, "cuBVH", None)
        if not callable(bvh_type):
            raise RuntimeError("the selected CuMesh build does not expose cuBVH")
        return bvh_type(mesh.vertices, mesh.faces.to(dtype=torch.int32))

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


__all__ = ["CUMESH_SOURCE_REVISION", "CUMESH_SOURCE_URL", "CuMeshBackend"]
