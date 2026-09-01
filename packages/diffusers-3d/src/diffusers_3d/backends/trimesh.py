from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import torch

from ..objects import CoordinateSystem, MeshAsset, PBRMaterial
from ._optional import load_selected_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability

_METADATA_KEY = "_diffusers_3d"
_COLOR_ATTRIBUTE = "_diffusers_3d_vertex_colors"
_SUPPORTED_FILE_TYPES = frozenset({"glb", "obj", "ply", "stl"})


def _cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        return result if dtype is None else result.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _cpu_material(material: PBRMaterial) -> PBRMaterial:
    return PBRMaterial(
        base_color=material.base_color.detach().cpu(),
        metallic=None if material.metallic is None else material.metallic.detach().cpu(),
        roughness=None if material.roughness is None else material.roughness.detach().cpu(),
        normal=None if material.normal is None else material.normal.detach().cpu(),
        emissive=None if material.emissive is None else material.emissive.detach().cpu(),
        opacity=None if material.opacity is None else material.opacity.detach().cpu(),
        extras={name: value.detach().cpu() for name, value in material.extras.items()},
        metadata=dict(material.metadata),
    )


def _constant_scalar(value: torch.Tensor | None, name: str) -> float | None:
    if value is None:
        return None
    if value.numel() != 1:
        raise ValueError(f"trimesh conversion cannot represent textured PBR channel {name!r}")
    return float(value.detach().cpu().reshape(()))


def _constant_color(value: torch.Tensor, name: str, *, channels: tuple[int, ...]) -> list[float]:
    if value.ndim != 1 or value.shape[0] not in channels:
        raise ValueError(f"trimesh conversion cannot represent textured PBR channel {name!r}")
    return value.detach().cpu().float().tolist()


class TrimeshBackend:
    """Portable CPU mesh conversion, file I/O, and conservative processing.

    Conversion crosses through NumPy and is intentionally non-differentiable.
    Optional ``trimesh`` imports happen only after explicit registry selection.
    """

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        self._trimesh = load_selected_backend(
            "trimesh",
            "trimesh",
            (
                BackendCapability.CONVERSION,
                BackendCapability.SERIALIZATION,
                BackendCapability.GEOMETRY_PROCESSING,
            ),
            registry=registry,
        )

    def _to_trimesh_material(self, material: PBRMaterial) -> Any:
        if material.extras:
            raise ValueError("trimesh conversion cannot represent PBR material extras")
        base_color = _constant_color(material.base_color, "base_color", channels=(3, 4))
        opacity = _constant_scalar(material.opacity, "opacity")
        if len(base_color) == 3:
            base_color.append(1.0 if opacity is None else opacity)
        elif opacity is not None:
            base_color[3] *= opacity
        if material.normal is not None:
            raise ValueError("trimesh conversion cannot represent a constant PBR normal channel")

        emissive = None
        if material.emissive is not None:
            emissive = _constant_color(material.emissive, "emissive", channels=(3,))
        name = material.metadata.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("PBR material metadata 'name' must be a string for trimesh conversion")
        return self._trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=base_color,
            metallicFactor=_constant_scalar(material.metallic, "metallic"),
            roughnessFactor=_constant_scalar(material.roughness, "roughness"),
            emissiveFactor=emissive,
            alphaMode="BLEND" if base_color[3] < 1.0 else None,
        )

    def to_trimesh(self, mesh: MeshAsset) -> Any:
        """Convert a package mesh to a detached CPU ``trimesh.Trimesh``."""

        if not isinstance(mesh, MeshAsset):
            raise TypeError("mesh must be a MeshAsset")
        mesh.validate(expensive=True)
        for name in mesh.extras:
            if name in {_COLOR_ATTRIBUTE, _METADATA_KEY}:
                raise ValueError(f"mesh extra name {name!r} is reserved by the trimesh adapter")

        vertex_attributes: dict[str, Any] = {}
        face_attributes: dict[str, Any] = {}
        vertex_count = mesh.vertices.shape[0]
        face_count = mesh.faces.shape[0]
        for name, value in mesh.extras.items():
            if value.shape[0] == vertex_count and value.shape[0] == face_count:
                raise ValueError(
                    f"mesh extra {name!r} is alignment-ambiguous because vertex and face counts are both "
                    f"{vertex_count}"
                )
            target = vertex_attributes if value.shape[0] == vertex_count else face_attributes
            target[name] = value.detach().cpu().numpy()

        visual = None
        if mesh.colors is not None:
            vertex_attributes[_COLOR_ATTRIBUTE] = mesh.colors.detach().cpu().numpy()
        if mesh.uvs is not None or mesh.materials:
            native_materials = [self._to_trimesh_material(material) for material in mesh.materials]
            material = None
            if len(native_materials) == 1:
                material = native_materials[0]
            elif native_materials:
                material = self._trimesh.visual.material.MultiMaterial(materials=native_materials)
            visual = self._trimesh.visual.TextureVisuals(
                uv=None if mesh.uvs is None else mesh.uvs.detach().cpu().numpy(),
                material=material,
                face_materials=(
                    None if mesh.face_material_ids is None else mesh.face_material_ids.detach().cpu().numpy()
                ),
            )
        elif mesh.colors is not None:
            visual = self._trimesh.visual.ColorVisuals(
                vertex_colors=(mesh.colors.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
            )

        package_metadata = {
            "coordinate_system": mesh.coordinate_system.value,
            "transform": mesh.transform.detach().cpu().tolist(),
            "metadata": dict(mesh.metadata),
            "has_normals": mesh.normals is not None,
            "has_colors": mesh.colors is not None,
            "has_uvs": mesh.uvs is not None,
            "has_face_material_ids": mesh.face_material_ids is not None,
            "material_count": len(mesh.materials),
            "materials": [
                {
                    "base_color_channels": material.base_color.shape[0],
                    "has_metallic": material.metallic is not None,
                    "has_roughness": material.roughness is not None,
                    "has_emissive": material.emissive is not None,
                    "has_opacity": material.opacity is not None,
                    "base_color": material.base_color.detach().cpu().tolist(),
                    "metallic": None if material.metallic is None else material.metallic.detach().cpu().tolist(),
                    "roughness": (None if material.roughness is None else material.roughness.detach().cpu().tolist()),
                    "emissive": None if material.emissive is None else material.emissive.detach().cpu().tolist(),
                    "opacity": None if material.opacity is None else material.opacity.detach().cpu().tolist(),
                    "metadata": dict(material.metadata),
                }
                for material in mesh.materials
            ],
        }
        return self._trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            vertex_normals=None if mesh.normals is None else mesh.normals.detach().cpu().numpy(),
            vertex_attributes=vertex_attributes,
            face_attributes=face_attributes,
            visual=visual,
            metadata={_METADATA_KEY: package_metadata},
            process=False,
            validate=False,
        )

    def _scene_mesh(self, scene: Any) -> tuple[Any, torch.Tensor]:
        nodes = tuple(scene.graph.nodes_geometry)
        if len(nodes) != 1:
            raise ValueError(
                "trimesh scene import requires exactly one geometry instance; "
                f"received {len(nodes)} instances and merging would be lossy"
            )
        transform, geometry_name = scene.graph[nodes[0]]
        return scene.geometry[geometry_name], _cpu_tensor(transform, dtype=torch.float32)

    def _from_trimesh_material(
        self,
        material: Any,
        descriptor: Mapping[str, Any] | None = None,
    ) -> PBRMaterial:
        if descriptor is not None and not isinstance(descriptor, Mapping):
            raise ValueError("trimesh package material metadata is malformed")
        if descriptor is not None and "base_color" in descriptor:
            stored_metadata = descriptor.get("metadata", {})
            if not isinstance(stored_metadata, Mapping):
                raise ValueError("trimesh package material metadata is malformed")

            def stored_tensor(name: str) -> torch.Tensor | None:
                value = descriptor.get(name)
                return None if value is None else _cpu_tensor(value, dtype=torch.float32)

            base_color = stored_tensor("base_color")
            if base_color is None:
                raise ValueError("trimesh package material metadata is missing base_color")
            return PBRMaterial(
                base_color=base_color,
                metallic=stored_tensor("metallic"),
                roughness=stored_tensor("roughness"),
                emissive=stored_tensor("emissive"),
                opacity=stored_tensor("opacity"),
                metadata=dict(stored_metadata),
            )
        pbr_type = self._trimesh.visual.material.PBRMaterial
        if isinstance(material, pbr_type):
            if any(
                getattr(material, name, None) is not None
                for name in (
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                    "normalTexture",
                    "occlusionTexture",
                    "emissiveTexture",
                )
            ):
                raise ValueError("textured trimesh materials cannot be converted without losing texture channels")
            color = _cpu_tensor(material.baseColorFactor, dtype=torch.float32) / 255.0
            base_color_channels = 4 if descriptor is None else int(descriptor.get("base_color_channels", 4))
            if base_color_channels not in (3, 4):
                raise ValueError("trimesh package material metadata has an invalid base color channel count")
            if descriptor is None:
                metadata = {"name": material.name} if material.name else {}
            else:
                stored_metadata = descriptor.get("metadata", {})
                if not isinstance(stored_metadata, Mapping):
                    raise ValueError("trimesh package material metadata is malformed")
                metadata = dict(stored_metadata)
            return PBRMaterial(
                base_color=color[:base_color_channels],
                metallic=(
                    None
                    if material.metallicFactor is None
                    or (descriptor is not None and not descriptor.get("has_metallic", False))
                    else torch.tensor(float(material.metallicFactor), dtype=torch.float32)
                ),
                roughness=(
                    None
                    if material.roughnessFactor is None
                    or (descriptor is not None and not descriptor.get("has_roughness", False))
                    else torch.tensor(float(material.roughnessFactor), dtype=torch.float32)
                ),
                emissive=(
                    None
                    if material.emissiveFactor is None
                    or (descriptor is not None and not descriptor.get("has_emissive", False))
                    else _cpu_tensor(material.emissiveFactor, dtype=torch.float32)
                ),
                opacity=(
                    color[3].clone()
                    if descriptor is not None and descriptor.get("has_opacity", False) and base_color_channels == 3
                    else None
                ),
                metadata=metadata,
            )

        diffuse = getattr(material, "diffuse", None)
        if diffuse is None or getattr(material, "image", None) is not None:
            raise ValueError(f"unsupported trimesh material type: {type(material).__name__}")
        color = _cpu_tensor(diffuse, dtype=torch.float32) / 255.0
        metadata = {"name": material.name} if getattr(material, "name", None) else {}
        return PBRMaterial(base_color=color, metadata=metadata)

    def from_trimesh(
        self,
        native_mesh: Any,
        *,
        coordinate_system: CoordinateSystem | str | None = None,
        transform: torch.Tensor | None = None,
    ) -> MeshAsset:
        """Convert one detached trimesh geometry or scene to ``MeshAsset``."""

        scene_type = self._trimesh.Scene
        mesh_type = self._trimesh.Trimesh
        scene_transform = None
        if isinstance(native_mesh, scene_type):
            native_mesh, scene_transform = self._scene_mesh(native_mesh)
        if not isinstance(native_mesh, mesh_type):
            raise TypeError("native_mesh must be a trimesh.Trimesh or single-instance trimesh.Scene")
        if native_mesh.faces.ndim != 2 or native_mesh.faces.shape[1] != 3:
            raise ValueError("trimesh conversion requires triangular faces")

        package_metadata = native_mesh.metadata.get(_METADATA_KEY, {})
        if not isinstance(package_metadata, Mapping):
            package_metadata = {}
        stored_coordinate_system = package_metadata.get("coordinate_system")
        resolved_coordinate_system = (
            coordinate_system
            if coordinate_system is not None
            else stored_coordinate_system or CoordinateSystem.RIGHT_HANDED_Y_UP
        )
        if transform is not None:
            resolved_transform = transform.detach().cpu()
        elif scene_transform is not None:
            resolved_transform = scene_transform
        elif "transform" in package_metadata:
            resolved_transform = _cpu_tensor(package_metadata["transform"], dtype=torch.float32)
        else:
            resolved_transform = torch.eye(4, dtype=torch.float32)

        extras: dict[str, torch.Tensor] = {}
        for attributes in (native_mesh.vertex_attributes, native_mesh.face_attributes):
            for name, value in attributes.items():
                if name == _COLOR_ATTRIBUTE:
                    continue
                if name in extras:
                    raise ValueError(f"trimesh contains duplicate vertex and face attribute name {name!r}")
                extras[name] = _cpu_tensor(value)

        colors = None
        if _COLOR_ATTRIBUTE in native_mesh.vertex_attributes:
            colors = _cpu_tensor(native_mesh.vertex_attributes[_COLOR_ATTRIBUTE], dtype=torch.float32)
        elif getattr(native_mesh.visual, "kind", None) == "vertex" and native_mesh.visual.defined:
            colors = _cpu_tensor(native_mesh.visual.vertex_colors, dtype=torch.float32) / 255.0

        uvs = None
        materials: tuple[PBRMaterial, ...] = ()
        face_material_ids = None
        if getattr(native_mesh.visual, "kind", None) == "texture":
            if native_mesh.visual.uv is not None:
                uvs = _cpu_tensor(native_mesh.visual.uv, dtype=torch.float32)
            native_material = native_mesh.visual.material
            material_descriptors = package_metadata.get("materials", ())
            if not isinstance(material_descriptors, (list, tuple)):
                raise ValueError("trimesh package material metadata is malformed")
            if isinstance(native_material, self._trimesh.visual.material.MultiMaterial):
                materials = tuple(
                    self._from_trimesh_material(
                        item,
                        material_descriptors[index] if index < len(material_descriptors) else None,
                    )
                    for index, item in enumerate(native_material.materials)
                )
            elif native_material is not None and int(package_metadata.get("material_count", 1)) > 0:
                descriptor = material_descriptors[0] if material_descriptors else None
                materials = (self._from_trimesh_material(native_material, descriptor),)
            if materials and native_mesh.visual.face_materials is not None:
                face_material_ids = _cpu_tensor(native_mesh.visual.face_materials, dtype=torch.int64)
                if not bool(package_metadata.get("has_face_material_ids", len(materials) > 1)):
                    face_material_ids = None

        has_normals = bool(package_metadata.get("has_normals", True))
        normals = (
            _cpu_tensor(native_mesh.vertex_normals, dtype=torch.float32)
            if has_normals and len(native_mesh.vertex_normals) == len(native_mesh.vertices)
            else None
        )
        metadata = package_metadata.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("trimesh package metadata is malformed")
        return MeshAsset(
            vertices=_cpu_tensor(native_mesh.vertices, dtype=torch.float32),
            faces=_cpu_tensor(native_mesh.faces, dtype=torch.int64),
            transform=resolved_transform,
            coordinate_system=resolved_coordinate_system,
            normals=normals,
            colors=colors,
            uvs=uvs,
            face_material_ids=face_material_ids,
            materials=tuple(_cpu_material(material) for material in materials),
            extras=extras,
            metadata=dict(metadata),
        )

    def _file_type(self, file_obj: str | os.PathLike[str] | BinaryIO | TextIO, file_type: str | None) -> str:
        if file_type is None and isinstance(file_obj, (str, os.PathLike)):
            file_type = Path(file_obj).suffix.removeprefix(".")
        if not isinstance(file_type, str) or not file_type:
            raise ValueError("file_type is required when it cannot be inferred from the file name")
        normalized = file_type.lower().removeprefix(".")
        if normalized not in _SUPPORTED_FILE_TYPES:
            choices = ", ".join(sorted(_SUPPORTED_FILE_TYPES))
            raise ValueError(f"unsupported mesh file type {file_type!r}; expected one of: {choices}")
        return normalized

    def _validate_export_channels(self, mesh: MeshAsset, file_type: str) -> None:
        if mesh.coordinate_system is not CoordinateSystem.RIGHT_HANDED_Y_UP:
            raise ValueError(
                f"{file_type.upper()} export cannot preserve coordinate system {mesh.coordinate_system.value!r}"
            )
        if file_type != "glb" and not torch.allclose(
            mesh.transform,
            torch.eye(4, device=mesh.device, dtype=mesh.transform.dtype),
        ):
            raise ValueError(f"{file_type.upper()} export cannot preserve a non-identity object transform")
        if mesh.extras:
            raise ValueError(f"{file_type.upper()} export cannot preserve package mesh extras")
        if any(material.base_color.shape[-1] == 4 and material.opacity is not None for material in mesh.materials):
            raise ValueError(
                f"{file_type.upper()} export cannot preserve separate base color alpha and opacity channels"
            )

        if file_type == "stl":
            unsupported = [
                name
                for name, value in (
                    ("normals", mesh.normals),
                    ("colors", mesh.colors),
                    ("uvs", mesh.uvs),
                    ("face material IDs", mesh.face_material_ids),
                )
                if value is not None
            ]
            if mesh.materials:
                unsupported.append("materials")
            if unsupported:
                raise ValueError(f"STL export cannot preserve {', '.join(unsupported)}")
        elif file_type == "ply" and (mesh.uvs is not None or mesh.materials or mesh.face_material_ids is not None):
            raise ValueError("PLY export cannot preserve UVs or materials through the trimesh adapter")
        elif file_type == "obj" and any(
            material.metallic is not None or material.roughness is not None or material.emissive is not None
            for material in mesh.materials
        ):
            raise ValueError("OBJ export cannot preserve metallic, roughness, or emissive PBR channels")
        elif file_type in {"glb", "obj"} and mesh.colors is not None and (mesh.uvs is not None or mesh.materials):
            raise ValueError(
                f"{file_type.upper()} export cannot preserve vertex colors together with UVs/materials through trimesh"
            )

    def export_mesh(
        self,
        mesh: MeshAsset,
        file_obj: str | os.PathLike[str] | BinaryIO | TextIO,
        *,
        file_type: str | None = None,
    ) -> Any:
        """Export OBJ, PLY, GLB, or STL after checking format lossiness."""

        normalized_type = self._file_type(file_obj, file_type)
        self._validate_export_channels(mesh, normalized_type)
        native_mesh = self.to_trimesh(mesh)
        if normalized_type == "glb":
            scene = self._trimesh.Scene()
            scene.add_geometry(
                native_mesh,
                geom_name="mesh",
                node_name="mesh",
                transform=mesh.transform.detach().cpu().numpy(),
            )
            return scene.export(file_obj=file_obj, file_type=normalized_type)
        return native_mesh.export(file_obj=file_obj, file_type=normalized_type)

    def import_mesh(
        self,
        file_obj: str | os.PathLike[str] | BinaryIO | TextIO,
        *,
        file_type: str | None = None,
        coordinate_system: CoordinateSystem | str = CoordinateSystem.RIGHT_HANDED_Y_UP,
    ) -> MeshAsset:
        """Import one OBJ, PLY, GLB, or STL geometry into detached CPU tensors."""

        normalized_type = self._file_type(file_obj, file_type)
        native_mesh = self._trimesh.load(
            file_obj,
            file_type=normalized_type,
            process=False,
            maintain_order=True,
        )
        return self.from_trimesh(native_mesh, coordinate_system=coordinate_system)

    def process_geometry(
        self,
        mesh: MeshAsset,
        *,
        operation: str,
        parameters: Mapping[str, object] | None = None,
    ) -> MeshAsset:
        """Run a conservative named trimesh operation and return detached CPU tensors."""

        if not isinstance(operation, str):
            raise TypeError("operation must be a string")
        options = {} if parameters is None else dict(parameters)
        native_mesh = self.to_trimesh(mesh).copy()
        if operation == "remove_unreferenced_vertices":
            if options:
                raise ValueError("remove_unreferenced_vertices does not accept parameters")
            native_mesh.remove_unreferenced_vertices()
        elif operation == "fix_normals":
            unknown = set(options) - {"multibody"}
            if unknown:
                raise ValueError(f"unsupported fix_normals parameters: {sorted(unknown)}")
            native_mesh.fix_normals(multibody=bool(options.get("multibody", False)))
            package_metadata = native_mesh.metadata[_METADATA_KEY]
            package_metadata["has_normals"] = True
        elif operation == "recompute_vertex_normals":
            if options:
                raise ValueError("recompute_vertex_normals does not accept parameters")
            native_mesh._cache.clear()
            package_metadata = native_mesh.metadata[_METADATA_KEY]
            package_metadata["has_normals"] = True
        elif operation == "merge_vertices":
            vertex_aligned = mesh.normals is not None or mesh.colors is not None or mesh.uvs is not None
            vertex_aligned = vertex_aligned or any(
                value.shape[0] == mesh.vertices.shape[0] for value in mesh.extras.values()
            )
            if vertex_aligned:
                raise ValueError("merge_vertices cannot safely merge a mesh with vertex-aligned channels")
            native_mesh.merge_vertices(**options)
        elif operation == "cleanup":
            if options:
                raise ValueError("cleanup does not accept parameters")
            if mesh.face_material_ids is not None or mesh.extras:
                raise ValueError("cleanup cannot safely remap face material IDs or custom extras")
            native_mesh.update_faces(native_mesh.nondegenerate_faces() & native_mesh.unique_faces())
            native_mesh.remove_unreferenced_vertices()
            native_mesh.fix_normals()
            package_metadata = native_mesh.metadata[_METADATA_KEY]
            package_metadata["has_normals"] = True
        else:
            choices = "cleanup, fix_normals, merge_vertices, recompute_vertex_normals, remove_unreferenced_vertices"
            raise ValueError(f"unsupported trimesh geometry operation {operation!r}; expected one of: {choices}")
        return self.from_trimesh(native_mesh)


__all__ = ["TrimeshBackend"]
