from __future__ import annotations

from collections.abc import Mapping

import torch

from ..objects import MeshAsset, PBRMaterial
from ._optional import load_selected_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability


def _detach_material(material: PBRMaterial) -> PBRMaterial:
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


class XAtlasBackend:
    """Portable xatlas UV unwrapping with explicit channel remapping."""

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        self._xatlas = load_selected_backend(
            "xatlas",
            "xatlas",
            (BackendCapability.GEOMETRY_PROCESSING,),
            registry=registry,
        )

    def process_geometry(
        self,
        mesh: MeshAsset,
        *,
        operation: str,
        parameters: Mapping[str, object] | None = None,
    ) -> MeshAsset:
        if operation != "unwrap_uv":
            raise ValueError("xatlas supports only the 'unwrap_uv' geometry operation")
        if parameters:
            raise ValueError("xatlas 'unwrap_uv' does not accept parameters")
        if not isinstance(mesh, MeshAsset):
            raise TypeError("mesh must be a MeshAsset")
        mesh.validate(expensive=True)

        positions = mesh.vertices.detach().cpu().to(torch.float32).contiguous().numpy()
        indices = mesh.faces.detach().cpu().to(torch.uint32).contiguous().numpy()
        normals = None if mesh.normals is None else mesh.normals.detach().cpu().to(torch.float32).contiguous().numpy()
        vertex_mapping, output_faces, output_uvs = self._xatlas.parametrize(positions, indices, normals)

        mapping = torch.from_numpy(vertex_mapping.copy()).to(torch.int64)
        faces = torch.from_numpy(output_faces.copy()).to(torch.int64)
        uvs = torch.from_numpy(output_uvs.copy()).to(torch.float32)
        if faces.shape != mesh.faces.shape:
            raise ValueError(
                "xatlas changed the number or ordering shape of input faces; face-aligned channels cannot be preserved"
            )
        if mapping.ndim != 1 or mapping.shape[0] != uvs.shape[0]:
            raise ValueError("xatlas returned an invalid vertex mapping")
        if bool((mapping < 0).any()) or bool((mapping >= mesh.vertices.shape[0]).any()):
            raise ValueError("xatlas returned a vertex mapping outside the input vertex range")

        vertex_count = mesh.vertices.shape[0]
        face_count = mesh.faces.shape[0]
        identity_mapping = mapping.shape[0] == vertex_count and torch.equal(
            mapping, torch.arange(vertex_count, dtype=torch.int64)
        )
        extras: dict[str, torch.Tensor] = {}
        for name, value in mesh.extras.items():
            cpu_value = value.detach().cpu()
            vertex_aligned = value.shape[0] == vertex_count
            face_aligned = value.shape[0] == face_count
            if vertex_aligned and face_aligned and not identity_mapping:
                raise ValueError(
                    f"mesh extra {name!r} is alignment-ambiguous after xatlas duplicated or reordered vertices"
                )
            extras[name] = cpu_value.index_select(0, mapping) if vertex_aligned else cpu_value

        def remap(channel: torch.Tensor | None) -> torch.Tensor | None:
            return None if channel is None else channel.detach().cpu().index_select(0, mapping)

        return MeshAsset(
            vertices=mesh.vertices.detach().cpu().index_select(0, mapping),
            faces=faces,
            transform=mesh.transform.detach().cpu(),
            coordinate_system=mesh.coordinate_system,
            normals=remap(mesh.normals),
            colors=remap(mesh.colors),
            uvs=uvs,
            face_material_ids=(None if mesh.face_material_ids is None else mesh.face_material_ids.detach().cpu()),
            materials=tuple(_detach_material(material) for material in mesh.materials),
            extras=extras,
            metadata=dict(mesh.metadata),
        )


__all__ = ["XAtlasBackend"]
