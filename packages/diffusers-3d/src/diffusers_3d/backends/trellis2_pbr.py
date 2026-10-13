"""TRELLIS.2 ``to_glb``: an O-Voxel cleaned, UV-unwrapped, and baked into a PBR-textured mesh."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import torch

from ..objects import MeshAsset, OVoxelAsset, PBRMaterial
from .cumesh import CuMeshBackend
from .defaults import BACKEND_REGISTRY
from .o_voxel import ovoxel_to_mesh
from .registry import BackendRegistry
from .texture_baking import fill_texture, rasterize_uv_atlas
from .types import BackendCapability, BackendSpec
from .voxel_index import VoxelIndex, trilinear_sample


class Trellis2PBRPostprocessFacade:
    """Explicit gate for the TRELLIS.2 textured-mesh path (upstream ``o_voxel.postprocess.to_glb``).

    Mesh repair, simplification, remeshing, UV unwrapping, and the BVH run on CuMesh, which users install
    themselves, in the same order as upstream. Texel rasterization and trilinear attribute baking, which upstream
    runs on nvdiffrast and FlexGEMM, are plain PyTorch. Construction and :meth:`requirements` are side-effect free.
    """

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        self.registry = registry

    def requirements(self, *, device: str | torch.device = "cuda") -> Mapping[str, BackendSpec]:
        return {
            "cumesh": self.registry.select(
                BackendCapability.GEOMETRY_PROCESSING,
                name="cumesh",
                device=device,
                dtype=torch.float32,
                differentiable=False,
            )
        }

    def to_textured_mesh(
        self,
        asset: OVoxelAsset,
        *,
        device: str | torch.device = "cuda",
        decimation_target: int = 1_000_000,
        texture_size: int = 2048,
        remesh: bool = True,
        remesh_band: float = 1.0,
        remesh_project: float = 0.0,
    ) -> MeshAsset:
        """Return ``asset``'s surface UV-unwrapped, with base colour, opacity, metallic, and roughness textures.

        ``decimation_target`` is the face count CuMesh simplifies to. The remesh defaults are the ones upstream's
        released demo uses. The result keeps the O-Voxel's Z-up frame;
        call :meth:`MeshAsset.to_coordinate_system` before exporting with :class:`TrimeshBackend`.
        """

        cumesh = CuMeshBackend(device=device, registry=self.registry)
        asset = asset.to(device)
        mesh = ovoxel_to_mesh(asset)
        grid_size = mesh.metadata["resolution"]
        aabb = torch.tensor(mesh.metadata["aabb"], dtype=torch.float32, device=mesh.device)

        native = cumesh._module.CuMesh()
        native.init(mesh.vertices, mesh.faces.to(torch.int32))
        native.fill_holes(max_hole_perimeter=3e-2)
        vertices, faces = native.read()
        bvh = cumesh._module.cuBVH(vertices, faces)
        if remesh:
            resolution = max(grid_size)
            scale = (aabb[1] - aabb[0]).max().item()
            native.init(
                *cumesh._module.remeshing.remesh_narrow_band_dc(
                    vertices,
                    faces,
                    center=aabb.mean(dim=0),
                    scale=(resolution + 3 * remesh_band) / resolution * scale,
                    resolution=resolution,
                    band=remesh_band,
                    project_back=remesh_project,
                    bvh=bvh,
                )
            )
            native.simplify(decimation_target)
        else:
            for target in (decimation_target * 3, decimation_target):
                native.simplify(target)
                native.remove_duplicate_faces()
                native.repair_non_manifold_edges()
                native.remove_small_connected_components(1e-5)
                native.fill_holes(max_hole_perimeter=3e-2)
            native.unify_face_orientations()
        out_vertices, out_faces, uvs, vertex_map = native.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": math.radians(90.0),
                "refine_iterations": 0,
                "global_iterations": 1,
                "smooth_strength": 1,
            },
            return_vmaps=True,
        )
        native.compute_vertex_normals()
        normals = native.read_vertex_normals()[vertex_map.to(mesh.device)]
        unwrapped = MeshAsset(
            vertices=out_vertices.to(mesh.device),
            faces=out_faces.to(device=mesh.device, dtype=torch.int64),
            transform=mesh.transform,
            coordinate_system=mesh.coordinate_system,
            normals=normals,
            # CuMesh's v points down the texture rows; MeshAsset uvs follow OpenGL, so v points up.
            uvs=torch.stack([uvs[:, 0], 1 - uvs[:, 1]], dim=1).to(mesh.device),
        )

        # Snap every texel back onto the full-resolution surface, then sample the voxel attributes there.
        surface = rasterize_uv_atlas(unwrapped, texture_size)
        _, face_id, uvw = bvh.unsigned_distance(surface.positions, return_uvw=True)
        points = (vertices[faces[face_id.long()].long()] * uvw.unsqueeze(-1)).sum(dim=1)
        opacity = torch.ones_like(asset.metallic) if asset.opacity is None else asset.opacity
        features = torch.cat(
            [asset.base_color, opacity.reshape(-1, 1), asset.metallic.reshape(-1, 1), asset.roughness.reshape(-1, 1)],
            dim=1,
        ).float()
        voxel_size = (aabb[1] - aabb[0]) / torch.tensor(grid_size, device=mesh.device)
        index = VoxelIndex(asset.active_coordinates, grid_size)
        samples = trilinear_sample(index, features, (points - aabb[0]) / voxel_size)

        texels = texture_size * texture_size
        texture = torch.zeros(texels, features.shape[1], device=mesh.device)
        known = torch.zeros(texels, dtype=torch.bool, device=mesh.device)
        texture[surface.texel_index] = samples
        known[surface.texel_index] = True
        known = known.reshape(texture_size, texture_size)
        texture = fill_texture(texture.reshape(texture_size, texture_size, -1), known, known)
        material = PBRMaterial(
            base_color=texture[..., :4],
            metallic=texture[..., 4],
            roughness=texture[..., 5],
            metadata={"double_sided": not remesh},
        )
        return replace(
            unwrapped,
            materials=(material,),
            metadata={**mesh.metadata, "texture_size": texture_size, "remesh": remesh},
        )


__all__ = ["Trellis2PBRPostprocessFacade"]
