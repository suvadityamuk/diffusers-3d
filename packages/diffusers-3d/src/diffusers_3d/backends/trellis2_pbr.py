"""TRELLIS.2 ``to_glb``: an O-Voxel cleaned, UV-unwrapped, and baked into a PBR-textured mesh."""

from __future__ import annotations

import numpy as np
import torch

from ..objects import MeshAsset, OVoxelAsset, PBRMaterial
from ._optional import require
from .cumesh import CuMeshBackend
from .o_voxel import ovoxel_to_mesh

NVDIFFRAST_INSTALL = "pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
FLEX_GEMM_INSTALL = (
    "pip install --no-build-isolation "
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e"
)
# Upstream writes GLB vertices and normals as ``(x, z, -y)``.
_GLB_ROTATION = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def textured_mesh_from_ovoxel(
    asset: OVoxelAsset,
    *,
    device: str | torch.device = "cuda",
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
    remesh: bool = True,
    remesh_band: float = 1.0,
    remesh_project: float = 0.0,
) -> MeshAsset:
    """Return ``asset``'s surface UV-unwrapped, with base colour, opacity, metallic, and roughness textures.

    This is TRELLIS.2's upstream ``o_voxel.postprocess.to_glb`` step for step, on its own CUDA libraries: CuMesh
    cleans, remeshes, simplifies, and unwraps the mesh; nvdiffrast rasterizes the UV atlas; every texel is snapped
    back onto the full-resolution surface with CuMesh's BVH and sampled from the voxels with FlexGEMM; and OpenCV
    inpaints the uncovered texels of the 8-bit textures. All of them are optional dependencies you install yourself.

    The defaults are the ones upstream's ``example.py`` passes. The result keeps the O-Voxel's Z-up frame, and
    :meth:`MeshAsset.save` writes it the way upstream does.
    """

    import cv2

    cumesh = CuMeshBackend(device=device)._module
    dr = require("nvdiffrast.torch", NVDIFFRAST_INSTALL)
    grid_sample_3d = require("flex_gemm.ops.grid_sample", FLEX_GEMM_INSTALL).grid_sample_3d
    asset = asset.to(device)
    mesh = ovoxel_to_mesh(asset)
    grid_size = torch.tensor(mesh.metadata["resolution"], dtype=torch.int32, device=mesh.device)
    aabb = torch.tensor(mesh.metadata["aabb"], dtype=torch.float32, device=mesh.device)
    voxel_size = (aabb[1] - aabb[0]) / grid_size

    native = cumesh.CuMesh()
    native.init(mesh.vertices, mesh.faces.to(torch.int32))
    native.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = native.read()
    bvh = cumesh.cuBVH(vertices, faces)
    if remesh:
        resolution = grid_size.max().item()
        scale = (aabb[1] - aabb[0]).max().item()
        native.init(
            *cumesh.remeshing.remesh_narrow_band_dc(
                vertices,
                faces,
                center=aabb.mean(dim=0),
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=remesh_project,
                verbose=False,
                bvh=bvh,
            )
        )
        native.simplify(decimation_target, verbose=False)
    else:
        for target in (decimation_target * 3, decimation_target):
            native.simplify(target, verbose=False)
            native.remove_duplicate_faces()
            native.repair_non_manifold_edges()
            native.remove_small_connected_components(1e-5)
            native.fill_holes(max_hole_perimeter=3e-2)
        native.unify_face_orientations()
    out_vertices, out_faces, uvs, vertex_map = native.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": np.radians(90.0),
            "refine_iterations": 0,
            "global_iterations": 1,
            "smooth_strength": 1,
        },
        return_vmaps=True,
        verbose=False,
    )
    out_vertices, out_faces, uvs = out_vertices.to(device), out_faces.to(device), uvs.to(device)
    native.compute_vertex_normals()
    normals = native.read_vertex_normals()[vertex_map.to(device)]

    # Rasterize the UV atlas in chunks of 100k faces, a later chunk winning, with face ids offset per chunk.
    context = dr.RasterizeCudaContext()
    clip = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)
    for start in range(0, out_faces.shape[0], 100000):
        chunk, _ = dr.rasterize(context, clip, out_faces[start : start + 100000], resolution=[texture_size] * 2)
        covered = chunk[..., 3:4] > 0
        chunk[..., 3:4] += start
        rast = torch.where(covered, chunk, rast)
    mask = rast[0, ..., 3] > 0

    # Snap every covered texel back onto the full-resolution surface, then sample the voxel attributes there.
    points = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0][mask]
    _, face_id, uvw = bvh.unsigned_distance(points, return_uvw=True)
    points = (vertices[faces[face_id.long()]] * uvw.unsqueeze(-1)).sum(dim=1)
    opacity = torch.ones_like(asset.metallic) if asset.opacity is None else asset.opacity
    features = torch.cat(
        [asset.base_color, opacity.reshape(-1, 1), asset.metallic.reshape(-1, 1), asset.roughness.reshape(-1, 1)],
        dim=1,
    ).float()
    coordinates = asset.active_coordinates.to(device)
    attributes = torch.zeros(texture_size, texture_size, features.shape[1], device=device)
    attributes[mask] = grid_sample_3d(
        features,
        torch.cat([torch.zeros_like(coordinates[:, :1]), coordinates], dim=-1),
        shape=torch.Size([1, features.shape[1], *grid_size.tolist()]),
        grid=((points - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )

    # 8-bit textures, with the uncovered texels inpainted as upstream does.
    texture = np.clip(attributes.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    holes = (~mask).cpu().numpy().astype(np.uint8)
    base_color = cv2.inpaint(np.ascontiguousarray(texture[..., :3]), holes, 3, cv2.INPAINT_TELEA)
    channels = [cv2.inpaint(np.ascontiguousarray(texture[..., i]), holes, 1, cv2.INPAINT_TELEA) for i in (3, 4, 5)]
    texture = torch.from_numpy(np.concatenate([base_color, np.stack(channels, axis=-1)], axis=-1)).float() / 255
    texture = texture.to(device)
    material = PBRMaterial(
        base_color=texture[..., :4],
        metallic=texture[..., 4],
        roughness=texture[..., 5],
        metadata={"double_sided": not remesh},
    )
    return MeshAsset(
        vertices=out_vertices,
        faces=out_faces.to(torch.int64),
        transform=mesh.transform,
        coordinate_system=mesh.coordinate_system,
        normals=normals,
        # CuMesh's v points down the texture rows; MeshAsset uvs follow OpenGL, so v points up.
        uvs=torch.stack([uvs[:, 0], 1 - uvs[:, 1]], dim=1),
        materials=(material,),
        metadata={**mesh.metadata, "texture_size": texture_size, "remesh": remesh, "export_rotation": _GLB_ROTATION},
    )


__all__ = ["textured_mesh_from_ovoxel"]
