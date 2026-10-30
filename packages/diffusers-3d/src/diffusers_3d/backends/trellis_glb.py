# Portions of this file reproduce ``postprocessing_utils.to_glb`` and the renderer and camera helpers it calls from
# Microsoft TRELLIS (https://github.com/microsoft/TRELLIS, revision 442aa1e1afb9014e80681d3bf604e8d728a86ee7, MIT
# License, Copyright (c) Microsoft Corporation) and the mesh and camera helpers they use from utils3d
# (https://github.com/EasternJournalist/utils3d, revision 9a4eb15e4021b67b12c460c7057d642626897ec8, MIT License).
"""TRELLIS ``to_glb``: a decoded FlexiCubes mesh, cleaned and textured from renders of the matching Gaussian splats."""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ..objects import GaussianSplatAsset, MeshAsset, PBRMaterial
from ._optional import require
from .trellis2_pbr import NVDIFFRAST_INSTALL

MIP_SPLATTING_INSTALL = (
    'pip install --no-build-isolation "git+https://github.com/autonomousvision/mip-splatting.git'
    '@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#subdirectory=submodules/diff-gaussian-rasterization"'
)
# Upstream writes GLB vertices as ``vertices @ [[1, 0, 0], [0, 0, -1], [0, 1, 0]]``, that is ``(x, z, -y)``.
_GLB_ROTATION = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def textured_mesh_from_splats(
    mesh: MeshAsset,
    gaussians: GaussianSplatAsset,
    *,
    device: str | torch.device = "cuda",
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
) -> MeshAsset:
    """Return ``mesh`` simplified, hole-filled, UV-unwrapped, and carrying a base-colour texture baked from splats.

    This is TRELLIS's upstream ``postprocessing_utils.to_glb`` step for step, on its own libraries: pyvista decimates
    away ``simplify`` of the faces; faces no view sees are cut away with an igraph min-cut over 1000 nvdiffrast
    visibility renders and the holes refilled by pymeshfix; xatlas unwraps the mesh; mip-splatting's rasterizer renders
    the splats from 100 views; and nvdiffrast fits the texture to them with 2500 Adam steps (L1 plus total variation)
    before OpenCV inpaints the texels outside the atlas. All of them are optional dependencies you install yourself.

    Like upstream, each optimisation step picks its view with NumPy's global ``np.random.randint``: seed it with
    ``np.random.seed`` to reproduce a texture. ``mesh`` and ``gaussians`` must come from one TRELLIS SLAT (same
    coordinate system, identity transforms). The result keeps TRELLIS' Z-up frame, and :meth:`MeshAsset.save` writes it
    the way upstream does.
    """

    if type(mesh) is not MeshAsset or type(gaussians) is not GaussianSplatAsset:
        raise TypeError("mesh must be a MeshAsset and gaussians a GaussianSplatAsset")
    if mesh.coordinate_system is not gaussians.coordinate_system:
        raise ValueError("mesh and gaussians must use the same coordinate system")
    for asset in (mesh, gaussians):
        if not torch.equal(asset.transform.cpu(), torch.eye(4, dtype=asset.transform.dtype)):
            raise ValueError("mesh and gaussians must have identity transforms; bake them into the data first")
    if not 0.0 <= simplify < 1.0:
        raise ValueError("simplify must be in [0, 1)")
    device = torch.device(device)
    dr = require("nvdiffrast.torch", NVDIFFRAST_INSTALL)

    vertices, faces = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
    if simplify > 0:
        pyvista = require("pyvista", "pip install pyvista")
        decimated = pyvista.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        decimated = decimated.decimate(simplify, progress_bar=False)
        vertices, faces = decimated.points, decimated.faces.reshape(-1, 4)[:, 1:]
    if fill_holes:
        filled = _fill_holes(
            dr,
            torch.tensor(vertices).to(device),
            torch.tensor(faces.astype(np.int32)).to(device),
            max_hole_size=fill_holes_max_size,
            max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            resolution=1024,
            num_views=1000,
        )
        vertices, faces = (value.cpu().numpy() for value in filled)
    vertex_map, faces, uvs = require("xatlas", "portable").parametrize(vertices, faces)
    vertices = vertices[vertex_map]

    observations, extrinsics, intrinsics = _render_multiview(gaussians.to(device), resolution=1024, num_views=100)
    texture = _bake_texture(
        dr,
        vertices,
        faces,
        uvs,
        observations,
        [np.any(observation > 0, axis=-1) for observation in observations],
        [extrinsic.cpu().numpy() for extrinsic in extrinsics],
        [intrinsic.cpu().numpy() for intrinsic in intrinsics],
        texture_size=texture_size,
        device=device,
    )
    material = PBRMaterial(
        base_color=torch.from_numpy(texture).to(device).float() / 255, roughness=torch.tensor(1.0, device=device)
    )
    return MeshAsset(
        vertices=torch.from_numpy(vertices).to(device),
        faces=torch.from_numpy(faces.astype(np.int64)).to(device),
        coordinate_system=mesh.coordinate_system,
        uvs=torch.from_numpy(uvs).to(device),
        materials=(material,),
        metadata={**mesh.metadata, "texture_size": texture_size, "export_rotation": _GLB_ROTATION},
    )


# --- Cameras: TRELLIS ``render_utils`` and ``random_utils``, and the utils3d transforms they call. ---


def _sphere_hammersley(index: int, count: int) -> tuple[float, float]:
    """Yaw and pitch of view ``index`` of ``count`` on TRELLIS' Hammersley sphere."""

    u, v, base = index / count, 0.0, 0.5
    while index > 0:
        v += index % 2 * base
        index //= 2
        base *= 0.5
    return v * 2 * np.pi, np.arccos(1 - 2 * u) - np.pi / 2


def _look_at(eye: torch.Tensor, up: torch.Tensor, *, opencv: bool) -> torch.Tensor:
    """utils3d ``extrinsics_look_at`` (OpenCV) or ``view_look_at`` (OpenGL) towards the origin, batch of one."""

    eye, look_at, up = eye[None], torch.zeros_like(eye)[None], up[None]
    z = look_at - eye if opencv else eye - look_at
    x = torch.cross(-up if opencv else up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    rotation = torch.stack(
        [x / x.norm(dim=-1, keepdim=True), y / y.norm(dim=-1, keepdim=True), z / z.norm(dim=-1, keepdim=True)], dim=-2
    )
    matrix = torch.zeros((1, 4, 4), dtype=eye.dtype, device=eye.device)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = -torch.matmul(rotation, eye[..., None])[:, :, 0]
    matrix[:, 3, 3] = 1.0
    return matrix[0]


def _intrinsics_from_fov(fov: torch.Tensor) -> torch.Tensor:
    focal = (0.5 / torch.tan(fov / 2)).reshape(1)
    zeros, ones, center = torch.zeros_like(focal), torch.ones_like(focal), torch.full_like(focal, 0.5)
    return torch.stack([focal, zeros, center, zeros, focal, center, zeros, zeros, ones], dim=-1).reshape(3, 3)


def _gl_perspective(intrinsics: torch.Tensor | None, near: float, far: float, fov: torch.Tensor | None = None):
    """utils3d ``intrinsics_to_perspective`` or, given ``fov``, ``perspective_from_fov_xy(fov, fov, near, far)``."""

    near_t = torch.tensor(near, device=(fov if intrinsics is None else intrinsics).device).reshape(1)
    far_t = torch.tensor(far, device=near_t.device).reshape(1)
    dtype = (fov if intrinsics is None else intrinsics).dtype
    matrix = torch.zeros((1, 4, 4), dtype=dtype, device=near_t.device)
    if intrinsics is None:
        fov = fov.reshape(1)
        aspect = torch.tan(fov / 2) / torch.tan(fov / 2)
        matrix[:, 0, 0] = 1.0 / (torch.tan(fov / 2) * aspect)
        matrix[:, 1, 1] = 1.0 / torch.tan(fov / 2)
    else:
        intrinsics = intrinsics[None]
        matrix[:, 0, 0] = 2 * intrinsics[:, 0, 0]
        matrix[:, 1, 1] = 2 * intrinsics[:, 1, 1]
        matrix[:, 0, 2] = -2 * intrinsics[:, 0, 2] + 1
        matrix[:, 1, 2] = 2 * intrinsics[:, 1, 2] - 1
    matrix[:, 2, 2] = (near_t + far_t) / (near_t - far_t)
    matrix[:, 2, 3] = 2.0 * near_t * far_t / (near_t - far_t)
    matrix[:, 3, 2] = -1.0
    return matrix[0]


def _rasterize(dr, context, vertices, faces, size, *, view=None, projection=None, uvs=None):
    """utils3d ``rasterize_triangle_faces``: ``(mask, face_id)`` flipped to image rows, plus ``uv`` and ``uv_dr``."""

    padding = [torch.ones_like(vertices[..., :1])]
    if vertices.shape[-1] == 2:
        padding.insert(0, torch.zeros_like(vertices[..., :1]))
    vertices = torch.cat([vertices, *padding], dim=-1)
    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view
    clip = vertices @ mvp.transpose(-1, -2)
    faces = faces.contiguous()
    rast, rast_db = dr.rasterize(context, clip, faces, resolution=[size[0], size[1]], grad_db=True)
    face_id = rast[..., 3].flip(1)
    if uvs is None:
        return (face_id > 0).float(), face_id
    return dr.interpolate(uvs, rast, faces, rast_db, diff_attrs="all")


def _render_multiview(gaussians: GaussianSplatAsset, *, resolution: int, num_views: int):
    """TRELLIS ``render_multiview``: 8-bit renders of the splats from ``num_views`` Hammersley views at radius 2."""

    rasterizer = require("diff_gaussian_rasterization", MIP_SPLATTING_INSTALL)
    warnings.warn(
        "mip-splatting's Gaussian rasterizer is licensed by Inria and MPII for non-commercial research and evaluation "
        "use only. You are responsible for making sure your use of it complies with that license.",
        UserWarning,
        stacklevel=3,
    )
    device = gaussians.means.device
    means = gaussians.means.float()
    opacities = gaussians.opacity_logits.float().reshape(-1, 1).sigmoid()
    scales, rotations = gaussians.log_scales.float().exp(), gaussians.quaternions_wxyz.float()
    shs = gaussians.sh_coefficients.float()
    observations, extrinsics, intrinsics = [], [], []
    for index in range(num_views):
        yaw, pitch = _sphere_hammersley(index, num_views)
        fov = torch.deg2rad(torch.tensor(float(40))).to(device)
        yaw, pitch = torch.tensor(float(yaw)).to(device), torch.tensor(float(pitch)).to(device)
        eye = torch.tensor([torch.sin(yaw) * torch.cos(pitch), torch.cos(yaw) * torch.cos(pitch), torch.sin(pitch)])
        extrinsic = _look_at(eye.to(device) * 2, torch.tensor([0, 0, 1]).float().to(device), opencv=True)
        intrinsic = _intrinsics_from_fov(fov)
        # TRELLIS ``GaussianRenderer.render`` (near 0.8, far 1.6, black background, mip filter 0.1).
        near, far = 0.8, 1.6
        projection = torch.zeros((4, 4), dtype=intrinsic.dtype, device=device)
        projection[0, 0], projection[1, 1] = 2 * intrinsic[0, 0], 2 * intrinsic[1, 1]
        projection[0, 2], projection[1, 2] = 2 * intrinsic[0, 2] - 1, -2 * intrinsic[1, 2] + 1
        projection[2, 2], projection[2, 3], projection[3, 2] = far / (far - near), near * far / (near - far), 1.0
        fov_x, fov_y = 2 * torch.atan(0.5 / intrinsic[0, 0]), 2 * torch.atan(0.5 / intrinsic[1, 1])
        settings = rasterizer.GaussianRasterizationSettings(
            image_height=resolution,
            image_width=resolution,
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            kernel_size=0.1,
            subpixel_offset=torch.zeros((resolution, resolution, 2), dtype=torch.float32, device=device),
            bg=torch.tensor((0, 0, 0), dtype=torch.float32, device=device),
            scale_modifier=1.0,
            viewmatrix=extrinsic.T.contiguous(),
            projmatrix=(projection @ extrinsic).T.contiguous(),
            sh_degree=gaussians.active_sh_degree,
            campos=torch.inverse(extrinsic)[:3, 3],
            prefiltered=False,
            debug=False,
        )
        with torch.no_grad():
            color, _ = rasterizer.GaussianRasterizer(raster_settings=settings)(
                means3D=means,
                means2D=torch.zeros_like(means),
                shs=shs,
                colors_precomp=None,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,
            )
        observations.append(np.clip(color.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
    return observations, extrinsics, intrinsics


# --- Mesh cleanup: TRELLIS ``_fill_holes`` and the utils3d mesh helpers it calls. ---


def _group(values: torch.Tensor, required_group_size: int | None = None):
    sorted_values, indices = torch.sort(values)
    first = torch.cat([torch.tensor([True], device=values.device), sorted_values[1:] != sorted_values[:-1]])
    group_index = torch.cumsum(first, dim=0) - 1
    counts = torch.bincount(group_index)
    if required_group_size is None:
        return torch.split(indices, counts.tolist())
    counts = counts[group_index]
    groups = indices[counts == required_group_size].reshape(-1, required_group_size)
    return groups, sorted_values[first][counts[first] == required_group_size]


def _edges(faces: torch.Tensor):
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0).sort(dim=1).values
    edges, inverse, counts = torch.unique(edges, return_inverse=True, return_counts=True, dim=0)
    return edges, inverse.view(3, faces.shape[0]).T, counts


def _connected_components(elements: torch.Tensor, element_to_node: torch.Tensor, num_nodes: int):
    """Label propagation of utils3d's ``compute_(edge_)connected_components``: groups of linked ``elements``."""

    labels = torch.arange(elements, dtype=torch.int32, device=element_to_node.device)
    while True:
        node_labels = torch.zeros(num_nodes, dtype=torch.int32, device=element_to_node.device).scatter_reduce(
            0,
            element_to_node.flatten().long(),
            labels.view(-1, 1).expand(-1, element_to_node.shape[1]).flatten(),
            reduce="amin",
            include_self=False,
        )
        new_labels = torch.min(node_labels[element_to_node], dim=-1).values
        if torch.equal(labels, new_labels):
            return _group(labels)
        labels = new_labels


def _face_components(faces: torch.Tensor, edges=None, face_to_edge=None):
    if edges is None:
        edges, face_to_edge, _ = _edges(faces)
    return _connected_components(faces.shape[0], face_to_edge, edges.shape[0])


def _fill_holes(dr, vertices, faces, *, max_hole_size, max_hole_nbe, resolution, num_views):
    """TRELLIS ``_fill_holes``: cut away faces no view sees with a min-cut, then fill small holes with pymeshfix."""

    igraph = require("igraph", "pip install python-igraph")
    meshfix = require("pymeshfix._meshfix", "pip install pymeshfix")
    device = vertices.device
    angles = [_sphere_hammersley(index, num_views) for index in range(num_views)]
    yaws = torch.tensor([yaw for yaw, _ in angles]).to(device)
    pitches = torch.tensor([pitch for _, pitch in angles]).to(device)
    projection = _gl_perspective(None, 1, 3, fov=torch.deg2rad(torch.tensor(40)).to(device))
    up = torch.tensor([0, 0, 1]).float().to(device)
    views = torch.stack(
        [
            _look_at(
                torch.tensor([torch.sin(y) * torch.cos(p), torch.cos(y) * torch.cos(p), torch.sin(p)])
                .to(device)
                .float()
                * 2.0,
                up,
                opencv=False,
            )
            for y, p in zip(yaws, pitches)
        ]
    )

    visibility = torch.zeros(faces.shape[0], dtype=torch.int32, device=device)
    context = dr.RasterizeCudaContext()
    for view in views:
        mask, face_id = _rasterize(
            dr, context, vertices[None], faces, (resolution, resolution), view=view, projection=projection
        )
        visibility[torch.unique(face_id[0][mask[0] > 0.95] - 1).long()] += 1
    visibility = visibility.float() / num_views

    # Outer faces are the well-seen ones of each component; inner faces are never seen.
    edges, face_to_edge, edge_degrees = _edges(faces)
    boundary_edges = torch.nonzero(edge_degrees == 1).reshape(-1)
    outer = torch.zeros(faces.shape[0], dtype=torch.bool, device=device)
    for component in _face_components(faces, edges, face_to_edge):
        seen = visibility[component]
        outer[component] = seen > min(max(seen.quantile(0.75).item(), 0.25), 0.5)
    outer = outer.nonzero().reshape(-1)
    inner = torch.nonzero(visibility == 0).reshape(-1)
    if inner.shape[0] == 0:
        return vertices, faces

    # Min-cut the face dual graph, weighted by shared edge length, between inner (source) and outer (sink) faces.
    dual_edges, dual_edge_to_edge = _group(face_to_edge.flatten(), required_group_size=2)
    dual_edges = dual_edges // face_to_edge.shape[1]
    dual_edge_to_edge = edges[dual_edge_to_edge]
    graph = igraph.Graph()
    graph.add_vertices(faces.shape[0])
    graph.add_edges(dual_edges.cpu().numpy())
    graph.es["weight"] = (
        torch.norm(vertices[dual_edge_to_edge[:, 0]] - vertices[dual_edge_to_edge[:, 1]], dim=1).cpu().numpy()
    )
    graph.add_vertex("s")
    graph.add_vertex("t")
    for faces_of, terminal in ((inner, "s"), (outer, "t")):
        graph.add_edges(
            [(face, terminal) for face in faces_of.tolist()],
            attributes={"weight": torch.ones(faces_of.shape[0], dtype=torch.float32).cpu().numpy()},
        )
    cut = graph.mincut("s", "t", (np.array(graph.es["weight"]) * 1000).tolist())
    removed = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=device)

    # Keep a cut piece only if it is barely visible and every new boundary loop it opens is small.
    valid = []
    for component in _face_components(faces[removed]):
        if visibility[removed[component]].median() > 0.25:
            continue
        piece_edges, piece_degrees = torch.unique(face_to_edge[removed[component]], return_counts=True)
        new_boundary = piece_edges[piece_degrees == 1]
        new_boundary = new_boundary[~torch.isin(new_boundary, boundary_edges)]
        if len(new_boundary) > 0:
            loop_vertices, loop_edges = torch.unique(edges[new_boundary].flatten(), return_inverse=True)
            loops = _connected_components(len(new_boundary), loop_edges.view(-1, 2), loop_vertices.shape[0])
            areas = []
            for loop in loops:
                segments = edges[new_boundary[loop]]
                center = vertices[segments].mean(dim=1).mean(dim=0)
                cross = torch.cross(vertices[segments[:, 0]] - center, vertices[segments[:, 1]] - center, dim=-1)
                areas.append(torch.norm(cross, dim=1).sum() * 0.5)
            if any(area > max_hole_size for area in areas):
                continue
        valid.append(component)
    if len(valid) > 0:
        keep = torch.ones(faces.shape[0], dtype=torch.bool, device=device)
        keep[removed[torch.cat(valid)]] = 0
        used, inverse = torch.unique(faces[keep], return_inverse=True)
        faces, vertices = inverse.to(torch.int32).reshape(-1, 3), vertices[used]

    fixer = meshfix.PyTMesh()
    fixer.load_array(vertices.cpu().numpy(), faces.cpu().numpy())
    fixer.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    vertices, faces = fixer.return_arrays()
    return torch.tensor(vertices, device=device, dtype=torch.float32), torch.tensor(
        faces, device=device, dtype=torch.int32
    )


# --- Texture: TRELLIS ``bake_texture`` in its default ``"opt"`` mode. ---


def _bake_texture(dr, vertices, faces, uvs, observations, masks, extrinsics, intrinsics, *, texture_size, device):
    """Fit a texture to the views with Adam on L1 + 0.01 total variation, then inpaint the texels off the atlas."""

    import cv2

    vertices = torch.tensor(vertices).to(device)
    faces = torch.tensor(faces.astype(np.int32)).to(device)
    uvs = torch.tensor(uvs).to(device)
    observations = [torch.tensor(observation / 255.0).float().to(device).flip(0) for observation in observations]
    masks = [torch.tensor(mask > 0).bool().to(device).flip(0) for mask in masks]
    flip = torch.tensor([1, -1, -1, 1], device=device)[:, None]
    context = dr.RasterizeCudaContext()
    rasterized = []
    with torch.no_grad():
        for observation, extrinsic, intrinsic in zip(observations, extrinsics, intrinsics):
            extrinsic, intrinsic = torch.tensor(extrinsic).to(device), torch.tensor(intrinsic).to(device)
            view = extrinsic * flip.to(extrinsic.dtype)
            projection = _gl_perspective(intrinsic, 0.1, 10.0)
            uv, uv_dr = _rasterize(
                dr,
                context,
                vertices[None],
                faces,
                observation.shape[:2],
                view=view,
                projection=projection,
                uvs=uvs[None],
            )
            rasterized.append((uv.detach(), uv_dr.detach()))

    with torch.enable_grad():
        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)
        total_steps = 2500
        for step in range(total_steps):
            optimizer.zero_grad()
            selected = np.random.randint(0, len(rasterized))
            (uv, uv_dr), observation, mask = rasterized[selected], observations[selected], masks[selected]
            render = dr.texture(texture, uv, uv_dr)[0]
            loss = F.l1_loss(render[mask], observation[mask])
            loss += 1e-2 * (
                F.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :])
                + F.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
            )
            loss.backward()
            optimizer.step()
            optimizer.param_groups[0]["lr"] = 1e-5 + 0.5 * (1e-2 - 1e-5) * (1 + np.cos(np.pi * step / total_steps))
    texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    covered, _ = _rasterize(dr, context, (uvs * 2 - 1)[None], faces, (texture_size, texture_size))
    holes = 1 - covered[0].detach().cpu().numpy().astype(np.uint8)
    return cv2.inpaint(texture, holes, 3, cv2.INPAINT_TELEA)


__all__ = ["textured_mesh_from_splats"]
