"""Pure-PyTorch texture baking: project multi-view renders of an object onto a UV-unwrapped mesh.

This is the portable half of the TRELLIS ``to_glb`` path. Upstream rasterizes the mesh into every view with
nvdiffrast and scatters observations into texels; here the atlas is rasterized once in UV space, every texel's
surface point is projected into every view, and observations are averaged where the point is visible. Visibility
uses the rendered depth and alpha, so any renderer that returns colour, depth, and alpha (gsplat, the voxel
rasterizer, a mesh renderer) can supply the observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..objects import CameraRig, CoordinateSystem, MeshAsset


@dataclass(frozen=True)
class TexelSurface:
    """Surface point behind every covered texel of a ``texture_size`` x ``texture_size`` atlas."""

    texel_index: torch.Tensor  # (T,) flat index row * size + column, row 0 at the top
    positions: torch.Tensor  # (T, 3) object-space points
    normals: torch.Tensor  # (T, 3) unit face normals


def sphere_hammersley_cameras(
    count: int,
    *,
    image_size: int,
    radius: float = 2.0,
    fov_degrees: float = 40.0,
    coordinate_system: CoordinateSystem | str = CoordinateSystem.RIGHT_HANDED_Z_UP,
    device: torch.device | str = "cpu",
) -> CameraRig:
    """``count`` OpenCV pinhole cameras on a sphere, placed by the 2D Hammersley sequence as TRELLIS does.

    Cameras look at the origin with the coordinate system's up axis; ``radius`` and ``fov_degrees`` default to the
    released multi-view settings. Intrinsics are in pixels for a square ``image_size`` image.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    coordinate_system = CoordinateSystem(coordinate_system)
    z_up = coordinate_system in (CoordinateSystem.RIGHT_HANDED_Z_UP, CoordinateSystem.LEFT_HANDED_Z_UP)
    up = torch.tensor([0.0, 0.0, 1.0] if z_up else [0.0, 1.0, 0.0])
    world_to_camera = []
    for index in range(count):
        u, v = index / count, _radical_inverse_base2(index)
        pitch = math.acos(1.0 - 2.0 * u) - math.pi / 2  # elevation in [-pi/2, pi/2]
        yaw = 2.0 * math.pi * v
        horizontal = radius * math.cos(pitch)
        if z_up:
            eye = torch.tensor([horizontal * math.sin(yaw), horizontal * math.cos(yaw), radius * math.sin(pitch)])
        else:
            eye = torch.tensor([horizontal * math.sin(yaw), radius * math.sin(pitch), horizontal * math.cos(yaw)])
        forward = F.normalize(-eye, dim=0)
        right = torch.cross(forward, up, dim=0)
        if float(right.norm()) < 1e-6:  # looking straight along the up axis
            right = torch.tensor([1.0, 0.0, 0.0])
        right = F.normalize(right, dim=0)
        down = torch.cross(forward, right, dim=0)
        camera_to_world = torch.eye(4)
        camera_to_world[:3, 0], camera_to_world[:3, 1], camera_to_world[:3, 2] = right, down, forward
        camera_to_world[:3, 3] = eye
        world_to_camera.append(torch.linalg.inv(camera_to_world))
    focal = 0.5 * image_size / math.tan(math.radians(fov_degrees) / 2)
    intrinsics = torch.tensor([[focal, 0.0, image_size / 2], [0.0, focal, image_size / 2], [0.0, 0.0, 1.0]])
    return CameraRig(
        world_to_camera=torch.stack(world_to_camera),
        intrinsics=intrinsics.expand(count, 3, 3).contiguous(),
        image_sizes=torch.full((count, 2), image_size, dtype=torch.int64),
        coordinate_system=coordinate_system,
    ).to(device)


def _radical_inverse_base2(index: int) -> float:
    result, scale = 0.0, 0.5
    while index:
        result += scale * (index & 1)
        index >>= 1
        scale *= 0.5
    return result


def pixel_barycentrics(
    triangles: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Barycentric weights of every pixel centre in the ``[lower, upper)`` box of each ``(F, 3, 2)`` pixel triangle.

    Coordinates are ``(column, row)`` with pixel centres at ``+ 0.5``. Returns the triangle, column, and row of
    every pair, triangle-major, and its ``(T, 3)`` weights (which may be negative); degenerate triangles are skipped.
    """

    extent = (upper - lower).clamp(min=0)
    counts = extent[:, 0] * extent[:, 1]
    face = torch.repeat_interleave(torch.arange(triangles.shape[0], device=triangles.device), counts)
    offsets = torch.arange(face.shape[0], device=triangles.device) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts, counts
    )
    width = extent[face, 0]
    column = lower[face, 0] + offsets % width
    row = lower[face, 1] + offsets // width
    a, b, c = triangles[face].unbind(dim=1)
    v0, v1, v2 = b - a, c - a, torch.stack([column, row], dim=1).to(triangles.dtype) + 0.5 - a
    denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = denominator.abs() > 1e-12
    denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / denominator
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / denominator
    weights = torch.stack([1.0 - w1 - w2, w1, w2], dim=1)
    return face[valid], column[valid], row[valid], weights[valid]


def rasterize_uv_atlas(mesh: MeshAsset, texture_size: int, *, chunk_faces: int = 65536) -> TexelSurface:
    """Find the surface point under every texel centre covered by a face of the UV atlas.

    ``mesh.uvs`` follow the OpenGL/trimesh convention (``v`` grows upwards), so ``v = 1`` is texture row 0. Texels
    covered by several faces (shared edges) keep the last face written.
    """

    if mesh.uvs is None:
        raise ValueError("mesh must carry uvs; unwrap it first (for example with XAtlasBackend)")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    device = mesh.vertices.device
    uv_pixels = torch.stack([mesh.uvs[:, 0], 1.0 - mesh.uvs[:, 1]], dim=1) * texture_size  # (V, 2) as (col, row)
    triangles_xyz = mesh.vertices[mesh.faces]  # (F, 3, 3)
    face_normals = F.normalize(
        torch.cross(triangles_xyz[:, 1] - triangles_xyz[:, 0], triangles_xyz[:, 2] - triangles_xyz[:, 0], dim=1), dim=1
    )
    texel_index, positions, normals = [], [], []
    for start in range(0, mesh.faces.shape[0], chunk_faces):
        tri_uv = uv_pixels[mesh.faces[start : start + chunk_faces]]  # (f, 3, 2)
        lower = tri_uv.amin(dim=1).floor().clamp(min=0).to(torch.int64)
        upper = (tri_uv.amax(dim=1).ceil().to(torch.int64)).clamp(max=texture_size)
        face_of_pair, column, row, weights = pixel_barycentrics(tri_uv, lower, upper)
        # Slightly negative weights keep texels whose centre sits just outside a face edge (conservative fill).
        inside = (weights >= -1e-3).all(dim=1)
        if not bool(inside.any()):
            continue
        face_of_pair, column, row = face_of_pair[inside], column[inside], row[inside]
        weights = weights[inside].clamp(min=0)
        weights = weights / weights.sum(dim=1, keepdim=True)
        corners = triangles_xyz[start + face_of_pair]  # (t, 3, 3)
        positions.append((weights.unsqueeze(-1) * corners).sum(dim=1))
        normals.append(face_normals[start + face_of_pair])
        texel_index.append(row * texture_size + column)
    if not texel_index:
        raise ValueError("the UV atlas covers no texel; increase texture_size or check the uvs")
    texel_index = torch.cat(texel_index)
    positions = torch.cat(positions)
    normals = torch.cat(normals)
    # Keep one surface point per texel (the last face written wins, like a plain rasterizer).
    order = torch.arange(texel_index.shape[0], device=device)
    last = torch.zeros(texture_size * texture_size, dtype=torch.int64, device=device).scatter(0, texel_index, order)
    keep = last[texel_index] == order
    return TexelSurface(texel_index[keep], positions[keep], normals[keep])


def bake_texture(
    surface: TexelSurface,
    cameras: CameraRig,
    colors: torch.Tensor,
    depths: torch.Tensor,
    alphas: torch.Tensor,
    *,
    texture_size: int,
    depth_tolerance: float = 0.03,
    alpha_threshold: float = 0.5,
    fill_iterations: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the observations of every texel's surface point over the views that see it.

    ``colors`` ``(N, 3, H, W)``, ``depths`` ``(N, 1, H, W)`` (camera-space z), and ``alphas`` ``(N, 1, H, W)`` are
    the renders for ``cameras``. A view counts when the point projects inside the image onto an opaque pixel
    (``alpha >= alpha_threshold``) and its depth agrees with the render within ``depth_tolerance``; each
    observation is weighted by the cosine between the face and the view direction (its magnitude, so a mesh
    with inverted winding still bakes; the depth test is what rejects occluded points). Texels nobody saw are filled from
    their neighbours (up to ``fill_iterations`` rings) so seams do not bleed the background colour.

    Returns the ``(texture_size, texture_size, 3)`` texture and a ``(texture_size, texture_size)`` boolean map of
    texels that were actually observed.
    """

    device = surface.positions.device
    if colors.shape[0] != cameras.world_to_camera.shape[0] or depths.shape[0] != colors.shape[0]:
        raise ValueError("colors, depths, and alphas must have one image per camera")
    homogeneous = torch.cat([surface.positions, torch.ones_like(surface.positions[:, :1])], dim=1)  # (T, 4)
    accumulated = torch.zeros(surface.positions.shape[0], 3, device=device, dtype=torch.float32)
    total_weight = torch.zeros(surface.positions.shape[0], device=device, dtype=torch.float32)
    for index in range(cameras.world_to_camera.shape[0]):
        camera_points = homogeneous @ cameras.world_to_camera[index].T  # (T, 4)
        z = camera_points[:, 2]
        in_front = z > 1e-6
        projected = camera_points[:, :3] @ cameras.intrinsics[index].T
        pixel = projected[:, :2] / projected[:, 2:3].clamp(min=1e-6)
        height, width = int(cameras.image_sizes[index, 0]), int(cameras.image_sizes[index, 1])
        grid = torch.stack([pixel[:, 0] / width * 2 - 1, pixel[:, 1] / height * 2 - 1], dim=1)
        inside = in_front & (grid.abs() <= 1).all(dim=1)
        if not bool(inside.any()):
            continue
        samples = F.grid_sample(
            torch.cat(
                [colors[index : index + 1], depths[index : index + 1], alphas[index : index + 1]], dim=1
            ).float(),
            grid.reshape(1, 1, -1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(5, -1)
        rendered_depth, alpha = samples[3], samples[4]
        camera_position = torch.linalg.inv(cameras.world_to_camera[index])[:3, 3]
        to_camera = F.normalize(camera_position - surface.positions, dim=1)
        facing = (surface.normals * to_camera).sum(dim=1).abs()
        visible = inside & (alpha >= alpha_threshold) & ((rendered_depth - z).abs() <= depth_tolerance)
        weight = torch.where(visible, facing, torch.zeros_like(facing))
        accumulated += weight.unsqueeze(1) * samples[:3].T
        total_weight += weight
    observed = total_weight > 0
    texel_colors = accumulated / total_weight.clamp(min=1e-8).unsqueeze(1)

    texture = torch.zeros(texture_size * texture_size, 3, device=device, dtype=torch.float32)
    known = torch.zeros(texture_size * texture_size, dtype=torch.bool, device=device)
    texture[surface.texel_index[observed]] = texel_colors[observed]
    known[surface.texel_index[observed]] = True
    covered = torch.zeros_like(known)
    covered[surface.texel_index] = True
    known = known.reshape(texture_size, texture_size)
    texture = fill_texture(
        texture.reshape(texture_size, texture_size, 3),
        known,
        covered.reshape(texture_size, texture_size),
        iterations=fill_iterations,
    )
    return texture, known


def fill_texture(
    texture: torch.Tensor, known: torch.Tensor, covered: torch.Tensor, *, iterations: int = 32
) -> torch.Tensor:
    """Fill the unknown texels of an ``(H, W, C)`` texture so seams do not bleed the background.

    Known values grow into unknown ``covered`` texels and a two-texel border around every chart, up to
    ``iterations`` rings; the rest of the atlas gets the mean known value so mipmaps do not pull in black.
    """

    height, width, channels = texture.shape
    known_values = texture[known]
    texture = texture.permute(2, 0, 1).unsqueeze(0)
    filled = known.reshape(1, 1, height, width).float()
    wanted = F.max_pool2d(covered.reshape(1, 1, height, width).float(), 5, stride=1, padding=2) > 0
    for _ in range(iterations):
        missing = wanted & (filled == 0)
        if not bool(missing.any()):
            break
        blurred = F.avg_pool2d(texture * filled, 3, stride=1, padding=1)
        weight = F.avg_pool2d(filled, 3, stride=1, padding=1)
        fillable = missing & (weight > 0)
        texture = torch.where(fillable, blurred / weight.clamp(min=1e-8), texture)
        filled = torch.where(fillable, torch.ones_like(filled), filled)
    if known_values.numel():
        background = known_values.mean(dim=0).reshape(1, channels, 1, 1)
        texture = torch.where(filled > 0, texture, background.expand_as(texture))
    return texture[0].permute(1, 2, 0).clamp(0, 1).contiguous()


__all__ = [
    "TexelSurface",
    "bake_texture",
    "fill_texture",
    "pixel_barycentrics",
    "rasterize_uv_atlas",
    "sphere_hammersley_cameras",
]
