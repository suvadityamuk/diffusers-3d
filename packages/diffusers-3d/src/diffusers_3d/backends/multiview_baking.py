"""Plain-PyTorch multi-view texture baking: back-project generated views onto a UV atlas, merge them, and inpaint.

This is an independent implementation of the ``MeshRender`` baker that Hunyuan3D-2.1 Paint and the Step1X-3D
texture stage share. It renders with :func:`~.mesh_rasterizer.rasterize` in ``custom_rasterizer`` mode
(``align_corners=True``) and needs no OpenCV or compiled extension.

Conventions follow ``MeshRender``. Vertices are in its frame, which for a right-handed Y-up mesh is
``(-x, z, -y)``, and cameras look at the origin (:func:`look_at_view`, :func:`orthographic_projection`). View images
are ``(H, W, C)`` with row ``r`` and column ``c`` at NDC ``y = 2 r / (H - 1) - 1`` and ``x = 2 c / (W - 1) - 1``; in
that frame row 0 is the top of the object. UVs are OpenGL style (``v`` up) and texture row 0 is ``v = 1``.
Faces must be wound counter-clockwise seen from outside in the mesh's own frame. The mirrored frame then turns the
winding normal inwards, which is what the view-angle cosine expects.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..objects import CoordinateSystem, MeshAsset
from .mesh_rasterizer import Rasterization, interpolate, rasterize
from .texture_baking import TexelSurface

CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 80
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_TAN_22_5 = 13573  # round(tan(22.5 degrees) * 2 ** 15), as in OpenCV


def look_at_view(elevation: float, azimuth: float, distance: float) -> torch.Tensor:
    """``(4, 4)`` float32 world-to-camera matrix of a camera at ``distance`` from the origin, looking at it.

    Angles are in degrees with ``MeshRender``'s ``get_mv_matrix`` convention: elevation is negated, azimuth is offset
    by 90 degrees, and up is ``+z``. The camera looks down its ``-z`` axis with ``+y`` up.
    """

    elevation, azimuth = math.radians(-elevation), math.radians(azimuth + 90)
    eye = torch.tensor(
        [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)],
        dtype=torch.float64,
    )
    eye = distance * eye
    forward = -eye / eye.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    up = up / up.norm()
    world_to_camera = torch.eye(4, dtype=torch.float64)
    world_to_camera[:3, :3] = torch.stack([right, up, -forward])
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ eye
    return world_to_camera.float()


def orthographic_projection(scale: float, near: float = 0.1, far: float = 100.0) -> torch.Tensor:
    """``(4, 4)`` float32 OpenGL orthographic projection of the ``scale`` x ``scale`` square around the view axis."""

    half = scale * 0.5
    projection = torch.eye(4, dtype=torch.float64)
    projection[0, 0] = projection[1, 1] = 2 / (half - -half)
    projection[2, 2] = -2 / (far - near)
    projection[2, 3] = -(far + near) / (far - near)
    return projection.float()


def canny(image: torch.Tensor, low: int = CANNY_LOW_THRESHOLD, high: int = CANNY_HIGH_THRESHOLD) -> torch.Tensor:
    """Edges of an ``(H, W)`` uint8 image as a boolean map, with ``cv2.Canny(image, low, high)`` semantics.

    3x3 Sobel gradients with replicated borders, L1 magnitude, non-maximum suppression along OpenCV's quantized
    gradient directions, and 8-connected hysteresis that keeps weak pixels (``> low``) joined to strong ones
    (``> high``).
    """

    values = image.to(torch.float32)[None, None]
    padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
    kernels = torch.stack([_SOBEL_X, _SOBEL_X.T]).to(values)[:, None]
    gradient_x, gradient_y = F.conv2d(padded, kernels)[0].to(torch.int64)
    magnitude = gradient_x.abs() + gradient_y.abs()
    neighbours = F.pad(magnitude, (1, 1, 1, 1))  # magnitudes outside the image are zero

    def shifted(row: int, column: int) -> torch.Tensor:
        height, width = magnitude.shape
        return neighbours[1 + row : 1 + row + height, 1 + column : 1 + column + width]

    x, y = gradient_x.abs(), gradient_y.abs() << 15
    tan_22 = x * _TAN_22_5
    tan_67 = tan_22 + (x << 16)
    horizontal = (magnitude > shifted(0, -1)) & (magnitude >= shifted(0, 1))
    vertical = (magnitude > shifted(-1, 0)) & (magnitude >= shifted(1, 0))
    same_sign = (gradient_x ^ gradient_y) >= 0
    diagonal = torch.where(
        same_sign,
        (magnitude > shifted(-1, -1)) & (magnitude > shifted(1, 1)),
        (magnitude > shifted(-1, 1)) & (magnitude > shifted(1, -1)),
    )
    maximum = torch.where(y < tan_22, horizontal, torch.where(y > tan_67, vertical, diagonal))
    candidates = (maximum & (magnitude > low)).float()[None, None]
    edges = candidates * (magnitude > high)
    while True:
        grown = F.max_pool2d(edges, 3, stride=1, padding=1) * candidates
        if torch.equal(grown, edges):
            return edges[0, 0] > 0
        edges = grown


def rasterize_texels(
    vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor, texture_size: int
) -> TexelSurface:
    """Rasterize the UV atlas of a ``texture_size`` square texture: the surface point under every covered texel.

    Texel ``(r, c)`` samples ``u = c / (size - 1)``, ``v = 1 - r / (size - 1)``; ``texel_index`` is ``r * size + c``.
    """

    clip = torch.stack(
        [uvs[:, 0] * 2 - 1, (1 - uvs[:, 1]) * 2 - 1, torch.zeros_like(uvs[:, 0]), torch.ones_like(uvs[:, 0])], dim=1
    )
    raster = rasterize(clip, faces, (texture_size, texture_size), align_corners=True)
    covered = raster.face_index[0].flatten() >= 0
    corners = vertices[faces]
    normals = F.normalize(torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1), dim=1)
    return TexelSurface(
        texel_index=covered.nonzero()[:, 0],
        positions=interpolate(vertices, raster, faces)[0].flatten(0, 1)[covered],
        normals=normals[raster.face_index[0].flatten()[covered]],
    )


def _render_view(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    image_size: tuple[int, int],
    *,
    unreliable_radius: int,
    max_angle: float,
    smooth_normals: bool,
) -> tuple[Rasterization, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize one view; return the rasterization, camera-space depth, the cosine weight, and reliable pixels.

    The weight is the cosine between the view direction and the surface normal, zeroed beyond ``max_angle`` and on
    pixels that are not reliable: the visibility mask eroded by a ``2 * unreliable_radius + 1`` box, minus the same
    box around Canny edges of the normalized depth.
    """

    camera = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1) @ world_to_camera.T
    raster = rasterize(camera @ projection.T, faces, image_size, align_corners=True)
    camera = camera[:, :3]
    visible = raster.face_index[0] >= 0
    corners = camera[faces]
    face_normals = F.normalize(torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1), dim=1)
    if smooth_normals:
        summed = torch.zeros_like(camera).index_put_((faces.flatten(),), face_normals.repeat_interleave(3, 0), True)
        normal = interpolate(F.normalize(summed, dim=1), raster, faces)[0]
    else:
        normal = torch.where(visible[..., None], face_normals[raster.face_index[0].clamp(min=0)], 0.0)
    depth = interpolate(camera[:, 2:], raster, faces)[0, ..., 0]

    near, far = depth[visible].max(), depth[visible].min()
    normalized = (depth - far) / (near - far) * visible
    edges = canny((normalized * 255).to(torch.uint8))
    size = 2 * unreliable_radius + 1
    hidden = F.max_pool2d((~visible).float()[None, None], size, stride=1, padding=size // 2)[0, 0] > 0
    near_edge = F.max_pool2d(edges.float()[None, None], size, stride=1, padding=size // 2)[0, 0] > 0
    reliable = ~hidden & ~near_edge

    view_direction = normal.new_tensor([0.0, 0.0, -1.0])
    cosine = F.cosine_similarity(view_direction.expand_as(normal), normal, dim=-1)
    cosine = torch.where((cosine < math.cos(max_angle / 180 * math.pi)) | ~reliable, 0.0, cosine)
    return raster, depth, cosine, reliable


def project_view(
    image: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    texels: TexelSurface,
    texture_size: int,
    *,
    unreliable_radius: int,
    max_angle: float = 75.0,
    depth_tolerance: float = 3e-3,
    smooth_normals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project one ``(H, W, C)`` view by sampling it at every texel's surface point (``back_sample``).

    A texel takes the view's colour when its point projects inside the image onto a pixel with a non-zero weight
    (see ``_render_view``) whose rendered camera-space depth agrees within ``depth_tolerance``. Colour is bilinear
    between the pixel ``floor(p)`` that contains the projection ``p`` and its right and lower neighbours, with
    fractions ``p - floor(p)`` (``MeshRender``'s half-pixel convention); depth and weight use pixel ``floor(p)``.
    Hunyuan3D-2.1 Paint bakes this way with flat face normals.

    Returns the ``(texture_size, texture_size, C)`` texture and the ``(texture_size, texture_size)`` weight, both
    zero on texels this view does not see.
    """

    height, width, channels = image.shape
    _, depth, cosine, reliable = _render_view(
        vertices,
        faces,
        world_to_camera,
        projection,
        (height, width),
        unreliable_radius=unreliable_radius,
        max_angle=max_angle,
        smooth_normals=smooth_normals,
    )
    points = torch.cat([texels.positions, torch.ones_like(texels.positions[:, :1])], dim=1) @ world_to_camera.T
    x, y, z = points[:, 0] * projection[0, 0], points[:, 1] * projection[1, 1], points[:, 2]
    column_position, row_position = (x * 0.5 + 0.5) * width, (y * 0.5 + 0.5) * height
    column = ((x.clamp(-1, 1) * 0.5 + 0.5) * width).long().clamp(0, width - 1)
    row = ((y.clamp(-1, 1) * 0.5 + 0.5) * height).long().clamp(0, height - 1)
    weight = cosine[row, column]
    valid = (x.abs() <= 1) & (y.abs() <= 1) & ((z - depth[row, column]).abs() < depth_tolerance)
    valid &= reliable[row, column] & (weight > 0)
    column, row, texel = column[valid], row[valid], texels.texel_index[valid]
    fraction_x = (column_position[valid] - column)[:, None]
    fraction_y = (row_position[valid] - row)[:, None]
    right, below = (column + 1).clamp(max=width - 1), (row + 1).clamp(max=height - 1)
    colors = (image[row, column] * (1 - fraction_x) + image[row, right] * fraction_x) * (1 - fraction_y)
    colors = colors + (image[below, column] * (1 - fraction_x) + image[below, right] * fraction_x) * fraction_y

    texture = image.new_zeros(texture_size * texture_size, channels)
    texture[texel] = colors
    weights = image.new_zeros(texture_size * texture_size)
    weights[texel] = weight[valid]
    return texture.view(texture_size, texture_size, channels), weights.view(texture_size, texture_size)


def splat_view(
    image: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    texture_size: int,
    *,
    unreliable_radius: int,
    max_angle: float = 75.0,
    smooth_normals: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project one ``(H, W, C)`` view by splatting its reliable pixels into UV space (``linear``).

    Every pixel with a non-zero weight (see ``_render_view``) spreads its colour and weight bilinearly over the four
    texels around its interpolated UV (texel ``(r, c)`` at ``u = c / (size - 1)``, ``v = 1 - r / (size - 1)``), and
    each texel averages what it received. The Hunyuan3D-2 ``MeshRender`` used by Step1X-3D bakes this way with
    smooth vertex normals.

    Returns the ``(texture_size, texture_size, C)`` texture and the ``(texture_size, texture_size)`` weight.
    """

    height, width, channels = image.shape
    raster, _, cosine, reliable = _render_view(
        vertices,
        faces,
        world_to_camera,
        projection,
        (height, width),
        unreliable_radius=unreliable_radius,
        max_angle=max_angle,
        smooth_normals=smooth_normals,
    )
    flipped = torch.stack([1 - uvs[:, 1], uvs[:, 0]], dim=1)
    position = interpolate(flipped, raster, faces)[0][reliable] * (texture_size - 1)  # (row, column)
    corner = position.floor().long().clamp(0, texture_size - 2)
    fraction = position - corner
    values = torch.cat([image[reliable], cosine[reliable][:, None], torch.ones_like(fraction[:, :1])], dim=1)
    accumulated = values.new_zeros(texture_size * texture_size, channels + 2)
    for step_row in (0, 1):
        for step_column in (0, 1):
            weight = (fraction[:, 0] if step_row else 1 - fraction[:, 0]) * (
                fraction[:, 1] if step_column else 1 - fraction[:, 1]
            )
            index = (corner[:, 0] + step_row) * texture_size + corner[:, 1] + step_column
            accumulated.index_put_((index,), values * weight[:, None], accumulate=True)
    count = accumulated[:, -1:]
    accumulated = torch.where(count > 0, accumulated[:, :-1] / count, 0.0)
    accumulated = accumulated.view(texture_size, texture_size, channels + 1)
    return accumulated[..., :channels], accumulated[..., channels]


def merge_views(
    textures: list[torch.Tensor] | tuple[torch.Tensor, ...], weights: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of per-view ``(T, T, C)`` textures under ``(T, T)`` weights, in view order (``fast_bake_texture``).

    A view is skipped when more than 99% of the texels it weights are already covered by earlier views. Callers
    pass each view's weight as ``view_weight * cosine ** exponent``. Returns the merged texture and the boolean map
    of texels some view covered.
    """

    texture, total = torch.zeros_like(textures[0]), torch.zeros_like(weights[0])
    for view_texture, weight in zip(textures, weights):
        seen = weight > 0
        if (seen & (total > 0)).sum() / seen.sum() > 0.99:
            continue
        texture = texture + view_texture * weight[..., None]
        total = total + weight
    return texture / total.clamp(min=1e-8)[..., None], total > 1e-8


def file_geometry(mesh: MeshAsset) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 CPU vertices and faces as ``mesh.save`` writes them to a GLB, which the released pipelines read."""

    rotation = mesh.metadata.get("export_rotation")
    faces = mesh.faces.detach().cpu()
    if rotation is None:
        mesh = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)
        vertices = mesh.vertices
    else:
        matrix = torch.tensor(rotation, dtype=mesh.vertices.dtype, device=mesh.vertices.device)
        vertices = mesh.vertices @ matrix.T
        faces = faces[:, [0, 2, 1]] if torch.det(matrix.double()) < 0 else faces
    transform = mesh.transform.to(vertices)
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    return vertices.detach().cpu().float(), faces


def _round_half_away(values: np.ndarray) -> np.ndarray:
    """C ``round``: halves go away from zero."""

    whole = np.trunc(values)
    return (whole + np.sign(values) * (np.abs(values - whole) >= 0.5)).astype(np.int64)


def _vertex_texels(uvs: np.ndarray, height: int, width: int, arithmetic: str) -> np.ndarray:
    """Flat texel of every vertex as ``meshVerticeInpaint`` finds it, from float32 OpenGL-style ``(V, 2)`` UVs.

    ``MeshRender`` stores ``1 - v`` and hands back ``1 - (1 - v)``, both in float32.
    """

    u, v = uvs[:, 0], np.float32(1) - (np.float32(1) - uvs[:, 1])
    if arithmetic == "cpp":
        # ``round(float * int)`` in float32, ``round((1.0 - float) * int)`` in double.
        column = _round_half_away((u * np.float32(width - 1)).astype(np.float64))
        row = _round_half_away((1.0 - v.astype(np.float64)) * (height - 1))
    else:
        # NumPy 1.x promotes these float32 scalars to float64; Python's ``round`` rounds halves to even.
        column = np.rint(u.astype(np.float64) * (width - 1)).astype(np.int64)
        row = np.rint((1.0 - v.astype(np.float64)) * (height - 1)).astype(np.int64)
    return row * width + column


def _edge_weights(vertices: np.ndarray, source: np.ndarray, target: np.ndarray, arithmetic: str):
    """Inverse squared distance of every graph edge: the float32 weight and the weight that is summed."""

    difference = vertices[source] - vertices[target]  # float32, as both implementations subtract
    if arithmetic == "cpp":
        # ``1.0f / max(sqrt(pow(dx, 2) + ...), 1E-4)`` in double, stored as float and squared in float.
        squared = difference.astype(np.float64) ** 2
        distance = np.sqrt((squared[:, 0] + squared[:, 1]) + squared[:, 2])
        weight = (1.0 / np.maximum(distance, 1e-4)).astype(np.float32)
        weight = weight * weight
        return weight, weight
    # ``np.sqrt(np.sum(d ** 2))`` in float32, then ``1.0 / max(dist, 1e-4)`` squared in float64.
    squared = difference * difference
    distance = np.sqrt((squared[:, 0] + squared[:, 1]) + squared[:, 2])
    distance = distance.astype(np.float64)
    weight = 1.0 / np.where(distance >= 1e-4, distance, 1e-4)
    weight = weight * weight
    return weight.astype(np.float32), weight


def mesh_vertex_inpaint(
    texture: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    arithmetic: str = "cpp",
) -> tuple[np.ndarray, np.ndarray]:
    """``meshVerticeInpaint`` (method ``"smooth"``): colour uncovered vertices from their mesh neighbours.

    ``texture`` is float32 ``(H, W, C)``, ``mask`` uint8 ``(H, W)`` (non-zero where covered), ``vertices`` float32
    ``(V, 3)`` in any mirrored frame, ``faces`` ``(F, 3)``, and ``uvs`` float32 OpenGL-style ``(V, 2)`` indexed by
    ``faces``. A vertex is coloured when the texel at its UV is covered. The others are listed once per face corner,
    in face order, and swept in that order until the count of vertices without a coloured neighbour stops changing
    twice: each takes the inverse-squared-distance weighted mean of its coloured neighbours (the next corner of every
    face it is in) as they are at that moment. Every coloured vertex is then written to its texel (in face-corner
    order) and marked 255 in the mask.

    ``arithmetic="cpp"`` follows the C++ extension of Hunyuan3D-2.1 (float32 sums, distances in double);
    ``"numpy"`` follows Step1X-3D's NumPy port under NumPy 1.x promotion (float32 colour sums, weights and their
    total in float64).
    """

    height, width, channels = texture.shape
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    corners = np.asarray(faces, dtype=np.int64).reshape(-1)
    texel = _vertex_texels(np.asarray(uvs, dtype=np.float32), height, width, arithmetic)[corners]
    flat_texture = texture.reshape(-1, channels)
    covered = mask.reshape(-1)[texel] > 0

    colored = np.zeros(len(vertices), dtype=bool)
    colors = np.zeros((len(vertices), channels), dtype=np.float32)
    # Every corner of a vertex shares its texel, so covered corners all agree on the colour.
    colored[corners[covered]] = True
    colors[corners[covered]] = flat_texture[texel[covered]]
    pending = corners[~covered]

    source = corners
    target = np.asarray(faces, dtype=np.int64)[:, [1, 2, 0]].reshape(-1)
    weight32, weight_sum = _edge_weights(vertices, source, target, arithmetic)
    order = np.argsort(source, kind="stable")
    starts = np.searchsorted(source[order], np.arange(len(vertices) + 1))
    neighbours = {
        vertex: (
            target[order[starts[vertex] : starts[vertex + 1]]],
            weight32[order[starts[vertex] : starts[vertex + 1]]],
            weight_sum[order[starts[vertex] : starts[vertex + 1]]],
        )
        for vertex in np.unique(pending).tolist()
    }

    smooth_count, last_uncolored = 2, 0
    while smooth_count > 0:
        uncolored = 0
        for vertex in pending.tolist():
            targets, weights, summed = neighbours[vertex]
            keep = colored[targets]
            if not keep.any():
                uncolored += 1
                continue
            contributions = colors[targets[keep]] * weights[keep][:, None]
            color_sum = np.add.accumulate(contributions, axis=0)[-1]
            total = np.add.accumulate(summed[keep])[-1]
            colors[vertex] = color_sum / np.float32(total)
            colored[vertex] = True
        smooth_count += -1 if uncolored == last_uncolored else 1
        last_uncolored = uncolored

    texture, mask = texture.copy(), mask.copy()
    # Corners write in face order, so the last corner on a texel wins.
    written = np.flatnonzero(colored[corners])[::-1]
    last = written[np.unique(texel[written], return_index=True)[1]]
    texture.reshape(-1, channels)[texel[last]] = colors[corners[last]]
    mask.reshape(-1)[texel[last]] = 255
    return texture, mask


def inpaint_texture(
    texture: torch.Tensor,
    mask: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    *,
    arithmetic: str = "cpp",
) -> torch.Tensor:
    """Fill the texels of an ``(H, W, C)`` texture outside the boolean ``mask`` (``MeshRender.uv_inpaint``).

    Uncovered vertices take colours from their mesh neighbours (:func:`mesh_vertex_inpaint`); the texture is then
    truncated to 8 bits and every texel still outside the mask is filled by ``cv2.inpaint(..., 3, INPAINT_NS)``,
    three channels at a time (``C`` must be a multiple of 3). Returns the 8-bit result divided by 255 in float32,
    as upstream hands it on.
    """

    import cv2

    height, width, channels = texture.shape
    if channels % 3:
        raise ValueError("inpaint_texture fills three channels at a time")
    texture_np, mask_np = mesh_vertex_inpaint(
        texture.detach().cpu().float().numpy(),
        mask.detach().cpu().numpy().astype(np.uint8) * 255,
        vertices.detach().cpu().float().numpy(),
        faces.detach().cpu().numpy(),
        uvs.detach().cpu().float().numpy(),
        arithmetic=arithmetic,
    )
    quantized = (texture_np * 255).astype(np.uint8)
    filled = [
        cv2.inpaint(np.ascontiguousarray(quantized[..., start : start + 3]), 255 - mask_np, 3, cv2.INPAINT_NS)
        for start in range(0, channels, 3)
    ]
    return torch.from_numpy(np.concatenate(filled, axis=-1) / 255).float().to(texture.device)


__all__ = [
    "CANNY_HIGH_THRESHOLD",
    "CANNY_LOW_THRESHOLD",
    "canny",
    "file_geometry",
    "inpaint_texture",
    "look_at_view",
    "merge_views",
    "mesh_vertex_inpaint",
    "orthographic_projection",
    "project_view",
    "rasterize_texels",
    "splat_view",
]
