"""Triangle rasterizer: per-pixel face index, barycentrics, and depth on any device.

On CUDA, ``align_corners=True`` (``custom_rasterizer`` mode) runs the ``custom_rasterizer`` extension itself, which
Hunyuan3D-2.1 Paint and the Step1X-3D texture baker call: its float32 edge tests differ from any restatement on a few
pixels per view. Everywhere else the rasterizer is plain PyTorch. It follows ``custom_rasterizer``'s semantics (an
independent implementation, no upstream code): clip-space vertices, pixel centres at
``+ 0.5``, row index growing with NDC ``y`` (no flip), closed barycentric coverage of both windings, a z-buffer on
window depth ``z/w * 0.49999 + 0.5`` quantized to ``2^-18`` with ties going to the lowest face index, and
perspective-correct barycentrics. Every triangle is enumerated over the pixel centres in its bounding box (shared
with :func:`~.texture_baking.rasterize_uv_atlas`) and the nearest fragment per pixel is kept with ``scatter_reduce``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..objects import CameraRig, MeshAsset
from ._optional import require
from .texture_baking import pixel_barycentrics

CUSTOM_RASTERIZER_INSTALL = (
    'pip install --no-build-isolation "git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git@'
    '82920d643c0dc2f7bfd7255f45f62d386edfe60c#subdirectory=hy3dpaint/custom_rasterizer" (needs the CUDA toolkit)'
)


@dataclass(frozen=True)
class Rasterization:
    """Per-pixel visibility of ``N`` rasterized views."""

    face_index: torch.Tensor  # (N, H, W) int64, -1 where no face covers the pixel centre
    barycentrics: torch.Tensor  # (N, H, W, 3) perspective-correct, zero on the background
    depth: torch.Tensor  # (N, H, W) NDC z/w of the visible surface, zero on the background


def rasterize(
    clip_positions: torch.Tensor,
    faces: torch.Tensor,
    image_size: tuple[int, int],
    *,
    align_corners: bool = False,
    cull_backfaces: bool = False,
    chunk_faces: int = 65536,
) -> Rasterization:
    """Rasterize ``(V, 4)`` or ``(N, V, 4)`` homogeneous clip-space vertices into ``N`` ``(height, width)`` views.

    ``align_corners=True`` puts NDC ``-1``/``+1`` on the centres of the border pixels, as ``custom_rasterizer``
    does; ``False`` puts them on the image edges, as OpenGL and nvdiffrast do. Fragments with negative window depth
    (just past NDC ``z = -1``) are dropped, there is no far plane, and triangles with a vertex at ``w <= 0`` are
    skipped rather than clipped. ``cull_backfaces`` keeps only triangles that are counter-clockwise in NDC (``x``
    right, ``y`` up). ``face_index`` is 0-based; ``custom_rasterizer`` and nvdiffrast report ``face_index + 1``.

    On CUDA, ``align_corners=True`` without culling runs ``custom_rasterizer_kernel`` (an optional dependency, see
    :data:`CUSTOM_RASTERIZER_INSTALL`) as the released pipelines do; its barycentrics are perspective-correct and
    ``depth`` interpolates NDC ``z/w`` with them.
    """

    positions = clip_positions if clip_positions.ndim == 3 else clip_positions[None]
    if align_corners and not cull_backfaces and positions.is_cuda:
        return _custom_rasterize(positions, faces, image_size)
    views, face_count = positions.shape[0], faces.shape[0]
    height, width = image_size
    w = positions[..., 3:]
    ndc = positions[..., :3] / w
    size = ndc.new_tensor([width, height])
    xy = (ndc[..., :2] * 0.5 + 0.5) * (size - 1) + 0.5 if align_corners else (ndc[..., :2] * 0.5 + 0.5) * size
    corners = xy[:, faces].reshape(-1, 3, 2)  # (N * F, 3, 2) as (column, row)
    corner_z, corner_w = ndc[..., 2][:, faces].reshape(-1, 3), w[..., 0][:, faces].reshape(-1, 3)
    corner_depth = corner_z * 0.49999 + 0.5
    keep = (corner_w > 0).all(dim=1)
    if cull_backfaces:
        edges = corners[:, 1:] - corners[:, :1]
        keep &= edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0] > 0
    lower = (corners.amin(dim=1) - 0.5).ceil().clamp(min=0).to(torch.int64)
    upper = torch.minimum((corners.amax(dim=1) - 0.5).floor().to(torch.int64) + 1, size.to(torch.int64))
    upper = torch.where(keep[:, None], upper, lower)

    pixels = height * width
    empty = torch.iinfo(torch.int64).max
    keys = torch.full((views * pixels,), empty, dtype=torch.int64, device=positions.device)
    for start in range(0, corners.shape[0], chunk_faces):
        stop = start + chunk_faces
        face, column, row, weights = pixel_barycentrics(corners[start:stop], lower[start:stop], upper[start:stop])
        face = face + start
        depth = (weights * corner_depth[face]).sum(dim=1)
        hit = (weights >= 0).all(dim=1) & (depth >= 0)
        face, depth = face[hit], depth[hit]
        key = (depth * 2**18).to(torch.int64) * 2**32 + face % face_count
        keys.scatter_reduce_(0, (face // face_count) * pixels + row[hit] * width + column[hit], key, "amin")

    covered = (keys != empty).nonzero()[:, 0]
    view, pixel = covered // pixels, covered % pixels
    face = view * face_count + keys[covered] % 2**32
    column, row = pixel % width, pixel // width
    origin = torch.stack([column, row], dim=1)
    _, _, _, weights = pixel_barycentrics(corners[face], origin, origin + 1)
    depth = (weights * corner_z[face]).sum(dim=1)
    weights = weights / corner_w[face]
    face_index = torch.full((views * pixels,), -1, dtype=torch.int64, device=positions.device)
    face_index[covered] = face % face_count
    barycentrics = torch.zeros(views * pixels, 3, dtype=ndc.dtype, device=positions.device)
    barycentrics[covered] = weights / weights.sum(dim=1, keepdim=True)
    depth_map = torch.zeros(views * pixels, dtype=ndc.dtype, device=positions.device)
    depth_map[covered] = depth
    return Rasterization(
        face_index.reshape(views, height, width),
        barycentrics.reshape(views, height, width, 3),
        depth_map.reshape(views, height, width),
    )


def _custom_rasterize(positions: torch.Tensor, faces: torch.Tensor, image_size: tuple[int, int]) -> Rasterization:
    """``custom_rasterizer.rasterize`` of every view, as ``MeshRender.raster_rasterize`` calls it."""

    kernel = require("custom_rasterizer_kernel", CUSTOM_RASTERIZER_INSTALL)
    height, width = image_size
    triangles = faces.to(torch.int32).contiguous()
    face_index, barycentrics = [], []
    for view in positions.to(torch.float32):
        found, weights = kernel.rasterize_image(view.contiguous(), triangles, torch.zeros(0), width, height, 1e-6, 0)
        face_index.append(found.long() - 1)
        barycentrics.append(weights)
    face_index, barycentrics = torch.stack(face_index), torch.stack(barycentrics)
    ndc_z = positions[..., 2] / positions[..., 3]
    view = torch.arange(len(positions), device=positions.device)[:, None, None]
    corners = faces.long()[face_index.clamp(min=0)]
    depth = sum(barycentrics[..., k] * ndc_z[view, corners[..., k]] for k in range(3))
    return Rasterization(face_index, barycentrics.to(positions.dtype), depth.to(positions.dtype))


def interpolate(attributes: torch.Tensor, rasterization: Rasterization, faces: torch.Tensor) -> torch.Tensor:
    """Interpolate ``(V, C)`` (or per-view ``(N, V, C)``) vertex attributes into ``(N, H, W, C)`` maps.

    ``faces`` indexes ``attributes`` and must list triangles in the rasterized order; it may differ from the
    rasterized faces (for example a separate UV index buffer). The background is zero.
    """

    views = rasterization.face_index.shape[0]
    attributes = attributes.expand(views, *attributes.shape) if attributes.ndim == 2 else attributes
    corners = faces[rasterization.face_index.clamp(min=0)]  # (N, H, W, 3)
    view = torch.arange(views, device=corners.device)[:, None, None]
    weights = rasterization.barycentrics.to(attributes.dtype)
    return sum(weights[..., k, None] * attributes[view, corners[..., k]] for k in range(3))


def rasterize_mesh(mesh: MeshAsset, cameras: CameraRig) -> dict[str, torch.Tensor]:
    """Rasterize ``mesh`` (with its ``transform``) for every OpenCV pinhole camera of ``cameras``.

    Pixel ``(row, column)`` spans ``[column, column + 1) x [row, row + 1)`` of the intrinsics' image plane, as in
    :func:`~.texture_baking.bake_texture`. Returns ``face_index`` ``(N, H, W)`` and ``barycentrics``
    ``(N, H, W, 3)`` as in :class:`Rasterization`, plus camera-space ``depth`` and binary ``alpha`` ``(N, 1, H, W)``.
    """

    if mesh.coordinate_system is not cameras.coordinate_system:
        raise ValueError("mesh and cameras must use the same coordinate system")
    if not bool((cameras.image_sizes == cameras.image_sizes[:1]).all()):
        raise ValueError("cameras must share one image size")
    height, width = (int(value) for value in cameras.image_sizes[0].tolist())
    world = mesh.vertices @ mesh.transform[:3, :3].T + mesh.transform[:3, 3]
    camera = torch.cat([world, torch.ones_like(world[:, :1])], dim=1) @ cameras.world_to_camera.transpose(1, 2)
    z = camera[..., 2]
    # Fit near/far around the mesh so the 2^-18 depth quantization spans only its depth range.
    near = 0.5 * torch.where(z > 0, z, torch.inf).amin(dim=1)
    far = 2.0 * z.amax(dim=1)
    projection = torch.zeros(z.shape[0], 4, 4, dtype=camera.dtype, device=camera.device)
    projection[:, :2, :3] = 2 * cameras.intrinsics[:, :2] / camera.new_tensor([width, height])[:, None]
    projection[:, :2, 2] -= 1
    projection[:, 2, 2] = (far + near) / (far - near)
    projection[:, 2, 3] = -2 * far * near / (far - near)
    projection[:, 3, 2] = 1
    rasterization = rasterize(camera @ projection.transpose(1, 2), mesh.faces, (height, width))
    depth = interpolate(z[..., None], rasterization, mesh.faces)
    return {
        "face_index": rasterization.face_index,
        "barycentrics": rasterization.barycentrics,
        "depth": depth.permute(0, 3, 1, 2),
        "alpha": (rasterization.face_index >= 0).to(depth.dtype)[:, None],
    }


__all__ = ["CUSTOM_RASTERIZER_INSTALL", "Rasterization", "interpolate", "rasterize", "rasterize_mesh"]
