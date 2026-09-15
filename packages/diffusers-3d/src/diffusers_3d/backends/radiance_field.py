"""Pure-PyTorch volume renderer for :class:`RadianceFieldAsset`.

This is an independent implementation of the tri-vector radiance field's definition (see the asset docstring):
rays are marched through the active voxels at ``samples_per_cell`` samples per trivec cell, the field is evaluated
by linear interpolation, and colours are alpha-composited front to back. It does not derive from the restricted
CUDA rasterizer TRELLIS uses and makes no claim of pixel parity with it; it exists so the asset can be inspected on
any device.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..objects import CameraRig, RadianceFieldAsset

SH_C0 = 0.28209479177387814


def sample_radiance_field(
    asset: RadianceFieldAsset,
    voxel_rows: torch.Tensor,
    local_coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Density and RGB of the field at ``local_coordinates`` ``(K, 3)`` in ``[0, 1]^3`` of voxels ``voxel_rows``.

    Samples sit at cell centres (``local * dim - 0.5``); the outermost half cells extrapolate the boundary
    segment, so the field is continuous across voxel faces only through the learned values.
    """

    dim = asset.dim
    position = local_coordinates * dim - 0.5
    lower = position.floor().clamp(0, dim - 2).to(torch.int64)  # (K, 3)
    weight = (position - lower).to(asset.trivec.dtype)
    trivec = asset.trivec[voxel_rows]  # (K, rank, 3, dim)
    gather_index = lower.unsqueeze(1).unsqueeze(-1).expand(-1, trivec.shape[1], -1, 1)
    start = torch.gather(trivec, 3, gather_index).squeeze(-1)
    end = torch.gather(trivec, 3, gather_index + 1).squeeze(-1)
    per_axis = start + (end - start) * weight.unsqueeze(1)  # (K, rank, 3)
    component = per_axis.prod(dim=2)  # (K, rank)
    raw_density = (asset.density[voxel_rows] * component).sum(dim=1)
    shift = float(asset.density_shift)
    density = F.softplus(raw_density - shift * 10.0) * min(1.0 / (1.0 - shift), 25.0)
    color = torch.sigmoid((SH_C0 * asset.color_coefficients[voxel_rows] * component.unsqueeze(-1)).sum(dim=1))
    return density, color


def render_radiance_field(
    asset: RadianceFieldAsset,
    cameras: CameraRig,
    *,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    samples_per_cell: int = 2,
    jitter: float = 0.5,
    ray_chunk: int = 4096,
) -> Mapping[str, torch.Tensor]:
    """Volume-render every camera; returns ``color`` ``(N, 3, H, W)``, ``depth`` and ``alpha`` ``(N, 1, H, W)``.

    ``depth`` is the expected camera-space z of the surface (alpha-weighted mean, zero where nothing was hit).
    ``jitter`` in ``[0, 1)`` offsets the sample positions along
    every ray by that fraction of a step (the released renderer draws it at random; ``0.5`` is deterministic).
    """

    if type(asset) is not RadianceFieldAsset or type(cameras) is not CameraRig:
        raise TypeError("asset must be a RadianceFieldAsset and cameras a CameraRig")
    if asset.coordinate_system is not cameras.coordinate_system:
        raise ValueError("asset and cameras must use the same coordinate system")
    if asset.device != cameras.device:
        raise ValueError("asset and cameras must be on the same device")
    if samples_per_cell <= 0 or not 0.0 <= jitter < 1.0:
        raise ValueError("samples_per_cell must be positive and jitter in [0, 1)")
    device, dtype = asset.device, torch.float32
    resolution = asset.resolution
    grid_to_world = (asset.transform @ asset.grid_transform).to(dtype)
    world_to_grid = torch.linalg.inv(grid_to_world)
    index_map = torch.full((resolution**3,), -1, dtype=torch.int64, device=device)
    flat_coordinates = (
        asset.coordinates[:, 0] * resolution + asset.coordinates[:, 1]
    ) * resolution + asset.coordinates[:, 2]
    index_map[flat_coordinates] = torch.arange(asset.coordinates.shape[0], device=device)
    step_grid = 1.0 / (samples_per_cell * asset.dim)
    background_color = torch.tensor(background, dtype=dtype, device=device)
    if background_color.shape != (3,):
        raise ValueError("background must have three components")

    outputs = {"color": [], "depth": [], "alpha": []}
    for camera_index in range(cameras.world_to_camera.shape[0]):
        height, width = int(cameras.image_sizes[camera_index, 0]), int(cameras.image_sizes[camera_index, 1])
        intrinsics = cameras.intrinsics[camera_index].to(dtype)
        camera_to_world = torch.linalg.inv(cameras.world_to_camera[camera_index].to(dtype))
        rows, columns = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype) + 0.5,
            torch.arange(width, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        camera_directions = torch.stack(
            [
                (columns - intrinsics[0, 2]) / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones_like(rows),
            ],
            dim=-1,
        ).reshape(-1, 3)
        world_directions = F.normalize(camera_directions @ camera_to_world[:3, :3].T, dim=1)
        cos_z = world_directions @ camera_to_world[:3, 2]
        # grid_transform maps coordinates to voxel centres; shift by half a cell so voxel g spans [g, g + 1).
        origin_grid = world_to_grid[:3, :3] @ camera_to_world[:3, 3] + world_to_grid[:3, 3] + 0.5
        directions_grid = world_directions @ world_to_grid[:3, :3].T
        grid_speed = directions_grid.norm(dim=1)  # grid units travelled per world unit
        color = torch.zeros(world_directions.shape[0], 3, dtype=dtype, device=device)
        depth = torch.zeros(world_directions.shape[0], dtype=dtype, device=device)
        alpha = torch.zeros(world_directions.shape[0], dtype=dtype, device=device)
        for start in range(0, world_directions.shape[0], ray_chunk):
            chunk = slice(start, start + ray_chunk)
            chunk_color, chunk_depth, chunk_alpha = _march(
                asset,
                index_map,
                origin_grid,
                directions_grid[chunk],
                grid_speed[chunk],
                cos_z[chunk],
                step_grid=step_grid,
                jitter=jitter,
            )
            color[chunk], depth[chunk], alpha[chunk] = chunk_color, chunk_depth, chunk_alpha
        color = color + (1.0 - alpha).unsqueeze(1) * background_color
        outputs["color"].append(color.T.reshape(3, height, width))
        outputs["depth"].append(depth.reshape(1, height, width))
        outputs["alpha"].append(alpha.reshape(1, height, width))
    return {name: torch.stack(values) for name, values in outputs.items()}


def _march(
    asset: RadianceFieldAsset,
    index_map: torch.Tensor,
    origin: torch.Tensor,
    directions: torch.Tensor,
    grid_speed: torch.Tensor,
    cos_z: torch.Tensor,
    *,
    step_grid: float,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite one chunk of rays given in grid units (``origin`` ``(3,)``, ``directions`` ``(R, 3)``).

    ``directions`` are the world unit vectors mapped into grid space, so ``grid_speed`` grid units correspond to
    one world unit; marching uses unit grid vectors and converts distances back through ``grid_speed``.
    """

    resolution = asset.resolution
    device = directions.device
    directions = directions / grid_speed.clamp(min=1e-12).unsqueeze(1)
    safe_directions = torch.where(directions.abs() < 1e-12, torch.full_like(directions, 1e-12), directions)
    inverse = 1.0 / safe_directions
    t_low = (0.0 - origin) * inverse
    t_high = (resolution - origin) * inverse
    t_enter = torch.minimum(t_low, t_high).amax(dim=1).clamp(min=0.0)
    t_exit = torch.maximum(t_low, t_high).amin(dim=1)
    hits = t_exit > t_enter
    num_rays = directions.shape[0]
    color = torch.zeros(num_rays, 3, dtype=torch.float32, device=device)
    depth = torch.zeros(num_rays, dtype=torch.float32, device=device)
    alpha = torch.zeros(num_rays, dtype=torch.float32, device=device)
    if not bool(hits.any()):
        return color, depth, alpha
    span = (t_exit - t_enter).clamp(min=0.0)
    num_steps = int(math.ceil(float(span.max()) / step_grid))
    if num_steps == 0:
        return color, depth, alpha
    # Sample positions are placed on a global lattice so neighbouring rays sample consistently.
    first = (t_enter / step_grid - jitter).ceil()
    steps = first.unsqueeze(1) + torch.arange(num_steps, device=device, dtype=torch.float32)
    t = (steps + jitter) * step_grid  # (R, S)
    valid = hits.unsqueeze(1) & (t <= t_exit.unsqueeze(1))
    points = origin + t.unsqueeze(-1) * directions.unsqueeze(1)  # (R, S, 3)
    cells = points.floor()
    inside = valid & ((cells >= 0) & (cells < resolution)).all(dim=-1)
    cells_long = cells.clamp(0, resolution - 1).to(torch.int64)
    flat = (cells_long[..., 0] * resolution + cells_long[..., 1]) * resolution + cells_long[..., 2]
    rows = index_map[flat]
    occupied = inside & (rows >= 0)
    density = torch.zeros(num_rays, num_steps, dtype=torch.float32, device=device)
    sample_color = torch.zeros(num_rays, num_steps, 3, dtype=torch.float32, device=device)
    if bool(occupied.any()):
        local = (points - cells)[occupied]
        sampled_density, sampled_color = sample_radiance_field(asset, rows[occupied], local)
        density[occupied] = sampled_density.to(torch.float32)
        sample_color[occupied] = sampled_color.to(torch.float32)
    step_world = step_grid / grid_speed.clamp(min=1e-12)  # (R,)
    sample_alpha = (1.0 - torch.exp(-density * step_world.unsqueeze(1))).clamp(max=0.999)
    transmittance = torch.cumprod(1.0 - sample_alpha, dim=1)
    transmittance_before = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    weights = sample_alpha * transmittance_before  # (R, S)
    color = (weights.unsqueeze(-1) * sample_color).sum(dim=1)
    alpha = 1.0 - transmittance[:, -1]
    depth = (weights * t / grid_speed.unsqueeze(1) * cos_z.unsqueeze(1)).sum(dim=1) / alpha.clamp(min=1e-8)
    depth = torch.where(alpha > 0, depth, torch.zeros_like(depth))
    return color, depth, alpha


__all__ = ["SH_C0", "render_radiance_field", "sample_radiance_field"]
