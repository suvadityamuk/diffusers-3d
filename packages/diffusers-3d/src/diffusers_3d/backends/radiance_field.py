"""Volume renderer for :class:`RadianceFieldAsset`, TRELLIS ``OctreeRenderer`` semantics.

On CUDA the field goes through upstream's ``diffoctreerast`` kernel with the exact arguments TRELLIS builds. Everywhere
else a plain-PyTorch path reproduces the kernel's semantics: two samples per trivec cell on a camera-anchored lattice
with a random phase per pixel, voxels whose centre is within 0.2 of the camera plane culled, early termination once
transmittance falls below 1e-3, depth ``sum(w * z)`` and ``ssaa``-times supersampling. It is not bitwise-identical.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ..objects import CameraRig, RadianceFieldAsset
from ._optional import require

SH_C0 = 0.28209479177387814
_DIFFOCTREERAST_INSTALL = (
    'pip install --no-build-isolation "git+https://github.com/JeffreyXiang/diffoctreerast.git'
    '@b09c20b84ec3aace4729e6e18a613112320eca3a" (CUDA build; research-only licence), '
    "or move the asset and cameras to the CPU for the plain-PyTorch renderer"
)


def sample_radiance_field(
    asset: RadianceFieldAsset,
    voxel_rows: torch.Tensor,
    local_coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Density and RGB of the field at ``local_coordinates`` ``(K, 3)`` in ``[0, 1]^3`` of voxels ``voxel_rows``.

    Samples sit at cell centres (``local * dim - 0.5``); the outermost half cells extrapolate the boundary
    segment, as in the ``diffoctreerast`` kernel.
    """

    dim, rank = asset.dim, asset.rank
    position = local_coordinates.float() * dim - 0.5
    lower = position.floor().clamp(0, dim - 2).to(torch.int64)  # (K, 3)
    weight = (position - lower).unsqueeze(1)  # (K, 1, 3)
    rows = voxel_rows.reshape(-1, 1, 1)
    ranks = torch.arange(rank, device=rows.device).reshape(1, -1, 1)
    axes = torch.arange(3, device=rows.device).reshape(1, 1, 3)
    trivec = asset.trivec.float()
    start = trivec[rows, ranks, axes, lower.unsqueeze(1)]  # (K, rank, 3)
    end = trivec[rows, ranks, axes, lower.unsqueeze(1) + 1]
    component = (start + (end - start) * weight).prod(dim=2)  # (K, rank)
    raw_density = (asset.density[voxel_rows].float() * component).sum(dim=1)
    shift = float(asset.density_shift)
    # The kernel's softplus switches to the identity above 10, not torch's default 20.
    density = F.softplus(raw_density - shift * 10.0, threshold=10.0) * min(1.0 / (1.0 - shift), 25.0)
    coefficients = SH_C0 * asset.color_coefficients[voxel_rows].float()
    color = torch.sigmoid((coefficients * component.unsqueeze(-1)).sum(dim=1))
    return density, color


def render_radiance_field(
    asset: RadianceFieldAsset,
    cameras: CameraRig,
    *,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    ssaa: int = 4,
    ray_chunk: int = 4096,
) -> Mapping[str, torch.Tensor]:
    """Volume-render every camera; returns ``color`` ``(N, 3, H, W)``, ``depth`` and ``alpha`` ``(N, 1, H, W)``.

    Each view renders at ``ssaa`` times its size and is downsampled with antialiased bilinear filtering, as TRELLIS'
    ``render_utils`` does (``ssaa=4``). ``depth`` is the weighted camera-space z, ``sum(w * z)``, not normalised by
    alpha. Each pixel draws a random sample phase from torch's global generator, so seed it for reproducible renders.
    CUDA assets use ``diffoctreerast`` (optional); other devices use the plain-PyTorch path, which processes
    ``ray_chunk`` rays at a time.
    """

    if type(asset) is not RadianceFieldAsset or type(cameras) is not CameraRig:
        raise TypeError("asset must be a RadianceFieldAsset and cameras a CameraRig")
    if asset.coordinate_system is not cameras.coordinate_system:
        raise ValueError("asset and cameras must use the same coordinate system")
    if asset.device != cameras.device:
        raise ValueError("asset and cameras must be on the same device")
    if isinstance(ssaa, bool) or not isinstance(ssaa, int) or ssaa <= 0:
        raise ValueError("ssaa must be a positive integer")
    if len(background) != 3:
        raise ValueError("background must have three components")
    render = _render_diffoctreerast if asset.device.type == "cuda" else _render_torch
    with torch.no_grad():
        return render(asset, cameras, background=background, ssaa=ssaa, ray_chunk=ray_chunk)


def _downsample(rendered: tuple[torch.Tensor, torch.Tensor, torch.Tensor], height: int, width: int, ssaa: int):
    """``OctreeRenderer.render``'s supersampling: color ``(3, h, w)``, depth and alpha ``(h, w)``, one by one."""

    color, depth, alpha = rendered
    if ssaa > 1:
        size = (height, width)
        options = {"mode": "bilinear", "align_corners": False, "antialias": True}
        color = F.interpolate(color[None], size=size, **options)[0]
        depth = F.interpolate(depth[None, None], size=size, **options)[0, 0]
        alpha = F.interpolate(alpha[None, None], size=size, **options)[0, 0]
    return color, depth.unsqueeze(0), alpha.unsqueeze(0)


def _intrinsics_to_projection(intrinsics: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """TRELLIS ``octree_renderer.intrinsics_to_projection`` (normalized OpenCV intrinsics to an OpenGL matrix)."""

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def _diffoctreerast_settings(
    world_to_camera: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple[int, int], ssaa: int
) -> dict:
    """Camera half of the raster settings, computed op for op like ``OctreeRenderer.render`` and ``render``.

    ``intrinsics`` are in pixels; TRELLIS' are normalized by the image size. Near and far (0.8, 1.6 in
    ``render_utils``) only enter rows of the projection the kernel never reads.
    """

    height, width = image_size
    normalized = intrinsics.clone()
    normalized[0] = normalized[0] / width
    normalized[1] = normalized[1] / height
    if abs(float(normalized[0, 2]) - 0.5) > 1e-6 or abs(float(normalized[1, 2]) - 0.5) > 1e-6:
        raise ValueError("diffoctreerast casts rays through the image centre; the principal point must be centred")
    perspective = _intrinsics_to_projection(normalized, 0.8, 1.6)
    fovx = 2 * torch.atan(0.5 / normalized[0, 0])
    fovy = 2 * torch.atan(0.5 / normalized[1, 1])
    return {
        "image_height": height * ssaa,
        "image_width": width * ssaa,
        "tanfovx": math.tan(fovx * 0.5),
        "tanfovy": math.tan(fovy * 0.5),
        "viewmatrix": world_to_camera.T.contiguous(),
        "projmatrix": (perspective @ world_to_camera).T.contiguous(),
        "campos": torch.inverse(world_to_camera)[:3, 3],
    }


def _render_diffoctreerast(asset, cameras, *, background, ssaa, ray_chunk=None):
    """``OctreeRenderer.render`` through ``diffoctreerast.OctreeTrivecRasterizer`` (TRELLIS ``Strivec`` inputs)."""

    del ray_chunk
    diffoctreerast = require("diffoctreerast", _DIFFOCTREERAST_INSTALL)
    warnings.warn(
        "diffoctreerast is licensed for non-commercial research and evaluation use only. "
        "You are responsible for making sure your use of it complies with that license.",
        UserWarning,
        stacklevel=3,
    )
    resolution, device = asset.resolution, asset.device
    tree_depth = round(math.log2(resolution))
    if 2**tree_depth != resolution or tree_depth > 10:
        raise ValueError("diffoctreerast needs a power-of-two resolution of at most 1024")
    scale = asset.grid_transform[0, 0]
    expected_grid = torch.eye(4, dtype=asset.grid_transform.dtype, device=device) * scale
    expected_grid[:3, 3] = asset.grid_transform[:3, 3]
    expected_grid[3, 3] = 1.0
    if not torch.equal(asset.grid_transform, expected_grid) or float(scale) <= 0:
        raise ValueError("diffoctreerast needs an axis-aligned grid_transform with one positive scale")
    transform = asset.transform.float()
    identity = torch.eye(4, device=device)
    if not torch.equal(transform, identity) and not torch.allclose(
        transform[:3, :3].T @ transform[:3, :3], identity[:3, :3], atol=1e-5
    ):
        raise ValueError("diffoctreerast needs a rigid asset transform")

    # DfsOctree inputs as SLatRadianceFieldDecoder.to_representation builds them; voxel g spans aabb cell g.
    corner = (asset.grid_transform[:3, 3] - 0.5 * scale).float()
    aabb = torch.cat([corner, (scale * resolution).float().expand(3)]).to(torch.float32)
    count, rank = asset.coordinates.shape[0], asset.rank
    positions = (asset.coordinates.float() + 0.5) / resolution
    trivecs = asset.trivec.float().contiguous()
    densities = asset.density.float().contiguous()
    shs = asset.color_coefficients.float().reshape(count, rank, 1, 3).contiguous()
    depths = torch.full((count, 1), tree_depth, dtype=torch.uint8, device=device)
    bg_color = torch.tensor(background, dtype=torch.float32, device=device)

    outputs = {"color": [], "depth": [], "alpha": []}
    for camera_index in range(cameras.world_to_camera.shape[0]):
        height, width = (int(value) for value in cameras.image_sizes[camera_index])
        view = cameras.world_to_camera[camera_index].float()
        if not torch.equal(transform, identity):
            view = view @ transform
        settings = SimpleNamespace(
            bg=bg_color,
            scale_modifier=1.0,
            sh_degree=0,
            with_distloss=False,
            jitter=False,
            debug=False,
            density_shift=float(asset.density_shift),
            used_rank=rank,
            **_diffoctreerast_settings(view, cameras.intrinsics[camera_index].float(), (height, width), ssaa),
        )
        rasterizer = diffoctreerast.OctreeTrivecRasterizer(raster_settings=settings)
        rgb, depth, alpha, _ = rasterizer(
            positions=positions, trivecs=trivecs, densities=densities, shs=shs, depths=depths, aabb=aabb
        )
        color, depth, alpha = _downsample((rgb, depth, alpha), height, width, ssaa)
        outputs["color"].append(color)
        outputs["depth"].append(depth)
        outputs["alpha"].append(alpha)
    return {name: torch.stack(values) for name, values in outputs.items()}


def _render_torch(asset, cameras, *, background, ssaa, ray_chunk=4096):
    """Plain-PyTorch path (any device) following the ``diffoctreerast`` trivec kernel's semantics."""

    device, dtype = asset.device, torch.float32
    resolution = asset.resolution
    grid_to_world = (asset.transform @ asset.grid_transform).to(dtype)
    world_to_grid = torch.linalg.inv(grid_to_world)
    flat_coordinates = (
        asset.coordinates[:, 0] * resolution + asset.coordinates[:, 1]
    ) * resolution + asset.coordinates[:, 2]
    homogeneous_centres = torch.cat(
        [asset.coordinates.to(dtype), torch.ones_like(asset.coordinates[:, :1], dtype=dtype)], dim=1
    )
    world_centres = homogeneous_centres @ grid_to_world.T
    step_grid = 0.5 / asset.dim
    background_color = torch.tensor(background, dtype=dtype, device=device)

    outputs = {"color": [], "depth": [], "alpha": []}
    for camera_index in range(cameras.world_to_camera.shape[0]):
        height, width = (int(value) for value in cameras.image_sizes[camera_index])
        rows_count, columns_count = height * ssaa, width * ssaa
        intrinsics = cameras.intrinsics[camera_index].to(dtype) * ssaa
        world_to_camera = cameras.world_to_camera[camera_index].to(dtype)
        camera_to_world = torch.linalg.inv(world_to_camera)
        # Same draw as diffoctreerast's random_image[..., 2]: one sample phase per (supersampled) pixel.
        jitter = torch.rand((rows_count, columns_count), dtype=dtype, device=device).reshape(-1)
        # Near culling: voxels whose centre sits at camera z <= 0.2 are dropped for this view.
        visible = (world_centres @ world_to_camera[2]) > 0.2
        index_map = torch.full((resolution**3,), -1, dtype=torch.int64, device=device)
        index_map[flat_coordinates[visible]] = torch.nonzero(visible).squeeze(1)
        rows, columns = torch.meshgrid(
            torch.arange(rows_count, device=device, dtype=dtype) + 0.5,
            torch.arange(columns_count, device=device, dtype=dtype) + 0.5,
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
        color = torch.empty(world_directions.shape[0], 3, dtype=dtype, device=device)
        depth = torch.empty(world_directions.shape[0], dtype=dtype, device=device)
        transmittance = torch.empty(world_directions.shape[0], dtype=dtype, device=device)
        for start in range(0, world_directions.shape[0], ray_chunk):
            chunk = slice(start, start + ray_chunk)
            color[chunk], depth[chunk], transmittance[chunk] = _march(
                asset,
                index_map,
                origin_grid,
                directions_grid[chunk],
                grid_speed[chunk],
                cos_z[chunk],
                jitter[chunk],
                step_grid=step_grid,
            )
        color = color + transmittance.unsqueeze(1) * background_color
        size = (rows_count, columns_count)
        rendered = (color.T.reshape(3, *size), depth.reshape(size), (1.0 - transmittance).reshape(size))
        color, depth, alpha = _downsample(rendered, height, width, ssaa)
        outputs["color"].append(color)
        outputs["depth"].append(depth)
        outputs["alpha"].append(alpha)
    return {name: torch.stack(values) for name, values in outputs.items()}


def _march(
    asset: RadianceFieldAsset,
    index_map: torch.Tensor,
    origin: torch.Tensor,
    directions: torch.Tensor,
    grid_speed: torch.Tensor,
    cos_z: torch.Tensor,
    jitter: torch.Tensor,
    *,
    step_grid: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite one chunk of rays given in grid units; returns colour, depth and final transmittance.

    ``directions`` are the world unit vectors mapped into grid space, so ``grid_speed`` grid units correspond to
    one world unit; marching uses unit grid vectors and converts distances back through ``grid_speed``. Samples sit
    at ``(k + jitter) * step`` from the camera, like the kernel's.
    """

    resolution = asset.resolution
    device = directions.device
    num_rays = directions.shape[0]
    color = torch.zeros(num_rays, 3, dtype=torch.float32, device=device)
    depth = torch.zeros(num_rays, dtype=torch.float32, device=device)
    transmittance = torch.ones(num_rays, dtype=torch.float32, device=device)
    directions = directions / grid_speed.clamp(min=1e-12).unsqueeze(1)
    safe_directions = torch.where(directions.abs() < 1e-12, torch.full_like(directions, 1e-12), directions)
    inverse = 1.0 / safe_directions
    t_low = (0.0 - origin) * inverse
    t_high = (resolution - origin) * inverse
    t_enter = torch.minimum(t_low, t_high).amax(dim=1).clamp(min=0.0)
    t_exit = torch.maximum(t_low, t_high).amin(dim=1)
    hits = t_exit > t_enter
    if not bool(hits.any()):
        return color, depth, transmittance
    span = torch.where(hits, t_exit - t_enter, torch.zeros_like(t_exit))
    num_steps = int(math.ceil(float(span.max()) / step_grid)) + 1
    first = (t_enter / step_grid - jitter).ceil()
    steps = first.unsqueeze(1) + torch.arange(num_steps, device=device, dtype=torch.float32)
    t = (steps + jitter.unsqueeze(1)) * step_grid  # (R, S)
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
        density[occupied], sample_color[occupied] = sample_radiance_field(
            asset, rows[occupied], (points - cells)[occupied]
        )
    step_world = step_grid / grid_speed.clamp(min=1e-12)  # (R,)
    sample_alpha = (1.0 - torch.exp(-density * step_world.unsqueeze(1))).clamp(max=0.999)
    survive = 1.0 - sample_alpha
    before = torch.cat([torch.ones_like(survive[:, :1]), torch.cumprod(survive, dim=1)[:, :-1]], dim=1)
    # Early termination: the sample that takes transmittance below 1e-3 is the last one composited.
    kept = before >= 1e-3
    weights = torch.where(kept, sample_alpha * before, torch.zeros_like(before))
    color = (weights.unsqueeze(-1) * sample_color).sum(dim=1)
    depth = (weights * t / grid_speed.unsqueeze(1) * cos_z.unsqueeze(1)).sum(dim=1)
    transmittance = torch.where(kept, survive, torch.ones_like(survive)).prod(dim=1)
    return color, depth, transmittance


__all__ = ["SH_C0", "render_radiance_field", "sample_radiance_field"]
