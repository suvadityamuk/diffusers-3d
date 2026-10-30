# Portions of this file reproduce Tencent ARC Pixal3D view-aligned projection conditioning:
# https://github.com/TencentARC/Pixal3D
# Revision: f7cf38429b0bd264f1995f0f8743a88b1c728b94
# MIT License. Copyright (c) 2026 Tencent.
#
# The feature upsampler reproduces the semantics of valeo.ai NAF (Neighborhood Attention Filtering):
# https://github.com/valeoai/NAF
# Revision: 37f2dfc180f2de53d98bd601109c0da0dd6b0f43
# Apache License 2.0. Copyright (c) valeo.ai.
#
# Modified for plain PyTorch (no NATTEN), Diffusers rotary helpers, and package-owned cameras.

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel
from ...objects import CameraRig

# Blender camera-to-world of the canonical front view, camera at (0, -distance, 0) looking along +y with +z up.
_FRONT_VIEW = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
# OpenCV camera axes (x right, y down, z forward) to Blender / OpenGL ones (x right, y up, -z forward).
_OPENCV_TO_BLENDER = (1.0, -1.0, -1.0, 1.0)


@dataclass
class Pixal3DNAFUpsamplerOutput(BaseOutput):
    sample: torch.Tensor


def _image_encoder(hidden: int, kernel_size: int, num_layers: int, groups: int) -> nn.Sequential:
    def conv(channels: int) -> nn.Conv2d:
        return nn.Conv2d(channels, hidden, kernel_size, padding=kernel_size // 2, padding_mode="reflect")

    layers: list[nn.Module] = [conv(3)]
    for _ in range(num_layers):
        layers += [nn.GroupNorm(groups, hidden), nn.SiLU(), conv(hidden), nn.GroupNorm(groups, hidden), nn.SiLU()]
        layers.append(conv(hidden))
    return nn.Sequential(*layers)


def neighborhood_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    kernel_size: int,
) -> torch.Tensor:
    """NAF's cross-scale neighborhood attention without NATTEN.

    Upstream nearest-upsamples low-resolution ``key``/``value`` by the integer factor ``d`` and runs a dilated
    ``kernel_size`` NATTEN window. Every dilation group then sees the low-resolution grid, so a query at
    high-resolution ``(i, j)`` attends to the border-clamped ``kernel_size`` window of low-resolution cells around
    ``(i // d, j // d)``; all ``d * d`` queries of a cell share one window, which is gathered once per row.
    """

    batch, query_channels, height, width = query.shape
    low_height, low_width = key.shape[-2:]
    if height % low_height or width % low_width:
        raise ValueError("the output size must be an integer multiple of the feature size")
    if min(low_height, low_width) < kernel_size:
        raise ValueError(f"features must span at least {kernel_size} cells per side")
    scale_y, scale_x = height // low_height, width // low_width
    value_channels = value.shape[1] // num_heads

    def windows(length: int) -> torch.Tensor:
        start = (torch.arange(length, device=query.device) - kernel_size // 2).clamp(0, length - kernel_size)
        return start[:, None] + torch.arange(kernel_size, device=query.device)

    rows, columns = windows(low_height), windows(low_width)
    key = key.to(query.dtype).reshape(batch, num_heads, -1, low_height, low_width)
    value = value.to(query.dtype).reshape(batch, num_heads, value_channels, low_height, low_width)
    query = query.reshape(batch, num_heads, -1, low_height, scale_y, low_width, scale_x)
    output = query.new_empty(batch, num_heads * value_channels, height, width)
    scale = (query_channels // num_heads) ** -0.5

    def window(tensor: torch.Tensor, row: int) -> torch.Tensor:
        # (batch, heads, channels, low_height, low_width) -> (batch, heads, low_width, window, channels)
        return tensor[:, :, :, rows[row]][..., columns].permute(0, 1, 4, 3, 5, 2).flatten(3, 4)

    for row in range(low_height):
        # (batch, heads, low_width, scale_y * scale_x, channels)
        row_queries = query[:, :, :, row].permute(0, 1, 4, 3, 5, 2).flatten(3, 4)
        weights = (row_queries @ window(key, row).transpose(-1, -2) * scale).softmax(dim=-1)
        attended = (weights @ window(value, row)).unflatten(3, (scale_y, scale_x))
        output[:, :, row * scale_y : (row + 1) * scale_y] = attended.permute(0, 1, 5, 3, 2, 4).reshape(
            batch, -1, scale_y, width
        )
    return output


class Pixal3DNAFUpsampler(Object3DModel):
    """NAF: image-guided, VFM-agnostic feature upsampling, used by Pixal3D's high-resolution projection branch.

    A two-branch convolutional encoder turns the guide image into position-aware (RoPE) queries at the output
    size; average-pooled copies of them are the keys over the low-resolution features, which are the values.
    """

    @register_to_config
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        rope_heads: int = 4,
        kernel_size: int = 9,
        rope_base: float = 100.0,
        encoder_layers: int = 2,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if dim % num_heads or dim % (4 * rope_heads):
            raise ValueError("dim must divide into attention heads and 4 * rope_heads")
        self.encoder = _image_encoder(dim // 2, 1, encoder_layers, norm_groups)
        self.semantic_encoder = _image_encoder(dim // 2, 3, encoder_layers, norm_groups)

    def _rotate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Axial 2D RoPE over ``[-1, 1]`` pixel-centre coordinates, rotating channel halves as DINOv3 does."""

        batch, channels, height, width = hidden_states.shape
        head_dim = channels // self.config.rope_heads
        device = hidden_states.device
        periods = self.config.rope_base ** (
            2 * torch.arange(head_dim // 4, device=device, dtype=torch.float32) / (head_dim // 2)
        )
        axes = (torch.arange(0.5, size, device=device, dtype=torch.float32) / size for size in (height, width))
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, 1) * 2.0 - 1.0
        angles = (2 * math.pi * coordinates[:, :, None] / periods).flatten(1, 2).tile(2)
        tokens = hidden_states.reshape(batch, self.config.rope_heads, head_dim, -1).transpose(2, 3)
        tokens = apply_rotary_emb(tokens, (angles.cos(), angles.sin()), use_real_unbind_dim=-2)
        return tokens.transpose(2, 3).reshape(batch, channels, height, width)

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
        *,
        return_dict: bool = True,
    ) -> Pixal3DNAFUpsamplerOutput | tuple[torch.Tensor]:
        """Upsample ``features`` ``(batch, channels, h, w)`` to ``output_size`` guided by ``image`` in ``[0, 1]``."""

        if image.ndim != 4 or image.shape[1] != 3 or features.ndim != 4 or image.shape[0] != features.shape[0]:
            raise ValueError("image and features must be batched (batch, channels, height, width) tensors")
        height, width = output_size
        parameter = next(self.parameters())
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        if image.shape[-2] > 4 * height or image.shape[-1] > 4 * width:
            size = (min(image.shape[-2], 4 * height, 4 * width), min(image.shape[-1], 4 * width, 4 * height))
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        guide = torch.cat([self.encoder(image), self.semantic_encoder(image)], dim=1)
        queries = self._rotate(F.adaptive_avg_pool2d(guide, output_size))
        keys = F.adaptive_avg_pool2d(queries, features.shape[-2:])
        sample = neighborhood_cross_attention(
            queries,
            keys,
            features.to(device=queries.device),
            num_heads=self.config.num_heads,
            kernel_size=self.config.kernel_size,
        )
        if not return_dict:
            return (sample,)
        return Pixal3DNAFUpsamplerOutput(sample=sample)


def front_view_camera(distance: torch.Tensor) -> torch.Tensor:
    """Blender camera-to-world of the canonical front view at ``distance`` (any leading shape)."""

    camera = distance.new_tensor(_FRONT_VIEW).expand(*distance.shape, 4, 4).clone()
    camera[..., 1, 3] = -distance
    return camera


def camera_distance_from_fov(camera_angle_x: float, mesh_scale: float = 1.0) -> float:
    """Upstream ``distance_from_fov``: the front-view distance at which the scaled unit cube spans the image width.

    Computed with upstream's float32 steps so the distance matches exactly.
    """

    focal_pixels = float(16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0)) * 512 / 32.0)
    edge = float(torch.tensor(-1.0) / mesh_scale / 2)
    return focal_pixels * edge / -256.0


def blender_camera_to_world(cameras: CameraRig) -> torch.Tensor:
    """Blender / OpenGL camera-to-world matrices of package (OpenCV world-to-camera) cameras."""

    world_to_camera = cameras.world_to_camera.float()
    return torch.linalg.inv(world_to_camera) * world_to_camera.new_tensor(_OPENCV_TO_BLENDER)


def relative_projection_cameras(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Upstream multi-view ``calc_mat``: ``F @ inv(C_0) @ C_i`` snaps view 0 onto the canonical front view."""

    camera_to_world = camera_to_world.float()
    front = front_view_camera(camera_to_world[0, :3, 3].norm())
    return front @ (torch.linalg.inv(camera_to_world[:1]) @ camera_to_world)


def horizontal_fov(cameras: CameraRig) -> torch.Tensor:
    return 2 * torch.atan(cameras.image_sizes[:, 1].float() / (2 * cameras.intrinsics[:, 0, 0].float()))


def project_grid_features(
    feature_maps: list[torch.Tensor],
    indices: torch.Tensor,
    *,
    grid_resolution: int,
    camera_to_world: torch.Tensor,
    camera_angle_x: torch.Tensor,
    mesh_scale: float,
    image_size: int,
) -> torch.Tensor:
    """Sample ``(channels, height, width)`` maps at the image projections of latent grid cells ``indices``.

    Cell ``i`` sits at ``linspace(-1, 1, grid_resolution)[i]`` (not a cell centre) in a Y-up grid frame that is
    rotated into the Z-up Blender world, scaled by ``1 / (2 * mesh_scale)``, and projected through a pinhole camera
    with a 32 mm sensor and ``image_size`` pixels. Features are sampled bilinearly with border padding and
    concatenated across maps, giving ``(cells, sum(channels))``.
    """

    axis = torch.linspace(-1, 1, grid_resolution, device=indices.device)
    x, y, z = axis[indices].unbind(-1)
    points = torch.stack((x, -z, y), dim=-1) / mesh_scale / 2
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
    world_to_camera = torch.linalg.inv(camera_to_world.float()).to(points.device)
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    focal = 16.0 / torch.tan(camera_angle_x.to(points.device) / 2.0) * image_size / 32.0
    depth = -camera[:, 2] + 1e-8
    pixels = torch.stack(
        (focal * camera[:, 0] / depth + image_size / 2.0, -(focal * camera[:, 1] / depth) + image_size / 2.0), -1
    )
    # Sampled in float32 like upstream: a bfloat16 grid would misplace samples by several pixels at 1024.
    grid = ((pixels + 0.5) / image_size * 2 - 1)[None, :, None]
    samples = [
        F.grid_sample(
            feature_map[None].float(),
            grid.to(device=feature_map.device),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].T
        for feature_map in feature_maps
    ]
    return torch.cat(samples, dim=-1)


# valeo.ai's released NAF checkpoint, which Pixal3D uses unchanged. It loads from this URL, checked against its SHA-256,
# and is never converted or redistributed.
NAF_CHECKPOINT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_CHECKPOINT_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
_NAF_BLOCK_OFFSETS = {"norm1": 0, "conv1": 2, "norm2": 3, "conv2": 5}


def _naf_block_key(key: str) -> str:
    """Map NAF ``image_encoder.{encoder,sem_encoder}.<block>.<layer>`` keys onto the flat ``nn.Sequential`` encoders."""

    branch, index, *rest = key.removeprefix("image_encoder.").split(".")
    if rest[0] in _NAF_BLOCK_OFFSETS:
        index, rest = str(1 + 6 * (int(index) - 1) + _NAF_BLOCK_OFFSETS[rest[0]]), rest[1:]
    return ".".join(["semantic_encoder" if branch == "sem_encoder" else branch, index, *rest])


def _naf_state_dict(path: Path, sha256: str = NAF_CHECKPOINT_SHA256) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path} is not the pinned NAF release checkpoint (sha256 {digest})")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # RoPE periods are derived from ``rope_base``; check the stored copy instead of loading it.
    periods = state_dict.pop("image_encoder.rope.periods")
    if not torch.equal(periods, 100.0 ** (2 * torch.arange(16, dtype=torch.float32) / 32)):
        raise ValueError("NAF RoPE periods differ from the configured rope_base")
    return {_naf_block_key(key): value for key, value in state_dict.items()}


def load_naf_upsampler(
    url: str = NAF_CHECKPOINT_URL,
    *,
    revision: str = NAF_CHECKPOINT_SHA256,
    dtype: torch.dtype | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    **hub_kwargs: Any,
) -> Pixal3DNAFUpsampler:
    """Download valeo.ai's NAF release (``url``, checked against the SHA-256 ``revision``) and convert it on load.

    ``url`` may also be a local ``naf_release.pth``. The file is cached under ``cache_dir`` (by default torch hub's
    checkpoint directory); float32 unless ``dtype`` is given.
    """

    path = Path(url)
    if not path.is_file():
        directory = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
        path = directory / f"naf-{revision[:12]}-{url.rsplit('/', 1)[-1]}"
        if force_download or not path.is_file():
            if local_files_only:
                raise FileNotFoundError(f"the NAF checkpoint is not cached at {path}; download it from {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path), hash_prefix=revision, progress=False)
    model = Pixal3DNAFUpsampler()
    model.load_state_dict(_naf_state_dict(path, revision), strict=True)
    return (model if dtype is None else model.to(dtype)).eval()


__all__ = [
    "NAF_CHECKPOINT_SHA256",
    "NAF_CHECKPOINT_URL",
    "load_naf_upsampler",
    "Pixal3DNAFUpsampler",
    "Pixal3DNAFUpsamplerOutput",
    "blender_camera_to_world",
    "camera_distance_from_fov",
    "front_view_camera",
    "horizontal_fov",
    "neighborhood_cross_attention",
    "project_grid_features",
    "relative_projection_cameras",
]
