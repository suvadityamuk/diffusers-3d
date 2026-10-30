# Portions of this file reproduce image preprocessing, mesh extraction, and the NeRF renderer from TripoSR:
# https://github.com/VAST-AI-Research/TripoSR
# Revision: 107cefdc244c39106fa830359024f6a2f1c78871
#
# MIT License. Copyright (c) 2024 Tripo AI & Stability AI.
# Modified for typed image conditions and device-agnostic PyTorch, with the package's plain-PyTorch port of torchmcubes'
# marching cubes; see this family's README.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers import ViTModel

from ...backends.marching_cubes import marching_cubes, unique_rows_in_order
from ...data import ImageCondition, ImageInput
from ...data.background import BackgroundRemoval, RembgBackgroundRemover, lacks_alpha
from ...data.image_processing import normalize_imagenet, validate_image_condition_pixels
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DPipelineOutput
from .models import TripoSRTransformerModel, TripoSRTriplaneDecoder


def prepare_triposr_image(condition: ImageCondition, *, foreground_ratio: float = 0.85) -> torch.Tensor:
    """Return the ``(3, H, W)`` RGB that upstream ``run.py`` feeds the model.

    With alpha (RGBA and/or ``mask``), the foreground is cropped, padded square, padded again so it fills
    ``foreground_ratio`` of the side, and composited on grey 0.5 in 8 bits, as upstream ``resize_foreground`` and
    ``run.py`` do (including the crop dropping the last foreground row and column). RGB without a mask is used as is,
    like upstream ``--no-remove-bg``.
    """

    validate_image_condition_pixels(condition)
    if not 0 < foreground_ratio <= 1:
        raise ValueError("foreground_ratio must be in (0, 1]")
    pixels = condition.image
    rgb = pixels[:3].expand(3, -1, -1)
    if pixels.shape[0] != 4 and condition.mask is None:
        return rgb
    alpha = pixels[3:4] if pixels.shape[0] == 4 else torch.ones_like(pixels[:1])
    if condition.mask is not None:
        alpha = alpha * condition.mask
    rgba = torch.cat([rgb, alpha]).mul(255).round().to(torch.uint8)
    rows, columns = torch.nonzero(rgba[3], as_tuple=True)
    if rows.numel() == 0:
        raise ValueError("alpha/mask has no foreground pixels")
    foreground = rgba[:, rows.min() : rows.max(), columns.min() : columns.max()]
    height, width = foreground.shape[1:]
    if height == 0 or width == 0:
        raise ValueError("the alpha/mask foreground must span at least two rows and two columns")
    size = max(height, width)
    top, left = (size - height) // 2, (size - width) // 2
    image = F.pad(foreground, (left, size - width - left, top, size - height - top))
    padding = int(size / foreground_ratio) - size
    image = F.pad(image, (padding // 2, padding - padding // 2) * 2).float() / 255
    image = image[:3] * image[3:] + (1 - image[3:]) * 0.5
    return image.mul(255).to(torch.uint8).float() / 255


def _spherical_rays(
    num_views: int, elevation_deg: float, camera_distance: float, fovy_deg: float, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray origins and unit directions ``(num_views, height, width, 3)`` of upstream ``get_spherical_cameras``.

    Cameras orbit the Z axis at evenly spaced azimuths starting on +X, look at the origin with +Z up, and shoot one
    ray through every pixel centre. Upstream crosses vectors without ``dim``, which picks the view axis when
    ``num_views`` is 3; here the cross product is always taken over the last axis.
    """

    azimuth_deg = torch.linspace(0, 360.0, num_views + 1)[:num_views]
    elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    camera_distances = torch.full_like(elevation_deg, camera_distance)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(num_views, 1)
    fovy = torch.full_like(elevation_deg, fovy_deg) * math.pi / 180
    lookat = F.normalize(torch.zeros_like(camera_positions) - camera_positions, dim=-1)
    right = F.normalize(torch.linalg.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.linalg.cross(right, lookat, dim=-1), dim=-1)
    rotation = torch.stack([right, up, -lookat], dim=-1)

    # Pixel-centre directions at unit focal length, normalized, then scaled by the focal length and rotated.
    columns, rows = torch.meshgrid(
        torch.arange(width, dtype=torch.float32) + 0.5, torch.arange(height, dtype=torch.float32) + 0.5, indexing="xy"
    )
    directions = torch.stack([(columns - width / 2) / 1.0, -(rows - height / 2) / 1.0, -torch.ones_like(columns)], -1)
    directions = F.normalize(directions, dim=-1)[None].repeat(num_views, 1, 1, 1)
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)
    directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]
    rays_d = F.normalize((directions[:, :, :, None, :] * rotation[:, None, None, :3, :3]).sum(-1), dim=-1)
    rays_o = camera_positions[:, None, None, :].expand(rays_d.shape)
    return rays_o, rays_d


class TripoSRImageTo3DPipeline(Object3DPipeline):
    """TripoSR feed-forward image-to-3D: ViT image tokens to triplanes, then a NeRF density grid and marching cubes.

    One forward pass, no sampling. Meshes carry per-vertex colours queried from the triplanes and are returned on the
    CPU in TripoSR's right-handed Z-up frame. Like upstream, inputs without usable alpha are cut out with rembg (u2net)
    at full size first; call :meth:`~diffusers_3d.Object3DPipeline.disable_background_removal` to use them as is.
    """

    background_removal = BackgroundRemoval(RembgBackgroundRemover, trigger=lacks_alpha)
    model_cpu_offload_seq = "image_encoder->transformer->decoder"

    def __init__(
        self, image_encoder: ViTModel, transformer: TripoSRTransformerModel, decoder: TripoSRTriplaneDecoder
    ) -> None:
        super().__init__()
        if transformer.config.cross_attention_dim != image_encoder.config.hidden_size:
            raise ValueError("transformer cross_attention_dim must match the image encoder hidden size")
        if transformer.config.out_channels != decoder.config.triplane_channels:
            raise ValueError("transformer out_channels must match the decoder triplane_channels")
        self.register_modules(image_encoder=image_encoder, transformer=transformer, decoder=decoder)

    def preprocess(self, image: ImageInput | list[ImageInput], *, foreground_ratio: float = 0.85) -> torch.Tensor:
        """Turn images into normalized ViT pixel values at the encoder's ``image_size`` (512 for the release)."""

        size = self.image_encoder.config.image_size
        images = [
            F.interpolate(
                prepare_triposr_image(condition, foreground_ratio=foreground_ratio)[None].float(),
                (size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for condition in self.image_conditions(image)
        ]
        pixel_values = normalize_imagenet(torch.cat(images).to(self._execution_device))
        return pixel_values.to(self.image_encoder.dtype)

    @torch.no_grad()
    def extract_mesh(
        self, triplanes: torch.Tensor, *, resolution: int = 256, threshold: float = 25.0, chunk_size: int = 8192
    ) -> MeshAsset:
        """Mesh one triplane ``(1, 3, channels, H, W)`` where its density crosses ``threshold``, as upstream.

        Density is sampled on a ``resolution^3`` grid spanning ``[-radius, radius]^3`` (corners included) and meshed
        with torchmcubes' marching cubes (its CUDA kernel on CUDA, as upstream); vertex colours are decoded at the
        unmerged vertices. As upstream's ``trimesh.Trimesh`` does by default, vertices that agree to 8 decimals are
        then merged, keeping the first. ``MeshAsset.save`` writes the raw Z-up coordinates, as ``run.py`` does.
        """

        if triplanes.ndim != 5 or triplanes.shape[0] != 1:
            raise ValueError("extract_mesh meshes one shape at a time: triplanes must have shape (1, 3, C, H, W)")
        radius = self.decoder.config.radius
        axis = torch.linspace(0, 1, resolution)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 3)
        grid = grid.to(triplanes) * (2 * radius) - radius
        density = self.decoder(triplanes, grid, chunk_size=chunk_size).density
        vertices, faces = marching_cubes((density - threshold).reshape(resolution, resolution, resolution))
        if vertices.shape[0] == 0:
            raise ValueError(f"the density never crosses threshold {threshold}, so there is no surface to mesh")
        vertices = (vertices / (resolution - 1.0)).to(triplanes) * (2 * radius) - radius
        colors = self.decoder(triplanes, vertices[None], chunk_size=chunk_size).color[0]
        keep, index = unique_rows_in_order(vertices.double().mul(1e8).round().long())
        vertices, colors, faces = vertices[keep], colors[keep], index[faces]
        return MeshAsset(
            vertices=vertices.cpu(),
            faces=faces.cpu(),
            colors=colors.cpu(),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
            metadata={
                "family": "triposr",
                "representation": "mesh",
                "resolution": resolution,
                "threshold": threshold,
                "export_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
        )

    @torch.no_grad()
    def render_views(
        self,
        triplanes: torch.Tensor,
        *,
        num_views: int = 30,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        num_samples_per_ray: int = 128,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        """Volume-render triplanes ``(batch, 3, C, H, W)`` from an orbit of cameras, as upstream ``TSR.render``.

        Defaults are upstream ``run.py --render``: 30 views at 256 x 256 around the Z axis, 1.9 units away with a
        40 degree vertical field of view, and 128 evenly spaced samples per ray through the ``[-radius, radius]^3``
        box, composited over white. Returns float frames ``(batch, num_views, height, width, 3)``, not clamped;
        upstream saves ``(frame * 255).astype(uint8)``.
        """

        radius = self.decoder.config.radius
        rays_o, rays_d = _spherical_rays(num_views, elevation_deg, camera_distance, fovy_deg, height, width)
        rays_o, rays_d = rays_o.to(triplanes.device), rays_d.to(triplanes.device)
        box = (1.0 - 1.0e-3) * torch.tensor([[-radius, radius]] * 3, device=triplanes.device)
        t_vals = torch.linspace(0, 1, num_samples_per_ray + 1, device=triplanes.device)
        t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
        deltas = t_vals[1:] - t_vals[:-1]
        frames = []
        for triplane in triplanes:
            for origins, directions in zip(rays_o.flatten(1, 2), rays_d.flatten(1, 2)):
                # Upstream ``rays_intersect_bbox``: the slab test against the slightly shrunken box.
                safe_directions = torch.where(directions.abs() < 1e-6, torch.full_like(directions, 1e-6), directions)
                hit_far = (box[..., 1] - origins) / safe_directions
                hit_near = (box[..., 0] - origins) / safe_directions
                t_near = torch.minimum(hit_far, hit_near).amax(dim=-1).clamp_min(0.0)
                t_far = torch.maximum(hit_far, hit_near).amin(dim=-1)
                hit = t_far - t_near > 0.01
                t_near, t_far = t_near[hit, None], t_far[hit, None]
                z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]
                points = origins[hit, None, :] + z_vals[..., None] * directions[hit, None, :]
                decoded = self.decoder(triplane[None], points.reshape(1, -1, 3), chunk_size=chunk_size)
                density = decoded.density.reshape(*points.shape[:2])
                color = decoded.color.reshape(points.shape)
                alpha = 1 - torch.exp(-deltas * density)
                transmittance = torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)
                weights = alpha * torch.cat([torch.ones_like(alpha[:, :1]), transmittance], dim=-1)
                rgb = torch.zeros(origins.shape[0], 3, device=triplanes.device)
                opacity = torch.zeros(origins.shape[0], device=triplanes.device)
                rgb[hit] = (weights[..., None] * color).sum(dim=-2)
                opacity[hit] = weights.sum(dim=-1)
                rgb += 1 - opacity[..., None]
                frames.append(rgb.view(height, width, 3))
        return torch.stack(frames).unflatten(0, (triplanes.shape[0], num_views))

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        foreground_ratio: float = 0.85,
        mc_resolution: int = 256,
        mc_threshold: float = 25.0,
        chunk_size: int = 8192,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Reconstruct one vertex-coloured mesh per image with upstream ``run.py`` defaults."""

        image_embeds = self.image_encoder(
            self.preprocess(image, foreground_ratio=foreground_ratio), interpolate_pos_encoding=True
        ).last_hidden_state
        triplanes = self.transformer(image_embeds).sample
        meshes = tuple(
            self.extract_mesh(triplane[None], resolution=mc_resolution, threshold=mc_threshold, chunk_size=chunk_size)
            for triplane in triplanes
        )
        latent_output = (
            Latent3DOutput(latents=triplanes, metadata={"family": "triposr", "stage": "triplanes"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["TripoSRImageTo3DPipeline", "prepare_triposr_image"]
