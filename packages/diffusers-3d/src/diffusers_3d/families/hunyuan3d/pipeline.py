# Portions of this file reproduce pipeline semantics from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: typed image inputs, MeshAsset outputs.

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from transformers import Dinov2Model

from ...backends import ScikitImageBackend
from ...backends._optional import require
from ...data import ImageInput, validate_image_condition_pixels
from ...data.background import BackgroundRemoval, RembgBackgroundRemover, lacks_alpha
from ...data.image_processing import _quantized_rgba, normalize_imagenet
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DPipelineOutput
from .models import DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, Hunyuan3DDiTModel, load_dinov2
from .scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from .vae import Hunyuan3DShapeVAE


def recenter_image(rgba: np.ndarray, border_ratio: float) -> np.ndarray:
    """Released ``ImageProcessorV2.recenter``: crop to the alpha box, rescale into a padded square, put on white.

    ``rgba`` is ``(height, width, 4)`` uint8; returns a square uint8 RGB image.
    """

    import cv2

    height, width = rgba.shape[:2]
    rows, columns = np.nonzero(rgba[..., 3])
    if rows.size == 0:
        raise ValueError("the image alpha and mask contain no foreground")
    # Upstream excludes the last foreground row and column from the crop.
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    crop_height, crop_width = bottom - top, right - left
    if crop_height == 0 or crop_width == 0:
        raise ValueError("the image foreground must span at least two rows and two columns")
    size = max(height, width)
    scale = int(size * (1 - border_ratio)) / max(crop_height, crop_width)
    resized_height, resized_width = int(crop_height * scale), int(crop_width * scale)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    offset_y, offset_x = (size - resized_height) // 2, (size - resized_width) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = cv2.resize(
        rgba[top:bottom, left:right], (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    alpha = canvas[..., 3:].astype(np.float32) / 255
    return (canvas[..., :3] * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)


def merge_vertices(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``trimesh.Trimesh(vertices, faces)`` with its default processing, as the released export: drops non-finite
    vertices and merges duplicates. Returns float32 vertices and int64 faces.
    """

    mesh = require("trimesh", "portable").Trimesh(vertices.cpu().numpy(), faces.cpu().numpy())
    return torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).to(torch.int64)


class Hunyuan3DImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """Hunyuan3D-2.1 image-to-shape: DINOv2 conditioning, flow-matching DiT, ShapeVAE decoding, marching cubes.

    Inputs must already have their background removed (an alpha channel or a mask); the pipeline recenters the
    foreground like the released ``ImageProcessorV2``. Surface extraction is the released default (dense occupancy
    grid plus scikit-image marching cubes); FlashVDM's hierarchical decoding is an opt-in upstream acceleration
    that is not ported.
    """

    # Upstream's demos never remove backgrounds by default; opting in makes upstream's rembg call.
    background_removal = BackgroundRemoval(
        partial(RembgBackgroundRemover, "u2net", bgcolor=[255, 255, 255, 0]), trigger=lacks_alpha, automatic=False
    )
    # The released conditioner is Meta's DINOv2-L cast to float16. It loads in the pipeline's dtype (float32 unless
    # one is given, like the other components); upstream runs everything in float16.
    external_components = {
        "image_encoder": ExternalComponent(DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, load_dinov2)
    }
    model_cpu_offload_seq = "image_encoder->transformer->vae"
    _lora_loadable_modules = ["transformer"]

    def __init__(
        self,
        image_encoder: Dinov2Model,
        transformer: Hunyuan3DDiTModel,
        vae: Hunyuan3DShapeVAE,
        scheduler: Hunyuan3DFlowMatchEulerDiscreteScheduler,
        image_size: int = 512,
        border_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        warnings.warn(
            "Hunyuan3D-2.1 weights are licensed under the Tencent Hunyuan 3D 2.1 Community License, which does not "
            "apply in the European Union, the United Kingdom, or South Korea. You are responsible for making sure "
            "your use of it complies with that license.",
            UserWarning,
            stacklevel=2,
        )
        self.register_modules(image_encoder=image_encoder, transformer=transformer, vae=vae, scheduler=scheduler)
        self.register_to_config(image_size=image_size, border_ratio=border_ratio)

    def preprocess(self, image: ImageInput | list[ImageInput]) -> torch.Tensor:
        """Recenter each image and resize it to ``image_size`` like upstream; returns float32 values in [-1, 1]."""

        import cv2

        conditions = self.image_conditions(image)
        size = self.config.image_size
        images = []
        for condition in conditions:
            validate_image_condition_pixels(condition)
            square = recenter_image(_quantized_rgba(condition)[1], self.config.border_ratio)
            resized = torch.from_numpy(cv2.resize(square, (size, size), interpolation=cv2.INTER_CUBIC)).permute(
                2, 0, 1
            )
            images.append(resized.float() / 255 * 2 - 1)
        return torch.stack(images).to(self._execution_device)

    def encode_image(self, images: torch.Tensor, do_classifier_free_guidance: bool) -> torch.Tensor:
        """DINOv2 CLS and patch tokens of ``[-1, 1]`` images, then the released all-zero unconditional context."""

        size = self.image_encoder.config.image_size
        images = ((images + 1) / 2).to(self.image_encoder.dtype)
        # torchvision's Resize, as used upstream, interpolates half-precision images in float32.
        images = F.interpolate(
            images.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
        ).to(images.dtype)
        context = self.image_encoder(pixel_values=normalize_imagenet(images)).last_hidden_state
        if do_classifier_free_guidance:
            context = torch.cat([context, torch.zeros_like(context)])
        return context

    def prepare_latents(
        self,
        batch_size: int,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.vae.config.num_latents, self.vae.config.embed_dim)
        if latents is None:
            return randn_tensor(
                shape, generator=generator, device=self._execution_device, dtype=self.transformer.dtype
            )
        if latents.shape != shape:
            raise ValueError(f"latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=self.transformer.dtype)

    def denoise(
        self, latents: torch.Tensor, context: torch.Tensor, num_inference_steps: int, guidance_scale: float
    ) -> torch.Tensor:
        do_classifier_free_guidance = context.shape[0] == 2 * latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            model_input = torch.cat([latents, latents]) if do_classifier_free_guidance else latents
            model_timestep = timestep.expand(model_input.shape[0]).to(latents.dtype)
            model_timestep = model_timestep / self.scheduler.config.num_train_timesteps
            velocity = self.transformer(model_input, model_timestep, context).sample
            if do_classifier_free_guidance:
                conditional, unconditional = velocity.chunk(2)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            latents = self.scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    def decode_occupancy(
        self,
        latents: torch.Tensor,
        bounds: tuple[float, ...],
        octree_resolution: int,
        num_chunks: int,
        query_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Occupancy logits on the ``(octree_resolution + 1) ** 3`` grid spanning ``bounds``, in float32.

        Grid points are cast to ``query_dtype`` (default: the VAE dtype) and then to the VAE dtype.
        """

        decoded = self.vae((1.0 / self.vae.config.scale_factor) * latents).sample
        axes = [
            np.linspace(bounds[axis], bounds[axis + 3], octree_resolution + 1, dtype=np.float32) for axis in range(3)
        ]
        points = torch.from_numpy(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)).reshape(-1, 3)
        points = points.to(device=decoded.device, dtype=query_dtype or decoded.dtype).to(decoded.dtype)
        logits = torch.cat(
            [
                self.vae.query_occupancy(decoded, chunk.expand(decoded.shape[0], -1, -1))
                for chunk in points.split(num_chunks)
            ],
            dim=1,
        )
        return logits.view(decoded.shape[0], *(octree_resolution + 1,) * 3).float()

    @staticmethod
    def extract_mesh(grid: torch.Tensor, bounds: tuple[float, ...], mc_level: float) -> MeshAsset:
        """Marching cubes on one occupancy grid, scaled into ``bounds`` and merged by trimesh like the released export."""

        mesh = ScikitImageBackend().extract_surface(
            grid, level=mc_level, gradient_direction="ascent", allow_degenerate=True
        )
        lower = torch.tensor(bounds[:3], dtype=torch.float64)
        extent = torch.tensor(bounds[3:], dtype=torch.float64) - lower
        # Upstream divides by the number of samples per axis, not the number of cells.
        vertices = mesh.vertices.double() / grid.shape[0] * extent + lower
        vertices, faces = merge_vertices(vertices.float(), mesh.faces)
        return MeshAsset(
            vertices=vertices,
            faces=faces,
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "hunyuan3d-2.1", "octree_resolution": grid.shape[0] - 1, "mc_level": mc_level},
        )

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        octree_resolution: int = 384,
        bounds: float | Sequence[float] = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per image with the released sampler and surface-extraction defaults.

        A negative ``guidance_scale`` disables classifier-free guidance, as upstream. ``bounds`` is a half extent
        or six ``(x_min, y_min, z_min, x_max, y_max, z_max)`` values; the occupancy grid has
        ``octree_resolution + 1`` samples per axis and is evaluated ``num_chunks`` points at a time.
        """

        bounds = (-bounds,) * 3 + (bounds,) * 3 if isinstance(bounds, (int, float)) else tuple(bounds)
        images = self.preprocess(image)
        context = self.encode_image(images, do_classifier_free_guidance=guidance_scale >= 0)
        latents = self.prepare_latents(images.shape[0], generator, latents)
        latents = self.denoise(latents, context, num_inference_steps, guidance_scale)
        grids = self.decode_occupancy(latents, bounds, octree_resolution, num_chunks)
        meshes = tuple(self.extract_mesh(grid, bounds, mc_level) for grid in grids)
        latent_output = (
            Latent3DOutput(latents=latents, metadata={"family": "hunyuan3d-2.1"}) if return_latents else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["Hunyuan3DImageTo3DPipeline", "merge_vertices", "recenter_image"]
