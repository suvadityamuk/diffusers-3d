# Portions of this file reproduce pipeline semantics from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: typed image inputs, OpenCV-free preprocessing (OpenCV resize semantics in NumPy), MeshAsset outputs.

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from transformers import Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...backends import ScikitImageBackend
from ...data import ImageCondition, validate_image_condition_pixels
from ...data.image_processing import _quantized_rgba
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.metadata import ContributionStatus, Object3DComponentSpec, ReviewStatus, fully_qualified_class_name
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DKind, Object3DPipelineOutput
from .models import Hunyuan3DDiTModel
from .scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from .vae import Hunyuan3DShapeVAE


def _box_weights(source: int, target: int) -> np.ndarray:
    """``(target, source)`` coverage weights of OpenCV's shrinking ``INTER_AREA`` filter."""

    scale = source / target
    weights = np.zeros((target, source), dtype=np.float32)
    for index in range(target):
        start, end = index * scale, (index + 1) * scale
        width = min(scale, source - start)
        last = min(math.floor(end), source - 1)
        first = min(math.ceil(start), last)
        if first - start > 1e-3:
            weights[index, first - 1] = (first - start) / width
        weights[index, first:last] = 1 / width
        if end - last > 1e-3:
            weights[index, last] = min(end - last, 1.0, width) / width
    return weights


def _fixed_point_taps(source: int, target: int, interpolation: str) -> tuple[np.ndarray, np.ndarray]:
    """Source indices and 11-bit weights of OpenCV's uint8 ``INTER_CUBIC`` and enlarging ``INTER_AREA`` filters."""

    if interpolation == "area":
        # Enlarging INTER_AREA is linear interpolation with area-aligned fractions.
        scale = source / target
        positions = np.floor(np.arange(target) * scale).astype(np.int64)
        fractions = ((np.arange(target) + 1) - (positions + 1) * (target / source)).astype(np.float32)
        fractions = np.where(fractions <= 0, 0, fractions - np.floor(fractions))
        fractions[positions >= source - 1] = 0
        positions = np.minimum(positions, source - 1)
        indices = np.stack([positions, np.minimum(positions + 1, source - 1)], axis=1)
        weights = np.stack([1 - fractions, fractions], axis=1)
    else:
        a = np.float32(-0.75)
        x = ((np.arange(target) + 0.5) * (source / target) - 0.5).astype(np.float32)
        start = np.floor(x).astype(np.int64)
        x = x - start
        w0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
        w1 = ((a + 2) * x - (a + 3)) * x * x + 1
        w2 = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
        indices = np.clip(start[:, None] + np.arange(-1, 3), 0, source - 1)
        weights = np.stack([w0, w1, w2, 1 - w0 - w1 - w2], axis=1)
    return indices, np.rint(weights * 2048).astype(np.int64)


def resize_like_opencv(image: np.ndarray, height: int, width: int, interpolation: str) -> np.ndarray:
    """``cv2.resize`` of a uint8 ``(height, width, channels)`` image with ``"area"`` or ``"cubic"`` interpolation.

    Shrinking ``INTER_AREA`` is a float box filter; enlarging ``INTER_AREA`` and ``INTER_CUBIC`` follow OpenCV's
    11-bit fixed-point separable passes and its vectorized rounding. Results match OpenCV to within one level;
    how many values differ depends on OpenCV's platform-specific SIMD path.
    """

    if interpolation == "area" and height <= image.shape[0] and width <= image.shape[1]:
        resized = np.einsum(
            "yh,hwc,xw->yxc",
            _box_weights(image.shape[0], height),
            image.astype(np.float32),
            _box_weights(image.shape[1], width),
            optimize=True,
        )
        return np.clip(np.floor(resized + 0.5), 0, 255).astype(np.uint8)
    column_index, column_weight = _fixed_point_taps(image.shape[1], width, interpolation)
    row_index, row_weight = _fixed_point_taps(image.shape[0], height, interpolation)
    rows = (image.astype(np.int64)[:, column_index] * column_weight[..., None]).sum(axis=2)
    taps = [(rows[row_index[:, tap]], row_weight[:, tap, None, None]) for tap in range(row_index.shape[1])]
    if interpolation == "area":
        resized = (sum(((values >> 4) * weight) >> 16 for values, weight in taps) + 2) >> 2
    else:
        resized = np.zeros(taps[0][0].shape, dtype=np.float32)
        for values, weight in taps:
            resized = resized + values.astype(np.float32) * (weight.astype(np.float32) / 2**22)
        resized = np.rint(resized)
    return np.clip(resized, 0, 255).astype(np.uint8)


def recenter_image(rgba: np.ndarray, border_ratio: float) -> np.ndarray:
    """Released ``ImageProcessorV2.recenter``: crop to the alpha box, rescale into a padded square, put on white.

    ``rgba`` is ``(height, width, 4)`` uint8; returns a square uint8 RGB image.
    """

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
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resize_like_opencv(
        rgba[top:bottom, left:right], resized_height, resized_width, "area"
    )
    alpha = canvas[..., 3:].astype(np.float32) / 255
    return (canvas[..., :3] * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)


class Hunyuan3DImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """Hunyuan3D-2.1 image-to-shape: DINOv2 conditioning, flow-matching DiT, ShapeVAE decoding, marching cubes.

    Inputs must already have their background removed (an alpha channel or a mask); the pipeline recenters the
    foreground like the released ``ImageProcessorV2``. Surface extraction is the released default (dense occupancy
    grid plus scikit-image marching cubes); FlashVDM's hierarchical decoding is an opt-in upstream acceleration
    that is not ported.
    """

    family_id = "hunyuan3d-2.1"
    task_ids = ("image-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("scikit-image",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    component_specs = tuple(
        Object3DComponentSpec(
            name=name,
            expected_class=fully_qualified_class_name(component_type),
            subfolder=name,
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        )
        for name, component_type in (
            ("image_encoder", Dinov2Model),
            ("transformer", Hunyuan3DDiTModel),
            ("vae", Hunyuan3DShapeVAE),
            ("scheduler", Hunyuan3DFlowMatchEulerDiscreteScheduler),
        )
    )
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

    def preprocess(self, image: ImageCondition | Sequence[ImageCondition] | torch.Tensor) -> torch.Tensor:
        """Recenter each image and resize it to ``image_size`` like upstream; returns float32 values in [-1, 1]."""

        if isinstance(image, torch.Tensor):
            conditions = [ImageCondition(image=item) for item in (image if image.ndim == 4 else image[None])]
        elif isinstance(image, ImageCondition):
            conditions = [image]
        else:
            conditions = list(image)
        size = self.config.image_size
        images = []
        for condition in conditions:
            validate_image_condition_pixels(condition)
            square = recenter_image(_quantized_rgba(condition)[1], self.config.border_ratio)
            resized = torch.from_numpy(resize_like_opencv(square, size, size, "cubic")).permute(2, 0, 1)
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
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, 3, 1, 1).to(images)
        std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, 3, 1, 1).to(images)
        context = self.image_encoder(pixel_values=(images - mean) / std).last_hidden_state
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
    ) -> torch.Tensor:
        """Occupancy logits on the ``(octree_resolution + 1) ** 3`` grid spanning ``bounds``, in float32."""

        decoded = self.vae((1.0 / self.vae.config.scale_factor) * latents).sample
        axes = [
            np.linspace(bounds[axis], bounds[axis + 3], octree_resolution + 1, dtype=np.float32) for axis in range(3)
        ]
        points = torch.from_numpy(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)).reshape(-1, 3)
        points = points.to(device=decoded.device, dtype=decoded.dtype)
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
        """Marching cubes on one occupancy grid, scaled into ``bounds`` like the released extractor."""

        mesh = ScikitImageBackend().extract_surface(
            grid, level=mc_level, gradient_direction="ascent", allow_degenerate=True
        )
        lower = torch.tensor(bounds[:3], dtype=torch.float64)
        extent = torch.tensor(bounds[3:], dtype=torch.float64) - lower
        # Upstream divides by the number of samples per axis, not the number of cells.
        vertices = mesh.vertices.double() / grid.shape[0] * extent + lower
        return MeshAsset(
            vertices=vertices.float(),
            faces=mesh.faces,
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "hunyuan3d-2.1", "octree_resolution": grid.shape[0] - 1, "mc_level": mc_level},
        )

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
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


__all__ = ["Hunyuan3DImageTo3DPipeline", "recenter_image", "resize_like_opencv"]
