# Portions of this file reproduce StepFun Step1X-3D geometry pipeline semantics:
# https://github.com/stepfun-ai/Step1X-3D
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified for typed image conditions, a dense occupancy grid, and the package marching-cubes backend.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Dinov2WithRegistersModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...backends import ScikitImageBackend
from ...data import ImageCondition
from ...data.image_processing import _quantized_rgba, validate_image_condition_pixels
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.metadata import ContributionStatus, Object3DComponentSpec, ReviewStatus, fully_qualified_class_name
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DKind, Object3DPipelineOutput
from .models import Step1X3DTransformer1DModel
from .scheduler import Step1X3DFlowMatchEulerScheduler
from .vae import Step1X3DShapeDecoder


def preprocess_step1x3d_image(
    condition: ImageCondition, *, image_size: int = 518, foreground_ratio: float = 0.95
) -> torch.Tensor:
    """Released Step1X-3D framing: crop to the non-transparent bounding box, composite on white, scale the crop to
    ``foreground_ratio`` of the original frame, centre it on a white square, then resize bicubically.

    Alpha (times the optional mask) selects the foreground; a fully opaque image is used as a full-frame crop, so no
    background-removal model ever runs. Returns a ``(3, image_size, image_size)`` RGB tensor in ``[0, 1]``.
    """

    validate_image_condition_pixels(condition)
    if not 0 < foreground_ratio <= 1:
        raise ValueError("foreground_ratio must lie in (0, 1]")
    pixels, rgba = _quantized_rgba(condition)
    image = Image.fromarray(rgba)
    width, height = image.size
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("alpha and mask must contain at least one non-transparent pixel")
    left, upper, right, lower = box
    scale = min(height * foreground_ratio / (lower - upper), width * foreground_ratio / (right - left))
    foreground = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image).crop(box)
    foreground = foreground.resize((int((right - left) * scale), int((lower - upper) * scale)))
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(
        foreground,
        (
            (side - width) // 2 + (width - foreground.width) // 2,
            (side - height) // 2 + (height - foreground.height) // 2,
        ),
    )
    canvas = np.asarray(canvas.resize((image_size, image_size), Image.Resampling.BICUBIC)).copy()
    return torch.from_numpy(canvas).permute(2, 0, 1).to(device=pixels.device, dtype=pixels.dtype) / 255


class Step1X3DImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """Step1X-3D image-to-geometry: DINOv2 tokens condition a rectified-flow transformer over the shape VAE's latent
    set, the VAE decoder evaluates occupancy on a dense grid, and marching cubes extracts a watertight mesh.

    Meshes are returned raw. The released pipeline's pymeshlab clean-up (floater removal, quadric decimation to 200k
    faces) and trimesh smooth shading are not applied.
    """

    family_id = "step1x3d"
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
            ("image_encoder", Dinov2WithRegistersModel),
            ("transformer", Step1X3DTransformer1DModel),
            ("vae", Step1X3DShapeDecoder),
            ("scheduler", Step1X3DFlowMatchEulerScheduler),
        )
    )
    model_cpu_offload_seq = "image_encoder->transformer->vae"
    _lora_loadable_modules = ["transformer"]

    def __init__(
        self,
        image_encoder: Dinov2WithRegistersModel,
        transformer: Step1X3DTransformer1DModel,
        vae: Step1X3DShapeDecoder,
        scheduler: Step1X3DFlowMatchEulerScheduler,
    ) -> None:
        super().__init__()
        if image_encoder.config.hidden_size != transformer.config.cross_attention_dim:
            raise ValueError("image encoder hidden size must match the transformer cross_attention_dim")
        if transformer.config.in_channels != vae.config.latent_channels:
            raise ValueError("transformer in_channels must match the VAE latent_channels")
        self.register_modules(image_encoder=image_encoder, transformer=transformer, vae=vae, scheduler=scheduler)

    def preprocess(
        self, image: ImageCondition | Sequence[ImageCondition] | torch.Tensor, *, foreground_ratio: float = 0.95
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim not in (3, 4):
                raise ValueError("image tensor must have shape (channels, height, width) or be batched")
            conditions = tuple(ImageCondition(image=value) for value in (image[None] if image.ndim == 3 else image))
        elif type(image) is ImageCondition:
            conditions = (image,)
        elif isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
            conditions = tuple(image)
            if not conditions or any(type(condition) is not ImageCondition for condition in conditions):
                raise TypeError("image sequences must contain exact ImageCondition values")
        else:
            raise TypeError("image must be an ImageCondition, sequence of ImageCondition values, or tensor")
        images = [
            preprocess_step1x3d_image(
                condition, image_size=self.image_encoder.config.image_size, foreground_ratio=foreground_ratio
            )
            for condition in conditions
        ]
        return torch.stack(images).to(device=self._execution_device)

    def encode_conditioning(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(batch, 3, height, width)`` RGB images in ``[0, 1]``; other sizes are resized bicubically.

        Returns the final layer-normed class and patch tokens with the registers dropped, and the all-zero null
        condition the released checkpoints were trained with.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        size = self.image_encoder.config.image_size
        images = images.to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)
        if images.shape[-2:] != (size, size):
            images = F.interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, 3, 1, 1).to(images)
        std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, 3, 1, 1).to(images)
        hidden_states = self.image_encoder((images - mean) / std).last_hidden_state
        registers = self.image_encoder.config.num_register_tokens
        conditional = torch.cat([hidden_states[:, :1], hidden_states[:, 1 + registers :]], dim=1)
        return conditional, torch.zeros_like(conditional)

    def prepare_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.vae.config.num_latents, self.transformer.config.in_channels)
        dtype = self.transformer.dtype
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}")
            return latents.to(device=self._execution_device, dtype=dtype)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=dtype)

    def sample_latents(
        self,
        latents: torch.Tensor,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        zero_init_steps: int = 1,
    ) -> torch.Tensor:
        """Euler flow sampling with ``[negative, conditional]`` classifier-free guidance when ``guidance_scale > 1``.

        The velocity of the first ``zero_init_steps`` steps is zeroed, as the released pipeline does.
        """

        do_guidance = guidance_scale > 1
        context = torch.cat([negative, conditional]) if do_guidance else conditional
        context = context.to(dtype=latents.dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_input = torch.cat([latents] * 2) if do_guidance else latents
            velocity = self.transformer(model_input, timestep.expand(model_input.shape[0]), context).sample
            if do_guidance:
                negative_velocity, conditional_velocity = velocity.chunk(2)
                velocity = negative_velocity + guidance_scale * (conditional_velocity - negative_velocity)
            if index < zero_init_steps:
                velocity = velocity * 0.0
            latents = self.scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    def extract_meshes(
        self,
        latents: torch.Tensor,
        *,
        bounds: float = 1.05,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        chunk_size: int = 65536,
    ) -> tuple[MeshAsset, ...]:
        """Decode a dense occupancy grid per latent set and run Lewiner marching cubes at ``mc_level``.

        Vertices use the released mapping ``index / (octree_resolution + 1) * 2 * bounds - bounds``.
        """

        backend = ScikitImageBackend()
        grids = self.vae.decode_grid(
            latents.to(dtype=self.vae.dtype), bounds=bounds, resolution=octree_resolution, chunk_size=chunk_size
        )
        meshes = []
        for grid in grids:
            surface = backend.extract_surface(grid, level=mc_level, gradient_direction="ascent", allow_degenerate=True)
            vertices = surface.vertices.double() / (octree_resolution + 1) * (2 * bounds) - bounds
            meshes.append(
                MeshAsset(
                    vertices=vertices.to(device=latents.device, dtype=torch.float32),
                    faces=surface.faces.to(device=latents.device),
                    coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
                    metadata={"family": "step1x3d", "representation": "mesh", "octree_resolution": octree_resolution},
                )
            )
        return tuple(meshes)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        foreground_ratio: float = 0.95,
        bounds: float = 1.05,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        chunk_size: int = 65536,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per input image (defaults follow the released ``inference.py``)."""

        images = self.preprocess(image, foreground_ratio=foreground_ratio)
        conditional, negative = self.encode_conditioning(images)
        latents = self.prepare_latents(images.shape[0], generator=generator, latents=latents)
        latents = self.sample_latents(
            latents, conditional, negative, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        meshes = self.extract_meshes(
            latents, bounds=bounds, octree_resolution=octree_resolution, mc_level=mc_level, chunk_size=chunk_size
        )
        latent_output = Latent3DOutput(latents=latents, metadata={"family": "step1x3d"}) if return_latents else None
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["Step1X3DImageTo3DPipeline", "preprocess_step1x3d_image"]
