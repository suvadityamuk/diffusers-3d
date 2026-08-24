# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for object-native Diffusers integration.

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...data import HunyuanImageProcessor, ImageCondition
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.pipelines import Object3DPipeline
from ...objects import Latent3DOutput, MeshAsset, Object3DKind, Object3DPipelineOutput
from .conditioner import Hunyuan3DDinov2Conditioner
from .models import Hunyuan3DShapeDiTModel
from .scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from .vae import Hunyuan3DShapeVAE


class Hunyuan3DImageToShapePipeline(Object3DPipeline):
    """Hunyuan3D-2.1 image-conditioned flow matching and dense shape decode."""

    family_id = "hunyuan3d-2.1"
    task_ids = ("image-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("scikit-image",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    model_cpu_offload_seq = "conditioner->denoiser->vae"

    def __init__(
        self,
        conditioner: Hunyuan3DDinov2Conditioner,
        denoiser: Hunyuan3DShapeDiTModel,
        vae: Hunyuan3DShapeVAE,
        scheduler: Hunyuan3DFlowMatchEulerDiscreteScheduler,
        image_processor_size: int = 512,
        image_processor_border_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        if denoiser.config.in_channels != vae.config.embed_dim:
            raise ValueError("denoiser channels must match the shape VAE embedding dimension")
        if denoiser.config.input_size != vae.config.num_latents:
            raise ValueError("denoiser and shape VAE latent sequence lengths must match")
        if denoiser.config.context_dim != conditioner.model.config.hidden_size:
            raise ValueError("denoiser context dimension must match the DINOv2 hidden size")
        if denoiser.config.text_len != conditioner.num_patches:
            raise ValueError("denoiser text_len must match the conditioner token count")

        self.register_modules(
            conditioner=conditioner,
            denoiser=denoiser,
            vae=vae,
            scheduler=scheduler,
        )
        self.register_to_config(
            image_processor_size=image_processor_size,
            image_processor_border_ratio=image_processor_border_ratio,
        )
        self.image_processor = HunyuanImageProcessor(
            size=image_processor_size,
            border_ratio=image_processor_border_ratio,
        )

    def preprocess(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
    ) -> torch.Tensor:
        """Recenter, white-composite, and normalize input images to ``[-1, 1]``."""

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                conditions = (ImageCondition(image=image),)
            elif image.ndim == 4:
                conditions = tuple(ImageCondition(image=item) for item in image)
            else:
                raise ValueError("image tensor must have shape (channels, height, width) or be batched")
        elif type(image) is ImageCondition:
            conditions = (image,)
        elif isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
            conditions = tuple(image)
            if not conditions or any(type(condition) is not ImageCondition for condition in conditions):
                raise TypeError("image sequences must contain exact ImageCondition values")
        else:
            raise TypeError("image must be an ImageCondition, a sequence of ImageCondition values, or a tensor")

        processed = [self.image_processor(condition).image for condition in conditions]
        parameter = next(self.conditioner.parameters())
        return torch.stack(processed).to(device=self._execution_device, dtype=parameter.dtype)

    def encode_conditioning(
        self,
        images: torch.Tensor,
        *,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        """Encode conditional tokens and append released all-zero unconditional tokens."""

        conditional = self.conditioner(images).embeddings
        if not do_classifier_free_guidance:
            return conditional
        unconditional = self.conditioner.unconditional_embedding(
            images.shape[0],
            device=conditional.device,
            dtype=conditional.dtype,
        )
        return torch.cat([conditional, unconditional], dim=0)

    def prepare_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create or validate one Hunyuan shape latent set per input image."""

        parameter = next(self.denoiser.parameters())
        expected_shape = (batch_size, self.vae.config.num_latents, self.vae.config.embed_dim)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        if latents is None:
            return randn_tensor(
                expected_shape,
                generator=generator,
                device=self._execution_device,
                dtype=parameter.dtype,
            )
        if not isinstance(latents, torch.Tensor) or latents.shape != expected_shape:
            raise ValueError(f"latents must have shape {expected_shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

    def denoise(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        sigmas: Sequence[float] | None = None,
        callback_on_step_end: Callable[
            [Hunyuan3DImageToShapePipeline, int, torch.Tensor, dict[str, torch.Tensor]],
            dict[str, torch.Tensor] | None,
        ]
        | None = None,
        callback_steps: int = 1,
    ) -> torch.Tensor:
        """Integrate the released noise-to-shape Euler flow with optional CFG."""

        if not isinstance(callback_steps, int) or isinstance(callback_steps, bool) or callback_steps <= 0:
            raise ValueError("callback_steps must be a positive integer")
        do_classifier_free_guidance = guidance_scale >= 0.0
        expected_context_batch = latents.shape[0] * (2 if do_classifier_free_guidance else 1)
        if encoder_hidden_states.shape[0] != expected_context_batch:
            raise ValueError("encoder_hidden_states batch does not match the guidance mode")

        self.scheduler.set_timesteps(
            None if sigmas is not None else num_inference_steps,
            device=latents.device,
            sigmas=sigmas,
        )
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            model_timestep = timestep.expand(model_input.shape[0]).to(latents.dtype)
            model_timestep = model_timestep / self.scheduler.config.num_train_timesteps
            velocity = self.denoiser(
                model_input,
                model_timestep,
                encoder_hidden_states,
            ).sample
            if do_classifier_free_guidance:
                conditional_velocity, unconditional_velocity = velocity.chunk(2)
                velocity = unconditional_velocity + guidance_scale * (conditional_velocity - unconditional_velocity)
            latents = self.scheduler.step(velocity, timestep, latents, return_dict=False)[0]

            if callback_on_step_end is not None and step_index % callback_steps == 0:
                callback_output = callback_on_step_end(
                    self,
                    step_index,
                    timestep,
                    {"latents": latents},
                )
                if callback_output is not None:
                    if not isinstance(callback_output, dict) or set(callback_output).difference({"latents"}):
                        raise ValueError("callback_on_step_end may only return a 'latents' value")
                    latents = callback_output.get("latents", latents)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        sigmas: Sequence[float] | None = None,
        bounds: float | Sequence[float] = 1.01,
        resolution: int = 384,
        level: float = 0.0,
        query_chunk_size: int = 8000,
        return_latents: bool = True,
        callback_on_step_end: Callable[
            [Hunyuan3DImageToShapePipeline, int, torch.Tensor, dict[str, torch.Tensor]],
            dict[str, torch.Tensor] | None,
        ]
        | None = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        images = self.preprocess(image)
        do_classifier_free_guidance = guidance_scale >= 0.0
        conditioning = self.encode_conditioning(
            images,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        latents = self.prepare_latents(
            images.shape[0],
            generator=generator,
            latents=latents,
        )
        latents = self.denoise(
            latents,
            conditioning,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sigmas=sigmas,
            callback_on_step_end=callback_on_step_end,
            callback_steps=callback_steps,
        )
        objects = self.vae.decode_to_meshes(
            latents,
            bounds=bounds,
            resolution=resolution,
            level=level,
            query_chunk_size=query_chunk_size,
        )
        latent_output = (
            Latent3DOutput(latents=latents, metadata={"family": self.family_id}) if return_latents else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return objects, latent_output
        return Object3DPipelineOutput(objects=objects, latents=latent_output)


__all__ = ["Hunyuan3DImageToShapePipeline"]
