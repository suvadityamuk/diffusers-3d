# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for typed, object-native Diffusers stages.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from ...data import ImageCondition
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    Latent3DOutput,
    Object3D,
    Object3DKind,
    Object3DPipelineOutput,
    SparseVoxelAsset,
)
from .conditioner import TrellisDinov2Conditioner
from .decoders import TrellisSLatGaussianDecoder, TrellisSLatMeshDecoder, TrellisSparseStructureDecoder
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .scheduler import TrellisFlowEulerScheduler
from .sparse import TrellisSparseTensor


class TrellisImageTo3DPipeline(Object3DPipeline):
    """Two-stage TRELLIS image pipeline with a reviewed portable sparse-structure path."""

    family_id = "trellis"
    task_ids = ("image-to-3d",)
    output_object_types = (SparseVoxelAsset,)
    output_representations = ("sparse-structure",)
    object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    model_cpu_offload_seq = (
        "conditioner->sparse_structure_flow_model->sparse_structure_decoder->"
        "slat_flow_model->gaussian_decoder->mesh_decoder"
    )
    _optional_components = [
        "slat_flow_model",
        "slat_scheduler",
        "gaussian_decoder",
        "mesh_decoder",
    ]

    def __init__(
        self,
        conditioner: TrellisDinov2Conditioner,
        sparse_structure_flow_model: TrellisSparseStructureFlowModel,
        sparse_structure_decoder: TrellisSparseStructureDecoder,
        sparse_structure_scheduler: TrellisFlowEulerScheduler,
        slat_flow_model: TrellisSLatFlowModel | None = None,
        slat_scheduler: TrellisFlowEulerScheduler | None = None,
        gaussian_decoder: TrellisSLatGaussianDecoder | None = None,
        mesh_decoder: TrellisSLatMeshDecoder | None = None,
        slat_mean: Sequence[float] | None = None,
        slat_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if sparse_structure_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
            raise ValueError("sparse-structure conditioner and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")
        if slat_flow_model is not None:
            if slat_scheduler is None:
                raise ValueError("slat_scheduler is required when slat_flow_model is provided")
            if slat_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
                raise ValueError("SLAT conditioner and flow context dimensions must match")
            if slat_mean is None or slat_std is None:
                raise ValueError("SLAT normalization is required when slat_flow_model is provided")
            if (
                len(slat_mean) != slat_flow_model.config.out_channels
                or len(slat_std) != slat_flow_model.config.out_channels
            ):
                raise ValueError("SLAT normalization must contain one value per output feature channel")
            if any(float(value) <= 0 for value in slat_std):
                raise ValueError("SLAT standard deviations must be positive")
        elif any(component is not None for component in (slat_scheduler, gaussian_decoder, mesh_decoder)):
            raise ValueError("SLAT decoders and scheduler require slat_flow_model")
        if gaussian_decoder is not None and slat_flow_model is not None:
            if gaussian_decoder.config.latent_channels != slat_flow_model.config.out_channels:
                raise ValueError("SLAT flow output channels must match the Gaussian decoder")

        self.register_modules(
            conditioner=conditioner,
            sparse_structure_flow_model=sparse_structure_flow_model,
            sparse_structure_decoder=sparse_structure_decoder,
            sparse_structure_scheduler=sparse_structure_scheduler,
            slat_flow_model=slat_flow_model,
            slat_scheduler=slat_scheduler,
            gaussian_decoder=gaussian_decoder,
            mesh_decoder=mesh_decoder,
        )
        self.register_to_config(
            slat_mean=None if slat_mean is None else [float(value) for value in slat_mean],
            slat_std=None if slat_std is None else [float(value) for value in slat_std],
        )

    def preprocess(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
    ) -> torch.Tensor:
        """Validate typed images, apply an explicit mask/alpha, and produce RGB ``[0, 1]`` tensors."""

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

        processed = []
        for condition in conditions:
            pixels = condition.image
            if pixels.shape[0] == 1:
                pixels = pixels.expand(3, -1, -1)
            elif pixels.shape[0] == 4:
                pixels = pixels[:3] * pixels[3:4]
            if condition.mask is not None:
                pixels = pixels * condition.mask
            if not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0) | (pixels > 1)).any()):
                raise ValueError("TRELLIS input images must contain finite values in [0, 1]")
            pixels = F.interpolate(
                pixels.unsqueeze(0),
                size=(self.conditioner.image_size, self.conditioner.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            processed.append(pixels)
        parameter = next(self.conditioner.parameters())
        return torch.stack(processed).to(device=self._execution_device, dtype=parameter.dtype)

    def encode_conditioning(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode released normalized image tokens and an all-zero unconditional branch."""

        conditional = self.conditioner(images, value_range=(0.0, 1.0)).embeddings
        unconditional = self.conditioner.unconditional_embedding(
            images.shape[0],
            device=conditional.device,
            dtype=conditional.dtype,
        )
        return conditional, unconditional

    def prepare_sparse_structure_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.sparse_structure_flow_model
        shape = (
            batch_size,
            model.config.in_channels,
            model.config.resolution,
            model.config.resolution,
            model.config.resolution,
        )
        parameter = next(model.parameters())
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        if latents is None:
            return randn_tensor(
                shape,
                generator=generator,
                device=self._execution_device,
                dtype=parameter.dtype,
            )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"sparse-structure latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

    def sample_sparse_structure(
        self,
        latents: torch.Tensor,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
    ) -> torch.Tensor:
        """Sample dense sparse-structure latents with released TRELLIS CFG semantics."""

        if len(guidance_interval) != 2 or not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        scheduler = self.sparse_structure_scheduler
        scheduler.set_timesteps(
            num_inference_steps,
            device=latents.device,
            rescale_t=rescale_t,
        )
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(
                timestep,
                latents.shape[0],
                device=latents.device,
            )
            conditional_velocity = self.sparse_structure_flow_model(
                latents,
                model_timestep,
                conditional,
            ).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unconditional_velocity = self.sparse_structure_flow_model(
                    latents,
                    model_timestep,
                    unconditional,
                ).sample
                velocity = scheduler.apply_guidance(
                    conditional_velocity,
                    unconditional_velocity,
                    guidance_scale,
                )
            else:
                velocity = conditional_velocity
            latents = scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    def extract_sparse_structure(self, latents: torch.Tensor) -> tuple[SparseVoxelAsset, ...]:
        return self.sparse_structure_decoder.decode_to_sparse_voxels(latents)

    def prepare_slat_latents(
        self,
        sparse_structures: tuple[SparseVoxelAsset, ...],
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        if self.slat_flow_model is None:
            raise RuntimeError("SLAT sampling requires slat_flow_model")
        coordinates = TrellisSparseTensor.from_sparse_voxel_assets(sparse_structures).coordinates
        expected_channels = self.slat_flow_model.config.in_channels
        if latents is not None:
            if not torch.equal(latents.coordinates, coordinates) or latents.channels != expected_channels:
                raise ValueError("SLAT latents must match extracted coordinates and model input channels")
            return latents.to(device=self._execution_device, dtype=next(self.slat_flow_model.parameters()).dtype)
        if isinstance(generator, list):
            if len(generator) != len(sparse_structures):
                raise ValueError("a generator list must contain one generator per sparse structure")
            feature_parts = []
            for batch_index, item_generator in enumerate(generator):
                count = int((coordinates[:, 0] == batch_index).sum())
                feature_parts.append(
                    randn_tensor(
                        (count, expected_channels),
                        generator=item_generator,
                        device=self._execution_device,
                        dtype=next(self.slat_flow_model.parameters()).dtype,
                    )
                )
            features = torch.cat(feature_parts)
        else:
            features = randn_tensor(
                (coordinates.shape[0], expected_channels),
                generator=generator,
                device=self._execution_device,
                dtype=next(self.slat_flow_model.parameters()).dtype,
            )
        return TrellisSparseTensor(coordinates.to(device=features.device), features)

    def sample_slat(
        self,
        latents: TrellisSparseTensor,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
    ) -> TrellisSparseTensor:
        """Sample and denormalize portable SLAT features over fixed sparse coordinates."""

        if self.slat_flow_model is None or self.slat_scheduler is None:
            raise RuntimeError("SLAT sampling requires slat_flow_model and slat_scheduler")
        if len(guidance_interval) != 2 or not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        self.slat_scheduler.set_timesteps(
            num_inference_steps,
            device=latents.device,
            rescale_t=rescale_t,
        )
        for timestep in self.slat_scheduler.timesteps:
            model_timestep = self.slat_scheduler.model_timestep(
                timestep,
                latents.batch_size,
                device=latents.device,
            )
            conditional_velocity = self.slat_flow_model(
                latents,
                model_timestep,
                conditional,
            ).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unconditional_velocity = self.slat_flow_model(
                    latents,
                    model_timestep,
                    unconditional,
                ).sample
                velocity = self.slat_scheduler.apply_guidance(
                    conditional_velocity,
                    unconditional_velocity,
                    guidance_scale,
                )
            else:
                velocity = conditional_velocity
            latents = self.slat_scheduler.step(velocity, timestep, latents).prev_sample
        mean = latents.features.new_tensor(self.config.slat_mean)
        std = latents.features.new_tensor(self.config.slat_std)
        return latents.denormalize(mean, std)

    def decode_slat(
        self,
        slat: TrellisSparseTensor,
        *,
        formats: tuple[str, ...],
    ) -> tuple[Object3D, ...]:
        objects: list[Object3D] = []
        if "slat" in formats:
            objects.extend(slat.to_sparse_voxel_assets(resolution=self.slat_flow_model.config.resolution))
        if "gaussian" in formats:
            if self.gaussian_decoder is None:
                raise RuntimeError("format 'gaussian' requires gaussian_decoder")
            objects.extend(self.gaussian_decoder(slat).assets)
        if "mesh" in formats:
            if self.mesh_decoder is None:
                raise RuntimeError("format 'mesh' requires mesh_decoder")
            mesh_output = self.mesh_decoder(slat)
            if mesh_output is not None:
                objects.extend(mesh_output)
        return tuple(objects)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        formats: tuple[str, ...] | list[str] = ("sparse_structure",),
        sparse_structure_num_inference_steps: int = 25,
        slat_num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: torch.Tensor | None = None,
        slat_latents: TrellisSparseTensor | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        formats = tuple(formats)
        allowed_formats = {"sparse_structure", "slat", "gaussian", "mesh"}
        if not formats or len(set(formats)) != len(formats) or set(formats).difference(allowed_formats):
            raise ValueError(f"formats must contain unique values from {sorted(allowed_formats)}")
        if any(name in formats for name in ("slat", "gaussian", "mesh")) and self.slat_flow_model is None:
            raise RuntimeError("formats requiring SLAT need slat_flow_model and slat_scheduler")
        if "gaussian" in formats and self.gaussian_decoder is None:
            raise RuntimeError("format 'gaussian' requires gaussian_decoder")
        if "mesh" in formats and self.mesh_decoder is None:
            raise RuntimeError("format 'mesh' requires mesh_decoder")
        images = self.preprocess(image)
        conditional, unconditional = self.encode_conditioning(images)
        sparse_structure_latents = self.prepare_sparse_structure_latents(
            images.shape[0],
            generator=generator,
            latents=sparse_structure_latents,
        )
        sparse_structure_latents = self.sample_sparse_structure(
            sparse_structure_latents,
            conditional,
            unconditional,
            num_inference_steps=sparse_structure_num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
        )
        sparse_structures = self.extract_sparse_structure(sparse_structure_latents)
        objects: list[Object3D] = []
        if "sparse_structure" in formats:
            objects.extend(sparse_structures)

        requires_slat = any(name in formats for name in ("slat", "gaussian", "mesh"))
        if requires_slat:
            slat_latents = self.prepare_slat_latents(
                sparse_structures,
                generator=generator,
                latents=slat_latents,
            )
            slat = self.sample_slat(
                slat_latents,
                conditional,
                unconditional,
                num_inference_steps=slat_num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                rescale_t=rescale_t,
            )
            objects.extend(self.decode_slat(slat, formats=formats))
        latent_output = (
            Latent3DOutput(
                latents=sparse_structure_latents,
                metadata={"family": self.family_id, "stage": "sparse_structure"},
            )
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(objects=tuple(objects), latents=latent_output)


__all__ = ["TrellisImageTo3DPipeline"]
