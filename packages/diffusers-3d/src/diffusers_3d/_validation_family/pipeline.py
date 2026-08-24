# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

import inspect

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

from ..data.conditions import ImageCondition
from ..execution.metadata import ContributionStatus, ReviewStatus
from ..execution.pipelines import Object3DPipeline
from ..objects import MeshAsset, Object3DKind, Object3DPipelineOutput
from .models import ContractReferenceDenoiser, ContractReferenceMeshDecoder


class ContractReferencePipeline(Object3DPipeline):
    """Private deterministic image-to-mesh pipeline for end-to-end validation."""

    family_id = "contract-reference"
    task_ids = ("image-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    model_cpu_offload_seq = "denoiser->mesh_decoder"

    def __init__(
        self,
        denoiser: ContractReferenceDenoiser,
        mesh_decoder: ContractReferenceMeshDecoder,
        scheduler: SchedulerMixin,
    ) -> None:
        super().__init__()
        if denoiser.config.latent_dim != mesh_decoder.config.latent_dim:
            raise ValueError("denoiser and mesh_decoder latent dimensions must match")

        # ModularPipeline cannot create a saveable ComponentSpec from a freshly
        # instantiated nn.Module unless ComponentSpec.load() created it first.
        # That prevents the ordinary local component save/load roundtrip this
        # contract fixture validates, so its stages are explicit public methods.
        self.register_modules(
            denoiser=denoiser,
            mesh_decoder=mesh_decoder,
            scheduler=scheduler,
        )

    def encode_conditioning(self, condition: ImageCondition) -> torch.Tensor:
        """Convert one exact image condition into deterministic channel features."""

        if type(condition) is not ImageCondition:
            raise TypeError("condition must be an exact ImageCondition")
        condition.validate()
        if condition.image.shape[0] != self.denoiser.config.condition_dim:
            raise ValueError(f"condition image must have exactly {self.denoiser.config.condition_dim} channels")
        image = condition.image.to(device=self._execution_device, dtype=self.denoiser.dtype)
        return image.mean(dim=(-2, -1)).unsqueeze(0)

    def prepare_latents(
        self,
        conditioning: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create or validate one latent vector per conditioning row."""

        expected_shape = (conditioning.shape[0], self.denoiser.config.latent_dim)
        if conditioning.ndim != 2 or conditioning.shape[-1] != self.denoiser.config.condition_dim:
            raise ValueError(f"conditioning must have shape (batch, {self.denoiser.config.condition_dim})")
        if latents is None:
            return randn_tensor(
                expected_shape,
                generator=generator,
                device=conditioning.device,
                dtype=conditioning.dtype,
            )
        if not isinstance(latents, torch.Tensor) or latents.shape != expected_shape:
            raise ValueError(f"latents must have shape {expected_shape}")
        return latents.to(device=conditioning.device, dtype=conditioning.dtype)

    def denoise_latents(
        self,
        latents: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        num_inference_steps: int = 2,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Denoise prepared latents with the registered Diffusers scheduler."""

        if (
            not isinstance(num_inference_steps, int)
            or isinstance(num_inference_steps, bool)
            or num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if latents.ndim != 2 or latents.shape != (
            conditioning.shape[0],
            self.denoiser.config.latent_dim,
        ):
            raise ValueError(
                f"latents must have shape (batch, {self.denoiser.config.latent_dim}) matching conditioning"
            )

        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        accepts_generator = "generator" in inspect.signature(self.scheduler.step).parameters
        for timestep in self.scheduler.timesteps:
            model_input = self.scheduler.scale_model_input(latents, timestep)
            noise_prediction = self.denoiser(model_input, timestep, conditioning)
            step_kwargs = {"generator": generator} if accepts_generator else {}
            latents = self.scheduler.step(
                noise_prediction,
                timestep,
                latents,
                return_dict=False,
                **step_kwargs,
            )[0]
        return latents

    def decode_mesh(self, latents: torch.Tensor) -> tuple[MeshAsset, ...]:
        """Decode latent rows into differentiable tensor-native mesh assets."""

        vertices = self.mesh_decoder(latents)
        meshes = []
        for batch_index in range(vertices.shape[0]):
            mesh_vertices = vertices[batch_index]
            meshes.append(
                MeshAsset(
                    vertices=mesh_vertices,
                    faces=self.mesh_decoder.faces.clone(),
                    transform=torch.eye(4, device=mesh_vertices.device, dtype=mesh_vertices.dtype),
                    metadata={"family": self.family_id},
                )
            )
        return tuple(meshes)

    @torch.no_grad()
    def __call__(
        self,
        condition: ImageCondition,
        *,
        num_inference_steps: int = 2,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> Object3DPipelineOutput:
        conditioning = self.encode_conditioning(condition)
        latents = self.prepare_latents(conditioning, generator=generator, latents=latents)
        latents = self.denoise_latents(
            latents,
            conditioning,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return Object3DPipelineOutput(objects=self.decode_mesh(latents), latents=latents)


__all__ = ["ContractReferencePipeline"]
