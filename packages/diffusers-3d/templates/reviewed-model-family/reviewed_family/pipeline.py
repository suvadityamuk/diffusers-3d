from __future__ import annotations

import torch

from diffusers_3d import (
    ContributionStatus,
    ImageCondition,
    MeshAsset,
    Object3DKind,
    Object3DPipeline,
    Object3DPipelineOutput,
    ReviewStatus,
)

from .model import ReviewedDenoiser


class ReviewedObject3DPipeline(Object3DPipeline):
    """Minimal reviewed pipeline structure using object-native outputs."""

    family_id = "reviewed-family"
    task_ids = ("image-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("triangle-mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("torch",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    model_cpu_offload_seq = "denoiser"

    def __init__(self, denoiser: ReviewedDenoiser) -> None:
        super().__init__()
        if denoiser.config.latent_dim < 9:
            raise ValueError("latent_dim must provide at least nine values for the mesh starter")
        self.register_modules(denoiser=denoiser)

    @torch.no_grad()
    def __call__(self, condition: ImageCondition) -> Object3DPipelineOutput:
        if type(condition) is not ImageCondition:
            raise TypeError("condition must be an exact ImageCondition")
        condition.validate()
        value = condition.image.to(device=self._execution_device, dtype=self.denoiser.dtype).mean()
        hidden_states = value.expand(1, self.denoiser.config.latent_dim)
        latents = self.denoiser(hidden_states, torch.zeros(1, device=hidden_states.device))
        mesh = MeshAsset(
            vertices=latents[0, :9].reshape(3, 3),
            faces=torch.tensor([[0, 1, 2]], device=latents.device, dtype=torch.int64),
            transform=torch.eye(4, device=latents.device, dtype=latents.dtype),
        )
        return Object3DPipelineOutput(objects=(mesh,), latents=latents)
