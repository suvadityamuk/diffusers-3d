# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from torch import nn

from ..execution.metadata import ContributionStatus, ReviewStatus
from ..execution.models import Object3DModel
from ..objects import Object3DKind


class ContractReferenceDenoiser(Object3DModel, PeftAdapterMixin):
    """Tiny reviewed denoiser used only to validate package contracts."""

    family_id = "contract-reference"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = []

    @register_to_config
    def __init__(self, latent_dim: int = 9, condition_dim: int = 3) -> None:
        super().__init__()
        if latent_dim <= 0 or condition_dim <= 0:
            raise ValueError("latent_dim and condition_dim must be positive")
        self.projection = nn.Linear(latent_dim, latent_dim)
        self.conditioning_projection = nn.Linear(condition_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.config.latent_dim:
            raise ValueError(f"hidden_states must have shape (batch, {self.config.latent_dim})")
        if conditioning.ndim != 2 or conditioning.shape != (
            hidden_states.shape[0],
            self.config.condition_dim,
        ):
            raise ValueError(
                f"conditioning must have shape (batch, {self.config.condition_dim}) matching hidden_states"
            )

        timestep = torch.as_tensor(timestep, device=hidden_states.device, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(hidden_states.shape[0])
        if timestep.ndim != 1 or timestep.shape[0] != hidden_states.shape[0]:
            raise ValueError("timestep must be a scalar or contain one value per batch item")
        timestep = timestep.unsqueeze(-1) / 1000

        hidden_states = self.projection(hidden_states)
        hidden_states = hidden_states + self.conditioning_projection(conditioning) + timestep
        hidden_states = torch.tanh(hidden_states)
        return self.output_projection(hidden_states)


class ContractReferenceMeshDecoder(Object3DModel):
    """Tiny tensor-native decoder whose vertex path remains differentiable."""

    family_id = "contract-reference"
    component_role = "mesh-decoder"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = []

    @register_to_config
    def __init__(self, latent_dim: int = 9, num_vertices: int = 3) -> None:
        super().__init__()
        if latent_dim <= 0 or num_vertices != 3:
            raise ValueError("latent_dim must be positive and this validation decoder requires exactly three vertices")
        self.vertex_projection = nn.Linear(latent_dim, num_vertices * 3)
        self.register_buffer(
            "template_vertices",
            torch.tensor(
                [
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.0, 0.5, 0.0],
                ]
            ),
        )
        self.register_buffer("faces", torch.tensor([[0, 1, 2]], dtype=torch.int64))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 2 or latents.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latents must have shape (batch, {self.config.latent_dim})")
        offsets = self.vertex_projection(latents).reshape(latents.shape[0], self.config.num_vertices, 3)
        return self.template_vertices.unsqueeze(0) + offsets


__all__ = ["ContractReferenceDenoiser", "ContractReferenceMeshDecoder"]
