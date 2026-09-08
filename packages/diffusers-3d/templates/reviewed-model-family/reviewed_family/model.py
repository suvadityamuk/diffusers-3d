from __future__ import annotations

import torch
from diffusers.configuration_utils import register_to_config
from torch import nn

from diffusers_3d import ContributionStatus, Object3DKind, Object3DModel, ReviewStatus


class ReviewedDenoiser(Object3DModel):
    """Tiny structure showing the required reviewed model declarations."""

    family_id = "reviewed-family"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ("torch",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = []

    @register_to_config
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
        while timestep.ndim < hidden_states.ndim:
            timestep = timestep.unsqueeze(-1)
        return self.projection(hidden_states) + timestep
