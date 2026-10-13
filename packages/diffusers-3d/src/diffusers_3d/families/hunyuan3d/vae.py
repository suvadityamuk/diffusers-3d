# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: decoder only, rebuilt from Diffusers building blocks; weights are remapped by `conversion.py`.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import FeedForward
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind
from .models import Hunyuan3DAttention


@dataclass
class Hunyuan3DShapeVAEOutput(BaseOutput):
    sample: torch.Tensor


class Hunyuan3DVAEBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Hunyuan3DAttention(width, heads, qk_norm="layer_norm")
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.ff = FeedForward(width, activation_fn="gelu")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class Hunyuan3DVAEDecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Hunyuan3DAttention(width, heads, cross_attention_dim=width, qk_norm="layer_norm")
        self.norm3 = nn.LayerNorm(width, eps=1e-6)
        self.ff = FeedForward(width, activation_fn="gelu")

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), self.norm2(encoder_hidden_states))
        return hidden_states + self.ff(self.norm3(hidden_states))


class Hunyuan3DShapeVAE(Object3DModel):
    """Decoder of the Hunyuan3D-2.1 ShapeVAE: latent set -> decoded tokens -> occupancy logits at 3D points.

    The point-cloud encoder is not ported: generation only decodes. Divide sampled latents by
    ``config.scale_factor`` before :meth:`forward`.
    """

    family_id = "hunyuan3d-2.1"
    component_role = "vae"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["Hunyuan3DVAEBlock", "Hunyuan3DVAEDecoderBlock"]
    _repeated_blocks = ["Hunyuan3DVAEBlock"]

    @register_to_config
    def __init__(
        self,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_freqs: int = 8,
        scale_factor: float = 1.0039506158752403,
    ) -> None:
        super().__init__()
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer_blocks = nn.ModuleList([Hunyuan3DVAEBlock(width, heads) for _ in range(num_decoder_layers)])
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        self.query_proj = nn.Linear(3 * (2 * num_freqs + 1), width)
        self.decoder_block = Hunyuan3DVAEDecoderBlock(width, heads)
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, 1)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "num_latents": 4096,
            "embed_dim": 64,
            "width": 1024,
            "heads": 16,
            "num_decoder_layers": 16,
            "num_freqs": 8,
            "scale_factor": 1.0039506158752403,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "num_latents": 6,
            "embed_dim": 8,
            "width": 32,
            "heads": 4,
            "num_decoder_layers": 2,
            "num_freqs": 2,
            "scale_factor": 1.25,
        }

    def forward(
        self, latents: torch.Tensor, return_dict: bool = True
    ) -> Hunyuan3DShapeVAEOutput | tuple[torch.Tensor]:
        """Decode ``(batch, num_latents, embed_dim)`` latents into the tokens :meth:`query_occupancy` attends to."""

        hidden_states = self.post_kl(latents)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return Hunyuan3DShapeVAEOutput(sample=hidden_states)

    def query_occupancy(self, decoded_latents: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Occupancy logits (positive inside) of ``(batch, points, 3)`` positions, shape ``(batch, points)``."""

        frequencies = (points[..., None] * self.frequencies).flatten(-2)
        embedding = torch.cat([points, frequencies.sin(), frequencies.cos()], dim=-1)
        hidden_states = self.decoder_block(self.query_proj(embedding.to(decoded_latents.dtype)), decoded_latents)
        return self.proj_out(self.norm_out(hidden_states)).squeeze(-1)


__all__ = ["Hunyuan3DShapeVAE", "Hunyuan3DShapeVAEOutput"]
