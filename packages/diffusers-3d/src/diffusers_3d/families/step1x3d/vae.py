# Portions of this file reproduce StepFun Step1X-3D shape-VAE decoding semantics:
# https://github.com/stepfun-ai/Step1X-3D
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified to keep only the decoder, built from stock Diffusers blocks, and to decode a dense grid.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class Step1X3DShapeDecoderOutput(BaseOutput):
    """Occupancy logits, positive inside the surface."""

    sample: torch.Tensor


def step1x3d_grid_points(bounds: float, resolution: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """The ``(resolution + 1) ** 3`` query points of the cube ``[-bounds, bounds]^3`` in ``ij`` order.

    Coordinates follow ``numpy.linspace(..., dtype=float32)`` bit for bit, as upstream builds its grid with NumPy.
    """

    if resolution <= 0 or bounds <= 0:
        raise ValueError("bounds and resolution must be positive")
    axis = torch.arange(resolution + 1, dtype=torch.float64) * (2 * bounds / resolution) - bounds
    axis[-1] = bounds
    axis = axis.to(device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)


class Step1X3DShapeDecoder(Object3DModel):
    """Decoder half of the Step1X-3D perceiver shape VAE (a 3DShape2VecSet-style latent set).

    ``post_kl`` lifts the ``(batch, num_latents, latent_channels)`` diffusion latents to the VAE width, a stack of
    self-attention blocks turns them into shape tokens, and one cross-attention block reads the occupancy logit of any
    query point in ``[-1, 1]^3`` from those tokens. The point encoder is not part of inference and is not ported.
    """

    family_id = "step1x3d"
    component_role = "shape_decoder"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["BasicTransformerBlock"]
    _repeated_blocks = ["BasicTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        num_latents: int = 2048,
        latent_channels: int = 64,
        width: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 16,
        num_freqs: int = 8,
        include_pi: bool = False,
        out_channels: int = 1,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if width % num_attention_heads:
            raise ValueError("width must be divisible by num_attention_heads")
        head_dim = width // num_attention_heads
        self.post_kl = nn.Linear(latent_channels, width)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(width, num_attention_heads, head_dim, activation_fn="gelu")
                for _ in range(num_layers)
            ]
        )
        self.query_proj = nn.Linear(3 * (2 * num_freqs + 1), width)
        self.norm1 = nn.LayerNorm(width)
        self.norm_context = nn.LayerNorm(width)
        self.attn = Attention(query_dim=width, cross_attention_dim=width, heads=num_attention_heads, dim_head=head_dim)
        self.norm2 = nn.LayerNorm(width)
        self.ff = FeedForward(width, activation_fn="gelu")
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, out_channels)

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "num_latents": 8,
            "latent_channels": 4,
            "width": 16,
            "num_attention_heads": 2,
            "num_layers": 1,
            "num_freqs": 2,
        }

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Turn diffusion latents into the shape tokens that ``query`` attends to."""

        hidden_states = self.post_kl(latents / self.config.scaling_factor)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def query(self, points: torch.Tensor, shape_tokens: torch.Tensor) -> torch.Tensor:
        """Occupancy logits for ``(batch, points, 3)`` query points.

        The Fourier embedding is computed in at least float32: half-precision coordinates cannot resolve a 384^3 grid.
        """

        points = points.to(torch.promote_types(points.dtype, torch.float32))
        frequencies = 2.0 ** torch.arange(self.config.num_freqs, device=points.device, dtype=points.dtype)
        if self.config.include_pi:
            frequencies = frequencies * torch.pi
        embedding = (points[..., None] * frequencies).flatten(-2)
        embedding = torch.cat([points, embedding.sin(), embedding.cos()], dim=-1)
        hidden_states = self.query_proj(embedding.to(self.query_proj.weight.dtype))
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), self.norm_context(shape_tokens))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return self.proj_out(self.norm_out(hidden_states))

    def forward(
        self, latents: torch.Tensor, points: torch.Tensor, *, return_dict: bool = True
    ) -> Step1X3DShapeDecoderOutput | tuple[torch.Tensor]:
        sample = self.query(points, self.decode(latents))
        if not return_dict:
            return (sample,)
        return Step1X3DShapeDecoderOutput(sample=sample)

    @torch.no_grad()
    def decode_grid(
        self,
        latents: torch.Tensor,
        *,
        bounds: float = 1.05,
        resolution: int = 384,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Dense ``(batch, resolution + 1, resolution + 1, resolution + 1)`` float32 occupancy-logit grid."""

        shape_tokens = self.decode(latents)
        points = step1x3d_grid_points(bounds, resolution, device=latents.device)
        logits = [
            self.query(chunk.expand(latents.shape[0], -1, -1), shape_tokens)[..., 0]
            for chunk in points[None].split(chunk_size, dim=1)
        ]
        return torch.cat(logits, dim=1).view(latents.shape[0], *([resolution + 1] * 3)).float()


__all__ = ["Step1X3DShapeDecoder", "Step1X3DShapeDecoderOutput", "step1x3d_grid_points"]
