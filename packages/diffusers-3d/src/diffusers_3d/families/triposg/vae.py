# Portions of this file reproduce the TripoSG shape VAE decoder:
# https://github.com/VAST-AI-Research/TripoSG
# Revision: fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
#
# MIT License. Copyright (c) 2025 VAST-AI-Research and contributors.
# Decoder half only, rewritten on Diffusers building blocks.

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel
from .models import TripoSGTransformerBlock


@dataclass
class TripoSGVAEDecoderOutput(BaseOutput):
    """Decoded field at each query point, shape ``(batch, points, 1)``; the surface is its zero level."""

    sample: torch.Tensor


class TripoSGVAEDecoder(Object3DModel):
    """Decoder half of the TripoSG shape VAE: latent tokens to a signed-distance-like field at query points.

    Self-attention blocks refine the latent set once; each query point is then Fourier-embedded and cross-attends to
    it. The encoder half is not ported because image-to-3D sampling never encodes shapes (it also needs
    ``torch_cluster`` farthest-point sampling). The decoded field is negated exactly as upstream, which extracts
    the surface at level 0.
    """

    _no_split_modules = ["TripoSGTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        num_attention_heads: int = 8,
        width_decoder: int = 1024,
        num_layers_decoder: int = 16,
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
    ) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(embed_frequency, dtype=torch.float32)
        if embed_include_pi:
            frequencies = frequencies * torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.post_quant = nn.Linear(latent_channels, width_decoder)
        self.blocks = nn.ModuleList(
            [TripoSGTransformerBlock(width_decoder, num_attention_heads) for _ in range(num_layers_decoder)]
            + [
                TripoSGTransformerBlock(
                    width_decoder,
                    num_attention_heads,
                    self_attention=False,
                    cross_attention_dim=width_decoder,
                    cross_attention_norm=True,
                )
            ]
        )
        self.proj_query = nn.Linear(in_channels * (2 * embed_frequency + 1), width_decoder)
        self.norm_out = nn.LayerNorm(width_decoder)
        self.proj_out = nn.Linear(width_decoder, 1)

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine ``latents`` ``(batch, tokens, latent_channels)`` once into the key/value every query attends to.

        Pass the result to :meth:`query` to decode many point sets of one shape without refining it again.
        """

        latent_states = self.post_quant(latents)
        for block in self.blocks[:-1]:
            latent_states = block(latent_states)
        return self.blocks[-1].attn2.project_key_value(latent_states)

    def query(
        self, context: tuple[torch.Tensor, torch.Tensor], queries: torch.Tensor, *, num_chunks: int | None = None
    ) -> torch.Tensor:
        """Negated field ``(batch, points, 1)`` at ``queries`` ``(batch, points, 3)`` given :meth:`decode_latents`.

        A context of batch one is shared by every query batch item.
        """

        key, value = context
        if key.shape[0] != queries.shape[0]:
            key, value = key.expand(queries.shape[0], -1, -1, -1), value.expand(queries.shape[0], -1, -1, -1)
        logits = []
        for chunk in queries.split(num_chunks or queries.shape[1], dim=1):
            chunk = chunk.to(key)
            embedding = (chunk[..., None] * self.frequencies.to(chunk.dtype)).flatten(-2)
            query_states = self.proj_query(torch.cat([chunk, embedding.sin(), embedding.cos()], dim=-1))
            query_states = self.blocks[-1](query_states, key_value=(key, value))
            logits.append(self.proj_out(self.norm_out(query_states)))
        return -torch.cat(logits, dim=1)

    def forward(
        self,
        latents: torch.Tensor,
        queries: torch.Tensor,
        *,
        num_chunks: int | None = None,
        return_dict: bool = True,
    ) -> TripoSGVAEDecoderOutput | tuple[torch.Tensor]:
        """Decode ``latents`` ``(batch, tokens, latent_channels)`` at ``queries`` ``(batch, points, 3)``.

        ``num_chunks`` bounds how many query points cross-attend at once; the latent set is refined once.
        """

        sample = self.query(self.decode_latents(latents), queries, num_chunks=num_chunks)
        if not return_dict:
            return (sample,)
        return TripoSGVAEDecoderOutput(sample=sample)


__all__ = ["TripoSGVAEDecoder", "TripoSGVAEDecoderOutput"]
