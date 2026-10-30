# Portions of this file reproduce the TripoSG rectified-flow transformer architecture:
# https://github.com/VAST-AI-Research/TripoSG
# Revision: fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
#
# MIT License. Copyright (c) 2025 VAST-AI-Research and contributors.
# Upstream states that its transformer is based on Tencent HunyuanDiT; see this family's NOTICE.
# Rewritten on Diffusers building blocks with the attention dispatcher.

from __future__ import annotations

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm, RMSNorm
from torch import nn

from ...execution.models import Object3DModel


class TripoSGAttnProcessor:
    """Scaled dot-product attention over TripoSG's head-major projection layout.

    The released checkpoints split heads before query/key/value: head ``h`` reads its query, key, and value from the
    ``h``-th chunk of the concatenated ``[to_q | to_k | to_v]`` (or ``[to_k | to_v]``) output, so the tensors load
    unchanged.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: TripoSGAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            projections = torch.cat([query, attn.to_k(hidden_states), attn.to_v(hidden_states)], dim=-1)
            query, key, value = projections.unflatten(-1, (attn.heads, -1)).chunk(3, dim=-1)
        else:
            if attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            projections = torch.cat([attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)], dim=-1)
            key, value = projections.unflatten(-1, (attn.heads, -1)).chunk(2, dim=-1)
            query = query.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return attn.to_out[0](hidden_states)


class TripoSGAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = TripoSGAttnProcessor
    _available_processors = [TripoSGAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        cross_attention_dim: int | None = None,
        qk_norm: bool = False,
        cross_attention_norm: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        context_dim = dim if cross_attention_dim is None else cross_attention_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.norm_q = RMSNorm(dim // heads, eps=1e-6) if qk_norm else None
        self.norm_k = RMSNorm(dim // heads, eps=1e-6) if qk_norm else None
        self.norm_cross = nn.LayerNorm(context_dim) if cross_attention_norm else None
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.set_processor(TripoSGAttnProcessor())

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states)


class TripoSGTransformerBlock(nn.Module):
    """Pre-norm self/cross-attention block with TripoSG's long skip: ``norm(linear([skip, x]))``."""

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        self_attention: bool = True,
        cross_attention_dim: int | None = None,
        qk_norm: bool = False,
        cross_attention_norm: bool = False,
        skip: bool = False,
    ) -> None:
        super().__init__()
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.skip_norm = FP32LayerNorm(dim, eps=1e-5) if skip else None
        self.norm1 = FP32LayerNorm(dim, eps=1e-5) if self_attention else None
        self.attn1 = TripoSGAttention(dim, heads, qk_norm=qk_norm) if self_attention else None
        self.norm2 = FP32LayerNorm(dim, eps=1e-5) if cross_attention_dim is not None else None
        self.attn2 = (
            TripoSGAttention(
                dim,
                heads,
                cross_attention_dim=cross_attention_dim,
                qk_norm=qk_norm,
                cross_attention_norm=cross_attention_norm,
            )
            if cross_attention_dim is not None
            else None
        )
        self.norm3 = FP32LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, activation_fn="gelu", inner_dim=4 * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.skip_linear is not None:
            hidden_states = self.skip_norm(self.skip_linear(torch.cat([skip, hidden_states], dim=-1)))
        if self.attn1 is not None:
            hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        if self.attn2 is not None:
            hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        return hidden_states + self.ff(self.norm3(hidden_states))


class TripoSGTransformerModel(Object3DModel, PeftAdapterMixin):
    """TripoSG rectified-flow transformer over a set of shape latent tokens, conditioned on DINOv2 image tokens.

    The timestep embedding is prepended as an extra token. Blocks after the middle one take a long skip from the
    mirrored block before it, as in U-ViT/HunyuanDiT. The model predicts the noise-to-data velocity.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TripoSGTransformerBlock"]
    _repeated_blocks = ["TripoSGTransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_embed", "time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        width: int = 2048,
        num_layers: int = 21,
        num_attention_heads: int = 16,
        cross_attention_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.time_embed = Timesteps(width, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_proj = TimestepEmbedding(width, 4 * width, act_fn="gelu", out_dim=width)
        self.proj_in = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList(
            [
                TripoSGTransformerBlock(
                    width,
                    num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,
                    skip=layer > num_layers // 2,
                )
                for layer in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, in_channels)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        num_tokens = hidden_states.shape[1]
        middle = self.config.num_layers // 2
        temb = self.time_proj(self.time_embed(timestep).to(hidden_states.dtype))
        hidden_states = torch.cat([temb.unsqueeze(1), self.proj_in(hidden_states)], dim=1)
        skips = []
        for layer, block in enumerate(self.blocks):
            skip = skips.pop() if layer > middle else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, skip)
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, skip)
            if layer < middle:
                skips.append(hidden_states)
        hidden_states = self.proj_out(self.norm_out(hidden_states)[:, -num_tokens:])
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)


__all__ = ["TripoSGAttention", "TripoSGAttnProcessor", "TripoSGTransformerBlock", "TripoSGTransformerModel"]
