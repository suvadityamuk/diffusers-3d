# Portions of this file reproduce StepFun Step1X-3D geometry flow semantics:
# https://github.com/stepfun-ai/Step1X-3D
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified to assemble the flow transformer from stock Diffusers blocks.

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZeroSingle
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel


@dataclass
class Step1X3DTransformerOutput(BaseOutput):
    sample: torch.Tensor


class Step1X3DAttnProcessor:
    """Flux joint attention: condition tokens are prepended to the latent tokens, and no rotary embedding is used."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if encoder_hidden_states is not None:
            context_query = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)))
            context_key = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)))
            context_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            query = torch.cat([context_query, query], dim=1)
            key = torch.cat([context_key, key], dim=1)
            value = torch.cat([context_value, value], dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return hidden_states
        encoder_hidden_states, hidden_states = hidden_states.split(
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
        )
        return attn.to_out[0](hidden_states), attn.to_add_out(encoder_hidden_states)


class Step1X3DSingleTransformerBlock(nn.Module):
    """Flux single-stream block (parallel attention and MLP) over the already joined condition and latent tokens."""

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, int(dim * mlp_ratio))
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + int(dim * mlp_ratio), dim)
        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
            processor=Step1X3DAttnProcessor(),
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(norm_hidden_states)
        hidden_states = hidden_states + gate.unsqueeze(1) * self.proj_out(
            torch.cat([attn_output, mlp_hidden_states], dim=2)
        )
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class Step1X3DTransformer1DModel(Object3DModel, PeftAdapterMixin):
    """Step1X-3D rectified-flow transformer over the shape VAE's latent token set.

    A Flux-style stack without positional embeddings: ``num_layers`` dual-stream blocks over latent and condition
    tokens, then ``num_single_layers`` single-stream blocks over their concatenation (conditions first).
    """

    _no_split_modules = ["JointTransformerBlock", "Step1X3DSingleTransformerBlock"]
    _repeated_blocks = ["JointTransformerBlock", "Step1X3DSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_embedding", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        width: int = 1280,
        num_layers: int = 12,
        num_single_layers: int = 24,
        num_attention_heads: int = 20,
        cross_attention_dim: int = 1024,
    ) -> None:
        super().__init__()
        if width % num_attention_heads:
            raise ValueError("width must be divisible by num_attention_heads")
        head_dim = width // num_attention_heads
        self.time_proj = Timesteps(width, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(width, 4 * width, act_fn="gelu", out_dim=width)
        self.proj_in = nn.Linear(in_channels, width)
        self.proj_cross_attention = nn.Linear(cross_attention_dim, width)
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(width, num_attention_heads, head_dim, qk_norm="rms_norm")
                for _ in range(num_layers)
            ]
        )
        for block in self.transformer_blocks:
            block.attn.set_processor(Step1X3DAttnProcessor())
        self.single_transformer_blocks = nn.ModuleList(
            [Step1X3DSingleTransformerBlock(width, num_attention_heads, head_dim) for _ in range(num_single_layers)]
        )
        self.norm_out = AdaLayerNormContinuous(width, width, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(width, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Step1X3DTransformerOutput | tuple[torch.Tensor]:
        """Predict the flow velocity for ``(batch, tokens, in_channels)`` latents at timesteps in ``[0, 1000]``."""

        timestep = torch.as_tensor(timestep, device=hidden_states.device).expand(hidden_states.shape[0])
        temb = self.time_embedding(self.time_proj(timestep).to(hidden_states.dtype))
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.proj_cross_attention(encoder_hidden_states)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, temb)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
        sample = self.proj_out(self.norm_out(hidden_states, temb))
        if not return_dict:
            return (sample,)
        return Step1X3DTransformerOutput(sample=sample)


__all__ = ["Step1X3DTransformer1DModel", "Step1X3DTransformerOutput"]
