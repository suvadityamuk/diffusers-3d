# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: rebuilt from Diffusers building blocks; the released weights are remapped by `conversion.py`.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel


@dataclass
class Hunyuan3DDiTOutput(BaseOutput):
    sample: torch.Tensor


class Hunyuan3DAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Hunyuan3DAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(attn.to_k(context).unflatten(-1, (attn.heads, -1)))
        value = attn.to_v(context).unflatten(-1, (attn.heads, -1))
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.to_out(hidden_states.flatten(2, 3).to(query.dtype))


class Hunyuan3DAttention(nn.Module, AttentionModuleMixin):
    """Multi-head attention with per-head q/k normalization, shared by the DiT and the shape VAE.

    The released checkpoints split heads after concatenating the q/k/v projections; `conversion.py` permutes the
    projection rows once so that the conventional per-projection head split here computes the same attention.
    """

    _default_processor_cls = Hunyuan3DAttnProcessor
    _available_processors = [Hunyuan3DAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        cross_attention_dim: int | None = None,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        context_dim = dim if cross_attention_dim is None else cross_attention_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        norm = nn.RMSNorm if qk_norm == "rms_norm" else nn.LayerNorm
        self.norm_q = norm(head_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = norm(head_dim, eps=1e-6, elementwise_affine=True)
        self.to_out = nn.Linear(dim, dim)
        self.set_processor(Hunyuan3DAttnProcessor())

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states)


class Hunyuan3DMoEFeedForward(nn.Module):
    """Top-k routed experts plus one shared expert; routing matches the released ``moe_infer`` path."""

    def __init__(self, dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(dim, activation_fn="gelu") for _ in range(num_experts)])
        self.shared_experts = FeedForward(dim, activation_fn="gelu")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        tokens = hidden_states.flatten(0, -2)
        weights, selected = self.gate(tokens).softmax(dim=-1).topk(self.top_k, dim=-1, sorted=False)
        routed = torch.zeros_like(tokens)
        for index, expert in enumerate(self.experts):
            token_index, slot = (selected == index).nonzero(as_tuple=True)
            routed.index_add_(0, token_index, expert(tokens[token_index]) * weights[token_index, slot, None])
        return routed.view_as(hidden_states) + self.shared_experts(hidden_states)


class Hunyuan3DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        cross_attention_dim: int,
        *,
        skip: bool,
        moe: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        if skip:
            self.skip_linear = nn.Linear(2 * dim, dim)
            self.skip_norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn1 = Hunyuan3DAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn2 = Hunyuan3DAttention(dim, heads, cross_attention_dim=cross_attention_dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim, activation_fn="gelu") if moe is None else Hunyuan3DMoEFeedForward(dim, *moe)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            hidden_states = self.skip_norm(self.skip_linear(torch.cat([skip, hidden_states], dim=-1)))
        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        return hidden_states + self.ff(self.norm3(hidden_states))


class Hunyuan3DDiTModel(Object3DModel, PeftAdapterMixin):
    """Hunyuan3D-DiT: the U-skip flow transformer that denoises ShapeVAE latent sets.

    The timestep embedding is prepended as an extra token; blocks in the second half take a skip connection from
    the mirrored first-half block, and the last ``num_moe_layers`` blocks use a mixture-of-experts feed-forward.
    ``timestep`` is the flow time in ``[0, 1]``.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Hunyuan3DDiTBlock"]
    _repeated_blocks = ["Hunyuan3DDiTBlock"]
    _skip_layerwise_casting_patterns = ["time_embedding", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, 4 * hidden_size, act_fn="gelu", out_dim=hidden_size)
        self.proj_in = nn.Linear(in_channels, hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                Hunyuan3DDiTBlock(
                    hidden_size,
                    num_heads,
                    context_dim,
                    skip=index > depth // 2,
                    moe=(num_experts, moe_top_k) if depth - index <= num_moe_layers else None,
                )
                for index in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Hunyuan3DDiTOutput | tuple[torch.Tensor]:
        embedding = self.time_embedding(self.time_proj(timestep).to(hidden_states.dtype))
        hidden_states = torch.cat([embedding.unsqueeze(1), self.proj_in(hidden_states)], dim=1)
        middle = self.config.depth // 2
        skips = []
        for index, block in enumerate(self.transformer_blocks):
            skip = skips.pop() if index > middle else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, skip)
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, skip)
            if index < middle:
                skips.append(hidden_states)
        sample = self.proj_out(self.norm_out(hidden_states)[:, 1:])
        if not return_dict:
            return (sample,)
        return Hunyuan3DDiTOutput(sample=sample)


# Meta's DINOv2-L/14. Hunyuan3D-2.1's conditioner is this model cast to float16, tensor for tensor; Hunyuan3D-Omni and
# TripoSG use it unchanged. It loads from Meta's repository and is never converted.
DINOV2_LARGE_REPOSITORY = "facebook/dinov2-large"
DINOV2_LARGE_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


def load_dinov2(
    repository: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    **hub_kwargs: Any,
):
    """Load a Transformers ``Dinov2Model`` from its repository in ``dtype``.

    Pipelines pass the dtype given to ``from_pretrained``, so it matches their other components: float32 when none is
    given (as Diffusers loads the rest), float16 to run as the upstream releases do.
    """

    from transformers import Dinov2Model

    model, loading_info = Dinov2Model.from_pretrained(
        repository,
        revision=revision,
        dtype=torch.float32 if dtype is None else dtype,
        output_loading_info=True,
        **hub_kwargs,
    )
    if any(loading_info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"the DINOv2 weights do not match its config: {loading_info}")
    return model.eval()


__all__ = [
    "DINOV2_LARGE_REPOSITORY",
    "DINOV2_LARGE_REVISION",
    "Hunyuan3DAttention",
    "Hunyuan3DDiTModel",
    "Hunyuan3DDiTOutput",
    "load_dinov2",
]
