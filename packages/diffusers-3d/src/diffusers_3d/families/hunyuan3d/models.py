# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for native Diffusers/PyTorch integration.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class Hunyuan3DShapeDiTOutput(BaseOutput):
    """Velocity prediction produced by :class:`Hunyuan3DShapeDiTModel`."""

    sample: torch.Tensor


class Hunyuan3DSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: Hunyuan3DSelfAttention, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        # Keep the released implementation's concatenate-then-head-split order.
        # Splitting each projection independently is more conventional, but it
        # changes the Hunyuan3D-2.1 computation for the same checkpoint.
        query_key_value = torch.cat(
            [attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)],
            dim=-1,
        ).reshape(batch_size, sequence_length, attn.num_heads, 3 * attn.head_dim)
        query, key, value = query_key_value.split(attn.head_dim, dim=-1)

        query = attn.q_norm(query)
        key = attn.k_norm(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, -1).to(query.dtype)
        return attn.out_proj(hidden_states)


class Hunyuan3DSelfAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Hunyuan3DSelfAttnProcessor
    _available_processors = [Hunyuan3DSelfAttnProcessor]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_bias = qkv_bias
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)
        self.set_processor(Hunyuan3DSelfAttnProcessor())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class Hunyuan3DCrossAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Hunyuan3DCrossAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.with_dca:
            dca_context = encoder_hidden_states[:, -attn.dca_dim :]
            encoder_hidden_states = encoder_hidden_states[:, : -attn.dca_dim]
        context_length = encoder_hidden_states.shape[1]
        query = attn.to_q(hidden_states).reshape(batch_size, sequence_length, attn.num_heads, attn.head_dim)
        key_value = torch.cat(
            [attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)],
            dim=-1,
        ).reshape(batch_size, context_length, attn.num_heads, 2 * attn.head_dim)
        key, value = key_value.split(attn.head_dim, dim=-1)
        query = attn.q_norm(query)
        key = attn.k_norm(key)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, -1).to(query.dtype)

        if attn.with_dca:
            dca_key_value = attn.kv_proj_dca(dca_context).reshape(
                batch_size,
                attn.dca_dim,
                2,
                attn.num_heads,
                attn.head_dim,
            )
            dca_key, dca_value = dca_key_value.unbind(dim=2)
            dca_key = attn.k_norm_dca(dca_key)
            dca_hidden_states = dispatch_attention_fn(
                query,
                dca_key,
                dca_value,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            dca_hidden_states = dca_hidden_states.reshape(batch_size, sequence_length, -1).to(query.dtype)
            hidden_states = hidden_states + attn.dca_weight * dca_hidden_states

        return attn.out_proj(hidden_states)


class Hunyuan3DCrossAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Hunyuan3DCrossAttnProcessor
    _available_processors = [Hunyuan3DCrossAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: type[nn.Module],
        with_decoupled_ca: bool = False,
        decoupled_ca_dim: int = 16,
        decoupled_ca_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.qdim = query_dim
        self.kdim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.use_bias = qkv_bias
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(query_dim, query_dim, bias=True)

        self.with_dca = with_decoupled_ca
        if with_decoupled_ca:
            self.kv_proj_dca = nn.Linear(context_dim, 2 * query_dim, bias=qkv_bias)
            self.k_norm_dca = (
                norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            )
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight
        self.set_processor(Hunyuan3DCrossAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.with_dca and encoder_hidden_states.shape[1] < self.dca_dim:
            raise ValueError("encoder_hidden_states is shorter than decoupled_ca_dim")
        return self.processor(self, hidden_states, encoder_hidden_states)


class Hunyuan3DTimesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a rank-one tensor")
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_dim,
            dtype=torch.float32,
            device=timesteps.device,
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        frequencies = torch.exp(exponent)
        embeddings = timesteps[:, None].float() * frequencies[None, :]
        embeddings = self.scale * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if self.num_channels % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class Hunyuan3DTimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int,
        cond_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        )
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)
        self.time_embed = Hunyuan3DTimesteps(hidden_size)

    def forward(
        self,
        timestep: torch.Tensor,
        condition: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_embedding = self.time_embed(timestep).to(dtype=dtype)
        if condition is not None:
            timestep_embedding = timestep_embedding + self.cond_proj(condition)
        return self.mlp(timestep_embedding).unsqueeze(1)


class Hunyuan3DMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(hidden_states)))


class _AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hidden_states: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        if loss.numel() != 1:
            raise ValueError("MoE auxiliary loss must be scalar")
        ctx.loss_dtype = loss.dtype
        ctx.requires_auxiliary_loss = loss.requires_grad
        return hidden_states

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        loss_gradient = None
        if ctx.requires_auxiliary_loss:
            loss_gradient = torch.ones(1, dtype=ctx.loss_dtype, device=gradient.device)
        return gradient, loss_gradient


class Hunyuan3DMoEGate(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_experts: int,
        num_experts_per_token: int,
        auxiliary_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_token
        self.n_routed_experts = num_experts
        self.alpha = auxiliary_loss_weight
        self.weight = nn.Parameter(torch.zeros((num_experts, embed_dim)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        scores = F.linear(hidden_states.reshape(-1, hidden_size), self.weight).softmax(dim=-1)
        topk_weight, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        auxiliary_loss = None
        if self.training and self.alpha > 0.0:
            expert_assignments = F.one_hot(
                topk_indices.reshape(batch_size, -1).reshape(-1),
                num_classes=self.n_routed_experts,
            )
            assignment_frequency = expert_assignments.float().mean(dim=0)
            mean_probability = scores.mean(dim=0)
            auxiliary_loss = (mean_probability * assignment_frequency * self.n_routed_experts).sum() * self.alpha
        return topk_indices, topk_weight, auxiliary_loss


class Hunyuan3DMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_experts: int,
        moe_top_k: int,
    ) -> None:
        super().__init__()
        self.moe_top_k = moe_top_k
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim,
                    dropout=0.0,
                    activation_fn="gelu",
                    final_dropout=False,
                    inner_dim=dim * 4,
                    bias=True,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = Hunyuan3DMoEGate(
            dim,
            num_experts=num_experts,
            num_experts_per_token=moe_top_k,
        )
        self.shared_experts = FeedForward(
            dim,
            dropout=0.0,
            activation_fn="gelu",
            final_dropout=False,
            inner_dim=dim * 4,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        topk_indices, topk_weight, auxiliary_loss = self.gate(hidden_states)
        flattened_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        routed_states = torch.zeros_like(flattened_states)

        flat_expert_indices = topk_indices.reshape(-1)
        token_indices = torch.arange(flattened_states.shape[0], device=hidden_states.device).repeat_interleave(
            self.moe_top_k
        )
        flat_weights = topk_weight.reshape(-1, 1)
        for expert_index, expert in enumerate(self.experts):
            route_mask = flat_expert_indices == expert_index
            selected_tokens = token_indices[route_mask]
            if selected_tokens.numel() == 0:
                continue
            expert_output = expert(flattened_states[selected_tokens])
            routed_states.index_add_(
                0,
                selected_tokens,
                expert_output * flat_weights[route_mask].to(expert_output.dtype),
            )

        routed_states = routed_states.reshape(original_shape)
        if auxiliary_loss is not None:
            routed_states = _AddAuxiliaryLoss.apply(routed_states, auxiliary_loss)
        return routed_states + self.shared_experts(hidden_states)


class Hunyuan3DDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_dim: int,
        *,
        norm_layer: type[nn.Module],
        qk_norm_layer: type[nn.Module],
        qk_norm: bool,
        qkv_bias: bool,
        skip_connection: bool,
        timestep_modulate: bool,
        use_moe: bool,
        num_experts: int,
        moe_top_k: int,
        with_decoupled_ca: bool,
        decoupled_ca_dim: int,
        decoupled_ca_weight: float,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Hunyuan3DSelfAttention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
        )
        self.norm2 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        self.timested_modulate = timestep_modulate
        if timestep_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        self.attn2 = Hunyuan3DCrossAttention(
            hidden_size,
            context_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
            with_decoupled_ca=with_decoupled_ca,
            decoupled_ca_dim=decoupled_ca_dim,
            decoupled_ca_weight=decoupled_ca_weight,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None
        self.use_moe = use_moe
        if use_moe:
            self.moe = Hunyuan3DMoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
        else:
            self.mlp = Hunyuan3DMLP(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        skip_value: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([skip_value, hidden_states], dim=-1))
            hidden_states = self.skip_norm(hidden_states)
        if self.timested_modulate:
            hidden_states = hidden_states + self.default_modulation(timestep_embedding).unsqueeze(1)

        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        feed_forward_input = self.norm3(hidden_states)
        if self.use_moe:
            hidden_states = hidden_states + self.moe(feed_forward_input)
        else:
            hidden_states = hidden_states + self.mlp(feed_forward_input)
        return hidden_states


class Hunyuan3DAttentionPool(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads
        self.spatial_dim = spatial_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[1] != self.spatial_dim:
            raise ValueError(f"encoder sequence length must be {self.spatial_dim} when attention pooling is enabled")
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = torch.cat([hidden_states.mean(dim=0, keepdim=True), hidden_states], dim=0)
        hidden_states = hidden_states + self.positional_embedding[:, None, :].to(hidden_states.dtype)
        hidden_states, _ = F.multi_head_attention_forward(
            query=hidden_states[:1],
            key=hidden_states,
            value=hidden_states,
            embed_dim_to_check=hidden_states.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True,
        )
        return hidden_states.squeeze(0)


class Hunyuan3DFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.final_hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_final(hidden_states)
        return self.linear(hidden_states[:, 1:])


class Hunyuan3DShapeDiTModel(Object3DModel, PeftAdapterMixin):
    """Production-compatible native port of Hunyuan3D-2.1 ``HunYuanDiTPlain``.

    Parameter names intentionally match the official denoiser state dict. The
    only conversion aliases are class/config names; tensor surgery is not used.
    """

    family_id = "hunyuan3d-2.1"
    component_role = "denoiser"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Hunyuan3DDiTBlock", "Hunyuan3DMoEBlock"]
    _repeated_blocks = ["Hunyuan3DDiTBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "norm"]

    @register_to_config
    def __init__(
        self,
        input_size: int = 4096,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qk_norm: bool = True,
        text_len: int = 1370,
        with_decoupled_ca: bool = False,
        additional_cond_hidden_state: int = 768,
        decoupled_ca_dim: int = 16,
        decoupled_ca_weight: float = 1.0,
        use_pos_emb: bool = False,
        use_attention_pooling: bool = False,
        guidance_cond_proj_dim: int | None = None,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        del mlp_ratio
        if min(input_size, in_channels, hidden_size, context_dim, depth, num_heads, text_len) <= 0:
            raise ValueError("model dimensions must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= num_moe_layers <= depth:
            raise ValueError("num_moe_layers must be between zero and depth")
        if num_experts <= 0 or not 0 < moe_top_k <= num_experts:
            raise ValueError("moe_top_k must be positive and no larger than num_experts")
        if norm_type not in {"layer", "rms"} or qk_norm_type not in {"layer", "rms"}:
            raise ValueError("norm_type and qk_norm_type must be 'layer' or 'rms'")

        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.text_len = text_len
        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim
        self.gradient_checkpointing = False

        norm_layer = nn.LayerNorm if norm_type == "layer" else nn.RMSNorm
        qk_norm_layer = nn.RMSNorm if qk_norm_type == "rms" else nn.LayerNorm
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = Hunyuan3DTimestepEmbedder(
            hidden_size,
            hidden_size * 4,
            cond_proj_dim=guidance_cond_proj_dim,
        )

        if use_pos_emb:
            positions = torch.arange(input_size, dtype=torch.float32)
            frequencies = torch.arange(hidden_size // 2, dtype=torch.float32)
            frequencies = 1.0 / (10000 ** (frequencies / (hidden_size / 2)))
            position_frequencies = positions[:, None] * frequencies[None, :]
            position_embedding = torch.cat(
                [position_frequencies.sin(), position_frequencies.cos()],
                dim=-1,
            )
            self.register_buffer("pos_embed", position_embedding.unsqueeze(0))

        if use_attention_pooling:
            if context_dim % 8 != 0:
                raise ValueError("context_dim must be divisible by eight for attention pooling")
            self.pooler = Hunyuan3DAttentionPool(text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )

        if with_decoupled_ca:
            self.additional_cond_hidden_state = additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(additional_cond_hidden_state, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, context_dim, bias=True),
            )

        self.blocks = nn.ModuleList(
            [
                Hunyuan3DDiTBlock(
                    hidden_size,
                    num_heads,
                    context_dim,
                    norm_layer=norm_layer,
                    qk_norm_layer=qk_norm_layer,
                    qk_norm=qk_norm,
                    qkv_bias=qkv_bias,
                    skip_connection=layer_index > depth // 2,
                    timestep_modulate=False,
                    use_moe=depth - layer_index <= num_moe_layers,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    with_decoupled_ca=with_decoupled_ca,
                    decoupled_ca_dim=decoupled_ca_dim,
                    decoupled_ca_weight=decoupled_ca_weight,
                )
                for layer_index in range(depth)
            ]
        )
        self.final_layer = Hunyuan3DFinalLayer(hidden_size, self.out_channels)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        """Return the exact architecture values from the released v2.1 config."""

        return {
            "input_size": 4096,
            "in_channels": 64,
            "hidden_size": 2048,
            "context_dim": 1024,
            "depth": 21,
            "num_heads": 16,
            "qk_norm": True,
            "text_len": 1370,
            "with_decoupled_ca": False,
            "use_attention_pooling": False,
            "qk_norm_type": "rms",
            "qkv_bias": False,
            "use_pos_emb": False,
            "num_moe_layers": 6,
            "num_experts": 8,
            "moe_top_k": 2,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        """Return a CPU-test configuration that executes every block type."""

        return {
            "input_size": 4,
            "in_channels": 8,
            "hidden_size": 32,
            "context_dim": 32,
            "depth": 3,
            "num_heads": 4,
            "qk_norm": True,
            "text_len": 5,
            "use_attention_pooling": False,
            "qkv_bias": False,
            "num_moe_layers": 1,
            "num_experts": 2,
            "moe_top_k": 1,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        additional_encoder_hidden_states: torch.Tensor | None = None,
        guidance_cond: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Hunyuan3DShapeDiTOutput | tuple[torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.in_channels:
            raise ValueError(f"hidden_states must have shape (batch, sequence, {self.in_channels})")
        batch_size, sequence_length, _ = hidden_states.shape
        if self.use_pos_emb and sequence_length != self.input_size:
            raise ValueError(f"sequence length must be {self.input_size} when use_pos_emb=True")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[:1] != (batch_size,):
            raise ValueError("encoder_hidden_states must be rank three and match the latent batch")
        if encoder_hidden_states.shape[-1] != self.context_dim:
            raise ValueError(f"encoder_hidden_states last dimension must be {self.context_dim}")
        timestep = torch.as_tensor(timestep, device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.ndim != 1 or timestep.shape[0] != batch_size:
            raise ValueError("timestep must be scalar or contain one value per batch item")

        timestep_embedding = self.t_embedder(
            timestep,
            guidance_cond,
            dtype=hidden_states.dtype,
        )
        hidden_states = self.x_embedder(hidden_states)
        if self.use_pos_emb:
            hidden_states = hidden_states + self.pos_embed.to(hidden_states.dtype)

        if self.use_attention_pooling:
            pooled_context = self.pooler(encoder_hidden_states)
            conditioning_embedding = timestep_embedding + self.extra_embedder(pooled_context).unsqueeze(1)
        else:
            conditioning_embedding = timestep_embedding

        if self.with_decoupled_ca:
            if additional_encoder_hidden_states is None:
                raise ValueError("additional_encoder_hidden_states is required with decoupled cross-attention")
            additional_context = self.additional_cond_proj(additional_encoder_hidden_states)
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, additional_context],
                dim=1,
            )

        hidden_states = torch.cat([conditioning_embedding, hidden_states], dim=1)
        skip_values = []
        for layer_index, block in enumerate(self.blocks):
            skip_value = None if layer_index <= self.depth // 2 else skip_values.pop()
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    conditioning_embedding,
                    encoder_hidden_states,
                    skip_value,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    conditioning_embedding,
                    encoder_hidden_states,
                    skip_value,
                )
            if layer_index < self.depth // 2:
                skip_values.append(hidden_states)

        hidden_states = self.final_layer(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return Hunyuan3DShapeDiTOutput(sample=hidden_states)


__all__ = ["Hunyuan3DShapeDiTModel", "Hunyuan3DShapeDiTOutput"]
