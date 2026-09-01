# Portions of this file are derived from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for native Diffusers attention and model integration.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import BACKEND_REGISTRY, BackendCapability
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind
from .sparse import TrellisSparseTensor


@dataclass
class TrellisSparseStructureFlowOutput(BaseOutput):
    """Dense sparse-structure velocity prediction."""

    sample: torch.Tensor


@dataclass
class TrellisSLatFlowOutput(BaseOutput):
    """Structured-latent velocity prediction with unchanged sparse coordinates."""

    sample: TrellisSparseTensor


class TrellisLayerNorm32(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(hidden_states.float(), self.normalized_shape, weight, bias, self.eps).to(
            dtype=hidden_states.dtype
        )


class TrellisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(dtype=get_parameter_dtype(self.mlp)))


class TrellisAbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        frequency_dimension = channels // in_channels // 2
        self.frequencies = 1.0 / (
            10000 ** (torch.arange(frequency_dimension, dtype=torch.float32) / frequency_dimension)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.in_channels:
            raise ValueError(f"coordinates must have shape (positions, {self.in_channels})")
        frequencies = self.frequencies.to(device=coordinates.device)
        embedding = torch.outer(coordinates.reshape(-1), frequencies)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        embedding = embedding.reshape(coordinates.shape[0], -1)
        if embedding.shape[1] < self.channels:
            embedding = torch.cat(
                [
                    embedding,
                    torch.zeros(
                        embedding.shape[0],
                        self.channels - embedding.shape[1],
                        device=embedding.device,
                        dtype=embedding.dtype,
                    ),
                ],
                dim=-1,
            )
        return embedding


class TrellisRotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for rotary embeddings")
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        frequency_dimension = hidden_size // in_channels // 2
        self.frequencies = 1.0 / (
            10000 ** (torch.arange(frequency_dimension, dtype=torch.float32) / frequency_dimension)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indices is None:
            indices = torch.arange(query.shape[1], device=query.device).reshape(1, -1, 1)
            indices = indices.expand(query.shape[0], -1, 1)
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
        frequencies = self.frequencies.to(device=indices.device)
        phases = torch.outer(indices.reshape(-1).float(), frequencies)
        phases = torch.polar(torch.ones_like(phases), phases)
        phases = phases.reshape(*indices.shape[:-1], -1)
        target_width = query.shape[-1] // 2
        if phases.shape[-1] < target_width:
            phases = torch.cat(
                [
                    phases,
                    torch.ones(
                        *phases.shape[:-1],
                        target_width - phases.shape[-1],
                        device=phases.device,
                        dtype=phases.dtype,
                    ),
                ],
                dim=-1,
            )
        phases = phases.unsqueeze(2)
        query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        query = torch.view_as_real(query_complex * phases).reshape_as(query).to(dtype=query.dtype)
        key = torch.view_as_real(key_complex * phases).reshape_as(key).to(dtype=key.dtype)
        return query, key


class TrellisMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(hidden_states.float(), dim=-1)
        return (normalized * self.gamma * self.scale).to(dtype=hidden_states.dtype)


class TrellisAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: TrellisAttention,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if attn._type == "self":
            query_key_value = attn.to_qkv(hidden_states).reshape(
                batch_size,
                sequence_length,
                3,
                attn.num_heads,
                attn.head_dim,
            )
            query, key, value = query_key_value.unbind(dim=2)
            if attn.use_rope:
                query, key = attn.rope(query, key, indices)
        else:
            if context is None:
                raise ValueError("context is required for cross-attention")
            context_length = context.shape[1]
            query = attn.to_q(hidden_states).reshape(
                batch_size,
                sequence_length,
                attn.num_heads,
                attn.head_dim,
            )
            key_value = attn.to_kv(context).reshape(
                batch_size,
                context_length,
                2,
                attn.num_heads,
                attn.head_dim,
            )
            key, value = key_value.unbind(dim=2)
        if attn.qk_rms_norm:
            query = attn.q_rms_norm(query)
            key = attn.k_rms_norm(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, -1).to(dtype=query.dtype)
        return attn.to_out(hidden_states)


class TrellisAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = TrellisAttnProcessor
    _available_processors = [TrellisAttnProcessor]

    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        context_channels: int | None = None,
        attention_type: str = "self",
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if attention_type not in {"self", "cross"}:
            raise ValueError("attention_type must be 'self' or 'cross'")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = channels if context_channels is None else context_channels
        self._type = attention_type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        if attention_type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=True)
        else:
            self.to_q = nn.Linear(channels, channels, bias=True)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=True)
        if qk_rms_norm:
            self.q_rms_norm = TrellisMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = TrellisMultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)
        if use_rope:
            self.rope = TrellisRotaryPositionEmbedder(channels)
        self.set_processor(TrellisAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, context, indices)


class TrellisFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class TrellisModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        use_rope: bool,
        share_mod: bool,
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
    ) -> None:
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = TrellisLayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = TrellisAttention(
            channels,
            num_heads,
            attention_type="self",
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = TrellisAttention(
            channels,
            num_heads,
            context_channels=context_channels,
            attention_type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = TrellisFeedForwardNet(channels, mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_indices is not None:
            output = torch.zeros_like(hidden_states)
            for batch_index in range(context.shape[0]):
                positions = torch.nonzero(batch_indices == batch_index, as_tuple=False).reshape(-1)
                batch_output = self(
                    hidden_states[positions].unsqueeze(0),
                    modulation[batch_index : batch_index + 1],
                    context[batch_index : batch_index + 1],
                    coordinates=(None if coordinates is None else coordinates[positions].unsqueeze(0)),
                )
                output = output.index_copy(0, positions, batch_output.squeeze(0))
            return output

        if self.share_mod:
            modulated = modulation
        else:
            modulated = self.adaLN_modulation(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulated.chunk(6, dim=1)
        residual = self.norm1(hidden_states)
        residual = residual * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        residual = self.self_attn(residual, indices=coordinates)
        hidden_states = hidden_states + residual * gate_msa.unsqueeze(1)
        hidden_states = hidden_states + self.cross_attn(self.norm2(hidden_states), context)
        residual = self.norm3(hidden_states)
        residual = residual * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        residual = self.mlp(residual)
        return hidden_states + residual * gate_mlp.unsqueeze(1)


def _patchify_3d(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, channels, depth, height, width = hidden_states.shape
    if depth % patch_size or height % patch_size or width % patch_size:
        raise ValueError("all spatial dimensions must be divisible by patch_size")
    hidden_states = hidden_states.reshape(
        batch_size,
        channels,
        depth // patch_size,
        patch_size,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return hidden_states.reshape(
        batch_size,
        channels * patch_size**3,
        depth // patch_size,
        height // patch_size,
        width // patch_size,
    )


def _unpatchify_3d(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, channels, depth, height, width = hidden_states.shape
    hidden_states = hidden_states.reshape(
        batch_size,
        channels // patch_size**3,
        patch_size,
        patch_size,
        patch_size,
        depth,
        height,
        width,
    )
    hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return hidden_states.reshape(
        batch_size,
        channels // patch_size**3,
        depth * patch_size,
        height * patch_size,
        width * patch_size,
    )


class TrellisSparseStructureFlowModel(Object3DModel, PeftAdapterMixin):
    """Production-name-compatible dense TRELLIS sparse-structure flow model."""

    family_id = "trellis"
    component_role = "sparse_structure_flow_model"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TrellisModulatedTransformerCrossBlock"]
    _repeated_blocks = ["TrellisModulatedTransformerCrossBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "pos_emb", "norm"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 16,
        in_channels: int = 8,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        out_channels: int = 8,
        num_blocks: int = 24,
        num_heads: int | None = 16,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
    ) -> None:
        super().__init__()
        if min(resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks, patch_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if resolution % patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        resolved_heads = model_channels // num_head_channels if num_heads is None else num_heads
        if resolved_heads <= 0 or model_channels % resolved_heads:
            raise ValueError("model_channels must be divisible by num_heads")

        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = resolved_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.gradient_checkpointing = use_checkpoint

        self.t_embedder = TrellisTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        if pe_mode == "ape":
            position_embedder = TrellisAbsolutePositionEmbedder(model_channels, 3)
            grid_resolution = resolution // patch_size
            coordinates = torch.meshgrid(
                *[torch.arange(grid_resolution) for _ in range(3)],
                indexing="ij",
            )
            coordinates = torch.stack(coordinates, dim=-1).reshape(-1, 3)
            self.register_buffer("pos_emb", position_embedder(coordinates))

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.blocks = nn.ModuleList(
            [
                TrellisModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    resolved_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=pe_mode == "rope",
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        self._initialize_weights()
        if use_fp16:
            self.blocks.to(dtype=torch.float16)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 1024,
            "cond_channels": 1024,
            "num_blocks": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 1,
            "pe_mode": "ape",
            "qk_rms_norm": True,
            "use_fp16": True,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 4,
            "in_channels": 2,
            "out_channels": 2,
            "model_channels": 16,
            "cond_channels": 12,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "patch_size": 2,
            "pe_mode": "ape",
            "qk_rms_norm": True,
            "use_fp16": False,
        }

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> TrellisSparseStructureFlowOutput | tuple[torch.Tensor]:
        expected_shape = (
            hidden_states.shape[0],
            self.in_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
        if tuple(hidden_states.shape) != expected_shape:
            raise ValueError(f"hidden_states must have shape {expected_shape}")
        batch_size = hidden_states.shape[0]
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must be rank three and match the latent batch")
        if encoder_hidden_states.shape[2] != self.cond_channels:
            raise ValueError(f"encoder_hidden_states last dimension must be {self.cond_channels}")
        timestep = torch.as_tensor(timestep, device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must be scalar or contain one value per batch item")

        hidden_dtype = hidden_states.dtype
        hidden_states = _patchify_3d(hidden_states, self.patch_size)
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1).contiguous()
        hidden_states = self.input_layer(hidden_states)
        if self.pe_mode == "ape":
            hidden_states = hidden_states + self.pos_emb[None].to(dtype=hidden_states.dtype)
        modulation = self.t_embedder(timestep)
        if self.share_mod:
            modulation = self.adaLN_modulation(modulation)
        inner_dtype = get_parameter_dtype(self.blocks)
        hidden_states = hidden_states.to(dtype=inner_dtype)
        modulation = modulation.to(dtype=inner_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=inner_dtype)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    modulation,
                    encoder_hidden_states,
                )
            else:
                hidden_states = block(hidden_states, modulation, encoder_hidden_states)
        hidden_states = F.layer_norm(hidden_states.to(dtype=hidden_dtype), hidden_states.shape[-1:])
        hidden_states = self.out_layer(hidden_states)
        grid_resolution = self.resolution // self.patch_size
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch_size,
            self.out_channels * self.patch_size**3,
            grid_resolution,
            grid_resolution,
            grid_resolution,
        )
        hidden_states = _unpatchify_3d(hidden_states, self.patch_size).contiguous()
        if not return_dict:
            return (hidden_states,)
        return TrellisSparseStructureFlowOutput(sample=hidden_states)


class TrellisSLatFlowModel(Object3DModel, PeftAdapterMixin):
    """Portable full-attention TRELLIS SLAT flow core.

    Official sparse-convolution IO blocks are capability-gated. The backend-free
    path is intended for tiny CPU parity and does not claim production checkpoint
    coverage for those blocks.
    """

    family_id = "trellis"
    component_role = "slat-denoiser"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ("spconv",)
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TrellisModulatedTransformerCrossBlock"]
    _repeated_blocks = ["TrellisModulatedTransformerCrossBlock"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 64,
        in_channels: int = 8,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        out_channels: int = 8,
        num_blocks: int = 24,
        num_heads: int | None = 16,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: tuple[int, ...] | list[int] | None = (128,),
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
    ) -> None:
        super().__init__()
        if io_block_channels is not None:
            BACKEND_REGISTRY.select(
                BackendCapability.SPARSE_COMPUTE,
                name="spconv",
                device="cuda",
                dtype="float16" if use_fp16 else "float32",
                differentiable=True,
            )
            raise NotImplementedError(
                "official TRELLIS sparse-convolution IO blocks require the separately tested spconv production "
                "implementation; use io_block_channels=None only for the portable full-attention core"
            )
        del num_io_res_blocks, use_skip_connection
        if min(resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks, patch_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        resolved_heads = model_channels // num_head_channels if num_heads is None else num_heads
        if resolved_heads <= 0 or model_channels % resolved_heads:
            raise ValueError("model_channels must be divisible by num_heads")

        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = resolved_heads
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.share_mod = share_mod
        self.gradient_checkpointing = use_checkpoint

        self.t_embedder = TrellisTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        if pe_mode == "ape":
            self.pos_embedder = TrellisAbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.input_blocks = nn.ModuleList()
        self.blocks = nn.ModuleList(
            [
                TrellisModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    resolved_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=pe_mode == "rope",
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_blocks = nn.ModuleList()
        self.out_layer = nn.Linear(model_channels, out_channels)
        self._initialize_weights()
        if use_fp16:
            self.blocks.to(dtype=torch.float16)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 64,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 1024,
            "cond_channels": 1024,
            "num_blocks": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "patch_size": 2,
            "num_io_res_blocks": 2,
            "io_block_channels": [128],
            "pe_mode": "ape",
            "qk_rms_norm": True,
            "use_fp16": True,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 8,
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 16,
            "cond_channels": 12,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "patch_size": 1,
            "num_io_res_blocks": 0,
            "io_block_channels": None,
            "pe_mode": "ape",
            "qk_rms_norm": True,
            "use_fp16": False,
        }

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> TrellisSLatFlowOutput | tuple[TrellisSparseTensor]:
        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        if hidden_states.channels != self.in_channels:
            raise ValueError(f"hidden_states must have {self.in_channels} feature channels")
        if bool((hidden_states.coordinates[:, 1:] >= self.resolution).any()):
            raise ValueError("hidden_states coordinates fall outside the configured resolution")
        batch_size = hidden_states.batch_size
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must be rank three and match the sparse batch")
        if encoder_hidden_states.shape[2] != self.cond_channels:
            raise ValueError(f"encoder_hidden_states last dimension must be {self.cond_channels}")
        timestep = torch.as_tensor(timestep, device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must be scalar or contain one value per batch item")

        output_dtype = hidden_states.dtype
        features = self.input_layer(hidden_states.features)
        if self.pe_mode == "ape":
            position_embedding = self.pos_embedder(hidden_states.coordinates[:, 1:])
            features = features + position_embedding.to(dtype=features.dtype)
        modulation = self.t_embedder(timestep)
        if self.share_mod:
            modulation = self.adaLN_modulation(modulation)
        inner_dtype = get_parameter_dtype(self.blocks)
        features = features.to(dtype=inner_dtype)
        modulation = modulation.to(dtype=inner_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=inner_dtype)
        batch_indices = hidden_states.coordinates[:, 0]
        spatial_coordinates = hidden_states.coordinates[:, 1:]
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                features = self._gradient_checkpointing_func(
                    block,
                    features,
                    modulation,
                    encoder_hidden_states,
                    batch_indices,
                    spatial_coordinates,
                )
            else:
                features = block(
                    features,
                    modulation,
                    encoder_hidden_states,
                    batch_indices,
                    spatial_coordinates,
                )
        features = F.layer_norm(features, features.shape[-1:])
        features = self.out_layer(features.to(dtype=output_dtype))
        sample = hidden_states.replace(features)
        if not return_dict:
            return (sample,)
        return TrellisSLatFlowOutput(sample=sample)


__all__ = [
    "TrellisSLatFlowModel",
    "TrellisSLatFlowOutput",
    "TrellisSparseStructureFlowModel",
    "TrellisSparseStructureFlowOutput",
]
