# Portions of this file reproduce Microsoft TRELLIS.2 model semantics:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for native Diffusers attention, loading, and package-owned sparse tensors.

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

from ...execution.models import Object3DModel
from ..trellis.models import (
    TrellisAbsolutePositionEmbedder,
    TrellisFeedForwardNet,
    TrellisLayerNorm32,
    TrellisMultiHeadRMSNorm,
    TrellisTimestepEmbedder,
)
from ..trellis.sparse import TrellisSparseTensor


@dataclass
class Trellis2SparseStructureFlowOutput(BaseOutput):
    sample: torch.Tensor


@dataclass
class Trellis2SLatFlowOutput(BaseOutput):
    sample: TrellisSparseTensor


class Trellis2RotaryPositionEmbedder(nn.Module):
    def __init__(self, head_dim: int, dimensions: int = 3, rope_freq: tuple[float, float] = (1.0, 10000.0)) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        self.head_dim = head_dim
        self.dimensions = dimensions
        frequency_dimension = head_dim // 2 // dimensions
        frequencies = torch.arange(frequency_dimension, dtype=torch.float32) / frequency_dimension
        self.frequencies = rope_freq[0] / (rope_freq[1] ** frequencies)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.dimensions:
            raise ValueError(f"coordinates must end with {self.dimensions} values")
        frequencies = self.frequencies.to(device=coordinates.device)
        phases = torch.outer(coordinates.reshape(-1).float(), frequencies)
        phases = torch.polar(torch.ones_like(phases), phases)
        phases = phases.reshape(*coordinates.shape[:-1], -1)
        # Head channels left over after splitting by axis are not rotated.
        padding = phases.new_ones(*phases.shape[:-1], self.head_dim // 2 - phases.shape[-1])
        return torch.cat([phases, padding], dim=-1)

    @staticmethod
    def apply_rotary_embedding(hidden_states: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        complex_states = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
        rotated = complex_states * phases.unsqueeze(-2)
        return torch.view_as_real(rotated).reshape_as(hidden_states).to(dtype=hidden_states.dtype)


class Trellis2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Trellis2Attention,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if attn._type == "self":
            query_key_value = attn.to_qkv(hidden_states).reshape(
                batch_size, sequence_length, 3, attn.num_heads, attn.head_dim
            )
            query, key, value = query_key_value.unbind(dim=2)
        else:
            if context is None:
                raise ValueError("context is required for cross-attention")
            query = attn.to_q(hidden_states).reshape(batch_size, sequence_length, attn.num_heads, attn.head_dim)
            key_value = attn.to_kv(context).reshape(batch_size, context.shape[1], 2, attn.num_heads, attn.head_dim)
            key, value = key_value.unbind(dim=2)
        if attn.qk_rms_norm:
            query = attn.q_rms_norm(query)
            key = attn.k_rms_norm(key)
        if attn.use_rope:
            if phases is None:
                raise ValueError("rotary phases are required for RoPE self-attention")
            query = Trellis2RotaryPositionEmbedder.apply_rotary_embedding(query, phases)
            key = Trellis2RotaryPositionEmbedder.apply_rotary_embedding(key, phases)
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, -1).to(dtype=query.dtype)
        return attn.to_out(hidden_states)


class Trellis2Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Trellis2AttnProcessor
    _available_processors = [Trellis2AttnProcessor]

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
        self.set_processor(Trellis2AttnProcessor())

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor | None = None, phases: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, context, phases)


class Trellis2ModulatedTransformerCrossBlock(nn.Module):
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
        self.self_attn = Trellis2Attention(channels, num_heads, use_rope=use_rope, qk_rms_norm=qk_rms_norm)
        self.cross_attn = Trellis2Attention(
            channels,
            num_heads,
            context_channels=context_channels,
            attention_type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = TrellisFeedForwardNet(channels, mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.share_mod:
            modulated = (self.modulation + modulation).to(dtype=modulation.dtype)
        else:
            modulated = self.adaLN_modulation(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulated.chunk(6, dim=1)
        residual = self.norm1(hidden_states)
        residual = residual * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        residual = self.self_attn(residual, phases=phases)
        hidden_states = hidden_states + residual * gate_msa.unsqueeze(1)
        hidden_states = hidden_states + self.cross_attn(self.norm2(hidden_states), context)
        residual = self.norm3(hidden_states)
        residual = residual * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        residual = self.mlp(residual)
        return hidden_states + residual * gate_mlp.unsqueeze(1)


class _Trellis2FlowInitialization:
    def _apply(self, fn, recurse: bool = True):
        # ``rope_phases`` is complex; keep it out of ``.to(dtype)`` casts and only follow device moves.
        rope_phases = self._buffers.pop("rope_phases", None)
        try:
            result = super()._apply(fn, recurse)
        finally:
            if rope_phases is not None:
                probe = fn(torch.empty(0, device=rope_phases.device, dtype=torch.float32))
                self._buffers["rope_phases"] = rope_phases.to(device=probe.device)
        return result

    def _convert_torso(self) -> None:
        # Upstream casts only the torso's Linear layers; norms and shared modulation stay fp32.
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                module.to(dtype=self.inner_dtype)

    def _initialize_weights(self) -> None:
        initialization = self.initialization

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if initialization == "vanilla":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=math.sqrt(2.0 / (5.0 * self.model_channels)))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        if initialization == "scaled":
            standard_deviation = 1.0 / math.sqrt(5 * self.num_blocks * self.model_channels)
            for block in self.blocks:
                for layer in (block.self_attn.to_out, block.cross_attn.to_out, block.mlp.mlp[2]):
                    nn.init.normal_(layer.weight, std=standard_deviation)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            nn.init.normal_(self.input_layer.weight, std=1.0 / math.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        else:
            for block in self.blocks:
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)


class Trellis2SparseStructureFlowModel(_Trellis2FlowInitialization, Object3DModel, PeftAdapterMixin):
    """TRELLIS.2 sparse-structure flow transformer over the dense occupancy latent."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Trellis2ModulatedTransformerCrossBlock"]
    _repeated_blocks = ["Trellis2ModulatedTransformerCrossBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "pos_emb", "rope_phases", "norm"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 16,
        in_channels: int = 8,
        model_channels: int = 1536,
        cond_channels: int = 1024,
        out_channels: int = 8,
        num_blocks: int = 30,
        num_heads: int | None = 12,
        num_head_channels: int = 64,
        mlp_ratio: float = 5.3334,
        pe_mode: str = "rope",
        rope_freq: tuple[float, float] | list[float] = (1.0, 10000.0),
        dtype: str = "bfloat16",
        use_checkpoint: bool = False,
        share_mod: bool = True,
        initialization: str = "scaled",
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ) -> None:
        super().__init__()
        if min(resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks) <= 0:
            raise ValueError("model dimensions must be positive")
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        if initialization not in {"vanilla", "scaled"}:
            raise ValueError("initialization must be 'vanilla' or 'scaled'")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
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
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        self.initialization = initialization
        self.inner_dtype = getattr(torch, dtype)
        self.gradient_checkpointing = use_checkpoint
        self.t_embedder = TrellisTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        coordinates = torch.stack(
            torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing="ij"), dim=-1
        ).reshape(-1, 3)
        if pe_mode == "ape":
            self.register_buffer("pos_emb", TrellisAbsolutePositionEmbedder(model_channels)(coordinates))
            self.rope_phases = None
        else:
            rope = Trellis2RotaryPositionEmbedder(
                model_channels // resolved_heads, rope_freq=tuple(float(item) for item in rope_freq)
            )
            # Derived from the config; the released checkpoints do not store it.
            self.register_buffer("rope_phases", rope(coordinates), persistent=False)
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                Trellis2ModulatedTransformerCrossBlock(
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
        self.out_layer = nn.Linear(model_channels, out_channels)
        self._initialize_weights()
        self._convert_torso()

    def _block_context(self, encoder_hidden_states: Any, batch_size: int, dtype: torch.dtype) -> Any:
        """Validate the cross-attention context and cast it for the blocks; subclasses may pass richer contexts."""

        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must be rank three and match the latent batch")
        if encoder_hidden_states.shape[2] != self.cond_channels:
            raise ValueError(f"encoder_hidden_states last dimension must be {self.cond_channels}")
        return encoder_hidden_states.to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Trellis2SparseStructureFlowOutput | tuple[torch.Tensor]:
        expected = (hidden_states.shape[0], self.in_channels, *([self.resolution] * 3))
        if tuple(hidden_states.shape) != expected:
            raise ValueError(f"hidden_states must have shape {expected}")
        batch_size = hidden_states.shape[0]
        timestep = torch.as_tensor(timestep, device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must be scalar or contain one value per batch item")
        output_dtype = hidden_states.dtype
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
        encoder_hidden_states = self._block_context(encoder_hidden_states, batch_size, inner_dtype)
        phases = None if self.rope_phases is None else self.rope_phases.to(device=hidden_states.device)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, modulation, encoder_hidden_states, phases
                )
            else:
                hidden_states = block(hidden_states, modulation, encoder_hidden_states, phases)
        hidden_states = F.layer_norm(hidden_states.to(dtype=output_dtype), hidden_states.shape[-1:])
        hidden_states = self.out_layer(hidden_states)
        hidden_states = (
            hidden_states.permute(0, 2, 1)
            .reshape(batch_size, self.out_channels, self.resolution, self.resolution, self.resolution)
            .contiguous()
        )
        if not return_dict:
            return (hidden_states,)
        return Trellis2SparseStructureFlowOutput(sample=hidden_states)


class Trellis2SLatFlowModel(_Trellis2FlowInitialization, Object3DModel, PeftAdapterMixin):
    """TRELLIS.2 shape / texture SLAT flow transformer (released ``SLatFlowModel`` layout).

    Unlike TRELLIS, there are no sparse-convolution stages: a linear input layer, RoPE full-attention DiT blocks
    over each item's active voxels, and a linear output layer. The texture variant takes the shape SLAT as
    ``concat_cond``.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Trellis2ModulatedTransformerCrossBlock"]
    _repeated_blocks = ["Trellis2ModulatedTransformerCrossBlock"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 32,
        in_channels: int = 32,
        model_channels: int = 1536,
        cond_channels: int = 1024,
        out_channels: int = 32,
        num_blocks: int = 30,
        num_heads: int | None = 12,
        num_head_channels: int = 64,
        mlp_ratio: float = 5.3334,
        pe_mode: str = "rope",
        rope_freq: tuple[float, float] | list[float] = (1.0, 10000.0),
        dtype: str = "bfloat16",
        use_checkpoint: bool = False,
        share_mod: bool = True,
        initialization: str = "scaled",
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ) -> None:
        super().__init__()
        if min(resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks) <= 0:
            raise ValueError("model dimensions must be positive")
        if pe_mode not in {"ape", "rope"} or initialization not in {"vanilla", "scaled"}:
            raise ValueError("unsupported positional embedding or initialization")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
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
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        self.initialization = initialization
        self.inner_dtype = getattr(torch, dtype)
        self.gradient_checkpointing = use_checkpoint
        self.t_embedder = TrellisTimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        if pe_mode == "ape":
            self.pos_embedder = TrellisAbsolutePositionEmbedder(model_channels)
            self.rope = None
        else:
            self.rope = Trellis2RotaryPositionEmbedder(
                model_channels // resolved_heads, rope_freq=tuple(float(item) for item in rope_freq)
            )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                Trellis2ModulatedTransformerCrossBlock(
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
        self.out_layer = nn.Linear(model_channels, out_channels)
        self._initialize_weights()
        self._convert_torso()

    def _validate_context(self, encoder_hidden_states: Any, hidden_states: TrellisSparseTensor) -> None:
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape != (
            hidden_states.batch_size,
            encoder_hidden_states.shape[1],
            self.cond_channels,
        ):
            raise ValueError("encoder_hidden_states must match sparse batch and context channels")

    def _batch_context(
        self, encoder_hidden_states: Any, batch_index: int, positions: torch.Tensor, dtype: torch.dtype
    ) -> Any:
        """One batch item's cross-attention context; ``positions`` are its rows of the sparse input."""

        return encoder_hidden_states[batch_index : batch_index + 1].to(dtype=dtype)

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        concat_cond: TrellisSparseTensor | None = None,
        return_dict: bool = True,
    ) -> Trellis2SLatFlowOutput | tuple[TrellisSparseTensor]:
        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        input_features = hidden_states.features
        if concat_cond is not None:
            if not isinstance(concat_cond, TrellisSparseTensor) or not hidden_states.has_coordinates_of(concat_cond):
                raise ValueError("concat_cond must be coordinate-aligned with hidden_states")
            input_features = torch.cat([input_features, concat_cond.features], dim=-1)
        if input_features.shape[1] != self.in_channels:
            raise ValueError(f"combined sparse input must have {self.in_channels} feature channels")
        # ``resolution`` is nominal: RoPE and the sinusoidal APE are unbounded, and the released 1536 cascade
        # runs the 1024 model on a 96^3 grid.
        batch_size = hidden_states.batch_size
        self._validate_context(encoder_hidden_states, hidden_states)
        timestep = torch.as_tensor(timestep, device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must be scalar or contain one value per batch item")
        features = self.input_layer(input_features)
        if self.pe_mode == "ape":
            features = features + self.pos_embedder(hidden_states.coordinates[:, 1:]).to(features)
        modulation = self.t_embedder(timestep)
        if self.share_mod:
            modulation = self.adaLN_modulation(modulation)
        inner_dtype = get_parameter_dtype(self.blocks)
        output = torch.zeros_like(features)
        # Positions and RoPE phases depend only on the coordinates, so a sampling run computes them once.
        all_positions = hidden_states.batch_positions()
        all_phases = (
            None
            if self.rope is None
            else hidden_states.cached(
                ("rope_phases", self.rope),
                lambda: tuple(self.rope(hidden_states.coordinates[p, 1:]).unsqueeze(0) for p in all_positions),
            )
        )
        for batch_index, positions in enumerate(all_positions):
            batch_features = features[positions].unsqueeze(0).to(dtype=inner_dtype)
            batch_modulation = modulation[batch_index : batch_index + 1].to(dtype=inner_dtype)
            batch_context = self._batch_context(encoder_hidden_states, batch_index, positions, inner_dtype)
            phases = None if all_phases is None else all_phases[batch_index]
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    batch_features = self._gradient_checkpointing_func(
                        block, batch_features, batch_modulation, batch_context, phases
                    )
                else:
                    batch_features = block(batch_features, batch_modulation, batch_context, phases)
            output = output.index_copy(0, positions, batch_features.squeeze(0).to(dtype=features.dtype))
        output = F.layer_norm(output, output.shape[-1:])
        sample = hidden_states.replace(self.out_layer(output).to(dtype=hidden_states.dtype))
        if not return_dict:
            return (sample,)
        return Trellis2SLatFlowOutput(sample=sample)


# The gated DINOv3 repositories upstream loads, at pinned revisions (TRELLIS.2 and Pixal3D: ViT-L/16; IDOL and
# TripoSplat: ViT-H/16+). Their weights load from these repositories and are never converted or redistributed.
DINOV3_VITL16_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_VITH16PLUS_REPO = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DINOV3_REVISIONS = {
    DINOV3_VITL16_REPO: "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    DINOV3_VITH16PLUS_REPO: "c807c9eeea853df70aec4069e6f56b28ddc82acc",
}


def load_dinov3(repository: str, *, revision: str | None = None, dtype: torch.dtype | None = None, **hub_kwargs: Any):
    """Load a Transformers ``DINOv3ViTModel`` from its repository, in ``dtype`` (float32, as released, by default)."""

    from transformers import DINOv3ViTModel

    model, loading_info = DINOv3ViTModel.from_pretrained(
        repository,
        revision=revision if revision is not None else DINOV3_REVISIONS.get(repository),
        dtype=torch.float32 if dtype is None else dtype,
        output_loading_info=True,
        **hub_kwargs,
    )
    if any(loading_info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"the DINOv3 weights do not match its config: {loading_info}")
    return model.eval()


__all__ = [
    "DINOV3_REVISIONS",
    "DINOV3_VITH16PLUS_REPO",
    "DINOV3_VITL16_REPO",
    "load_dinov3",
    "Trellis2SLatFlowModel",
    "Trellis2SLatFlowOutput",
    "Trellis2SparseStructureFlowModel",
    "Trellis2SparseStructureFlowOutput",
]
