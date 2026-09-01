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
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import BACKEND_REGISTRY, BackendCapability
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind
from ..trellis.sparse import TrellisSparseTensor


@dataclass
class Trellis2SparseStructureFlowOutput(BaseOutput):
    sample: torch.Tensor


@dataclass
class Trellis2SLatFlowOutput(BaseOutput):
    sample: TrellisSparseTensor


class Trellis2LayerNorm32(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(hidden_states.float(), self.normalized_shape, weight, bias, self.eps).to(
            dtype=hidden_states.dtype
        )


class Trellis2TimestepEmbedder(nn.Module):
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


class Trellis2AbsolutePositionEmbedder(nn.Module):
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


class Trellis2RotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dimensions: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
    ) -> None:
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
        if phases.shape[-1] < self.head_dim // 2:
            padding = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat(
                [
                    phases,
                    torch.ones(*phases.shape[:-1], padding, device=phases.device, dtype=phases.dtype),
                ],
                dim=-1,
            )
        return phases

    @staticmethod
    def apply_rotary_embedding(hidden_states: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        complex_states = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
        rotated = complex_states * phases.unsqueeze(-2)
        return torch.view_as_real(rotated).reshape_as(hidden_states).to(dtype=hidden_states.dtype)


class Trellis2MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (F.normalize(hidden_states.float(), dim=-1) * self.gamma * self.scale).to(hidden_states.dtype)


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
                batch_size,
                sequence_length,
                3,
                attn.num_heads,
                attn.head_dim,
            )
            query, key, value = query_key_value.unbind(dim=2)
        else:
            if context is None:
                raise ValueError("context is required for cross-attention")
            query = attn.to_q(hidden_states).reshape(
                batch_size,
                sequence_length,
                attn.num_heads,
                attn.head_dim,
            )
            key_value = attn.to_kv(context).reshape(
                batch_size,
                context.shape[1],
                2,
                attn.num_heads,
                attn.head_dim,
            )
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
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
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
            self.q_rms_norm = Trellis2MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = Trellis2MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)
        self.set_processor(Trellis2AttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, context, phases)


class Trellis2FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


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
        self.norm1 = Trellis2LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = Trellis2LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = Trellis2LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = Trellis2Attention(
            channels,
            num_heads,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = Trellis2Attention(
            channels,
            num_heads,
            context_channels=context_channels,
            attention_type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = Trellis2FeedForwardNet(channels, mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )

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
        rope_phases = self._buffers.pop("rope_phases", None)
        try:
            result = super()._apply(fn, recurse)
        finally:
            if rope_phases is not None:
                probe = fn(torch.empty(0, device=rope_phases.device, dtype=torch.float32))
                self._buffers["rope_phases"] = rope_phases.to(device=probe.device)
        return result

    def _convert_torso(self) -> None:
        # Upstream converts only primitive Linear layers in the transformer
        # torso. Norm, RMSNorm, and shared-modulation parameters remain fp32.
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


class Trellis2SparseStructureFlowModel(_Trellis2FlowInitialization, Object3DModel):
    """Reviewed dense TRELLIS.2 sparse-structure flow transformer."""

    family_id = "trellis2"
    component_role = "sparse_structure_flow_model"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
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
        self.t_embedder = Trellis2TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        coordinates = torch.stack(
            torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        if pe_mode == "ape":
            self.register_buffer("pos_emb", Trellis2AbsolutePositionEmbedder(model_channels)(coordinates))
            self.rope_phases = None
        else:
            rope = Trellis2RotaryPositionEmbedder(
                model_channels // resolved_heads,
                rope_freq=tuple(float(item) for item in rope_freq),
            )
            self.register_buffer("rope_phases", rope(coordinates))
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

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 16,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 1536,
            "cond_channels": 1024,
            "num_blocks": 30,
            "num_heads": 12,
            "mlp_ratio": 5.3334,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "scaled",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "dtype": "bfloat16",
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 2,
            "in_channels": 2,
            "out_channels": 2,
            "model_channels": 12,
            "cond_channels": 12,
            "num_blocks": 2,
            "num_heads": 3,
            "mlp_ratio": 2,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "scaled",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "dtype": "float32",
        }

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
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("encoder_hidden_states must be rank three and match the latent batch")
        if encoder_hidden_states.shape[2] != self.cond_channels:
            raise ValueError(f"encoder_hidden_states last dimension must be {self.cond_channels}")
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
        encoder_hidden_states = encoder_hidden_states.to(dtype=inner_dtype)
        phases = None if self.rope_phases is None else self.rope_phases.to(device=hidden_states.device)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    modulation,
                    encoder_hidden_states,
                    phases,
                )
            else:
                hidden_states = block(hidden_states, modulation, encoder_hidden_states, phases)
        hidden_states = F.layer_norm(hidden_states.to(dtype=output_dtype), hidden_states.shape[-1:])
        hidden_states = self.out_layer(hidden_states)
        hidden_states = (
            hidden_states.permute(0, 2, 1)
            .reshape(
                batch_size,
                self.out_channels,
                self.resolution,
                self.resolution,
                self.resolution,
            )
            .contiguous()
        )
        if not return_dict:
            return (hidden_states,)
        return Trellis2SparseStructureFlowOutput(sample=hidden_states)


class Trellis2SLatFlowModel(_Trellis2FlowInitialization, Object3DModel):
    """Experimental backend-free tiny core for TRELLIS.2 shape and texture SLAT."""

    family_id = "trellis2"
    component_role = "slat_flow_model"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.O_VOXEL)
    required_backends = ("flex_gemm",)
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED
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
        require_flex_gemm: bool = True,
    ) -> None:
        super().__init__()
        if require_flex_gemm:
            BACKEND_REGISTRY.select(
                BackendCapability.SPARSE_COMPUTE,
                name="flex_gemm",
                device="cuda",
                dtype=dtype,
                differentiable=True,
            )
            raise NotImplementedError(
                "production TRELLIS.2 sparse IO requires a pinned, parity-tested FlexGEMM integration; "
                "require_flex_gemm=False is only the backend-free tiny full-attention core"
            )
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
        self.t_embedder = Trellis2TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )
        if pe_mode == "ape":
            self.pos_embedder = Trellis2AbsolutePositionEmbedder(model_channels)
            self.rope = None
        else:
            self.rope = Trellis2RotaryPositionEmbedder(
                model_channels // resolved_heads,
                rope_freq=tuple(float(item) for item in rope_freq),
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

    @classmethod
    def production_config(cls, *, texture: bool = False) -> dict[str, Any]:
        return {
            "resolution": 32,
            "in_channels": 64 if texture else 32,
            "out_channels": 32,
            "model_channels": 1536,
            "cond_channels": 1024,
            "num_blocks": 30,
            "num_heads": 12,
            "mlp_ratio": 5.3334,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "scaled",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "dtype": "bfloat16",
            "require_flex_gemm": True,
        }

    @classmethod
    def tiny_config(cls, *, texture: bool = False) -> dict[str, Any]:
        return {
            "resolution": 8,
            "in_channels": 8 if texture else 4,
            "out_channels": 4,
            "model_channels": 12,
            "cond_channels": 12,
            "num_blocks": 2,
            "num_heads": 3,
            "mlp_ratio": 2,
            "pe_mode": "rope",
            "share_mod": True,
            "initialization": "scaled",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "dtype": "float32",
            "require_flex_gemm": False,
        }

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
            if not isinstance(concat_cond, TrellisSparseTensor) or not torch.equal(
                hidden_states.coordinates, concat_cond.coordinates
            ):
                raise ValueError("concat_cond must be coordinate-aligned with hidden_states")
            input_features = torch.cat([input_features, concat_cond.features], dim=-1)
        if input_features.shape[1] != self.in_channels:
            raise ValueError(f"combined sparse input must have {self.in_channels} feature channels")
        if bool((hidden_states.coordinates[:, 1:] >= self.resolution).any()):
            raise ValueError("hidden_states coordinates fall outside the configured resolution")
        batch_size = hidden_states.batch_size
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape != (
            batch_size,
            encoder_hidden_states.shape[1],
            self.cond_channels,
        ):
            raise ValueError("encoder_hidden_states must match sparse batch and context channels")
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
        output = torch.zeros_like(features)
        for batch_index in range(batch_size):
            positions = torch.nonzero(hidden_states.coordinates[:, 0] == batch_index, as_tuple=False).reshape(-1)
            inner_dtype = get_parameter_dtype(self.blocks)
            batch_features = features[positions].unsqueeze(0).to(dtype=inner_dtype)
            batch_modulation = modulation[batch_index : batch_index + 1].to(dtype=inner_dtype)
            batch_context = encoder_hidden_states[batch_index : batch_index + 1].to(dtype=inner_dtype)
            phases = None if self.rope is None else self.rope(hidden_states.coordinates[positions, 1:]).unsqueeze(0)
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    batch_features = self._gradient_checkpointing_func(
                        block,
                        batch_features,
                        batch_modulation,
                        batch_context,
                        phases,
                    )
                else:
                    batch_features = block(
                        batch_features,
                        batch_modulation,
                        batch_context,
                        phases,
                    )
            output = output.index_copy(0, positions, batch_features.squeeze(0).to(dtype=features.dtype))
        output = F.layer_norm(output, output.shape[-1:])
        sample = hidden_states.replace(self.out_layer(output).to(dtype=hidden_states.dtype))
        if not return_dict:
            return (sample,)
        return Trellis2SLatFlowOutput(sample=sample)


__all__ = [
    "Trellis2SLatFlowModel",
    "Trellis2SLatFlowOutput",
    "Trellis2SparseStructureFlowModel",
    "Trellis2SparseStructureFlowOutput",
]
