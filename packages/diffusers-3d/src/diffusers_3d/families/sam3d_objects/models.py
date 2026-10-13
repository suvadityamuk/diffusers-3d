# Portions of this file are derived from SAM 3D Objects:
# https://github.com/facebookresearch/sam-3d-objects
# Revision: f91db411c50efee93d8db7aeb323885650f6f722
#
# SAM License. Copyright (c) Meta Platforms, Inc. and affiliates.
# This file has been modified to compose the package's TRELLIS transformer components.

from __future__ import annotations

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

from ...execution.lora import Object3DPeftAdapterMixin
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind
from ..trellis.models import (
    TrellisAbsolutePositionEmbedder,
    TrellisAttention,
    TrellisFeedForwardNet,
    TrellisLayerNorm32,
    TrellisTimestepEmbedder,
)

_STREAMS = ("shape", "pose")


@dataclass
class Sam3DSparseStructureFlowOutput(BaseOutput):
    """Per-modality velocities: ``shape`` tokens followed by the pose modalities."""

    sample: dict[str, torch.Tensor]


class Sam3DMoTAttnProcessor:
    """Mixture-of-transformers self-attention: shape tokens attend only to themselves, pose tokens to everything."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self, attn: Sam3DMoTSelfAttention, shape: torch.Tensor, pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = {}
        for name, hidden_states in zip(_STREAMS, (shape, pose)):
            stream = attn.streams[name]
            batch_size, length, _ = hidden_states.shape
            query, key, value = (
                stream.to_qkv(hidden_states).reshape(batch_size, length, 3, stream.num_heads, -1).unbind(2)
            )
            if stream.qk_rms_norm:
                query, key = stream.q_rms_norm(query), stream.k_rms_norm(key)
            projected[name] = query, key, value
        (shape_query, shape_key, shape_value), (pose_query, pose_key, pose_value) = projected.values()
        attended = (
            dispatch_attention_fn(
                shape_query,
                shape_key,
                shape_value,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ),
            dispatch_attention_fn(
                pose_query,
                torch.cat([pose_key, shape_key], dim=1),
                torch.cat([pose_value, shape_value], dim=1),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            ),
        )
        return tuple(
            attn.streams[name].to_out(hidden_states.flatten(2).to(dtype=query.dtype))
            for name, hidden_states, (query, _, _) in zip(_STREAMS, attended, projected.values())
        )


class Sam3DMoTSelfAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Sam3DMoTAttnProcessor
    _available_processors = [Sam3DMoTAttnProcessor]

    def __init__(self, channels: int, num_heads: int, *, qk_rms_norm: bool) -> None:
        super().__init__()
        self.streams = nn.ModuleDict(
            {name: TrellisAttention(channels, num_heads, qk_rms_norm=qk_rms_norm) for name in _STREAMS}
        )
        self.set_processor(Sam3DMoTAttnProcessor())

    def forward(self, shape: torch.Tensor, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, shape, pose)


class Sam3DMoTBlock(nn.Module):
    """Modulated cross-attention block with separate shape/pose weights and one shared adaLN modulation."""

    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
    ) -> None:
        super().__init__()

        def per_stream(factory):
            return nn.ModuleDict({name: factory() for name in _STREAMS})

        self.norm1 = per_stream(lambda: TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6))
        self.norm2 = per_stream(lambda: TrellisLayerNorm32(channels, elementwise_affine=True, eps=1e-6))
        self.norm3 = per_stream(lambda: TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6))
        self.self_attn = Sam3DMoTSelfAttention(channels, num_heads, qk_rms_norm=qk_rms_norm)
        self.cross_attn = per_stream(
            lambda: TrellisAttention(
                channels,
                num_heads,
                context_channels=context_channels,
                attention_type="cross",
                qk_rms_norm=qk_rms_norm_cross,
            )
        )
        self.mlp = per_stream(lambda: TrellisFeedForwardNet(channels, mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def forward(
        self, shape: torch.Tensor, pose: torch.Tensor, modulation: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            value.unsqueeze(1) for value in self.adaLN_modulation(modulation).chunk(6, dim=1)
        )
        streams = dict(zip(_STREAMS, (shape, pose)))
        attended = self.self_attn(
            *(self.norm1[name](hidden_states) * (1 + scale_msa) + shift_msa for name, hidden_states in streams.items())
        )
        outputs = []
        for (name, hidden_states), residual in zip(streams.items(), attended):
            hidden_states = hidden_states + residual * gate_msa
            hidden_states = hidden_states + self.cross_attn[name](self.norm2[name](hidden_states), context)
            residual = self.norm3[name](hidden_states) * (1 + scale_mlp) + shift_mlp
            outputs.append(hidden_states + self.mlp[name](residual) * gate_mlp)
        return tuple(outputs)


class Sam3DLatentProjection(nn.Module):
    """Input/output projection and position embedding of one generated modality (upstream ``mm_latent.Latent``)."""

    def __init__(self, channels: int, model_channels: int, position_embedding: torch.Tensor | None = None) -> None:
        super().__init__()
        self.input_layer = nn.Linear(channels, model_channels)
        self.out_layer = nn.Linear(model_channels, channels)
        if position_embedding is None:
            self.pos_emb = nn.Parameter(torch.randn(1, model_channels))
        else:
            self.register_buffer("pos_emb", position_embedding)


class Sam3DSparseStructureFlowModel(Object3DModel, Object3DPeftAdapterMixin):
    """SAM 3D Objects geometry model: a shortcut flow over the 16^3 shape latent and the object layout.

    The dense shape latent is the TRELLIS sparse-structure latent, flattened to ``resolution**3`` tokens. The layout
    modalities (6D rotation, translation, log-scale, log translation scale) share a second "pose" stream; both
    streams run every block with their own weights and a single timestep modulation.
    """

    family_id = "sam3d_objects"
    component_role = "sparse_structure_flow_model"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Sam3DMoTBlock"]
    _repeated_blocks = ["Sam3DMoTBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "d_embedder", "pos_emb", "norm"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 16,
        in_channels: int = 8,
        pose_latents: list[tuple[str, int]] | None = None,
        model_channels: int = 1024,
        cond_channels: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        # Ordered (name, channels) pairs: the order is the pose stream's token order.
        pose_channels = dict(self.production_pose_latents() if pose_latents is None else pose_latents)
        if min(resolution, in_channels, model_channels, cond_channels, num_blocks, *pose_channels.values()) <= 0:
            raise ValueError("model dimensions must be positive")
        if "shape" in pose_channels or len(pose_channels) != len(pose_latents or pose_channels):
            raise ValueError("pose latent names must be unique and must not be 'shape'")
        self.resolution = resolution
        self.pose_names = tuple(pose_channels)
        self.gradient_checkpointing = use_checkpoint

        self.t_embedder = TrellisTimestepEmbedder(model_channels)
        self.d_embedder = TrellisTimestepEmbedder(model_channels)
        coordinates = torch.stack(torch.meshgrid(*[torch.arange(resolution)] * 3, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        self.latent_mapping = nn.ModuleDict(
            {
                "shape": Sam3DLatentProjection(
                    in_channels, model_channels, TrellisAbsolutePositionEmbedder(model_channels)(coordinates)
                ),
                **{name: Sam3DLatentProjection(channels, model_channels) for name, channels in pose_channels.items()},
            }
        )
        self.blocks = nn.ModuleList(
            [
                Sam3DMoTBlock(
                    model_channels,
                    cond_channels,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self._initialize_weights()

    @staticmethod
    def production_pose_latents() -> list[tuple[str, int]]:
        return [("rotation", 6), ("translation", 3), ("scale", 3), ("translation_scale", 1)]

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 16,
            "in_channels": 8,
            "pose_latents": cls.production_pose_latents(),
            "model_channels": 1024,
            "cond_channels": 1024,
            "num_blocks": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "qk_rms_norm": True,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 2,
            "in_channels": 2,
            "pose_latents": cls.production_pose_latents(),
            "model_channels": 16,
            "cond_channels": 12,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "qk_rms_norm": True,
        }

    def latent_shapes(self, batch_size: int) -> dict[str, tuple[int, ...]]:
        """Token-form latent shape of every modality, keyed like :meth:`forward` inputs."""

        return {
            name: (batch_size, mapping.pos_emb.shape[0], mapping.input_layer.in_features)
            for name, mapping in self.latent_mapping.items()
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
        nn.init.constant_(self.d_embedder.mlp[2].weight, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shortcut_step: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
    ) -> Sam3DSparseStructureFlowOutput | tuple[dict[str, torch.Tensor]]:
        """Predict per-modality velocities at ``timestep`` (``0`` is noise, ``1000`` is data).

        ``shortcut_step`` is the shortcut-model step size on the same scale; the released default sampler uses
        ``0`` (plain flow matching), which is also what ``None`` means.
        """

        expected = self.latent_shapes(encoder_hidden_states.shape[0])
        if set(latents) != set(expected) or any(tuple(latents[name].shape) != expected[name] for name in expected):
            raise ValueError(f"latents must contain exactly these shapes: {expected}")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.config.cond_channels:
            raise ValueError(f"encoder_hidden_states must have shape (batch, tokens, {self.config.cond_channels})")
        batch_size = encoder_hidden_states.shape[0]
        reference = latents["shape"]
        timestep = torch.as_tensor(timestep, device=reference.device).expand(batch_size)
        shortcut_step = torch.zeros_like(timestep) if shortcut_step is None else torch.as_tensor(shortcut_step)
        modulation = self.t_embedder(timestep) + self.d_embedder(shortcut_step.to(timestep).expand(batch_size))

        hidden = {
            name: mapping.input_layer(latents[name]) + mapping.pos_emb[None].to(reference.dtype)
            for name, mapping in self.latent_mapping.items()
        }
        inner_dtype = get_parameter_dtype(self.blocks)
        shape = hidden["shape"].to(inner_dtype)
        pose = torch.cat([hidden[name] for name in self.pose_names], dim=1).to(inner_dtype)
        modulation = modulation.to(inner_dtype)
        encoder_hidden_states = encoder_hidden_states.to(inner_dtype)
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                shape, pose = self._gradient_checkpointing_func(block, shape, pose, modulation, encoder_hidden_states)
            else:
                shape, pose = block(shape, pose, modulation, encoder_hidden_states)

        outputs = {"shape": shape, **dict(zip(self.pose_names, pose.split(1, dim=1)))}
        sample = {}
        for name, hidden_states in outputs.items():
            hidden_states = hidden_states.to(reference.dtype)
            sample[name] = self.latent_mapping[name].out_layer(F.layer_norm(hidden_states, hidden_states.shape[-1:]))
        if not return_dict:
            return (sample,)
        return Sam3DSparseStructureFlowOutput(sample=sample)


__all__ = ["Sam3DSparseStructureFlowModel", "Sam3DSparseStructureFlowOutput"]
