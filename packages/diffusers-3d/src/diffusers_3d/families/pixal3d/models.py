# Portions of this file reproduce Tencent ARC Pixal3D model semantics:
# https://github.com/TencentARC/Pixal3D
# Revision: f7cf38429b0bd264f1995f0f8743a88b1c728b94
#
# MIT License. Copyright (c) 2026 Tencent.
# Modified to extend the package's TRELLIS.2 transformers instead of duplicating them.

from __future__ import annotations

from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from torch import nn

from ..trellis.sparse import TrellisSparseTensor
from ..trellis2.models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel


class Pixal3DProjectedCrossAttention(nn.Module):
    """Upstream ``ProjectAttention``: global cross-attention plus a per-block projection of pixel-aligned features."""

    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int) -> None:
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels)

    @property
    def to_out(self) -> nn.Linear:
        # The inherited scaled initialization addresses the output projection of ``block.cross_attn``.
        return self.cross_attn_block.to_out

    def forward(self, hidden_states: torch.Tensor, context: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        global_context, projected = context
        return self.proj_linear(projected) + self.cross_attn_block(hidden_states, global_context)


def _add_projection(model: nn.Module, proj_in_channels: int) -> None:
    for block in model.blocks:
        block.cross_attn = Pixal3DProjectedCrossAttention(block.cross_attn, model.model_channels, proj_in_channels)
    model._initialize_weights()
    model._convert_torso()


class Pixal3DSparseStructureFlowModel(Trellis2SparseStructureFlowModel):
    """TRELLIS.2 sparse-structure flow whose cross-attention also adds view-aligned DINOv3 grid features.

    As upstream, ``encoder_hidden_states`` is the pair ``(global_tokens, projected)``: the DINOv3 class and register
    tokens ``(batch, tokens, cond_channels)`` and one pixel-aligned feature per grid token
    ``(batch, resolution**3, proj_in_channels)`` in ``ij`` flattening order.
    """

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
        proj_in_channels: int = 1024,
    ) -> None:
        super().__init__(
            resolution,
            in_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            pe_mode,
            rope_freq,
            dtype,
            use_checkpoint,
            share_mod,
            initialization,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        _add_projection(self, proj_in_channels)

    def _block_context(self, context: Any, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states, projected = context
        expected = (batch_size, self.resolution**3, self.config.proj_in_channels)
        if tuple(projected.shape) != expected:
            raise ValueError(f"projected_hidden_states must have shape {expected}")
        return super()._block_context(encoder_hidden_states, batch_size, dtype), projected.to(dtype=dtype)


class Pixal3DSLatFlowModel(Trellis2SLatFlowModel):
    """TRELLIS.2 shape / texture SLAT flow whose cross-attention also adds per-voxel pixel-aligned features.

    ``encoder_hidden_states`` is the pair ``(global_tokens, projected)``, where ``projected`` is a
    ``TrellisSparseTensor`` coordinate-aligned with ``hidden_states`` carrying ``proj_in_channels`` features.
    """

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
        proj_in_channels: int = 2048,
    ) -> None:
        super().__init__(
            resolution,
            in_channels,
            model_channels,
            cond_channels,
            out_channels,
            num_blocks,
            num_heads,
            num_head_channels,
            mlp_ratio,
            pe_mode,
            rope_freq,
            dtype,
            use_checkpoint,
            share_mod,
            initialization,
            qk_rms_norm,
            qk_rms_norm_cross,
        )
        _add_projection(self, proj_in_channels)

    def _validate_context(self, context: Any, hidden_states: TrellisSparseTensor) -> None:
        encoder_hidden_states, projected = context
        super()._validate_context(encoder_hidden_states, hidden_states)
        if not isinstance(projected, TrellisSparseTensor) or not torch.equal(
            projected.coordinates, hidden_states.coordinates
        ):
            raise ValueError("projected_hidden_states must be coordinate-aligned with hidden_states")
        if projected.channels != self.config.proj_in_channels:
            raise ValueError(f"projected_hidden_states must have {self.config.proj_in_channels} channels")

    def _batch_context(
        self, context: Any, batch_index: int, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states, projected = context
        return (
            super()._batch_context(encoder_hidden_states, batch_index, positions, dtype),
            projected.features[positions].unsqueeze(0).to(dtype=dtype),
        )


__all__ = ["Pixal3DProjectedCrossAttention", "Pixal3DSLatFlowModel", "Pixal3DSparseStructureFlowModel"]
