# Portions of this file reproduce the MV-Adapter decoupled row/column attention used by Step1X-3D:
# https://github.com/stepfun-ai/Step1X-3D (step1x3d_texture/models/attention_processor.py)
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
# Derived from https://github.com/huanngzh/MV-Adapter
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified to use dispatch_attention_fn and to keep the released weights in a separate model component.

from __future__ import annotations

import math

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from torch import nn

from ...execution.models import Object3DModel

STEP1X3D_NUM_VIEWS = 6


class Step1X3DMultiViewAttnProcessor(nn.Module):
    """Attention plus MV-Adapter's decoupled multi-view and reference branches.

    Built with ``hidden_size`` for a self-attention layer; built without it, it is plain attention that accepts
    the same keyword arguments. Batches hold groups of six square views (front, right, back, left, top, bottom).
    The multi-view branch attends along image rows across views 0-3 and along image columns across views 0, 2, 4
    and 5 (view 0 mirrored), averaging the two for views 0 and 2. The reference branch attends to this layer's
    input cached from the reference image.

    ``reference_cache`` (a dict) stores this layer's input under the processor and disables both branches.
    ``reference_hidden_states`` maps processors to cached inputs, one per batch item.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, hidden_size: int | None = None) -> None:
        super().__init__()
        if hidden_size is None:
            return
        self.to_q_mv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_mv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_mv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_mv = nn.ModuleList([nn.Linear(hidden_size, hidden_size), nn.Dropout(0.0)])
        self.to_q_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_ref = nn.ModuleList([nn.Linear(hidden_size, hidden_size), nn.Dropout(0.0)])

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=mask, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return hidden_states.flatten(2, 3).to(query.dtype)

    def _line_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, rows: bool) -> torch.Tensor:
        """Attention among the tokens sharing an image row (or column) across ``(batch, views, h, w, heads, dim)``."""

        order = (0, 2, 1, 3, 4, 5) if rows else (0, 3, 1, 2, 4, 5)
        batch, views = query.shape[:2]
        query, key, value = (x.permute(order).flatten(0, 1).flatten(1, 2) for x in (query, key, value))
        hidden_states = self._attend(query, key, value).unflatten(0, (batch, -1)).unflatten(2, (views, -1))
        return hidden_states.permute(0, 2, 1, 3, 4) if rows else hidden_states.permute(0, 2, 3, 1, 4)

    def _multiview(self, hidden_states: torch.Tensor, heads: int) -> torch.Tensor:
        length = hidden_states.shape[1]
        side = math.isqrt(length)
        query, key, value = (
            layer(hidden_states)
            .unflatten(-1, (heads, -1))
            .unflatten(1, (side, side))
            .unflatten(0, (-1, STEP1X3D_NUM_VIEWS))
            for layer in (self.to_q_mv, self.to_k_mv, self.to_v_mv)
        )
        rows = self._line_attention(query[:, :4], key[:, :4], value[:, :4], rows=True)
        columns = self._line_attention(
            *(torch.cat([x[:, :1].flip(3), x[:, [2, 4, 5]]], dim=1) for x in (query, key, value)), rows=False
        )
        columns = torch.cat([columns[:, :1].flip(3), columns[:, 1:]], dim=1)
        hidden_states = torch.stack(
            [
                (rows[:, 0] + columns[:, 0]) / 2,
                rows[:, 1],
                (rows[:, 2] + columns[:, 1]) / 2,
                rows[:, 3],
                columns[:, 2],
                columns[:, 3],
            ],
            dim=1,
        ).flatten(0, 1)
        return self.to_out_mv[1](self.to_out_mv[0](hidden_states.flatten(1, 2)))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        reference_cache: dict[nn.Module, torch.Tensor] | None = None,
        reference_hidden_states: dict[nn.Module, torch.Tensor] | None = None,
        mv_scale: float = 1.0,
        ref_scale: float = 1.0,
    ) -> torch.Tensor:
        heads = attn.heads
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            context = attn.norm_encoder_hidden_states(context)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, context.shape[1], hidden_states.shape[0])
            attention_mask = attention_mask.view(hidden_states.shape[0], heads, -1, attention_mask.shape[-1])
        query, key, value = (
            projection(states).unflatten(-1, (heads, -1))
            for projection, states in ((attn.to_q, hidden_states), (attn.to_k, context), (attn.to_v, context))
        )
        output = attn.to_out[1](attn.to_out[0](self._attend(query, key, value, attention_mask)))

        if reference_cache is not None:
            if hasattr(self, "to_q_ref"):
                reference_cache[self] = hidden_states
        elif hasattr(self, "to_q_mv"):
            output = output + mv_scale * self._multiview(hidden_states, heads)
            if reference_hidden_states is not None:
                reference = reference_hidden_states[self]
                query, key, value = (
                    projection(states).unflatten(-1, (heads, -1))
                    for projection, states in (
                        (self.to_q_ref, hidden_states),
                        (self.to_k_ref, reference),
                        (self.to_v_ref, reference),
                    )
                )
                output = output + ref_scale * self.to_out_ref[1](self.to_out_ref[0](self._attend(query, key, value)))

        if attn.residual_connection:
            output = output + hidden_states
        return output / attn.rescale_output_factor


class Step1X3DMultiViewAdapter(Object3DModel):
    """The released MV-Adapter attention weights, one processor per UNet self-attention layer.

    ``hidden_sizes`` maps the UNet processor names (``unet.attn_processors`` keys) to their widths. Each processor
    sits at its UNet path, so the parameter names (``...attn1.processor.to_q_mv.weight``) are those of the released
    file.
    """

    @register_to_config
    def __init__(self, hidden_sizes: dict[str, int]) -> None:
        super().__init__()
        for name, hidden_size in hidden_sizes.items():
            parent = self
            *path, leaf = name.split(".")
            for part in path:
                if part not in parent._modules:
                    parent.add_module(part, nn.Module())
                parent = parent._modules[part]
            parent.add_module(leaf, Step1X3DMultiViewAttnProcessor(hidden_size))

    @staticmethod
    def hidden_sizes_for(unet: nn.Module) -> dict[str, int]:
        """The ``hidden_sizes`` covering every self-attention layer of a ``UNet2DConditionModel``."""

        return {
            name: unet.get_submodule(name.removesuffix(".processor")).to_q.in_features
            for name in unet.attn_processors
            if name.endswith("attn1.processor")
        }

    def attn_processors_for(self, unet: nn.Module) -> dict[str, Step1X3DMultiViewAttnProcessor]:
        """Processors for ``unet.set_attn_processor``: this adapter's where it has weights, plain ones elsewhere."""

        return {
            name: self.get_submodule(name) if name in self.config.hidden_sizes else Step1X3DMultiViewAttnProcessor()
            for name in unet.attn_processors
        }


__all__ = ["STEP1X3D_NUM_VIEWS", "Step1X3DMultiViewAdapter", "Step1X3DMultiViewAttnProcessor"]
