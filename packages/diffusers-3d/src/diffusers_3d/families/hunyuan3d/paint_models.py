# Portions of this file reproduce model semantics from Tencent Hunyuan3D-2.1 Paint:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: reimplemented as attention processors and a BasicTransformerBlock subclass on stock Diffusers UNets.
#
# Hunyuan3DPaintRRDBNet reimplements the RRDBNet generator of Real-ESRGAN (BSD-3-Clause, Copyright (c) 2021 Xintao
# Wang) for its RealESRGAN_x4plus weights; see NOTICE.

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin, UNet2DConditionModel  # noqa: F401 - ModelMixin is required by external loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import ImageProjection, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from torch import nn

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, heads: int) -> torch.Tensor:
    query, key, value = (tensor.unflatten(-1, (heads, -1)).transpose(1, 2) for tensor in (query, key, value))
    return F.scaled_dot_product_attention(query, key, value).transpose(1, 2).flatten(2)


class Hunyuan3DPaintMaterialAttnProcessor(nn.Module):
    """Self-attention within each view, with the stock projections for albedo and separate ones for metallic-roughness.

    Batch rows are ordered ``(batch, material, view)``.
    """

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.to_q_mr = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_mr = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_mr = nn.Linear(dim, inner_dim, bias=False)
        self.to_out_mr = nn.Linear(inner_dim, dim)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        num_views: int = 1,
    ) -> torch.Tensor:
        albedo, metallic_roughness = (
            part.flatten(0, 1) for part in hidden_states.unflatten(0, (-1, 2, num_views)).unbind(1)
        )
        albedo = _attention(attn.to_q(albedo), attn.to_k(albedo), attn.to_v(albedo), attn.heads)
        metallic_roughness = _attention(
            self.to_q_mr(metallic_roughness),
            self.to_k_mr(metallic_roughness),
            self.to_v_mr(metallic_roughness),
            attn.heads,
        )
        outputs = (attn.to_out[1](attn.to_out[0](albedo)), self.to_out_mr(metallic_roughness))
        return torch.stack([output.unflatten(0, (-1, num_views)) for output in outputs], dim=1).flatten(0, 2)


class Hunyuan3DPaintReferenceAttnProcessor(nn.Module):
    """Cross-attention from the albedo tokens of all views to reference-UNet features, decoded per material.

    Values for both materials are concatenated before the heads are split, as in the released model, so each head
    reads a ``2 * head_dim`` slice of the joint values; its first half feeds albedo and its second half
    metallic-roughness. Returns ``(batch, 2, tokens, dim)``.
    """

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.to_v_mr = nn.Linear(dim, inner_dim, bias=False)
        self.to_out_mr = nn.Linear(inner_dim, dim)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = encoder_hidden_states.expand(hidden_states.shape[0], -1, -1)
        value = torch.cat([attn.to_v(context), self.to_v_mr(context)], dim=-1)
        output = _attention(attn.to_q(hidden_states), attn.to_k(context), value, attn.heads)
        albedo, metallic_roughness = output.unflatten(-1, (attn.heads, 2, -1)).unbind(-2)
        albedo = attn.to_out[1](attn.to_out[0](albedo.flatten(2)))
        return torch.stack([albedo, self.to_out_mr(metallic_roughness.flatten(2))], dim=1)


class Hunyuan3DPaintMultiviewAttnProcessor:
    """Self-attention across the tokens of all views, with 3D rotary embeddings of the surface position."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query, key, value = (
            projection(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            for projection in (attn.to_q, attn.to_k, attn.to_v)
        )
        query, key = apply_rotary_emb(query, rotary_emb), apply_rotary_emb(key, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](hidden_states))


def position_voxel_indices(position_maps: torch.Tensor, latent_size: int) -> dict[int, tuple[torch.Tensor, int]]:
    """Quantized surface positions of every view for the multiview attention at each UNet resolution.

    ``position_maps`` is ``(views, 3, H, W)`` in ``[0, 1]`` with a white background. Each of the grids ``latent_size
    / 2 ** level`` averages positions per cell in float16 and rounds them on a ``8 * latent_size / 2 ** level``
    voxel grid. The finest grid averages foreground pixels only and drops cells less than a sixteenth covered. The
    released pipeline zeroes the background of its float16 maps in place while doing so, so the coarser grids
    average whole cells with a zero background; this reproduces that. Returns ``{views * cells: ((views * cells, 3)
    indices, voxel resolution)}``.
    """

    views, channels, height, width = position_maps.shape
    position = position_maps.half()
    valid = (position != 1).all(dim=1, keepdim=True).expand_as(position)
    position = torch.where(valid, position, 0)
    indices = {}
    for level in range(4):
        grid, resolution = latent_size >> level, (8 * latent_size) >> level

        def cells(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(views, channels, grid, height // grid, grid, width // grid)
            return tensor.permute(0, 2, 4, 1, 3, 5).sum(dim=(-2, -1))

        count = cells(valid)
        mean = cells(position) / count.clamp(min=1)
        mean = torch.where(count < (height // grid) * (width // grid) // 16, 0, mean)
        voxels = (mean.clamp(0, 1) * (resolution - 1)).round().long().flatten(0, 2)
        indices[voxels.shape[0]] = (voxels, resolution)
        valid = torch.ones_like(valid)
    return indices


def rotary_embedding(voxels: torch.Tensor, resolution: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """3D rotary ``(cos, sin)`` of ``(tokens, 3)`` voxel indices: 3/8 of ``head_dim`` each for x and y, 2/8 for z."""

    grid = torch.arange(resolution, dtype=torch.float32, device=voxels.device)
    xy = get_1d_rotary_pos_embed(head_dim // 8 * 3, grid, use_real=True)
    z = get_1d_rotary_pos_embed(head_dim // 8 * 2, grid, use_real=True)
    tables = (xy, xy, z)
    return tuple(
        torch.cat([table[part][voxels[:, axis]] for axis, table in enumerate(tables)], dim=-1) for part in (0, 1)
    )


class Hunyuan3DPaintTransformerBlock(BasicTransformerBlock):
    """``BasicTransformerBlock`` with Paint's material, reference, multiview, and DINO attention.

    After the per-material self-attention (``attn1``), the block adds reference attention scaled per batch row and
    multiview attention, both reading the ``norm1`` output; after the text cross-attention (``attn2``) it adds a
    cross-attention to the projected DINOv2 tokens, both reading the ``norm2`` output. Only the layer-norm
    configuration of Stable Diffusion 2 is supported. Per-call inputs arrive through ``cross_attention_kwargs``.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int) -> None:
        super().__init__(dim, num_attention_heads, attention_head_dim, cross_attention_dim=cross_attention_dim)
        inner_dim = num_attention_heads * attention_head_dim
        heads = {"heads": num_attention_heads, "dim_head": attention_head_dim}
        self.attn1.set_processor(Hunyuan3DPaintMaterialAttnProcessor(dim, inner_dim))
        self.attn_refview = Attention(dim, **heads, processor=Hunyuan3DPaintReferenceAttnProcessor(dim, inner_dim))
        self.attn_multiview = Attention(dim, **heads, processor=Hunyuan3DPaintMultiviewAttnProcessor())
        self.attn_dino = Attention(dim, cross_attention_dim=cross_attention_dim, **heads)
        self.reference_key = ""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        class_labels: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_views = cross_attention_kwargs["num_views"]
        norm_hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(norm_hidden_states, num_views=num_views) + hidden_states

        tokens = norm_hidden_states.unflatten(0, (-1, 2, num_views)).flatten(
            2, 3
        )  # (batch, material, view tokens, dim)
        features = cross_attention_kwargs["reference_features"][self.reference_key]
        reference = self.attn_refview(tokens[:, 0], encoder_hidden_states=features)
        reference = reference.unflatten(2, (num_views, -1)).flatten(0, 2)
        hidden_states = cross_attention_kwargs["reference_scale"] * reference + hidden_states
        if num_views > 1:
            voxels, resolution = cross_attention_kwargs["voxel_indices"][tokens.shape[2]]
            rotary_emb = rotary_embedding(voxels, resolution, self.attention_head_dim)
            multiview = self.attn_multiview(tokens.flatten(0, 1), rotary_emb=rotary_emb)
            hidden_states = multiview.unflatten(1, (num_views, -1)).flatten(0, 1) + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states) + hidden_states
        image_embeds = cross_attention_kwargs["image_embeds"]
        hidden_states = self.attn_dino(norm_hidden_states, encoder_hidden_states=image_embeds) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class Hunyuan3DPaintUNet(Object3DModel):
    """Hunyuan3D-Paint 2.1's multiview PBR UNet: two stock Stable Diffusion 2 UNets and Paint's added attention.

    ``unet`` denoises the albedo and metallic-roughness latents of every view, with the normal and position latents
    concatenated to its input and its transformer blocks replaced by :class:`Hunyuan3DPaintTransformerBlock`.
    ``reference_unet`` runs once on the reference-image latents at timestep 0 (:meth:`encode_reference`); its
    self-attention inputs are the keys and values of the reference attention. Text conditioning is replaced by
    learned token sequences per material and for the reference, and DINOv2 tokens are projected by a stock
    ``ImageProjection``.
    """

    family_id = "hunyuan3d-2.1"
    component_role = "unet"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        latent_channels: int = 4,
        down_block_types: tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: tuple[str, ...] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        attention_head_dim: int | tuple[int, ...] = (5, 10, 20, 20),
        cross_attention_dim: int = 1024,
        norm_num_groups: int = 32,
        num_text_tokens: int = 77,
        image_embed_dim: int = 1536,
        num_image_text_embeds: int = 4,
    ) -> None:
        super().__init__()
        unet_config = {
            "sample_size": sample_size,
            "out_channels": latent_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "layers_per_block": layers_per_block,
            "attention_head_dim": attention_head_dim,
            "cross_attention_dim": cross_attention_dim,
            "norm_num_groups": norm_num_groups,
            "use_linear_projection": True,
        }
        self.unet = UNet2DConditionModel(in_channels=3 * latent_channels, **unet_config)
        self.reference_unet = UNet2DConditionModel(in_channels=latent_channels, **unet_config)
        blocks = [
            (name, module) for name, module in self.unet.named_modules() if isinstance(module, BasicTransformerBlock)
        ]
        for name, block in blocks:
            parent, _, index = name.rpartition(".")
            paint_block = Hunyuan3DPaintTransformerBlock(
                block.dim, block.num_attention_heads, block.attention_head_dim, block.cross_attention_dim
            )
            paint_block.reference_key = name
            self.unet.get_submodule(parent)[int(index)] = paint_block
        self.material_embeds = nn.Parameter(torch.zeros(2, num_text_tokens, cross_attention_dim))
        self.reference_embeds = nn.Parameter(torch.zeros(num_text_tokens, cross_attention_dim))
        self.image_projection = ImageProjection(image_embed_dim, cross_attention_dim, num_image_text_embeds)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "sample_size": 8,
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "block_out_channels": (16, 32),
            "layers_per_block": 1,
            "attention_head_dim": (1, 2),
            "cross_attention_dim": 12,
            "norm_num_groups": 8,
            "num_text_tokens": 3,
            "image_embed_dim": 10,
            "num_image_text_embeds": 2,
        }

    def encode_reference(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run ``reference_unet`` on ``(batch, C, h, w)`` reference latents and return each block's normed input."""

        features = {}

        def capture(name: str, module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            features[name] = args[0]

        hooks = [
            module.attn1.register_forward_pre_hook(partial(capture, name))
            for name, module in self.reference_unet.named_modules()
            if isinstance(module, BasicTransformerBlock)
        ]
        try:
            context = self.reference_embeds.expand(latents.shape[0], -1, -1)
            self.reference_unet(latents, 0, encoder_hidden_states=context)
        finally:
            for hook in hooks:
                hook.remove()
        return features

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
        condition_latents: torch.Tensor,
        position_maps: torch.Tensor,
        image_embeds: torch.Tensor,
        reference_features: dict[str, torch.Tensor],
        reference_scale: torch.Tensor,
        return_dict: bool = True,
    ) -> UNet2DConditionOutput | tuple[torch.Tensor]:
        """Predict ``(batch, 2, views, C, h, w)`` outputs for noisy albedo and metallic-roughness latents.

        Every batch row denoises the same object: ``condition_latents`` ``(views, 2 C, h, w)`` (normal then position
        latents) and ``position_maps`` ``(views, 3, H, W)`` are shared. ``image_embeds`` ``(batch, tokens,
        image_embed_dim)`` are DINOv2 tokens, ``reference_features`` come from :meth:`encode_reference`, and
        ``reference_scale`` ``(batch,)`` weights the reference attention of each row.
        """

        batch, materials, views = sample.shape[:3]
        conditions = condition_latents.expand(batch, materials, *condition_latents.shape)
        hidden_states = torch.cat([sample, conditions], dim=3).flatten(0, 2)
        rows = materials * views
        text = self.material_embeds[None, :, None].expand(batch, -1, views, -1, -1).flatten(0, 2)
        image = self.image_projection(image_embeds.flatten(0, 1)).unflatten(0, (batch, -1)).flatten(1, 2)
        output = self.unet(
            hidden_states,
            timestep,
            encoder_hidden_states=text,
            cross_attention_kwargs={
                "num_views": views,
                "reference_features": reference_features,
                "reference_scale": reference_scale.repeat_interleave(rows)[:, None, None],
                "voxel_indices": position_voxel_indices(position_maps, sample.shape[-1]),
                "image_embeds": image.repeat_interleave(rows, dim=0),
            },
        ).sample.unflatten(0, (batch, materials, views))
        if not return_dict:
            return (output,)
        return UNet2DConditionOutput(sample=output)


class _ResidualDenseBlock(nn.Module):
    def __init__(self, features: int, growth: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = [hidden_states]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            features.append(F.leaky_relu(conv(torch.cat(features, dim=1)), 0.2))
        return self.conv5(torch.cat(features, dim=1)) * 0.2 + hidden_states


class _ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, features: int, growth: int) -> None:
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(features, growth)
        self.rdb2 = _ResidualDenseBlock(features, growth)
        self.rdb3 = _ResidualDenseBlock(features, growth)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(hidden_states))) * 0.2 + hidden_states


class Hunyuan3DPaintRRDBNet(Object3DModel):
    """Real-ESRGAN's RRDBNet generator, the 4x super-resolution Hunyuan3D-Paint 2.1 applies to every generated view.

    ``forward`` maps ``(batch, 3, H, W)`` images in ``[0, 1]`` to ``(batch, 3, 4 H, 4 W)``.
    """

    family_id = "hunyuan3d-2.1"
    component_role = "upscaler"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(self, channels: int = 3, num_features: int = 64, num_blocks: int = 23, growth_channels: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *(_ResidualInResidualDenseBlock(num_features, growth_channels) for _ in range(num_blocks))
        )
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_last = nn.Conv2d(num_features, channels, 3, padding=1)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {"num_features": 8, "num_blocks": 1, "growth_channels": 4}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_first(images)
        features = features + self.conv_body(self.body(features))
        for conv in (self.conv_up1, self.conv_up2):
            features = F.leaky_relu(conv(F.interpolate(features, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(features), 0.2))


__all__ = [
    "Hunyuan3DPaintMaterialAttnProcessor",
    "Hunyuan3DPaintMultiviewAttnProcessor",
    "Hunyuan3DPaintRRDBNet",
    "Hunyuan3DPaintReferenceAttnProcessor",
    "Hunyuan3DPaintTransformerBlock",
    "Hunyuan3DPaintUNet",
    "position_voxel_indices",
    "rotary_embedding",
]
