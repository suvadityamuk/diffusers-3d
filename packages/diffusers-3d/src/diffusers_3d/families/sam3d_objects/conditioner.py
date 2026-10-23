# Portions of this file reproduce the condition embedders of SAM 3D Objects:
# https://github.com/facebookresearch/sam-3d-objects
# Revision: f91db411c50efee93d8db7aeb323885650f6f722
#
# SAM License. Copyright (c) Meta Platforms, Inc. and affiliates.
# This file has been modified to use the Diffusers transformer block and SwiGLU feed-forward.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class Sam3DConditionerOutput(BaseOutput):
    """Fused condition tokens: image, mask, then (when configured) pointmap tokens, each cropped then full."""

    embeddings: torch.Tensor


class Sam3DPointmapEmbedder(nn.Module):
    """One token per ``patch_size`` window of a camera-space pointmap (upstream ``PointPatchEmbed``).

    Points are projected per pixel (non-finite points become a learned token), every window runs one ViT block with
    a window class token, and the class tokens plus an interpolated window position embedding are returned.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        input_size: int = 256,
        patch_size: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        windows = input_size // patch_size
        self.num_tokens = windows**2
        self.point_proj = nn.Linear(3, embed_dim)
        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, windows, windows))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, 1 + patch_size**2, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # timm's ``Block``: pre-norm self-attention and exact-GELU MLP, both with biases.
        self.block = BasicTransformerBlock(
            embed_dim,
            num_heads,
            embed_dim // num_heads,
            activation_fn="gelu",
            attention_bias=True,
            norm_eps=1e-6,
            ff_inner_dim=int(embed_dim * mlp_ratio),
        )

    def forward(self, pointmaps: torch.Tensor) -> torch.Tensor:
        points = F.interpolate(pointmaps, size=self.input_size, mode="nearest").permute(0, 2, 3, 1)
        valid = points.isfinite().all(dim=-1, keepdim=True)
        hidden_states = self.point_proj(torch.where(valid, points, 0.0))
        hidden_states = torch.where(valid, hidden_states, self.invalid_xyz_token)

        batch_size, windows, patch = hidden_states.shape[0], self.input_size // self.patch_size, self.patch_size
        hidden_states = hidden_states.reshape(batch_size, windows, patch, windows, patch, self.embed_dim)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(-1, patch * patch, self.embed_dim)
        hidden_states = torch.cat([self.cls_token.expand(hidden_states.shape[0], -1, -1), hidden_states], dim=1)
        hidden_states = self.block(hidden_states + self.pos_embed_window)
        position = F.interpolate(self.pos_embed, size=(windows, windows), mode="bilinear", align_corners=False)
        position = position.permute(0, 2, 3, 1).reshape(1, self.num_tokens, self.embed_dim)
        return hidden_states[:, 0].reshape(batch_size, self.num_tokens, self.embed_dim) + position


def _projection(input_dim: int, output_dim: int, multiplier: float) -> nn.Sequential:
    # Upstream's Llama-3 SwiGLU sizing: two thirds of the nominal width, rounded up to a multiple of 256.
    hidden_dim = 256 * math.ceil(int(2 * int(multiplier * output_dim) / 3) / 256)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        FeedForward(input_dim, output_dim, inner_dim=hidden_dim, activation_fn="swiglu", bias=False),
    )


class Sam3DConditionEmbedder(Object3DModel):
    """SAM 3D Objects condition fuser: projects DINOv2 tokens of the image and its mask, for both the object crop and
    the full frame, into one context.

    The sparse-structure embedder adds a pointmap encoder and takes DINOv2's normed class and patch tokens; the
    structured-latent embedder takes its pre-norm tokens under an unparameterized layer norm (``prenorm_features``).
    """

    family_id = "sam3d_objects"
    component_role = "conditioner"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.GAUSSIAN_SPLAT, Object3DKind.MESH)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        image_embed_dim: int = 1024,
        *,
        prenorm_features: bool = False,
        projection_hidden_multiplier: float = 4.0,
        pointmap_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.pointmap_encoder = None if pointmap_config is None else Sam3DPointmapEmbedder(**pointmap_config)
        widths = [image_embed_dim, image_embed_dim]
        if self.pointmap_encoder is not None:
            widths.append(self.pointmap_encoder.embed_dim)
        self.embed_dim = max(widths)
        self.projections = nn.ModuleList(
            _projection(width, self.embed_dim, projection_hidden_multiplier) for width in widths
        )
        # Learned offsets for the two view groups (0: object crop, 1: full frame); the third row is never used.
        self.idx_emb = nn.Parameter(torch.randn(3, self.embed_dim) / math.sqrt(self.embed_dim))

    @staticmethod
    def production_pointmap_config() -> dict[str, Any]:
        return {"embed_dim": 512, "input_size": 256, "patch_size": 8, "num_heads": 16, "mlp_ratio": 2.0}

    @classmethod
    def production_config(cls, *, prenorm_features: bool = False, pointmap: bool = True) -> dict[str, Any]:
        return {
            "image_embed_dim": 1024,
            "prenorm_features": prenorm_features,
            "projection_hidden_multiplier": 4.0,
            "pointmap_config": cls.production_pointmap_config() if pointmap else None,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "image_embed_dim": 12,
            "prenorm_features": False,
            "projection_hidden_multiplier": 1.0,
            "pointmap_config": {"embed_dim": 8, "input_size": 8, "patch_size": 4, "num_heads": 2, "mlp_ratio": 2.0},
        }

    def forward(
        self,
        image_tokens: torch.Tensor,
        mask_tokens: torch.Tensor,
        pointmaps: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
    ) -> Sam3DConditionerOutput | tuple[torch.Tensor]:
        """Fuse ``(batch, 2, tokens, image_embed_dim)`` DINOv2 tokens of views stacked as (object crop, full frame).

        ``pointmaps`` are ``(batch, 2, 3, height, width)`` normalized camera-space points. Without ``pointmaps`` the
        pointmap tokens are zero, exactly like the modality dropout the model was trained with; embedders built
        without a pointmap encoder ignore them.
        """

        if image_tokens.ndim != 4 or image_tokens.shape[1] != 2 or mask_tokens.shape[:2] != image_tokens.shape[:2]:
            raise ValueError("image_tokens and mask_tokens must have shape (batch, 2, tokens, image_embed_dim)")
        dtype = self.idx_emb.dtype
        offsets = self.idx_emb[:2, None]
        tokens = [
            self.projections[0](image_tokens.to(dtype)) + offsets,
            self.projections[1](mask_tokens.to(dtype)) + offsets,
        ]
        if self.pointmap_encoder is not None:
            if pointmaps is None:
                shape = (image_tokens.shape[0], 2, self.pointmap_encoder.num_tokens, self.embed_dim)
                tokens.append(image_tokens.new_zeros(shape, dtype=dtype))
            else:
                embedded = self.pointmap_encoder(pointmaps.flatten(0, 1).to(dtype)).unflatten(0, (-1, 2))
                tokens.append(self.projections[2](embedded) + offsets)
        embeddings = torch.cat([value.flatten(1, 2) for value in tokens], dim=1)
        if not return_dict:
            return (embeddings,)
        return Sam3DConditionerOutput(embeddings=embeddings)


__all__ = ["Sam3DConditionEmbedder", "Sam3DConditionerOutput", "Sam3DPointmapEmbedder"]
