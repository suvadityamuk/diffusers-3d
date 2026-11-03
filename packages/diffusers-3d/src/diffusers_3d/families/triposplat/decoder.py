# Portions of this file reproduce the TripoSplat octree point sampler and elastic Gaussian decoder:
# https://github.com/VAST-AI-Research/TripoSplat
# Revision: d8db9e018b413dd9c4a9fe22463781bf98e8e68d
#
# MIT License. Copyright (c) 2026 VAST.
# Rewritten on Diffusers building blocks; the octree sampler takes an explicit generator.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel
from ...objects import CoordinateSystem, GaussianSplatAsset
from .models import TripoSplatAttention

# Children of an octree cell, x fastest, in the order upstream lays out the 8 logits.
_CHILD_OFFSETS = tuple((i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1))


def _fourier_embedding(points: torch.Tensor, dim: int) -> torch.Tensor:
    """Upstream ``PcdAbsolutePositionEmbedderV2``: ``sin``/``cos`` of ``x * 2**linspace(0, 10) * pi`` per axis."""

    width = dim // 3 // 2
    angles = points.float().unsqueeze(-1) * torch.pow(2.0, torch.linspace(0.0, 10.0, width, device=points.device))
    angles = angles * torch.pi
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


def _hammersley(count: int) -> torch.Tensor:
    def radical_inverse(base: int, index: int) -> float:
        value, inverse = 0.0, 1.0 / base
        scale = inverse
        while index:
            value += index % base * scale
            index //= base
            scale *= inverse
        return value

    return torch.tensor([[i / count, radical_inverse(2, i), radical_inverse(3, i)] for i in range(count)])


def _systematic_counts(probs: torch.Tensor, counts: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Split ``counts[b]`` points among the ``probs[b]`` categories by systematic resampling, as upstream."""

    batch_shape, categories = counts.shape, probs.shape[-1]
    probs = probs.reshape(-1, categories).float().clamp_min(0)
    counts = counts.reshape(-1).long()
    row_sums = probs.sum(1, keepdim=True)
    probs = torch.where(row_sums.eq(0), 1.0 / categories, probs / row_sums.clamp_min(1))
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    out = torch.zeros_like(probs, dtype=torch.long)
    unique, inverse = counts.unique(sorted=False, return_inverse=True)
    for index, count in enumerate(unique.tolist()):
        if count == 0:
            continue
        rows = (inverse == index).nonzero().squeeze(1)
        start = torch.rand(rows.numel(), 1, device=probs.device, generator=generator) / float(count)
        grid = torch.arange(count, device=probs.device, dtype=torch.float32)[None] / float(count)
        samples = (start + grid).clamp(max=1.0 - 1e-12)
        bins = torch.searchsorted(cdf.index_select(0, rows), samples).clamp_max(categories - 1)
        hits = torch.zeros(rows.numel(), categories, device=probs.device).scatter_add_(
            1, bins, torch.ones_like(bins, dtype=torch.float32)
        )
        out.index_copy_(0, rows, hits.long())
    return out.view(*batch_shape, categories)


class TripoSplatOctreeBlock(nn.Module):
    """adaLN-modulated cross-attention-only block of the octree occupancy decoder."""

    def __init__(self, dim: int, context_dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = TripoSplatAttention(dim, heads, cross_attention_dim=context_dim)
        self.norm2 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(self, hidden_states: torch.Tensor, modulation: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.unsqueeze(1).chunk(6, dim=-1)
        attention_input = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + self.attn(attention_input, context) * gate_msa
        return hidden_states + self.ff(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp) * gate_mlp


class TripoSplatGaussianBlock(nn.Module):
    """Pre-norm self-attention, cross-attention, and MLP block of the Gaussian decoder."""

    def __init__(self, dim: int, context_dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = TripoSplatAttention(dim, heads)
        self.norm2 = FP32LayerNorm(dim, eps=1e-6)
        self.attn2 = TripoSplatAttention(dim, heads, cross_attention_dim=context_dim)
        self.norm3 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = FeedForward(dim, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), context)
        return hidden_states + self.ff(self.norm3(hidden_states))


@dataclass
class TripoSplatGaussianDecoderOutput(BaseOutput):
    """One :class:`~diffusers_3d.GaussianSplatAsset` per latent set."""

    assets: tuple[GaussianSplatAsset, ...]


class TripoSplatGaussianDecoder(Object3DModel):
    """TripoSplat VAE decoder: latent tokens to a variable number of 3D Gaussians.

    An octree occupancy transformer splits ``num_gaussians / gaussians_per_token`` points down ``octree_depth``
    levels by systematic resampling of its child probabilities, and jitters each inside its leaf cell. A
    self/cross-attention transformer then decodes ``gaussians_per_token`` Gaussians around every point. Gaussians are
    returned in TripoSplat's (TRELLIS-style) right-handed Z-up frame, inside ``[-0.5, 0.5]^3``.
    """

    _no_split_modules = ["TripoSplatOctreeBlock", "TripoSplatGaussianBlock"]

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 16,
        width: int = 1024,
        num_attention_heads: int = 16,
        num_octree_layers: int = 4,
        num_gaussian_layers: int = 16,
        mlp_ratio: float = 4.0,
        octree_depth: int = 8,
        gaussians_per_token: int = 32,
        offset_scale: float = 0.05,
        offset_extent: float = 1.5,
        scaling_bias: float = 0.004,
        opacity_bias: float = 0.1,
        rotation_scale: float = 0.1,
        filter_kernel_size: float = 0.0009,
    ) -> None:
        super().__init__()
        self.octree_proj_in = nn.Linear(3, width)
        self.octree_input = nn.Linear(width, width)
        self.octree_level_embed = TimestepEmbedding(256, width)
        self.octree_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.octree_blocks = nn.ModuleList(
            [
                TripoSplatOctreeBlock(width, latent_channels, num_attention_heads, mlp_ratio)
                for _ in range(num_octree_layers)
            ]
        )
        self.octree_proj_out = nn.Linear(width, 8)
        self.gaussian_proj_in = nn.Linear(3, width)
        self.gaussian_input = nn.Linear(width, width)
        self.gaussian_blocks = nn.ModuleList(
            [
                TripoSplatGaussianBlock(width, latent_channels, num_attention_heads, mlp_ratio)
                for _ in range(num_gaussian_layers)
            ]
        )
        # Per Gaussian: offset 3, SH DC 3, scale 3, rotation 4, opacity 1, offset scale 1.
        self.gaussian_proj_out = nn.Linear(width, 15 * gaussians_per_token)
        perturbation = torch.atanh((_hammersley(gaussians_per_token).float() * 2 - 1) / offset_extent)
        self.register_buffer("offset_perturbation", perturbation)
        self.register_buffer("base_offset_scale", torch.log(torch.exp(torch.tensor(offset_scale)) - 1.0))

    def _embed(self, points: torch.Tensor, proj_in: nn.Linear, input_layer: nn.Linear) -> torch.Tensor:
        dtype = proj_in.weight.dtype
        return input_layer(proj_in(points.to(dtype)) + _fourier_embedding(points, self.config.width).to(dtype))

    def octree_logits(self, parents: torch.Tensor, resolution: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Child-occupancy logits ``(batch, parents, 8)`` for parent-cell centres in ``[0, 1]^3`` at ``resolution``."""

        dtype, block_dtype = self.octree_input.weight.dtype, self.octree_blocks[0].attn.to_out.weight.dtype
        hidden_states = self._embed(parents, self.octree_proj_in, self.octree_input).to(block_dtype)
        level = get_timestep_embedding(
            resolution, 256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=2 * torch.pi, max_period=1024
        )
        modulation = self.octree_modulation(self.octree_level_embed(level.to(dtype))).to(block_dtype)
        for block in self.octree_blocks:
            hidden_states = block(hidden_states, modulation, latents.to(block_dtype))
        return self.octree_proj_out(F.layer_norm(hidden_states.float(), hidden_states.shape[-1:]).to(dtype))

    @torch.no_grad()
    def sample_points(
        self, latents: torch.Tensor, num_points: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draw ``num_points`` points ``(batch, num_points, 3)`` in ``[0, 1]^3`` by descending the octree."""

        batch, device = latents.shape[0], latents.device
        offsets = torch.tensor(_CHILD_OFFSETS, dtype=torch.long, device=device)
        cells = torch.zeros(batch, 1, 3, dtype=torch.long, device=device)
        counts = torch.full((batch, 1), num_points, dtype=torch.long, device=device)
        batch_index = torch.arange(batch, device=device)[:, None]
        for level in range(1, self.config.octree_depth + 1):
            parents = (cells.float() + 0.5) / (1 << (level - 1))
            resolution = torch.full((batch,), 1 << level, dtype=torch.long, device=device)
            probs = torch.softmax(self.octree_logits(parents, resolution, latents), dim=-1)
            children_counts = _systematic_counts(probs, counts, generator).flatten(1, 2)
            children = (cells[:, :, None] * 2 + offsets[None, None]).flatten(1, 2)
            occupied = children_counts > 0
            slots = (occupied.cumsum(dim=1) - 1)[occupied]
            rows = batch_index.expand_as(occupied)[occupied]
            width = int(occupied.sum(dim=1).max())
            cells = children.new_zeros(batch, width, 3)
            cells[rows, slots] = children[occupied]
            counts = children_counts.new_zeros(batch, width)
            counts[rows, slots] = children_counts[occupied]
        cells = torch.repeat_interleave(cells.flatten(0, 1), counts.flatten(), dim=0).reshape(batch, num_points, 3)
        jitter = torch.rand(cells.shape, generator=generator, device=device, dtype=torch.float32)
        return (cells.float() + jitter) / (1 << self.config.octree_depth)

    def gaussian_features(self, points: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Per-point Gaussian parameters ``(batch, points, 15 * gaussians_per_token)`` before activation."""

        dtype, block_dtype = self.gaussian_input.weight.dtype, self.gaussian_blocks[0].attn1.to_out.weight.dtype
        hidden_states = self._embed(points, self.gaussian_proj_in, self.gaussian_input).to(block_dtype)
        for block in self.gaussian_blocks:
            hidden_states = block(hidden_states, latents.to(block_dtype))
        return self.gaussian_proj_out(F.layer_norm(hidden_states.float(), hidden_states.shape[-1:]).to(dtype))

    def _assets(self, points: torch.Tensor, features: torch.Tensor) -> tuple[GaussianSplatAsset, ...]:
        config = self.config
        count = config.gaussians_per_token
        # Half-precision positions and quaternions are too coarse for the asset (unit-norm check).
        features = features.float() if features.dtype.itemsize < 4 else features
        # Channels are grouped by attribute, each holding all ``count`` Gaussians.
        xyz, features_dc, scaling, rotation, opacity, offset_scale = (
            value.unflatten(-1, (count, -1))
            for value in features.split([width * count for width in (3, 3, 3, 4, 1, 1)], dim=-1)
        )
        offset_scale = F.softplus(offset_scale + self.base_offset_scale.to(offset_scale))
        offsets = torch.tanh(xyz + self.offset_perturbation.to(xyz)) * 0.5 * config.offset_extent * offset_scale
        means = offsets + points[:, :, None].to(offsets) - 0.5
        # Upstream builds its activation biases from float32 tensors.
        scaling_bias = torch.tensor(config.scaling_bias)
        scaling_bias = (scaling_bias + torch.log(-torch.expm1(-scaling_bias))).to(scaling)
        scales = torch.sqrt(F.softplus(scaling + scaling_bias).square() + config.filter_kernel_size**2)
        opacity_bias = torch.tensor(config.opacity_bias)
        opacity_logits = opacity + torch.log(opacity_bias / (1 - opacity_bias)).to(opacity)
        quaternions = F.normalize(rotation * config.rotation_scale + rotation.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=-1)
        return tuple(
            GaussianSplatAsset(
                means=means[index].flatten(0, 1),
                log_scales=scales[index].log().flatten(0, 1),
                quaternions_wxyz=quaternions[index].flatten(0, 1),
                opacity_logits=opacity_logits[index].flatten(0, 1),
                sh_coefficients=features_dc[index].flatten(0, 1).unsqueeze(1),
                active_sh_degree=0,
                coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                metadata={"family": "triposplat", "representation": "gaussian", "gaussians_per_token": count},
            )
            for index in range(points.shape[0])
        )

    def forward(
        self,
        latents: torch.Tensor,
        num_gaussians: int = 262144,
        *,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> TripoSplatGaussianDecoderOutput | tuple[tuple[GaussianSplatAsset, ...]]:
        """Decode ``latents`` ``(batch, tokens, latent_channels)`` into ``num_gaussians`` Gaussians each.

        ``num_gaussians`` must be a positive multiple of ``gaussians_per_token``. ``generator`` drives the octree
        resampling and the in-cell jitter.
        """

        count = self.config.gaussians_per_token
        if isinstance(num_gaussians, bool) or not isinstance(num_gaussians, int) or num_gaussians < count:
            raise ValueError(f"num_gaussians must be an integer of at least {count}")
        if num_gaussians % count:
            raise ValueError(f"num_gaussians must be a multiple of {count}")
        points = self.sample_points(latents, num_gaussians // count, generator)
        assets = self._assets(points, self.gaussian_features(points, latents))
        if not return_dict:
            return (assets,)
        return TripoSplatGaussianDecoderOutput(assets=assets)


__all__ = [
    "TripoSplatGaussianBlock",
    "TripoSplatGaussianDecoder",
    "TripoSplatGaussianDecoderOutput",
    "TripoSplatOctreeBlock",
]
