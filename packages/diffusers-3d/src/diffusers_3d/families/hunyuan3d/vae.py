# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for native Diffusers/PyTorch integration.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import ScikitImageBackend
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import MeshAsset, Object3DKind


@dataclass
class Hunyuan3DShapeVAEOutput(BaseOutput):
    """Decoded Hunyuan shape tokens."""

    sample: torch.Tensor


@dataclass
class Hunyuan3DShapeFieldOutput(BaseOutput):
    """Dense scalar field evaluated on a regular XYZ grid."""

    field: torch.Tensor
    bounds: tuple[float, float, float, float, float, float]


class Hunyuan3DFourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 8,
        *,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = False,
    ) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies = frequencies * torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = input_dim * (2 * num_freqs + int(include_input or num_freqs == 0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return hidden_states
        frequency_states = (hidden_states[..., None].contiguous() * self.frequencies).reshape(
            *hidden_states.shape[:-1], -1
        )
        encoded = (frequency_states.sin(), frequency_states.cos())
        return torch.cat((hidden_states, *encoded) if self.include_input else encoded, dim=-1)


class Hunyuan3DDropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return hidden_states
        keep_probability = 1.0 - self.probability
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        mask = hidden_states.new_ones(shape).bernoulli_(keep_probability)
        return hidden_states * mask.div_(keep_probability)


class Hunyuan3DVAEFeedForward(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        expand_ratio: int = 4,
        output_width: int | None = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width or width)
        self.gelu = nn.GELU()
        self.drop_path = Hunyuan3DDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(hidden_states))))


class Hunyuan3DVAESelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: Hunyuan3DVAESelfAttention, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_key_value = attn.c_qkv(hidden_states).reshape(batch_size, sequence_length, attn.heads, -1)
        query, key, value = query_key_value.chunk(3, dim=-1)
        query = attn.attention.q_norm(query)
        key = attn.attention.k_norm(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, sequence_length, attn.width)
        return attn.drop_path(attn.c_proj(hidden_states))


class Hunyuan3DVAESelfAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Hunyuan3DVAESelfAttnProcessor
    _available_processors = [Hunyuan3DVAESelfAttnProcessor]

    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        head_dim = width // heads
        self.attention = nn.Module()
        self.attention.q_norm = nn.LayerNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attention.k_norm = nn.LayerNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.drop_path = Hunyuan3DDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.set_processor(Hunyuan3DVAESelfAttnProcessor())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class Hunyuan3DVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.attn = Hunyuan3DVAESelfAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = Hunyuan3DVAEFeedForward(width, drop_path_rate=drop_path_rate)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class Hunyuan3DVAELatentTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                Hunyuan3DVAEAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states)
        return hidden_states


class Hunyuan3DVAECrossAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Hunyuan3DVAECrossAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        context_length = encoder_hidden_states.shape[1]
        query = attn.c_q(hidden_states).reshape(batch_size, query_length, attn.heads, -1)
        key_value = attn.c_kv(encoder_hidden_states).reshape(batch_size, context_length, attn.heads, -1)
        key, value = key_value.chunk(2, dim=-1)
        query = attn.attention.q_norm(query)
        key = attn.attention.k_norm(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, query_length, attn.width)
        return attn.c_proj(hidden_states)


class Hunyuan3DVAECrossAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = Hunyuan3DVAECrossAttnProcessor
    _available_processors = [Hunyuan3DVAECrossAttnProcessor]

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        head_dim = width // heads
        self.attention = nn.Module()
        self.attention.q_norm = nn.LayerNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attention.k_norm = nn.LayerNorm(head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.set_processor(Hunyuan3DVAECrossAttnProcessor())

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states)


class Hunyuan3DVAECrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        mlp_expand_ratio: int,
        qkv_bias: bool,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.attn = Hunyuan3DVAECrossAttention(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = Hunyuan3DVAEFeedForward(width, expand_ratio=mlp_expand_ratio)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states),
            self.ln_2(encoder_hidden_states),
        )
        return hidden_states + self.mlp(self.ln_3(hidden_states))


class Hunyuan3DGeometricDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        fourier_embedder: Hunyuan3DFourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int,
        downsample_ratio: int,
        enable_ln_post: bool,
        qkv_bias: bool,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        if downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        self.cross_attn_decoder = Hunyuan3DVAECrossAttentionBlock(
            width=width,
            heads=heads,
            mlp_expand_ratio=mlp_expand_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm if enable_ln_post else False,
        )
        if enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, 1)
        self.num_latents = num_latents

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        hidden_states = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            hidden_states = self.ln_post(hidden_states)
        return self.output_proj(hidden_states)


class Hunyuan3DShapeVAE(Object3DModel):
    """Production-checkpoint-compatible Hunyuan3D-2.1 shape decoder.

    The released ``post_kl``, latent transformer, geometric query decoder, and
    dense marching-cubes path are implemented. The point-cloud encoder is
    intentionally not included because its released implementation requires
    ``torch_cluster.fps``; :meth:`encode` therefore fails explicitly.
    """

    family_id = "hunyuan3d-2.1"
    component_role = "shape-vae"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ("scikit-image",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    _no_split_modules = ["Hunyuan3DVAEAttentionBlock", "Hunyuan3DVAECrossAttentionBlock"]
    _repeated_blocks = ["Hunyuan3DVAEAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        *,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_encoder_layers: int = 8,
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        point_feats: int = 4,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0039506158752403,
        use_ln_post: bool = True,
        decoder_type: str = "vanilla",
    ) -> None:
        super().__init__()
        del num_encoder_layers, pc_size, pc_sharpedge_size, point_feats, downsample_ratio, use_ln_post
        if min(num_latents, embed_dim, width, heads, num_decoder_layers) <= 0:
            raise ValueError("VAE dimensions must be positive")
        if width % heads != 0:
            raise ValueError("width must be divisible by heads")
        if geo_decoder_downsample_ratio <= 0 or width % geo_decoder_downsample_ratio != 0:
            raise ValueError("geo_decoder_downsample_ratio must divide width")
        if heads % geo_decoder_downsample_ratio != 0:
            raise ValueError("geo_decoder_downsample_ratio must divide heads")
        if decoder_type != "vanilla":
            raise NotImplementedError(
                "Only the portable dense decoder is supported; FlashVDM, hierarchical decoding, and DISO are unsupported"
            )
        if label_type != "binary":
            raise ValueError("the released Hunyuan3D-2.1 shape checkpoint requires label_type='binary'")
        if not math.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")

        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.width = width
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.fourier_embedder = Hunyuan3DFourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
        )
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Hunyuan3DVAELatentTransformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        geo_width = width // geo_decoder_downsample_ratio
        self.geo_decoder = Hunyuan3DGeometricDecoder(
            num_latents=num_latents,
            fourier_embedder=self.fourier_embedder,
            width=geo_width,
            heads=heads // geo_decoder_downsample_ratio,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=geo_decoder_ln_post,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "num_latents": 4096,
            "embed_dim": 64,
            "num_freqs": 8,
            "include_pi": False,
            "heads": 16,
            "width": 1024,
            "num_encoder_layers": 8,
            "num_decoder_layers": 16,
            "qkv_bias": False,
            "qk_norm": True,
            "scale_factor": 1.0039506158752403,
            "geo_decoder_mlp_expand_ratio": 4,
            "geo_decoder_downsample_ratio": 1,
            "geo_decoder_ln_post": True,
            "point_feats": 4,
            "pc_size": 81920,
            "pc_sharpedge_size": 0,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "num_latents": 4,
            "embed_dim": 8,
            "width": 32,
            "heads": 4,
            "num_decoder_layers": 2,
            "num_encoder_layers": 0,
            "num_freqs": 2,
            "include_pi": False,
            "qkv_bias": False,
            "qk_norm": True,
            "point_feats": 0,
            "pc_size": 4,
            "pc_sharpedge_size": 0,
        }

    def encode(self, surface_samples: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del surface_samples, args, kwargs
        raise NotImplementedError(
            "Hunyuan3DShapeVAE is decode-only: released surface encoding depends on torch_cluster.fps and is unsupported"
        )

    def enable_flashvdm_decoder(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError("FlashVDM, hierarchical decoding, and DISO extraction are unsupported")

    def decode(
        self,
        latents: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Hunyuan3DShapeVAEOutput | tuple[torch.Tensor]:
        if latents.ndim != 3 or latents.shape[1:] != self.latent_shape:
            raise ValueError(f"latents must have shape (batch, {self.num_latents}, {self.embed_dim})")
        hidden_states = self.transformer(self.post_kl(latents))
        if not return_dict:
            return (hidden_states,)
        return Hunyuan3DShapeVAEOutput(sample=hidden_states)

    def forward(
        self,
        latents: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> Hunyuan3DShapeVAEOutput | tuple[torch.Tensor]:
        return self.decode(latents, return_dict=return_dict)

    def evaluate_field(
        self,
        decoded_latents: torch.Tensor,
        queries: torch.Tensor,
        *,
        query_chunk_size: int = 8000,
    ) -> torch.Tensor:
        if decoded_latents.ndim != 3 or decoded_latents.shape != (
            decoded_latents.shape[0],
            self.num_latents,
            self.width,
        ):
            raise ValueError(f"decoded_latents must have shape (batch, {self.num_latents}, {self.width})")
        if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")
        if queries.ndim == 2:
            if queries.shape[-1] != 3:
                raise ValueError("queries must end in XYZ coordinates")
            queries = queries.unsqueeze(0).expand(decoded_latents.shape[0], -1, -1)
        if queries.ndim != 3 or queries.shape[0] != decoded_latents.shape[0] or queries.shape[-1] != 3:
            raise ValueError("queries must have shape (points, 3) or (batch, points, 3)")

        logits = [
            self.geo_decoder(queries[:, start : start + query_chunk_size], decoded_latents)
            for start in range(0, queries.shape[1], query_chunk_size)
        ]
        return torch.cat(logits, dim=1).squeeze(-1)

    @staticmethod
    def _normalize_bounds(
        bounds: float | Sequence[float],
    ) -> tuple[float, float, float, float, float, float]:
        if isinstance(bounds, (int, float)) and not isinstance(bounds, bool):
            value = float(bounds)
            resolved = (-value, -value, -value, value, value, value)
        else:
            resolved = tuple(float(value) for value in bounds)
            if len(resolved) != 6:
                raise ValueError("bounds must be a scalar or six XYZ min/max values")
        if any(not math.isfinite(value) for value in resolved):
            raise ValueError("bounds must be finite")
        if any(resolved[index] >= resolved[index + 3] for index in range(3)):
            raise ValueError("each lower bound must be smaller than its upper bound")
        return resolved

    def decoded_latents_to_field(
        self,
        decoded_latents: torch.Tensor,
        *,
        bounds: float | Sequence[float] = 1.01,
        resolution: int = 384,
        query_chunk_size: int = 8000,
    ) -> Hunyuan3DShapeFieldOutput:
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        resolved_bounds = self._normalize_bounds(bounds)
        minimum = decoded_latents.new_tensor(resolved_bounds[:3])
        maximum = decoded_latents.new_tensor(resolved_bounds[3:])
        axes = [
            torch.linspace(minimum[index], maximum[index], resolution + 1, device=decoded_latents.device)
            for index in range(3)
        ]
        queries = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        field = self.evaluate_field(
            decoded_latents,
            queries,
            query_chunk_size=query_chunk_size,
        ).reshape(decoded_latents.shape[0], resolution + 1, resolution + 1, resolution + 1)
        return Hunyuan3DShapeFieldOutput(field=field.float(), bounds=resolved_bounds)

    def extract_meshes(
        self,
        field_output: Hunyuan3DShapeFieldOutput,
        *,
        level: float = 0.0,
        backend: ScikitImageBackend | None = None,
    ) -> tuple[MeshAsset, ...]:
        if type(field_output) is not Hunyuan3DShapeFieldOutput:
            raise TypeError("field_output must be an exact Hunyuan3DShapeFieldOutput")
        field = field_output.field
        if field.ndim != 4:
            raise ValueError("field_output.field must have shape (batch, x, y, z)")
        bounds = field_output.bounds
        spacing = tuple((bounds[index + 3] - bounds[index]) / (field.shape[index + 1] - 1) for index in range(3))
        extractor = ScikitImageBackend() if backend is None else backend
        offset = torch.tensor(bounds[:3], dtype=torch.float32)
        meshes = []
        for scalar_field in field:
            mesh = extractor.extract_surface(scalar_field, level=level, spacing=spacing)
            meshes.append(
                MeshAsset(
                    vertices=mesh.vertices + offset,
                    faces=mesh.faces,
                    normals=mesh.normals,
                    metadata={
                        "family": self.family_id,
                        "level": float(level),
                        "resolution": field.shape[1] - 1,
                    },
                )
            )
        return tuple(meshes)

    @torch.no_grad()
    def decode_to_meshes(
        self,
        latents: torch.Tensor,
        *,
        bounds: float | Sequence[float] = 1.01,
        resolution: int = 384,
        level: float = 0.0,
        query_chunk_size: int = 8000,
    ) -> tuple[MeshAsset, ...]:
        decoded_latents = self.decode(latents / self.scale_factor).sample
        field = self.decoded_latents_to_field(
            decoded_latents,
            bounds=bounds,
            resolution=resolution,
            query_chunk_size=query_chunk_size,
        )
        return self.extract_meshes(field, level=level)


__all__ = [
    "Hunyuan3DShapeFieldOutput",
    "Hunyuan3DShapeVAE",
    "Hunyuan3DShapeVAEOutput",
]
