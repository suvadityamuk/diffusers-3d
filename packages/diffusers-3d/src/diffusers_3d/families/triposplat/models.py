# Portions of this file reproduce the TripoSplat flow transformer and its Flux.2 VAE image-latent conditioning:
# https://github.com/VAST-AI-Research/TripoSplat
# Revision: d8db9e018b413dd9c4a9fe22463781bf98e8e68d
#
# MIT License. Copyright (c) 2026 VAST.
# Rewritten on Diffusers building blocks with the attention dispatcher.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import GELU
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.vae import Encoder
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from ...execution.models import Object3DModel


class TripoSplatRMSNorm(nn.Module):
    """Per-head RMS norm: ``normalize(x) * gamma[head] * sqrt(head_dim)`` in float32."""

    def __init__(self, head_dim: int, heads: int) -> None:
        super().__init__()
        self.scale = head_dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(hidden_states.float(), dim=-1) * self.gamma.float() * self.scale
        return normalized.to(hidden_states.dtype)


class TripoSplatAttnProcessor:
    """Fused-projection attention; rotary embedding (complex, per head) is applied before the query/key norms."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: TripoSplatAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            query, key, value = attn.to_qkv(hidden_states).unflatten(-1, (3, attn.heads, -1)).unbind(2)
        else:
            query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            key, value = attn.to_kv(encoder_hidden_states).unflatten(-1, (2, attn.heads, -1)).unbind(2)
        if rotary_emb is not None:
            query, key = (
                torch.view_as_real(torch.view_as_complex(x.float().unflatten(-1, (-1, 2))) * rotary_emb)
                .flatten(-2)
                .type_as(x)
                for x in (query, key)
            )
        if attn.norm_q is not None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        hidden_states = dispatch_attention_fn(
            query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config
        )
        return attn.to_out(hidden_states.flatten(2, 3).to(query.dtype))


class TripoSplatAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = TripoSplatAttnProcessor
    _available_processors = [TripoSplatAttnProcessor]

    def __init__(self, dim: int, heads: int, *, cross_attention_dim: int | None = None) -> None:
        super().__init__()
        self.heads = heads
        if cross_attention_dim is None:
            self.to_qkv = nn.Linear(dim, 3 * dim)
        else:
            self.to_q = nn.Linear(dim, dim)
            self.to_kv = nn.Linear(cross_attention_dim, 2 * dim)
        self.norm_q = TripoSplatRMSNorm(dim // heads, heads)
        self.norm_k = TripoSplatRMSNorm(dim // heads, heads)
        self.to_out = nn.Linear(dim, dim)
        self.set_processor(TripoSplatAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, rotary_emb)


class TripoSplatRePoEmbedding(nn.Module):
    """Content-dependent 3D rotary positions ("RePo"): each token predicts a per-head 3D offset that is rotated in."""

    def __init__(self, dim: int, heads: int, head_dim: int, max_frequency: float = 16.0) -> None:
        super().__init__()
        self.heads = heads
        self.norm = FP32LayerNorm(dim, eps=1e-5)
        self.gate_map = nn.Linear(dim, dim // 8, bias=False)
        self.content_map = nn.Linear(dim, dim // 8, bias=False)
        self.final_map = nn.Linear(dim // 8, 3 * heads, bias=False)
        widths = (head_dim // 6, head_dim // 6, (head_dim - 4 * (head_dim // 6)) // 2)
        for axis, width in enumerate(widths):
            setattr(self, f"freqs_{axis}", nn.Parameter(torch.linspace(1.0, max_frequency, width)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The rotary layers stay outside IDOL's float16 blocks, so they run in their own precision.
        hidden_states = self.norm(hidden_states.to(self.gate_map.weight.dtype))
        offsets = self.final_map(F.silu(self.gate_map(hidden_states)) * self.content_map(hidden_states))
        offsets = offsets.unflatten(-1, (self.heads, 3))
        angles = []
        for axis in range(3):
            offset, frequencies = offsets[..., axis, None], getattr(self, f"freqs_{axis}")
            # Upstream's straight-through clamp ``x * tanh(f) + x.detach() * (f - tanh(f))``, evaluated forward.
            angles.append((offset * frequencies.tanh() + offset * (frequencies - frequencies.tanh())) * torch.pi)
        angles = torch.cat(angles, dim=-1).float()
        return torch.polar(torch.ones_like(angles), angles)


class TripoSplatTransformerBlock(nn.Module):
    """Self-attention DiT block; with ``modulation`` it takes the shared adaLN vector plus its own learned shift."""

    def __init__(self, dim: int, heads: int, *, modulation: bool = True, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=not modulation)
        self.attn = TripoSplatAttention(dim, heads)
        self.norm2 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=not modulation)
        self.ff = FeedForward(dim, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.shift_table = nn.Parameter(torch.randn(1, 6 * dim) / dim**0.5) if modulation else None

    def forward(
        self, hidden_states: torch.Tensor, modulation: torch.Tensor | None, rotary_emb: torch.Tensor
    ) -> torch.Tensor:
        if self.shift_table is None:
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb=rotary_emb)
            return hidden_states + self.ff(self.norm2(hidden_states))
        modulation = (modulation + self.shift_table.to(modulation.dtype)).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attention_input = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + self.attn(attention_input, rotary_emb=rotary_emb) * gate_msa
        return hidden_states + self.ff(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp) * gate_mlp


def _latent_position_embedding(num_tokens: int, dim: int) -> torch.Tensor:
    """Fourier features of upstream's fixed scrambled-Sobol point per latent token, computed in float32."""

    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123)
    points = sobol.draw(num_tokens, dtype=torch.float32).reshape(-1)
    width = dim // 3 // 2
    residual = torch.arange(max(0, width - 16), dtype=torch.float32) / max(width - 16, 1) * 16
    exponents = torch.cat([torch.arange(16, dtype=torch.float32), residual])
    angles = torch.outer(points, torch.pow(2.0, exponents[:width])) * 2 * torch.pi
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(1, num_tokens, -1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


@dataclass
class TripoSplatTransformerOutput(BaseOutput):
    """Velocity for the latent tokens ``(batch, tokens, in_channels)`` and the camera token ``(batch, 1, camera)``."""

    sample: torch.Tensor
    camera: torch.Tensor


class TripoSplatTransformerModel(Object3DModel):
    """TripoSplat rectified-flow transformer over 8192 Gaussian-field latent tokens and one camera token.

    Noise tokens get a fixed Sobol-point Fourier embedding; noise and image tokens are refined separately, then the
    noise, image, and camera tokens run jointly through ``num_layers`` adaLN blocks. Every block computes its own
    content-dependent 3D rotary positions. Image tokens are DINOv3 features plus projected Flux.2 VAE latents.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TripoSplatTransformerBlock"]
    _repeated_blocks = ["TripoSplatTransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_embed", "time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        camera_channels: int = 5,
        num_latent_tokens: int = 8192,
        width: int = 1024,
        num_layers: int = 24,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 1280,
        cross_attention_dim_2: int = 128,
    ) -> None:
        super().__init__()
        head_dim = width // num_attention_heads
        self.time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(256, width)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        # Upstream computes this in float32 and casts the result, so the buffer is cast with the model.
        self.register_buffer(
            "position_embedding", _latent_position_embedding(num_latent_tokens, width), persistent=False
        )
        self.proj_in = nn.Linear(in_channels, width)
        self.context_proj = nn.Linear(cross_attention_dim, width)
        self.context_proj_2 = nn.Linear(cross_attention_dim_2, width)
        # Upstream's camera MLP has ``num_refiner_layers`` linear layers; extra hidden layers go after the first, so
        # the released two-layer checkpoints keep the ``FeedForward`` layout.
        if num_refiner_layers < 2:
            raise ValueError("num_refiner_layers must be at least 2: upstream's camera MLP needs two linear layers")
        self.camera_proj = FeedForward(
            camera_channels, dim_out=width, inner_dim=width, activation_fn="gelu-approximate"
        )
        for _ in range(num_refiner_layers - 2):
            self.camera_proj.net.insert(1, GELU(width, width, approximate="tanh"))

        def blocks(count: int, modulation: bool = True) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    TripoSplatTransformerBlock(width, num_attention_heads, modulation=modulation, mlp_ratio=mlp_ratio)
                    for _ in range(count)
                ]
            )

        def rope(count: int) -> nn.ModuleList:
            return nn.ModuleList([TripoSplatRePoEmbedding(width, num_attention_heads, head_dim) for _ in range(count)])

        self.noise_refiner, self.noise_rope = blocks(num_refiner_layers), rope(num_refiner_layers)
        self.context_refiner, self.context_rope = blocks(num_refiner_layers, False), rope(num_refiner_layers)
        self.blocks, self.rope = blocks(num_layers), rope(num_layers)
        self.shift_table = nn.Parameter(torch.randn(1, 2, width) / width**0.5)
        self.proj_out = nn.Linear(width, in_channels)
        self.camera_proj_out = nn.Linear(width, camera_channels)
        self.gradient_checkpointing = False

    def _block(self, block: nn.Module, *args: torch.Tensor | None) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(block, *args)
        return block(*args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        camera: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> TripoSplatTransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Predict velocities for latents ``(batch, num_latent_tokens, in_channels)`` and ``camera``.

        ``timestep`` is in ``[0, 1000]``. ``encoder_hidden_states`` (DINOv3) and ``encoder_hidden_states_2``
        (Flux.2 VAE) are token-aligned image features that are summed after projection.
        """

        # The blocks may run in lower precision than the projections around them, as IDOL's float16 blocks do.
        dtype, block_dtype = self.proj_in.weight.dtype, self.blocks[0].attn.to_out.weight.dtype
        num_tokens = hidden_states.shape[1]
        temb = self.time_embed(self.time_proj(timestep).to(dtype))
        modulation = self.time_modulation(temb).to(block_dtype)
        hidden_states = self.proj_in(hidden_states.to(dtype)).to(block_dtype)
        hidden_states = hidden_states + self.position_embedding.to(block_dtype)
        context = self.context_proj(encoder_hidden_states.to(dtype))
        context = (context + self.context_proj_2(encoder_hidden_states_2.to(dtype))).to(block_dtype)

        for block, rope in zip(self.noise_refiner, self.noise_rope):
            hidden_states = self._block(block, hidden_states, modulation, rope(hidden_states))
        for block, rope in zip(self.context_refiner, self.context_rope):
            context = self._block(block, context, None, rope(context))
        hidden_states = torch.cat([hidden_states, context, self.camera_proj(camera.to(dtype)).to(block_dtype)], dim=1)
        for block, rope in zip(self.blocks, self.rope):
            hidden_states = self._block(block, hidden_states, modulation, rope(hidden_states))

        shift, scale = (self.shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        latent_states, camera_states = (
            F.layer_norm(states.float(), states.shape[-1:]).to(dtype) * (1 + scale) + shift
            for states in (hidden_states[:, :num_tokens], hidden_states[:, -camera.shape[1] :])
        )
        sample, camera = self.proj_out(latent_states), self.camera_proj_out(camera_states)
        if not return_dict:
            return sample, camera
        return TripoSplatTransformerOutput(sample=sample, camera=camera)


class TripoSplatImageVAE(Object3DModel):
    """Encoder half of the Flux.2 VAE, as TripoSplat uses it for its second set of image tokens.

    The stock Diffusers ``Encoder`` loads Flux.2 VAE weights unchanged. A posterior sample is 2x2-patchified to
    ``4 * latent_channels`` channels and standardized by the VAE's batch-norm running statistics.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        batch_norm_eps: float = 1e-5,
        clamp_logvar: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=("DownEncoderBlock2D",) * len(block_out_channels),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.bn = nn.BatchNorm1d(4 * latent_channels, eps=batch_norm_eps, affine=False)

    def forward(self, images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Encode ``images`` in ``[-1, 1]`` to ``(batch, (H / 16) * (W / 16), 4 * latent_channels)`` tokens.

        The posterior is sampled with ``generator``. ``clamp_logvar`` clamps the log-variance to ``[-30, 20]`` first,
        as Diffusers' ``DiagonalGaussianDistribution`` does (IDOL); TripoSplat's own code does not clamp.
        """

        mean, logvar = self.quant_conv(self.encoder(images)).chunk(2, dim=1)
        if self.config.clamp_logvar:
            logvar = logvar.clamp(-30.0, 20.0)
        noise = randn_tensor(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        latents = mean + torch.exp(0.5 * logvar) * noise
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, 4 * channels, height // 2, width // 2)
        mean = self.bn.running_mean.view(1, -1, 1, 1).to(latents)
        std = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn.eps).to(latents)
        return ((latents - mean) / std).flatten(2).transpose(1, 2)


# The FLUX.2-dev VAE (gated; accept its license on the Hub first), whose encoder TripoSplat and IDOL condition on.
# Its weights load from this repository and are never converted or redistributed.
FLUX2_REPO = "black-forest-labs/FLUX.2-dev"
FLUX2_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
_VAE_FILES = ("vae/config.json", "vae/diffusion_pytorch_model.safetensors")


def convert_triposplat_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep the encoder half of a Diffusers-layout Flux.2 VAE checkpoint; its keys load unchanged."""

    return {key: value for key, value in state_dict.items() if not key.startswith(("decoder.", "post_quant_conv."))}


def load_flux2_vae_encoder(
    repository: str | Path = FLUX2_REPO,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    batch_norm_eps: float | None = None,
    clamp_logvar: bool = False,
    **hub_kwargs: Any,
) -> TripoSplatImageVAE:
    """Load the encoder half of a Diffusers-layout Flux.2 VAE from ``<repository>/vae`` (a local folder or Hub id).

    ``revision`` defaults to :data:`FLUX2_REVISION` for :data:`FLUX2_REPO`. ``batch_norm_eps`` defaults to the VAE
    config's (IDOL's ``AutoencoderKLFlux2``); TripoSplat's own encoder uses ``1e-5``. ``clamp_logvar`` clamps the
    posterior log-variance as ``AutoencoderKLFlux2`` does (IDOL). The weights stay float32 unless ``dtype`` is given;
    ``hub_kwargs`` (``token``, ``cache_dir``, ``local_files_only``, ``force_download``) go to the download.
    """

    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    root = Path(repository)
    if root.joinpath("vae").is_dir():
        config_path, weights_path = (root / name for name in _VAE_FILES)
    else:
        if revision is None and str(repository) == FLUX2_REPO:
            revision = FLUX2_REVISION
        names = ("token", "cache_dir", "local_files_only", "force_download")
        download = {key: hub_kwargs[key] for key in names if hub_kwargs.get(key) is not None}
        config_path, weights_path = (
            Path(hf_hub_download(str(repository), name, revision=revision, **download)) for name in _VAE_FILES
        )
    config = json.loads(config_path.read_text())
    keys = ("in_channels", "latent_channels", "block_out_channels", "layers_per_block", "norm_num_groups")
    vae_config = {key: config[key] for key in keys if key in config}
    vae_config["batch_norm_eps"] = config.get("batch_norm_eps", 1e-4) if batch_norm_eps is None else batch_norm_eps
    vae_config["clamp_logvar"] = clamp_logvar
    with init_empty_weights():
        model = TripoSplatImageVAE(**vae_config)
    model.load_state_dict(convert_triposplat_vae_state_dict(load_file(weights_path)), strict=True, assign=True)
    return (model if dtype is None else model.to(dtype)).eval()


__all__ = [
    "FLUX2_REPO",
    "FLUX2_REVISION",
    "convert_triposplat_vae_state_dict",
    "load_flux2_vae_encoder",
    "TripoSplatAttention",
    "TripoSplatAttnProcessor",
    "TripoSplatImageVAE",
    "TripoSplatRMSNorm",
    "TripoSplatRePoEmbedding",
    "TripoSplatTransformerBlock",
    "TripoSplatTransformerModel",
    "TripoSplatTransformerOutput",
]
