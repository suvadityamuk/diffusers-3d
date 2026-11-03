# Portions of this file reproduce the TripoSR triplane tokenizer, backbone, upsampler, and NeRF decoder:
# https://github.com/VAST-AI-Research/TripoSR
# Revision: 107cefdc244c39106fa830359024f6a2f1c78871
#
# MIT License. Copyright (c) 2024 Tripo AI & Stability AI.
# The backbone is the stock Diffusers Transformer2DModel; the NeRF renderer's ray marching lives in the pipeline.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import (
    ModelMixin,  # noqa: F401 - required by external-component loading
    Transformer2DModel,
)
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import BaseOutput
from torch import nn

from ...execution.models import Object3DModel


class TripoSRTransformerModel(Object3DModel):
    """Image tokens to triplanes: learned triplane tokens cross-attend to the image, then each plane is upsampled 2x.

    Upstream's ``Transformer1D`` (group norm, linear in, Diffusers ``BasicTransformerBlock``s, linear out, residual)
    over the ``3 * size * size`` tokens is the stock ``Transformer2DModel`` with linear projections, run on a
    ``(3 * size, size)`` view of the three planes, so the released weights load unchanged under ``backbone``.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        triplane_size: int = 32,
        num_channels: int = 1024,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        num_layers: int = 16,
        cross_attention_dim: int = 768,
        norm_num_groups: int = 32,
        out_channels: int = 40,
    ) -> None:
        super().__init__()
        if num_attention_heads * attention_head_dim != num_channels:
            raise ValueError("num_attention_heads * attention_head_dim must equal num_channels")
        self.embeddings = nn.Parameter(
            torch.randn(3, num_channels, triplane_size, triplane_size) / math.sqrt(num_channels)
        )
        self.backbone = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=num_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=norm_num_groups,
            activation_fn="geglu",
            use_linear_projection=True,
        )
        self.upsample = nn.ConvTranspose2d(num_channels, out_channels, kernel_size=2, stride=2)

    def forward(
        self, encoder_hidden_states: torch.Tensor, *, return_dict: bool = True
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Map image tokens ``(batch, tokens, cross_attention_dim)`` to triplanes ``(batch, 3, out_channels, 2s, 2s)``."""

        batch_size = encoder_hidden_states.shape[0]
        size = self.config.triplane_size
        tokens = self.embeddings.transpose(0, 1).reshape(1, -1, 3 * size, size).expand(batch_size, -1, -1, -1)
        tokens = self.backbone(tokens, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        planes = tokens.unflatten(2, (3, size)).transpose(1, 2).flatten(0, 1)
        triplanes = self.upsample(planes).unflatten(0, (batch_size, 3))
        if not return_dict:
            return (triplanes,)
        return Transformer2DModelOutput(sample=triplanes)


@dataclass
class TripoSRTriplaneDecoderOutput(BaseOutput):
    """Activated density ``(batch, points, 1)`` and RGB colour in ``[0, 1]`` ``(batch, points, 3)``."""

    density: torch.Tensor
    color: torch.Tensor


class TripoSRTriplaneDecoder(Object3DModel):
    """NeRF MLP queried on triplanes: bilinear plane features are concatenated, then decoded to density and colour.

    Points inside ``[-radius, radius]^3`` sample the XY, XZ, and YZ planes. Density is ``exp(raw + density_bias)``
    and colour is ``sigmoid``, the released ``TriplaneNeRFRenderer`` settings.
    """

    @register_to_config
    def __init__(
        self,
        triplane_channels: int = 40,
        hidden_size: int = 64,
        num_hidden_layers: int = 9,
        radius: float = 0.87,
        density_bias: float = -1.0,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(3 * triplane_channels, hidden_size), nn.SiLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.SiLU()]
        self.layers = nn.Sequential(*layers, nn.Linear(hidden_size, 4))

    def forward(
        self,
        triplanes: torch.Tensor,
        positions: torch.Tensor,
        *,
        chunk_size: int | None = None,
        return_dict: bool = True,
    ) -> TripoSRTriplaneDecoderOutput | tuple[torch.Tensor, torch.Tensor]:
        """Decode triplanes ``(batch, 3, channels, H, W)`` at ``positions`` ``(batch, points, 3)``.

        ``chunk_size`` bounds how many points are decoded at once.
        """

        radius = self.config.radius
        features = []
        for chunk in positions.split(chunk_size or positions.shape[1], dim=1):
            grid = (chunk + radius) / (2 * radius) * 2 - 1
            grid = torch.stack((grid[..., [0, 1]], grid[..., [0, 2]], grid[..., [1, 2]]), dim=1)
            sampled = F.grid_sample(
                triplanes.flatten(0, 1), grid.flatten(0, 1)[:, None], align_corners=False, mode="bilinear"
            )
            # Upstream feeds the MLP one shape at a time as this transposed ``(points, 3 * channels)`` view, and
            # activates contiguous outputs; memory layout changes the rounding, so both are kept.
            sampled = sampled.reshape(positions.shape[0], -1, sampled.shape[-1]).transpose(1, 2)
            features.append(torch.stack([self.layers(item) for item in sampled]))
        features = torch.cat(features, dim=1)
        density = torch.exp(features[..., :1] + self.config.density_bias)
        color = torch.sigmoid(features[..., 1:4].contiguous())
        if not return_dict:
            return density, color
        return TripoSRTriplaneDecoderOutput(density=density, color=color)


__all__ = ["TripoSRTransformerModel", "TripoSRTriplaneDecoder", "TripoSRTriplaneDecoderOutput"]
