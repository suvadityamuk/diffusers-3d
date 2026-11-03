# Portions of this file are derived from Tencent Hunyuan3D-Omni:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni
# Revision: 4d47c0cc2bd0c4281963a7314ab330a5af36bfa8
#
# Tencent Hunyuan 3D Omni is licensed under the Tencent Hunyuan 3D Omni Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: the control branch of `OmniEncoder` only (DINOv2 is a separate Transformers component), inference only;
# the released weights are remapped by `conversion.py`.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ...execution.models import Object3DModel

# Position is the released control-signal id (upstream's pose, bbox, voxel, point).
CONTROL_TYPES = ("skeleton", "bbox", "voxels", "point_cloud")


@dataclass
class Hunyuan3DOmniControlEncoderOutput(BaseOutput):
    embeddings: torch.Tensor


class Hunyuan3DOmniControlEncoder(Object3DModel):
    """Hunyuan3D-Omni control encoder: turns one 3D control signal into DiT cross-attention tokens.

    Every control becomes 6-channel rows (bone start and end points; other controls repeat their xyz), each row a
    Fourier-embedded ``Linear -> RMSNorm -> GELU`` token, followed by ``num_type_tokens`` copies of a learned
    control-type token. The pipeline appends these tokens to the DINOv2 image tokens.

    Controls are ``(batch, rows, channels)`` in the generated mesh's normalized frame (right-handed, Y up, inside
    ``[-1, 1]``): ``"skeleton"`` bones ``(x1, y1, z1, x2, y2, z2)``, ``"bbox"`` one ``(x, y, z)`` row of box side lengths,
    ``"point_cloud"`` points, and ``"voxels"`` points that are first reduced to the centers of the occupied cells of
    a ``voxel_resolution`` cube grid (zero-padded to the longest sample, as upstream).
    """

    @register_to_config
    def __init__(
        self,
        width: int = 1024,
        num_freqs: int = 8,
        voxel_resolution: int = 16,
        num_type_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(len(CONTROL_TYPES), 8)
        self.type_proj = nn.Linear(8, width)
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        self.proj_in = nn.Linear(6 * (2 * num_freqs + 1), width)
        self.norm = nn.RMSNorm(width)

    def voxelize(self, points: torch.Tensor) -> torch.Tensor:
        """Centers of the grid cells hit by ``(batch, points, 3)`` positions in ``[-1, 1]``, in linear-index order."""

        resolution = self.config.voxel_resolution
        cells = ((points + 1) / 2 * resolution).floor().long().clamp(0, resolution - 1)
        indices = cells[..., 0] + cells[..., 1] * resolution + cells[..., 2] * resolution**2
        size = 2.0 / resolution
        centers = []
        for sample in indices:
            unique = sample.unique()
            cell = torch.stack([unique % resolution, unique // resolution % resolution, unique // resolution**2], -1)
            centers.append(cell.float() * size + size / 2 - 1)
        return pad_sequence(centers, batch_first=True).to(points.dtype)

    def forward(
        self, control: torch.Tensor, control_type: str, return_dict: bool = True
    ) -> Hunyuan3DOmniControlEncoderOutput | tuple[torch.Tensor]:
        if control_type not in CONTROL_TYPES:
            raise ValueError(f"control_type must be one of {CONTROL_TYPES}, got {control_type!r}")
        channels = 6 if control_type == "skeleton" else 3
        if control.ndim != 3 or control.shape[-1] != channels:
            raise ValueError(f"{control_type} control must have shape (batch, rows, {channels})")
        if control_type == "voxels":
            control = self.voxelize(control)
        if channels == 3:
            control = control.repeat(1, 1, 2)
        frequencies = (control[..., None] * self.frequencies.to(control.dtype)).flatten(-2)
        embedding = torch.cat([control, frequencies.sin(), frequencies.cos()], dim=-1)
        hidden_states = F.gelu(self.norm(self.proj_in(embedding)))
        type_token = self.type_proj(self.type_embedding.weight[CONTROL_TYPES.index(control_type)])
        type_tokens = type_token.expand(control.shape[0], self.config.num_type_tokens, -1)
        embeddings = torch.cat([hidden_states, type_tokens], dim=1)
        if not return_dict:
            return (embeddings,)
        return Hunyuan3DOmniControlEncoderOutput(embeddings=embeddings)


__all__ = ["CONTROL_TYPES", "Hunyuan3DOmniControlEncoder", "Hunyuan3DOmniControlEncoderOutput"]
