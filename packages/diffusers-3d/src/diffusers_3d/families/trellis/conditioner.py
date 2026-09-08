# Portions of this file reproduce image-conditioning semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified to use Transformers DINOv2.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class TrellisConditionerOutput(BaseOutput):
    """Layer-normalized DINOv2 class, register, and patch tokens."""

    embeddings: torch.Tensor


class TrellisDinov2Conditioner(Object3DModel):
    """TRELLIS DINOv2-L/14-register conditioner using Transformers blocks."""

    family_id = "trellis"
    component_role = "conditioner"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.GAUSSIAN_SPLAT, Object3DKind.MESH)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["Dinov2Layer"]

    @register_to_config
    def __init__(
        self,
        dinov2_config: dict[str, Any] | None = None,
        *,
        image_size: int = 518,
        num_register_tokens: int = 4,
    ) -> None:
        super().__init__()
        if dinov2_config is None:
            dinov2_config = self.production_dinov2_config()
        if not isinstance(dinov2_config, dict):
            raise TypeError("dinov2_config must be a dictionary")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if (
            not isinstance(num_register_tokens, int)
            or isinstance(num_register_tokens, bool)
            or num_register_tokens < 0
        ):
            raise ValueError("num_register_tokens must be a non-negative integer")
        config = Dinov2Config.from_dict(dinov2_config)
        if image_size % config.patch_size:
            raise ValueError("image_size must be divisible by the DINOv2 patch size")

        self.model = Dinov2Model(config)
        self.model.eval()
        self.image_size = image_size
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, config.hidden_size))
        nn.init.trunc_normal_(self.register_tokens, std=config.initializer_range)
        self.num_tokens = (image_size // config.patch_size) ** 2 + 1 + num_register_tokens
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def production_dinov2_config() -> dict[str, Any]:
        return {
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 1024,
            "image_size": 518,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-6,
            "layerscale_value": 1.0,
            "mlp_ratio": 4,
            "model_type": "dinov2",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "qkv_bias": True,
            "use_swiglu_ffn": False,
        }

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "dinov2_config": cls.production_dinov2_config(),
            "image_size": 518,
            "num_register_tokens": 4,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        config = Dinov2Config(
            image_size=8,
            patch_size=4,
            num_channels=3,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            mlp_ratio=2,
            qkv_bias=True,
        )
        return {
            "dinov2_config": config.to_dict(),
            "image_size": 8,
            "num_register_tokens": 2,
        }

    @classmethod
    def from_dinov2_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        image_size: int = 518,
        num_register_tokens: int = 0,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> TrellisDinov2Conditioner:
        model = Dinov2Model.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )
        conditioner = cls(
            dinov2_config=model.config.to_dict(),
            image_size=image_size,
            num_register_tokens=num_register_tokens,
        )
        conditioner.model = model
        conditioner.model.requires_grad_(False)
        conditioner.model.eval()
        return conditioner

    def forward(
        self,
        images: torch.Tensor,
        *,
        value_range: tuple[float, float] | None = (0.0, 1.0),
        return_dict: bool = True,
    ) -> TrellisConditionerOutput | tuple[torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        parameter = next(self.model.parameters())
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        if value_range is not None:
            low, high = (float(value) for value in value_range)
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("value_range must contain finite increasing bounds")
            images = (images - low) / (high - low)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)

        embeddings = self.model.embeddings(images)
        if self.num_register_tokens:
            register_tokens = self.register_tokens.expand(images.shape[0], -1, -1).to(embeddings)
            embeddings = torch.cat([embeddings[:, :1], register_tokens, embeddings[:, 1:]], dim=1)
        embeddings = self.model.encoder(embeddings).last_hidden_state
        # TRELLIS consumes DINOv2 ``x_prenorm`` and applies an unparameterized
        # final layer norm, rather than DINOv2's learned output norm.
        embeddings = F.layer_norm(embeddings, embeddings.shape[-1:])
        if not return_dict:
            return (embeddings,)
        return TrellisConditionerOutput(embeddings=embeddings)

    def unconditional_embedding(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.model.parameters())
        return torch.zeros(
            batch_size,
            self.num_tokens,
            self.model.config.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )


__all__ = ["TrellisConditionerOutput", "TrellisDinov2Conditioner"]
