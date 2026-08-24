# Portions of this file are derived from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for native Diffusers/PyTorch integration.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from transformers import Dinov2Config, Dinov2Model

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class Hunyuan3DConditionerOutput(BaseOutput):
    """DINOv2 image tokens used by the shape denoiser."""

    embeddings: torch.Tensor


class Hunyuan3DDinov2Conditioner(Object3DModel):
    """Frozen DINOv2 conditioner with released Hunyuan normalization semantics."""

    family_id = "hunyuan3d-2.1"
    component_role = "conditioner"
    supported_object_kinds = (Object3DKind.MESH,)
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
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        if dinov2_config is None:
            dinov2_config = self.production_dinov2_config()
        if not isinstance(dinov2_config, dict):
            raise TypeError("dinov2_config must be a dictionary")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        config = Dinov2Config.from_dict(dinov2_config)
        if image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by the DINOv2 patch size")

        self.model = Dinov2Model(config)
        self.model.requires_grad_(False)
        self.model.eval()
        self.image_size = image_size
        self.use_cls_token = use_cls_token
        self.num_patches = (image_size // config.patch_size) ** 2 + int(use_cls_token)
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
            "use_cls_token": True,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        config = Dinov2Config(
            image_size=8,
            patch_size=4,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            mlp_ratio=2,
            qkv_bias=True,
        )
        return {
            "dinov2_config": config.to_dict(),
            "image_size": 8,
            "use_cls_token": True,
        }

    @classmethod
    def from_dinov2_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        image_size: int = 518,
        use_cls_token: bool = True,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> Hunyuan3DDinov2Conditioner:
        model = Dinov2Model.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )
        conditioner = cls(
            dinov2_config=model.config.to_dict(),
            image_size=image_size,
            use_cls_token=use_cls_token,
        )
        conditioner.model = model
        conditioner.model.requires_grad_(False)
        conditioner.model.eval()
        return conditioner

    def _resize_and_center_crop(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        scale = self.image_size / min(height, width)
        resized_height = max(self.image_size, int(math.floor(height * scale)))
        resized_width = max(self.image_size, int(math.floor(width * scale)))
        images = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - self.image_size) // 2
        left = (resized_width - self.image_size) // 2
        return images[:, :, top : top + self.image_size, left : left + self.image_size]

    def forward(
        self,
        images: torch.Tensor,
        *,
        value_range: tuple[float, float] | None = (-1.0, 1.0),
        return_dict: bool = True,
    ) -> Hunyuan3DConditionerOutput | tuple[torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        parameter = next(self.model.parameters())
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        if value_range is not None:
            low, high = (float(value) for value in value_range)
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("value_range must contain finite increasing bounds")
            images = (images - low) / (high - low)
        images = self._resize_and_center_crop(images)
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        embeddings = self.model(pixel_values=images).last_hidden_state
        if not self.use_cls_token:
            embeddings = embeddings[:, 1:]
        if not return_dict:
            return (embeddings,)
        return Hunyuan3DConditionerOutput(embeddings=embeddings)

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
            self.num_patches,
            self.model.config.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )


__all__ = ["Hunyuan3DConditionerOutput", "Hunyuan3DDinov2Conditioner"]
