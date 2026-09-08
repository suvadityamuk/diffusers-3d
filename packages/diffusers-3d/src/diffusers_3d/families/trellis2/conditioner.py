# Portions of this file reproduce Microsoft TRELLIS.2 image conditioning:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified to use the public Transformers DINOv3 implementation without downloads.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind


@dataclass
class Trellis2ConditionerOutput(BaseOutput):
    embeddings: torch.Tensor


class Trellis2Dinov3Conditioner(Object3DModel):
    """DINOv3-L/16 token conditioner with TRELLIS.2's manual final normalization."""

    family_id = "trellis2"
    component_role = "conditioner"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.O_VOXEL)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["DINOv3ViTLayer"]

    @register_to_config
    def __init__(
        self,
        dinov3_config: dict[str, Any] | None = None,
        *,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        if dinov3_config is None:
            dinov3_config = self.production_dinov3_config()
        if not isinstance(dinov3_config, dict):
            raise TypeError("dinov3_config must be a dictionary")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        config = DINOv3ViTConfig.from_dict(dinov3_config)
        if image_size % config.patch_size:
            raise ValueError("image_size must be divisible by the DINOv3 patch size")
        self.model = DINOv3ViTModel(config)
        self.model.eval()
        self.image_size = image_size
        self.num_tokens = (image_size // config.patch_size) ** 2 + 1 + config.num_register_tokens
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
    def production_dinov3_config() -> dict[str, Any]:
        return DINOv3ViTConfig(
            image_size=512,
            patch_size=16,
            num_channels=3,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_register_tokens=4,
            hidden_act="gelu",
            layerscale_value=1.0,
            drop_path_rate=0.0,
            use_gated_mlp=False,
        ).to_dict()

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {"dinov3_config": cls.production_dinov3_config(), "image_size": 512}

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        config = DINOv3ViTConfig(
            image_size=8,
            patch_size=4,
            num_channels=3,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_register_tokens=2,
            hidden_act="gelu",
        )
        return {"dinov3_config": config.to_dict(), "image_size": 8}

    @classmethod
    def from_dinov3_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        image_size: int = 512,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> Trellis2Dinov3Conditioner:
        model = DINOv3ViTModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )
        conditioner = cls(dinov3_config=model.config.to_dict(), image_size=image_size)
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
    ) -> Trellis2ConditionerOutput | tuple[torch.Tensor]:
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
        hidden_states = self.model.embeddings(images, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(images)
        for layer in self.model.model.layer:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        embeddings = F.layer_norm(hidden_states, hidden_states.shape[-1:])
        if not return_dict:
            return (embeddings,)
        return Trellis2ConditionerOutput(embeddings=embeddings)

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


__all__ = ["Trellis2ConditionerOutput", "Trellis2Dinov3Conditioner"]
