# Portions of this file reproduce text-conditioning semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified to wrap the Transformers CLIP text encoder as a Diffusers model.

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import Object3DKind

_TOKENIZER_SUBFOLDER = "tokenizer"
_TOKENIZER_LOADING_KWARGS = ("cache_dir", "force_download", "local_files_only", "proxies", "revision", "token")


@dataclass
class TrellisTextConditionerOutput(BaseOutput):
    """CLIP text-encoder ``last_hidden_state`` over the padded 77-token prompt."""

    embeddings: torch.Tensor


class TrellisClipTextConditioner(Object3DModel):
    """TRELLIS text conditioner: CLIP ViT-L/14 text encoder tokens, exactly as ``TrellisTextTo3DPipeline`` uses them.

    Upstream tokenizes with ``max_length=77``, ``padding="max_length"``, ``truncation=True`` and feeds
    ``last_hidden_state`` to both flow models; the unconditional branch is the encoding of the empty string.
    The tokenizer is saved next to the weights in a ``tokenizer/`` subfolder so the component round-trips
    through ``save_pretrained`` / ``from_pretrained`` as a single Diffusers module.
    """

    family_id = "trellis"
    component_role = "text-conditioner"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.GAUSSIAN_SPLAT, Object3DKind.MESH)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["CLIPEncoderLayer"]

    @register_to_config
    def __init__(
        self,
        clip_config: dict[str, Any] | None = None,
        *,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        if clip_config is None:
            clip_config = self.production_clip_config()
        if not isinstance(clip_config, dict):
            raise TypeError("clip_config must be a dictionary")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        config = CLIPTextConfig.from_dict(clip_config)
        if max_length > config.max_position_embeddings:
            raise ValueError("max_length must not exceed the CLIP text encoder's max_position_embeddings")
        self.model = CLIPTextModel(config)
        self.model.eval()
        self.max_length = max_length
        self.tokenizer: CLIPTokenizer | None = None

    @staticmethod
    def production_clip_config() -> dict[str, Any]:
        """Text tower of ``openai/clip-vit-large-patch14``."""

        return {
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "dropout": 0.0,
            "eos_token_id": 2,
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 77,
            "model_type": "clip_text_model",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "projection_dim": 768,
            "vocab_size": 49408,
        }

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {"clip_config": cls.production_clip_config(), "max_length": 77}

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        config = CLIPTextConfig(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            max_position_embeddings=8,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        )
        return {"clip_config": config.to_dict(), "max_length": 8}

    @classmethod
    def from_clip_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        max_length: int | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> TrellisClipTextConditioner:
        """Build from a Transformers CLIP checkpoint such as ``openai/clip-vit-large-patch14``.

        ``max_length`` defaults to the checkpoint's ``max_position_embeddings`` (77 for the released encoder).
        """

        model = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only, **kwargs
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only, **kwargs
        )
        if max_length is None:
            max_length = int(model.config.max_position_embeddings)
        conditioner = cls(clip_config=model.config.to_dict(), max_length=max_length)
        conditioner.model = model
        conditioner.tokenizer = tokenizer
        conditioner.requires_grad_(False)
        conditioner.eval()
        return conditioner

    def save_pretrained(self, save_directory: str | os.PathLike[str], **kwargs: Any) -> None:
        super().save_pretrained(save_directory, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(save_directory, _TOKENIZER_SUBFOLDER))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any):
        conditioner = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        subfolder = kwargs.get("subfolder")
        tokenizer_subfolder = _TOKENIZER_SUBFOLDER if not subfolder else f"{subfolder}/{_TOKENIZER_SUBFOLDER}"
        loading_kwargs = {name: kwargs[name] for name in _TOKENIZER_LOADING_KWARGS if name in kwargs}
        # Weights saved without a tokenizer (tiny test models) still load; tokenize() then reports the gap.
        # Transformers builds an empty two-token tokenizer from a missing folder, so check for files first.
        local_tokenizer = os.path.join(os.fspath(pretrained_model_name_or_path), tokenizer_subfolder)
        if os.path.isdir(os.fspath(pretrained_model_name_or_path)) and not os.path.isdir(local_tokenizer):
            conditioner.tokenizer = None
            return conditioner
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder=tokenizer_subfolder, **loading_kwargs
            )
        except OSError:
            tokenizer = None
        conditioner.tokenizer = tokenizer if tokenizer is not None and len(tokenizer) > 2 else None
        return conditioner

    def tokenize(self, prompts: Sequence[str]) -> torch.Tensor:
        """Released tokenization: ``max_length`` padding and truncation, returning ``(batch, max_length)`` ids."""

        if self.tokenizer is None:
            raise RuntimeError(
                "no tokenizer is attached; load the conditioner with from_pretrained() or from_clip_pretrained()"
            )
        if isinstance(prompts, str) or not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("prompts must be a sequence of strings")
        encoding = self.tokenizer(
            list(prompts),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding["input_ids"]

    def forward(
        self,
        prompts: Sequence[str] | torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> TrellisTextConditionerOutput | tuple[torch.Tensor]:
        """Encode prompts (or pre-tokenized ``(batch, max_length)`` ids) to ``last_hidden_state``."""

        input_ids = prompts if isinstance(prompts, torch.Tensor) else self.tokenize(prompts)
        if input_ids.ndim != 2 or input_ids.shape[1] != self.max_length:
            raise ValueError(f"input ids must have shape (batch, {self.max_length})")
        embeddings = self.model(input_ids=input_ids.to(device=self.device)).last_hidden_state
        if not return_dict:
            return (embeddings,)
        return TrellisTextConditionerOutput(embeddings=embeddings)

    def unconditional_embedding(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Released null condition: the encoding of the empty prompt, repeated per batch item."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        embeddings = self.forward([""]).embeddings.expand(batch_size, -1, -1)
        return embeddings.to(device=device, dtype=dtype)


__all__ = ["TrellisClipTextConditioner", "TrellisTextConditionerOutput"]
