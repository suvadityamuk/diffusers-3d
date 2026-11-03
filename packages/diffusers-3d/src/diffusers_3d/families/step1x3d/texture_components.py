"""Third-party components of the Step1X-3D texture stage, loaded from their original repositories at pinned revisions.

The released pipeline runs the SDXL base (``stabilityai/stable-diffusion-xl-base-1.0``) unchanged: its text encoders
and UNet (the release adds only the T2I-Adapter and the MV-Adapter weights), with the
``madebyollin/sdxl-vae-fp16-fix`` VAE. None of these networks is converted or redistributed with the converted
pipeline; the tokenizers (vocabulary files, no weights) are saved with it.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from ...execution.pipelines import ExternalComponent

SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_BASE_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
SDXL_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
SDXL_VAE_REVISION = "207b116dae70ace3637169f1ddd2434b91b3a8cd"


def load_sdxl_component(
    repository: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    component: str,
    subfolder: str | None = None,
    **hub_kwargs: Any,
):
    """Load one component (``"vae"``, ``"text_encoder"``, ``"text_encoder_2"``, ``"tokenizer"``, ``"tokenizer_2"``,
    or ``"unet"``) of an SDXL repository, in ``dtype`` when it has weights."""

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    classes = {
        "vae": AutoencoderKL,
        "text_encoder": CLIPTextModel,
        "text_encoder_2": CLIPTextModelWithProjection,
        "tokenizer": CLIPTokenizer,
        "tokenizer_2": CLIPTokenizer,
        "unet": UNet2DConditionModel,
    }
    loaded = classes[component].from_pretrained(repository, subfolder=subfolder, revision=revision, **hub_kwargs)
    if isinstance(loaded, torch.nn.Module):
        loaded = loaded.to(dtype) if dtype is not None else loaded
        return loaded.eval()
    return loaded


STEP1X3D_TEXTURE_EXTERNAL_COMPONENTS = {
    "vae": ExternalComponent(SDXL_VAE_MODEL, SDXL_VAE_REVISION, partial(load_sdxl_component, component="vae")),
    **{
        name: ExternalComponent(
            SDXL_BASE_MODEL, SDXL_BASE_REVISION, partial(load_sdxl_component, component=name, subfolder=name)
        )
        for name in ("text_encoder", "text_encoder_2", "unet")
    },
}


__all__ = [
    "SDXL_BASE_MODEL",
    "SDXL_BASE_REVISION",
    "SDXL_VAE_MODEL",
    "SDXL_VAE_REVISION",
    "STEP1X3D_TEXTURE_EXTERNAL_COMPONENTS",
    "load_sdxl_component",
]
