"""Third-party components of Hunyuan3D-2.1 Paint, loaded from their original sources at pinned revisions.

The Paint release ships the Stable Diffusion 2.1 VAE unchanged (tensor for tensor), and the released pipeline loads
DINOv2-giant from ``facebook/dinov2-giant`` and Real-ESRGAN x4plus from its GitHub release. None of them is converted
or redistributed with the converted pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from ...execution.pipelines import ExternalComponent
from .paint_models import Hunyuan3DPaintRRDBNet

DINOV2_REPOSITORY = "facebook/dinov2-giant"
DINOV2_REVISION = "611a9d42f2335e0f921f1e313ad3c1b7178d206d"
# Stability AI took ``stabilityai/stable-diffusion-2-1`` down; this community copy holds the same VAE tensors as the
# Paint release's ``hunyuan3d-paintpbr-v2-1/vae`` (and the same config).
SD21_REPOSITORY = "sd2-community/stable-diffusion-2-1"
SD21_REVISION = "bb2154823665391b4fb29b0b9cf82a198964ee05"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def load_dinov2_processor(repository: str, *, revision: str | None = None, dtype: Any = None, **hub_kwargs: Any):
    """The DINOv2 image processor of ``repository``."""

    from transformers import BitImageProcessorPil

    return BitImageProcessorPil.from_pretrained(repository, revision=revision, **hub_kwargs)


def load_dinov2(repository: str, *, revision: str | None = None, dtype: torch.dtype | None = None, **hub_kwargs: Any):
    """DINOv2 from ``repository``, in ``dtype`` (float32 by default)."""

    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(
        repository, revision=revision, dtype=torch.float32 if dtype is None else dtype, **hub_kwargs
    )
    return model.eval()


def load_sd21_vae(
    repository: str, *, revision: str | None = None, dtype: torch.dtype | None = None, **hub_kwargs: Any
):
    """The Stable Diffusion 2.1 VAE from ``<repository>/vae``."""

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(repository, subfolder="vae", revision=revision, **hub_kwargs)
    return vae.to(dtype).eval() if dtype is not None else vae.eval()


def load_realesrgan(
    url: str,
    *,
    revision: str,
    dtype: torch.dtype | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    **hub_kwargs: Any,
) -> Hunyuan3DPaintRRDBNet:
    """Real-ESRGAN x4plus from its release ``url``, checked against the SHA-256 ``revision`` and converted on load.

    The file is cached in ``cache_dir`` (``torch.hub``'s checkpoint folder by default).
    """

    directory = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
    path = directory / url.rsplit("/", 1)[-1]
    if force_download or not path.is_file():
        if local_files_only:
            raise FileNotFoundError(f"{path} is not cached and local_files_only is set; download it from {url}")
        directory.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path), hash_prefix=revision)
    if hashlib.sha256(path.read_bytes()).hexdigest() != revision:
        raise ValueError(f"{path} is not the released {url.rsplit('/', 1)[-1]} (SHA-256 {revision})")
    upscaler = Hunyuan3DPaintRRDBNet()
    upscaler.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params_ema"], strict=True)
    return upscaler.to(dtype).eval() if dtype is not None else upscaler.eval()


HUNYUAN3D_PAINT_EXTERNAL_COMPONENTS = {
    "feature_extractor": ExternalComponent(DINOV2_REPOSITORY, DINOV2_REVISION, load_dinov2_processor),
    "image_encoder": ExternalComponent(DINOV2_REPOSITORY, DINOV2_REVISION, load_dinov2),
    "vae": ExternalComponent(SD21_REPOSITORY, SD21_REVISION, load_sd21_vae),
    "upscaler": ExternalComponent(REALESRGAN_URL, REALESRGAN_SHA256, load_realesrgan),
}


__all__ = [
    "DINOV2_REPOSITORY",
    "DINOV2_REVISION",
    "HUNYUAN3D_PAINT_EXTERNAL_COMPONENTS",
    "REALESRGAN_SHA256",
    "REALESRGAN_URL",
    "SD21_REPOSITORY",
    "SD21_REVISION",
    "load_dinov2",
    "load_dinov2_processor",
    "load_realesrgan",
    "load_sd21_vae",
]
