"""Convert the released Step1X-3D texture adapter (``stepfun-ai/Step1X-3D/Step1X-3D-Texture``) to Diffusers.

The SDXL base components are loaded with stock ``from_pretrained``; the release supplies the T2I-Adapter and the
MV-Adapter attention weights.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .conversion import STEP1X3D_CHECKPOINT_REVISION, STEP1X3D_REFERENCE_REVISION
from .texture_attention import Step1X3DMultiViewAdapter
from .texture_pipeline import Step1X3DMeshTexturePipeline

SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_BASE_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
SDXL_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
SDXL_VAE_REVISION = "207b116dae70ace3637169f1ddd2434b91b3a8cd"
STEP1X3D_TEXTURE_WEIGHTS_NAME = "step1x-3d-ig2v.safetensors"


def shifted_snr_betas(alphas_cumprod: torch.Tensor, shift_scale: float = 8.0) -> torch.Tensor:
    """Released ``ShiftSNRScheduler`` ``"interpolated"`` betas: log-SNR moves from the base schedule at ``t = 0``
    to the base SNR divided by ``shift_scale`` at the last timestep."""

    # The operation order follows upstream so the betas match it bit for bit.
    snr = (alphas_cumprod**0.5 / (1 - alphas_cumprod) ** 0.5) ** 2
    weighting = torch.arange(len(alphas_cumprod)).float() / (len(alphas_cumprod) - 1)
    snr = torch.exp(torch.log(snr) * (1 - weighting) + torch.log(snr / shift_scale) * weighting)
    alphas_cumprod = ((snr / (1 + snr)) ** 0.5) ** 2
    return 1 - alphas_cumprod / torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]])


def convert_step1x3d_texture_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    base_model: str | Path = SDXL_BASE_MODEL,
    base_revision: str | None = SDXL_BASE_REVISION,
    vae_model: str | Path = SDXL_VAE_MODEL,
    vae_revision: str | None = SDXL_VAE_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Write a ``Step1X3DMeshTexturePipeline`` folder from a local copy of ``Step1X-3D-Texture``.

    The SDXL base (CreativeML Open RAIL++-M) is redistributed in the output, so its ``LICENSE.md`` is copied there
    as ``LICENSE-SDXL.md``.
    """

    def base(component_type, subfolder):
        return component_type.from_pretrained(base_model, subfolder=subfolder, revision=base_revision)

    unet = base(UNet2DConditionModel, "unet")
    release = load_file(Path(source_directory) / STEP1X3D_TEXTURE_WEIGHTS_NAME)
    channels = unet.config.block_out_channels
    with init_empty_weights():
        adapter = T2IAdapter(
            in_channels=6,
            channels=(*channels, channels[-1]),
            num_res_blocks=len({key.split(".")[4] for key in release if key.startswith("adapter.body.0.resnets.")}),
            downscale_factor=math.isqrt(release["adapter.conv_in.weight"].shape[1] // 6),
            adapter_type="full_adapter_xl",
        )
        mv_adapter = Step1X3DMultiViewAdapter(hidden_sizes=Step1X3DMultiViewAdapter.hidden_sizes_for(unet))
    unexpected = set(release) - set(adapter.state_dict()) - set(mv_adapter.state_dict())
    if unexpected:
        raise ValueError(f"unexpected Step1X-3D texture keys {sorted(unexpected)[:8]} ({len(unexpected)} in total)")
    adapter.load_state_dict({key: value for key, value in release.items() if key.startswith("adapter.")}, assign=True)
    mv_adapter.load_state_dict(
        {key: value for key, value in release.items() if not key.startswith("adapter.")}, assign=True
    )

    scheduler_config = DDPMScheduler.load_config(base_model, subfolder="scheduler", revision=base_revision)
    betas = shifted_snr_betas(DDPMScheduler.from_config(scheduler_config).alphas_cumprod)
    scheduler = DDPMScheduler.from_config(scheduler_config, trained_betas=betas.tolist())

    pipeline = Step1X3DMeshTexturePipeline(
        vae=AutoencoderKL.from_pretrained(vae_model, revision=vae_revision),
        text_encoder=base(CLIPTextModel, "text_encoder"),
        text_encoder_2=base(CLIPTextModelWithProjection, "text_encoder_2"),
        tokenizer=base(CLIPTokenizer, "tokenizer"),
        tokenizer_2=base(CLIPTokenizer, "tokenizer_2"),
        unet=unet,
        adapter=adapter.eval(),
        mv_adapter=mv_adapter.eval(),
        scheduler=scheduler,
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    local_license = Path(base_model) / "LICENSE.md"
    shutil.copyfile(
        local_license
        if local_license.is_file()
        else hf_hub_download(str(base_model), "LICENSE.md", revision=base_revision),
        destination / "LICENSE-SDXL.md",
    )
    report = {
        "reference_revision": STEP1X3D_REFERENCE_REVISION,
        "checkpoint_revision_checked": STEP1X3D_CHECKPOINT_REVISION,
        "source": str(Path(source_directory).resolve()),
        "base_model": [str(base_model), base_revision],
        "vae_model": [str(vae_model), vae_revision],
        "scheduler": "DDPMScheduler with interpolated shifted-SNR betas (shift_scale 8)",
    }
    (destination / "step1x3d_texture_conversion.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path, help="local copy of Step1X-3D-Texture")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--base-model", default=SDXL_BASE_MODEL)
    parser.add_argument("--base-revision", default=SDXL_BASE_REVISION)
    parser.add_argument("--vae-model", default=SDXL_VAE_MODEL)
    parser.add_argument("--vae-revision", default=SDXL_VAE_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_step1x3d_texture_checkpoint(
        args.source_directory,
        args.output_directory,
        base_model=args.base_model,
        base_revision=args.base_revision,
        vae_model=args.vae_model,
        vae_revision=args.vae_revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SDXL_BASE_MODEL",
    "SDXL_BASE_REVISION",
    "SDXL_VAE_MODEL",
    "SDXL_VAE_REVISION",
    "convert_step1x3d_texture_checkpoint",
    "main",
    "shifted_snr_betas",
]
