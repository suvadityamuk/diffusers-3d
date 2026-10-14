"""Convert the released Hunyuan3D-Paint 2.1 checkpoint (``hunyuan3d-paintpbr-v2-1``) into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import BitImageProcessorPil, Dinov2Model

from .conversion import HUNYUAN3D_HUB_REPOSITORY, HUNYUAN3D_HUB_REVISION, HUNYUAN3D_REFERENCE_REVISION
from .paint_models import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
from .paint_pipeline import Hunyuan3DPaintPipeline

HUNYUAN3D_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"
DINOV2_REPOSITORY = "facebook/dinov2-giant"
DINOV2_REVISION = "611a9d42f2335e0f921f1e313ad3c1b7178d206d"
# A Hub copy of Real-ESRGAN's release asset RealESRGAN_x4plus.pth, checked against its SHA-256.
REALESRGAN_REPOSITORY = "lllyasviel/Annotators"
REALESRGAN_REVISION = "982e7edaec38759d914a963c48c4726685de7d96"
REALESRGAN_FILENAME = "RealESRGAN_x4plus.pth"
REALESRGAN_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

# Released UNet config values this port fixes, and the ones it maps onto Hunyuan3DPaintUNet.
_UNET_FIXED = {
    "_class_name": "UNet2DConditionModel",
    "act_fn": "silu",
    "center_input_sample": False,
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "use_linear_projection": True,
}
_UNET_MAPPED = {
    "sample_size": "sample_size",
    "out_channels": "latent_channels",
    "down_block_types": "down_block_types",
    "up_block_types": "up_block_types",
    "block_out_channels": "block_out_channels",
    "layers_per_block": "layers_per_block",
    "attention_head_dim": "attention_head_dim",
    "cross_attention_dim": "cross_attention_dim",
    "norm_num_groups": "norm_num_groups",
}
_UNET_RENAMES = (
    (r"^unet_dual\.", "reference_unet."),
    (r"\.transformer_blocks\.(\d+)\.transformer\.", r".transformer_blocks.\1."),
    (r"\.processor\.to_out_mr\.0\.", ".processor.to_out_mr."),
    (r"^unet\.image_proj_model_dino\.proj\.", "image_projection.image_embeds."),
    (r"^unet\.image_proj_model_dino\.norm\.", "image_projection.norm."),
)


def unet_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map the released ``unet/config.json`` onto :class:`Hunyuan3DPaintUNet`, rejecting unsupported options."""

    unknown = set(config) - set(_UNET_FIXED) - set(_UNET_MAPPED) - {"_diffusers_version"}
    unsupported = {key: config.get(key) for key, value in _UNET_FIXED.items() if config.get(key, value) != value}
    if unknown or unsupported:
        raise ValueError(f"unsupported UNet configuration: unknown={sorted(unknown)}, unsupported={unsupported}")
    return {target: config[source] for source, target in _UNET_MAPPED.items()}


def convert_unet_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a released ``UNet2p5DConditionModel`` state dict onto :class:`Hunyuan3DPaintUNet`."""

    converted = {}
    for key, value in state_dict.items():
        for pattern, replacement in _UNET_RENAMES:
            key = re.sub(pattern, replacement, key)
        converted[key] = value
    materials = (converted.pop("unet.learned_text_clip_albedo"), converted.pop("unet.learned_text_clip_mr"))
    converted["material_embeds"] = torch.stack(materials)
    converted["reference_embeds"] = converted.pop("unet.learned_text_clip_ref")
    return converted


def _load_upscaler() -> Hunyuan3DPaintRRDBNet:
    path = Path(hf_hub_download(REALESRGAN_REPOSITORY, REALESRGAN_FILENAME, revision=REALESRGAN_REVISION))
    if hashlib.sha256(path.read_bytes()).hexdigest() != REALESRGAN_SHA256:
        raise ValueError(f"{path} is not the released {REALESRGAN_FILENAME}")
    upscaler = Hunyuan3DPaintRRDBNet()
    upscaler.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params_ema"], strict=True)
    return upscaler.eval()


def convert_hunyuan3d_paint_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str = HUNYUAN3D_HUB_REVISION,
    upscaler: bool = True,
    safe_serialization: bool = True,
) -> Path:
    """Convert the released Paint model into a :class:`Hunyuan3DPaintPipeline` folder.

    ``source`` is a local folder holding ``unet/``, ``vae/``, and ``scheduler/`` (the release's
    ``hunyuan3d-paintpbr-v2-1`` folder or its parent), or a Hub repository id downloaded at ``revision``. The
    ``.bin`` checkpoints are read with ``torch.load(weights_only=True)`` and keep their dtype. The unused CLIP text
    and image encoders are dropped; DINOv2-giant and, unless ``upscaler`` is false, Real-ESRGAN x4plus are fetched
    from the Hub at pinned revisions. The DDIM scheduler config becomes the ``UniPCMultistepScheduler`` with
    trailing timesteps that the release samples with.
    """

    root = Path(source)
    if not root.is_dir():
        patterns = [f"{HUNYUAN3D_PAINT_SUBFOLDER}/{name}" for name in ("unet/*", "vae/*", "scheduler/*")]
        root = Path(snapshot_download(str(source), revision=revision, allow_patterns=patterns, ignore_patterns="*.py"))
    folder = root / HUNYUAN3D_PAINT_SUBFOLDER if (root / HUNYUAN3D_PAINT_SUBFOLDER).is_dir() else root

    config = unet_config(json.loads((folder / "unet" / "config.json").read_text(encoding="utf-8")))
    state_dict = torch.load(
        folder / "unet" / "diffusion_pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    with init_empty_weights():
        unet = Hunyuan3DPaintUNet(**config)
    unet.load_state_dict(convert_unet_state_dict(state_dict), strict=True, assign=True)
    scheduler_config = DDIMScheduler.load_config(folder / "scheduler")
    pipeline = Hunyuan3DPaintPipeline(
        feature_extractor=BitImageProcessorPil.from_pretrained(DINOV2_REPOSITORY, revision=DINOV2_REVISION),
        image_encoder=Dinov2Model.from_pretrained(DINOV2_REPOSITORY, revision=DINOV2_REVISION).eval(),
        unet=unet.eval(),
        vae=AutoencoderKL.from_pretrained(folder / "vae", use_safetensors=False).eval(),
        scheduler=UniPCMultistepScheduler.from_config(scheduler_config, timestep_spacing="trailing"),
        upscaler=_load_upscaler() if upscaler else None,
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    report = {
        "reference_revision": HUNYUAN3D_REFERENCE_REVISION,
        "source": str(folder.resolve()),
        "hub_revision": revision if not Path(source).is_dir() else None,
        "image_encoder": {"repository": DINOV2_REPOSITORY, "revision": DINOV2_REVISION},
        "upscaler": (
            {"repository": REALESRGAN_REPOSITORY, "revision": REALESRGAN_REVISION, "sha256": REALESRGAN_SHA256}
            if upscaler
            else None
        ),
        "dropped_components": ["feature_extractor (CLIP)", "image_encoder (CLIP)", "text_encoder", "tokenizer"],
    }
    (destination / "hunyuan3d_paint_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help=f"local release folder or Hub repository id, e.g. {HUNYUAN3D_HUB_REPOSITORY}")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=HUNYUAN3D_HUB_REVISION)
    parser.add_argument("--no-upscaler", action="store_true", help="omit the optional Real-ESRGAN upscaler")
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_hunyuan3d_paint_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        upscaler=not args.no_upscaler,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DINOV2_REVISION",
    "REALESRGAN_SHA256",
    "convert_hunyuan3d_paint_checkpoint",
    "convert_unet_state_dict",
    "main",
    "unet_config",
]
