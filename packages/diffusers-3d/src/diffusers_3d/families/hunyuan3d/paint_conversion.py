"""Convert the released Hunyuan3D-Paint 2.1 checkpoint (``hunyuan3d-paintpbr-v2-1``) into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from huggingface_hub import snapshot_download
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

from .conversion import HUNYUAN3D_HUB_REPOSITORY, HUNYUAN3D_HUB_REVISION, HUNYUAN3D_REFERENCE_REVISION
from .paint_components import (
    DINOV2_REPOSITORY,
    DINOV2_REVISION,
    REALESRGAN_SHA256,
    REALESRGAN_URL,
    SD21_REPOSITORY,
    SD21_REVISION,
)
from .paint_models import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
from .paint_pipeline import Hunyuan3DPaintPipeline

HUNYUAN3D_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"

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


def convert_hunyuan3d_paint_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str = HUNYUAN3D_HUB_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Convert the released Paint model into a :class:`Hunyuan3DPaintPipeline` folder.

    ``source`` is a local folder holding ``unet/`` and ``scheduler/`` (the release's ``hunyuan3d-paintpbr-v2-1``
    folder or its parent), or a Hub repository id downloaded at ``revision``. The UNet ``.bin`` checkpoint is read
    with ``torch.load(weights_only=True)`` and keeps its dtype. Only the UNet and the scheduler are written. The
    unused CLIP text and image encoders are dropped; the DINOv2-giant processor and encoder, the VAE (the release's
    copy is Stable Diffusion 2.1's, tensor for tensor), and Real-ESRGAN x4plus are third-party and load from their
    original sources (:mod:`.paint_components`), so they are written as empty entries of ``model_index.json``. The
    DDIM scheduler config becomes the ``UniPCMultistepScheduler`` with trailing timesteps that the release samples
    with.
    """

    root = Path(source)
    if not root.is_dir():
        patterns = [f"{HUNYUAN3D_PAINT_SUBFOLDER}/{name}" for name in ("unet/*", "scheduler/*")]
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
    # Weightless shells of the third-party components; save_pretrained records their classes only.
    with init_empty_weights():
        image_encoder = Dinov2Model(Dinov2Config.from_pretrained(DINOV2_REPOSITORY, revision=DINOV2_REVISION))
        vae = AutoencoderKL.from_config(
            AutoencoderKL.load_config(SD21_REPOSITORY, subfolder="vae", revision=SD21_REVISION)
        )
        upscaler = Hunyuan3DPaintRRDBNet()
    pipeline = Hunyuan3DPaintPipeline(
        feature_extractor=BitImageProcessorPil.from_pretrained(DINOV2_REPOSITORY, revision=DINOV2_REVISION),
        image_encoder=image_encoder,
        unet=unet.eval(),
        vae=vae,
        scheduler=UniPCMultistepScheduler.from_config(scheduler_config, timestep_spacing="trailing"),
        upscaler=upscaler,
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    report = {
        "reference_revision": HUNYUAN3D_REFERENCE_REVISION,
        "source": str(folder.resolve()),
        "hub_revision": revision if not Path(source).is_dir() else None,
        "external_components": {
            "feature_extractor": {"repository": DINOV2_REPOSITORY, "revision": DINOV2_REVISION},
            "image_encoder": {"repository": DINOV2_REPOSITORY, "revision": DINOV2_REVISION},
            "vae": {"repository": SD21_REPOSITORY, "revision": SD21_REVISION},
            "upscaler": {"url": REALESRGAN_URL, "sha256": REALESRGAN_SHA256},
        },
        "dropped_components": ["feature_extractor (CLIP)", "image_encoder (CLIP)", "text_encoder", "tokenizer"],
    }
    (destination / "hunyuan3d_paint_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help=f"local release folder or Hub repository id, e.g. {HUNYUAN3D_HUB_REPOSITORY}")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=HUNYUAN3D_HUB_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_hunyuan3d_paint_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "convert_hunyuan3d_paint_checkpoint",
    "convert_unet_state_dict",
    "main",
    "unet_config",
]
