"""Convert the released speridlabs/idol checkpoints into a diffusers-3d pipeline folder.

IDOL keeps TripoSplat's networks unchanged, so its denoiser and decoders load into the TripoSplat classes through the
TripoSplat key renames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from ..trellis2.models import DINOV3_VITH16PLUS_REPO
from ..triposplat.conversion import (
    DINOV3_VITH16_CONFIG,
    convert_triposplat_decoder_state_dict,
    convert_triposplat_transformer_state_dict,
)
from ..triposplat.decoder import TripoSplatGaussianDecoder
from ..triposplat.models import FLUX2_REPO, TripoSplatImageVAE, TripoSplatTransformerModel
from .pipeline import IdolImageTo3DPipeline

IDOL_REFERENCE_REVISION = "4702a7f00850e34b6e83a2170cc123c3e9570744"
IDOL_WEIGHTS_REPOSITORY = "speridlabs/idol"
IDOL_WEIGHTS_REVISION = "2731c2291fc65a797c5cb627a5e5526fc1490902"
# BiRefNet (birefnet.safetensors) is not converted; the pipeline loads ZhengPeng7/BiRefNet, the same tensors.
_FILES = ("config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors")


def convert_idol_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename upstream ``LatentSeqMMFlowModel`` keys and drop ``pos_pe``, which the transformer recomputes.

    Upstream stores its latent token positions; they must equal TripoSplat's scrambled Sobol points (seed 123).
    """

    state_dict = dict(state_dict)
    positions = state_dict.pop("pos_pe").reshape(-1, 3).float()
    expected = torch.quasirandom.SobolEngine(3, scramble=True, seed=123).draw(len(positions), dtype=torch.float32)
    if not torch.equal(positions, expected):
        raise ValueError("pos_pe is not TripoSplat's Sobol sequence, which TripoSplatTransformerModel recomputes")
    return convert_triposplat_transformer_state_dict(state_dict)


def convert_idol_decoder_state_dict(
    octree: dict[str, torch.Tensor], gaussian: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Merge upstream's octree (``decoder``) and Gaussian (``decoder_gs``) decoders into one decoder state dict."""

    merged = {f"octree.{key}": value for key, value in octree.items()}
    merged.update({f"gs.{key}": value for key, value in gaussian.items()})
    return convert_triposplat_decoder_state_dict(merged)


def idol_component_configs(config: dict) -> dict[str, dict]:
    """Map upstream ``config.json`` to ``transformer`` and ``decoder`` configs, rejecting what they can't express."""

    denoiser, octree, gaussian = (config[name]["args"] for name in ("denoiser", "decoder", "decoder_gs"))
    representation = gaussian["representation_config"]
    rates = representation["lr"]
    if {key: value for key, value in rates.items() if key != "_rotation"} != dict.fromkeys(
        rates.keys() - {"_rotation"}, 1.0
    ):
        raise ValueError("only the rotation learning-rate scale may differ from 1.0")
    if denoiser.get("num_head_channels", 64) * denoiser["num_heads"] != denoiser["model_channels"]:
        raise ValueError("denoiser num_head_channels * num_heads must equal model_channels")
    shared = ("model_channels", "cond_channels", "num_heads", "mlp_ratio")
    if any(octree[key] != gaussian[key] for key in shared) or gaussian["in_channels"] != 3:
        raise ValueError("the octree and Gaussian decoders must share width, heads, MLP ratio, and latent channels")
    return {
        "transformer": {
            "in_channels": denoiser["in_channels"],
            "camera_channels": denoiser["cam_channels"],
            "num_latent_tokens": denoiser["q_token_length"],
            "width": denoiser["model_channels"],
            "num_layers": denoiser["num_blocks"],
            "num_refiner_layers": denoiser["num_refiner_blocks"],
            "num_attention_heads": denoiser["num_heads"],
            "mlp_ratio": denoiser["mlp_ratio"],
            "cross_attention_dim": denoiser["cond_channels"],
            "cross_attention_dim_2": denoiser["cond2_channels"],
        },
        "decoder": {
            "latent_channels": gaussian["cond_channels"],
            "width": gaussian["model_channels"],
            "num_attention_heads": gaussian["num_heads"],
            "num_octree_layers": octree["num_blocks"],
            "num_gaussian_layers": gaussian["num_blocks"],
            "mlp_ratio": gaussian["mlp_ratio"],
            "octree_depth": config["max_voxel_level"],
            "gaussians_per_token": representation["num_gaussians"],
            "offset_extent": representation["perturbe_size"],
            "scaling_bias": representation["scaling_bias"],
            "opacity_bias": representation["opacity_bias"],
            "rotation_scale": rates["_rotation"],
            "filter_kernel_size": representation["3d_filter_kernel_size"],
        },
    }


def convert_idol_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = IDOL_WEIGHTS_REVISION,
    encoder_configs: dict[str, dict] | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Convert a local IDOL snapshot, or download ``source`` from the Hub at ``revision``.

    Tensors keep their released precision, and upstream's shifted Euler sampler becomes
    ``FlowMatchEulerDiscreteScheduler(shift=3.0)``. The gated DINOv3 ViT-H/16+ and FLUX.2-dev VAE that ``config.json``
    names are not downloaded: the pipeline loads them from those repositories at run time, so they are built here as
    empty shells that ``model_index.json`` names but whose weights are never saved. The bundled BiRefNet background
    remover is not converted. ``encoder_configs`` overrides the architecture of those shells (``image_encoder``, ``vae``).
    """

    root = Path(source)
    if not root.is_dir():
        root = Path(snapshot_download(str(source), revision=revision, allow_patterns=list(_FILES)))
    config = json.loads((root / "config.json").read_text())
    configs = idol_component_configs(config)
    encoders = config["image_encoders"]
    expected = {"dinov3": DINOV3_VITH16PLUS_REPO, "flux2_vae": FLUX2_REPO}
    if {name: encoders.get(name) for name in expected} != expected:
        raise ValueError(f"config.json must name the image encoders the pipeline loads, {expected}; got {encoders}")

    shells = {"image_encoder": DINOV3_VITH16_CONFIG, "vae": {"batch_norm_eps": 1e-4, "clamp_logvar": True}}
    shells.update(encoder_configs or {})
    with init_empty_weights():
        image_encoder = DINOv3ViTModel(DINOv3ViTConfig(**shells["image_encoder"]))
        vae = TripoSplatImageVAE(**shells["vae"])
        transformer = TripoSplatTransformerModel(**configs["transformer"])
        decoder = TripoSplatGaussianDecoder(**configs["decoder"])
    transformer_state = convert_idol_transformer_state_dict(load_file(root / "denoiser.safetensors"))
    decoder_state = convert_idol_decoder_state_dict(
        load_file(root / "decoder.safetensors"), load_file(root / "decoder_gs.safetensors")
    )
    transformer.load_state_dict(transformer_state, strict=True, assign=True)
    decoder.load_state_dict(decoder_state, strict=True, assign=True)
    pipeline = IdolImageTo3DPipeline(
        image_encoder=image_encoder.eval(),
        vae=vae.eval(),
        transformer=transformer.eval(),
        decoder=decoder.eval(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local snapshot directory or Hub repository id (e.g. speridlabs/idol)")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=IDOL_WEIGHTS_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_idol_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IDOL_REFERENCE_REVISION",
    "IDOL_WEIGHTS_REPOSITORY",
    "IDOL_WEIGHTS_REVISION",
    "convert_idol_checkpoint",
    "convert_idol_decoder_state_dict",
    "convert_idol_transformer_state_dict",
    "idol_component_configs",
    "main",
]
