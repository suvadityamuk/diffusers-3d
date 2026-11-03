"""Convert the released VAST-AI/TripoSG Diffusers-style snapshot into a diffusers-3d pipeline folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

from .models import TripoSGTransformerModel
from .pipeline import TripoSGImageTo3DPipeline
from .vae import TripoSGVAEDecoder

TRIPOSG_REFERENCE_REVISION = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
TRIPOSG_WEIGHTS_REPOSITORY = "VAST-AI/TripoSG"
TRIPOSG_WEIGHTS_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
# VAE config keys that only describe the encoder half, which the image-to-3D pipeline never runs.
_VAE_ENCODER_CONFIG = {"width_encoder", "num_layers_encoder", "embedding_type"}


def _config(path: Path) -> dict[str, Any]:
    return {
        key: value for key, value in json.loads(path.read_text(encoding="utf-8")).items() if not key.startswith("_")
    }


def _load(model_type: type[torch.nn.Module], config: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    with init_empty_weights():
        model = model_type(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_triposg_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = TRIPOSG_WEIGHTS_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Convert a local TripoSG snapshot, or download ``source`` from the Hub at ``revision``.

    Transformer weights load unchanged. The VAE keeps its decoder: ``encoder.*`` and ``quant.*`` are dropped and
    the ``decoder.`` prefix is removed. The DINOv2 encoder and its preprocessor equal Meta's ``facebook/dinov2-large``
    and are not written; the pipeline loads them from there. The rectified-flow scheduler becomes an equivalent
    ``FlowMatchEulerDiscreteScheduler``.
    """

    root = Path(source)
    if not root.is_dir():
        root = Path(snapshot_download(str(source), revision=revision))
    if json.loads((root / "model_index.json").read_text(encoding="utf-8")).get("_class_name") != "TripoSGPipeline":
        raise ValueError(f"{root} is not a TripoSGPipeline snapshot")

    transformer_config = _config(root / "transformer" / "config.json")
    if transformer_config.pop("use_cross_attention_2", False):
        raise ValueError("TripoSG-scribble (a second cross-attention) is not supported")
    transformer_config.pop("cross_attention_2_dim", None)
    transformer = _load(
        TripoSGTransformerModel,
        transformer_config,
        load_file(root / "transformer" / "diffusion_pytorch_model.safetensors"),
    )

    vae_config = _config(root / "vae" / "config.json")
    if vae_config.get("embedding_type", "frequency") != "frequency":
        raise ValueError("only the frequency positional embedding is supported")
    vae_state = load_file(root / "vae" / "diffusion_pytorch_model.safetensors")
    vae = _load(
        TripoSGVAEDecoder,
        {key: value for key, value in vae_config.items() if key not in _VAE_ENCODER_CONFIG},
        {
            key.removeprefix("decoder."): value
            for key, value in vae_state.items()
            if not key.startswith(("encoder.", "quant."))
        },
    )

    scheduler_config = _config(root / "scheduler" / "scheduler_config.json")
    if scheduler_config.get("use_dynamic_shifting", False):
        raise ValueError("dynamic timestep shifting is not supported")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
        shift=float(scheduler_config.get("shift", 1.0)),
    )

    # The release's DINOv2 and its preprocessor equal Meta's facebook/dinov2-large, so neither is written: the
    # pipeline loads them from Meta's repository.
    with init_empty_weights():
        image_encoder = Dinov2Model(Dinov2Config.from_pretrained(root / "image_encoder_dinov2"))
    pipeline = TripoSGImageTo3DPipeline(
        feature_extractor=BitImageProcessorPil.from_pretrained(root / "feature_extractor_dinov2"),
        image_encoder=image_encoder,
        scheduler=scheduler,
        transformer=transformer,
        vae=vae,
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local snapshot directory or Hub repository id (e.g. VAST-AI/TripoSG)")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=TRIPOSG_WEIGHTS_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_triposg_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TRIPOSG_REFERENCE_REVISION",
    "TRIPOSG_WEIGHTS_REPOSITORY",
    "TRIPOSG_WEIGHTS_REVISION",
    "convert_triposg_checkpoint",
    "main",
]
