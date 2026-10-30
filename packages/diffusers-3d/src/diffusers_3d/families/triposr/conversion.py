"""Convert the released stabilityai/TripoSR checkpoint into a diffusers-3d pipeline folder."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import ViTConfig, ViTModel

from .models import TripoSRTransformerModel, TripoSRTriplaneDecoder
from .pipeline import TripoSRImageTo3DPipeline

TRIPOSR_REFERENCE_REVISION = "107cefdc244c39106fa830359024f6a2f1c78871"
TRIPOSR_WEIGHTS_REPOSITORY = "stabilityai/TripoSR"
TRIPOSR_WEIGHTS_REVISION = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
DINO_REPOSITORY = "facebook/dino-vitb16"
DINO_REVISION = "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2"
# Checkpoint prefix -> (component, prefix in that component).
_KEY_PREFIXES = (
    ("image_tokenizer.model.", "image_encoder", ""),
    ("tokenizer.", "transformer", ""),
    ("backbone.", "transformer", "backbone."),
    ("post_processor.", "transformer", ""),
    ("decoder.", "decoder", ""),
)
# Released renderer/decoder settings the ported decoder hard-codes.
_FIXED_SETTINGS = {
    ("decoder", "activation"): "silu",
    ("renderer", "feature_reduction"): "concat",
    ("renderer", "density_activation"): "exp",
    ("renderer", "color_activation"): "sigmoid",
}


def interpolate_position_embeddings(position_embeddings: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Resample ViT position embeddings to a ``grid_size^2`` patch grid exactly as transformers 4.35 did.

    TripoSR pins transformers 4.35, whose ``interpolate_pos_encoding`` uses bicubic ``scale_factor=(grid + 0.1) /
    side``; transformers 5 interpolates to an exact ``size`` instead, which samples different points. Baking the
    4.35 table into the checkpoint lets the stock ``ViTModel`` skip interpolation at the release resolution.
    """

    class_embedding, patch_embeddings = position_embeddings[:, :1], position_embeddings[:, 1:]
    side = int(math.sqrt(patch_embeddings.shape[1]))
    if grid_size == side:
        return position_embeddings
    scale = (grid_size + 0.1) / math.sqrt(patch_embeddings.shape[1])
    patch_embeddings = F.interpolate(
        patch_embeddings.reshape(1, side, side, -1).permute(0, 3, 1, 2),
        scale_factor=(scale, scale),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat([class_embedding, patch_embeddings.permute(0, 2, 3, 1).flatten(1, 2)], dim=1)


def _split_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    components = {"image_encoder": {}, "transformer": {}, "decoder": {}}
    for key, value in state_dict.items():
        for prefix, component, replacement in _KEY_PREFIXES:
            if key.startswith(prefix):
                components[component][replacement + key.removeprefix(prefix)] = value
                break
        else:
            raise ValueError(f"unexpected TripoSR checkpoint key {key!r}")
    return components


def _load(model_type: type[torch.nn.Module], config: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    with init_empty_weights():
        model = model_type(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_triposr_components(
    config: dict[str, Any], state_dict: dict[str, torch.Tensor], encoder_config: ViTConfig
) -> dict[str, torch.nn.Module]:
    """Build the pipeline components from a parsed ``config.yaml``, a ``model.ckpt`` state dict, and the ViT config.

    The image tokenizer becomes a stock Transformers ``ViTModel`` at the conditioning size, with transformers 4.35's
    position interpolation baked in (see :func:`interpolate_position_embeddings`); Transformers maps its legacy
    tensor names. Every other tensor keeps its values; only key prefixes change.
    """

    for (section, key), expected in _FIXED_SETTINGS.items():
        if config[section].get(key, expected) != expected:
            raise ValueError(f"only {section}.{key}={expected!r} is supported")
    components = _split_state_dict(state_dict)

    encoder_config = ViTConfig.from_dict({**encoder_config.to_dict(), "image_size": config["cond_image_size"]})
    encoder_state = components["image_encoder"]
    encoder_state["embeddings.position_embeddings"] = interpolate_position_embeddings(
        encoder_state["embeddings.position_embeddings"], encoder_config.image_size // encoder_config.patch_size
    )
    with tempfile.TemporaryDirectory() as folder:
        encoder_config.save_pretrained(folder)
        save_file({key: value.contiguous() for key, value in encoder_state.items()}, Path(folder, "model.safetensors"))
        image_encoder = ViTModel.from_pretrained(folder).eval()

    tokenizer, backbone, post_processor = config["tokenizer"], config["backbone"], config["post_processor"]
    transformer = _load(
        TripoSRTransformerModel,
        {
            "triplane_size": tokenizer["plane_size"],
            "num_channels": tokenizer["num_channels"],
            "num_attention_heads": backbone["num_attention_heads"],
            "attention_head_dim": backbone["attention_head_dim"],
            "num_layers": backbone["num_layers"],
            "cross_attention_dim": backbone["cross_attention_dim"],
            "norm_num_groups": backbone.get("norm_num_groups", 32),
            "out_channels": post_processor["out_channels"],
        },
        components["transformer"],
    )
    decoder = _load(
        TripoSRTriplaneDecoder,
        {
            "triplane_channels": post_processor["out_channels"],
            "hidden_size": config["decoder"]["n_neurons"],
            "num_hidden_layers": config["decoder"]["n_hidden_layers"],
            "radius": config["renderer"]["radius"],
            "density_bias": config["renderer"].get("density_bias", -1.0),
        },
        components["decoder"],
    )
    return {"image_encoder": image_encoder, "transformer": transformer, "decoder": decoder}


def convert_triposr_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = TRIPOSR_WEIGHTS_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Convert a folder holding ``config.yaml`` and ``model.ckpt``, or download them from ``source`` at ``revision``.

    The ViT config comes from the repository ``config.yaml`` names (``facebook/dino-vitb16``, pinned), or from a
    local folder it names. See :func:`convert_triposr_components`.
    """

    root = Path(source)
    if root.is_dir():
        config_path, checkpoint_path = root / "config.yaml", root / "model.ckpt"
    else:
        config_path, checkpoint_path = (
            hf_hub_download(str(source), name, revision=revision) for name in ("config.yaml", "model.ckpt")
        )
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    encoder_name = config["image_tokenizer"].get("pretrained_model_name_or_path", DINO_REPOSITORY)
    encoder_config = ViTConfig.from_pretrained(
        encoder_name, revision=DINO_REVISION if encoder_name == DINO_REPOSITORY else None
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pipeline = TripoSRImageTo3DPipeline(**convert_triposr_components(config, state_dict, encoder_config))
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="folder with config.yaml and model.ckpt, or a Hub id (stabilityai/TripoSR)")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=TRIPOSR_WEIGHTS_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_triposr_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DINO_REPOSITORY",
    "DINO_REVISION",
    "TRIPOSR_REFERENCE_REVISION",
    "TRIPOSR_WEIGHTS_REPOSITORY",
    "TRIPOSR_WEIGHTS_REVISION",
    "convert_triposr_checkpoint",
    "convert_triposr_components",
    "interpolate_position_embeddings",
    "main",
]
