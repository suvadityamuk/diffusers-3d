"""Convert a released Step1X-3D geometry folder (``stepfun-ai/Step1X-3D/Step1X-3D-Geometry-1300m``) to Diffusers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from .models import Step1X3DTransformer1DModel
from .pipeline import Step1X3DImageTo3DPipeline
from .scheduler import Step1X3DFlowMatchEulerScheduler
from .vae import Step1X3DShapeDecoder

STEP1X3D_REFERENCE_REVISION = "cb5ac944709c6c913109070c7b90c3447f57f3d4"
STEP1X3D_CHECKPOINT_REVISION = "bf7084495b3a72222f36549b7942948aa4d9daa7"

_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_MLP_KEYS = {"mlp.c_fc": "ff.net.0.proj", "mlp.c_proj": "ff.net.2"}
_VAE_BLOCK_KEYS = {"ln_1": "norm1", "ln_2": "norm3", "attn.c_proj": "attn1.to_out.0", **_MLP_KEYS}
_VAE_DECODER_KEYS = {
    "query_proj": "query_proj",
    "cross_attn_decoder.ln_1": "norm1",
    "cross_attn_decoder.ln_2": "norm_context",
    "cross_attn_decoder.ln_3": "norm2",
    "cross_attn_decoder.attn.c_q": "attn.to_q",
    "cross_attn_decoder.attn.c_proj": "attn.to_out.0",
    **{f"cross_attn_decoder.{key}": value for key, value in _MLP_KEYS.items()},
    "ln_post": "norm_out",
    "output_proj": "proj_out",
}
# The point encoder needs farthest-point sampling (torch_cluster) and is unused by image-to-3D inference.
_VAE_ENCODER_PREFIXES = ("encoder.", "pre_kl.")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return value


def _require(config: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    mismatched = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise ValueError(f"unsupported {name} configuration {mismatched}; expected {expected}")


def _split_heads(weight: torch.Tensor, num_heads: int, parts: int) -> list[torch.Tensor]:
    """Split a fused projection whose output rows are grouped per head (``[q_h, k_h, v_h]`` for each head ``h``)."""

    return [
        part.reshape(-1, weight.shape[-1]) for part in weight.view(num_heads, parts, -1, weight.shape[-1]).unbind(1)
    ]


def convert_step1x3d_vae_state_dict(
    state_dict: dict[str, torch.Tensor], *, num_attention_heads: int
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map the released shape VAE onto ``Step1X3DShapeDecoder``; returns the converted state and the dropped keys."""

    converted: dict[str, torch.Tensor] = {}
    dropped = []
    for key, value in state_dict.items():
        if key.startswith(_VAE_ENCODER_PREFIXES):
            dropped.append(key)
            continue
        module, parameter = key.rsplit(".", 1)
        if module == "post_kl":
            converted[key] = value
        elif module.startswith("transformer.resblocks."):
            index, name = module.removeprefix("transformer.resblocks.").split(".", 1)
            prefix = f"transformer_blocks.{index}"
            if name == "attn.c_qkv":
                for target, part in zip(("to_q", "to_k", "to_v"), _split_heads(value, num_attention_heads, 3)):
                    converted[f"{prefix}.attn1.{target}.{parameter}"] = part
            else:
                converted[f"{prefix}.{_VAE_BLOCK_KEYS[name]}.{parameter}"] = value
        elif module == "decoder.cross_attn_decoder.attn.c_kv":
            for target, part in zip(("to_k", "to_v"), _split_heads(value, num_attention_heads, 2)):
                converted[f"attn.{target}.{parameter}"] = part
        elif module.startswith("decoder.") and module.removeprefix("decoder.") in _VAE_DECODER_KEYS:
            converted[f"{_VAE_DECODER_KEYS[module.removeprefix('decoder.')]}.{parameter}"] = value
        else:
            raise ValueError(f"unexpected Step1X-3D VAE key {key!r}")
    return converted, sorted(dropped)


def convert_step1x3d_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the ``FluxDenoiser`` wrapper prefix and rename the timestep MLP to the Diffusers ``time_embedding``."""

    converted = {}
    for key, value in state_dict.items():
        if not key.startswith("dit_model."):
            raise ValueError(f"unexpected Step1X-3D transformer key {key!r}")
        key = key.removeprefix("dit_model.")
        converted[f"time_embedding.{key.removeprefix('time_proj.')}" if key.startswith("time_proj.") else key] = value
    return converted


def _dinov2_with_registers_large() -> Dinov2WithRegistersConfig:
    """``facebook/dinov2-with-registers-large``, the backbone the released visual encoder was trained from."""

    return Dinov2WithRegistersConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        image_size=518,
        patch_size=14,
        num_register_tokens=4,
        layer_norm_eps=1e-6,
        layerscale_value=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )


def _load(model_type: type[torch.nn.Module], config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> Any:
    with init_empty_weights():
        model = model_type(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_step1x3d_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    dinov2_config: dict[str, Any] | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Convert the default image-only geometry release into a ``Step1X3DImageTo3DPipeline`` folder.

    ``source_directory`` is a local copy of ``Step1X-3D-Geometry-1300m``. Its visual encoder config only names
    ``facebook/dinov2-with-registers-large``, whose architecture is the default ``dinov2_config``. The
    label-conditioned variant uses a DINOv2+CLIP visual encoder and a label encoder that are not ported, so it is
    rejected.
    """

    source = Path(source_directory)
    model_index = _load_json(source / "model_index.json")
    _require(
        model_index,
        {
            "_class_name": "Step1X3DGeometryPipeline",
            "caption_encoder": [None, None],
            "label_encoder": [None, None],
            "visual_encoder": ["step1x3d_geometry.models.conditional_encoders.dinov2_encoder", "Dinov2Encoder"],
        },
        "model_index",
    )

    transformer_config = _load_json(source / "transformer" / "config.json")
    _require(
        transformer_config,
        {
            "multi_condition_type": "in_context",
            "use_visual_condition": True,
            "use_caption_condition": False,
            "use_label_condition": False,
            "visual_condition_dim": transformer_config.get("condition_dim"),
            "n_views": 1,
        },
        "transformer",
    )
    transformer = _load(
        Step1X3DTransformer1DModel,
        {
            "in_channels": transformer_config["input_channels"],
            "width": transformer_config["width"],
            "num_layers": transformer_config["layers"],
            "num_single_layers": transformer_config["num_single_layers"],
            "num_attention_heads": transformer_config["num_heads"],
            "cross_attention_dim": transformer_config["condition_dim"],
        },
        convert_step1x3d_transformer_state_dict(load_file(source / "transformer" / _WEIGHTS_NAME)),
    )

    vae_config = _load_json(source / "vae" / "config.json")
    _require(
        vae_config,
        {"embed_type": "fourier", "qkv_bias": False, "qk_norm": False, "use_flash": True},
        "vae",
    )
    vae_state, dropped_vae_keys = convert_step1x3d_vae_state_dict(
        load_file(source / "vae" / _WEIGHTS_NAME), num_attention_heads=vae_config["heads"]
    )
    vae = _load(
        Step1X3DShapeDecoder,
        {
            "num_latents": vae_config["num_latents"],
            "latent_channels": vae_config["embed_dim"],
            "width": vae_config["width"],
            "num_attention_heads": vae_config["heads"],
            "num_layers": vae_config["num_decoder_layers"],
            "num_freqs": vae_config["num_freqs"],
            "include_pi": vae_config["include_pi"],
            "out_channels": vae_config["out_dim"],
            "scaling_factor": vae_config["z_scale_factor"],
        },
        vae_state,
    )

    visual_config = _load_json(source / "visual_encoder" / "config.json")
    _require(
        visual_config,
        {
            "dino_type": "facebook/dinov2-with-registers-large",
            "encode_camera": False,
            "zero_uncond_embeds": True,
            "normalize_embeds": False,
            "n_views": 1,
        },
        "visual_encoder",
    )
    config = (
        _dinov2_with_registers_large() if dinov2_config is None else Dinov2WithRegistersConfig.from_dict(dinov2_config)
    )
    if visual_config["image_size"] != config.image_size:
        raise ValueError("the visual encoder image_size must match the DINOv2 config")
    # Released keys use the Hub's DINOv2 layout, which Transformers maps onto its current module names.
    encoder_state = {
        key.removeprefix("dino_model."): value
        for key, value in load_file(source / "visual_encoder" / _WEIGHTS_NAME).items()
    }
    image_encoder, loading_info = Dinov2WithRegistersModel.from_pretrained(
        None, config=config, state_dict=encoder_state, output_loading_info=True
    )
    if any(loading_info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"the visual encoder weights do not match the DINOv2 config: {loading_info}")
    scheduler = Step1X3DFlowMatchEulerScheduler.from_config(_load_json(source / "scheduler" / "scheduler_config.json"))

    destination = Path(output_directory)
    pipeline = Step1X3DImageTo3DPipeline(
        image_encoder=image_encoder.eval(), transformer=transformer, vae=vae, scheduler=scheduler
    )
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    report = {
        "reference_revision": STEP1X3D_REFERENCE_REVISION,
        "checkpoint_revision_checked": STEP1X3D_CHECKPOINT_REVISION,
        "source": str(source.resolve()),
        "dropped_vae_keys": dropped_vae_keys,
        "upstream_extraction": {
            "volume_decoder_type": vae_config.get("volume_decoder_type"),
            "surface_extractor_type": vae_config.get("surface_extractor_type"),
            "replaced_by": "dense grid + scikit-image Lewiner marching cubes",
        },
    }
    (destination / "step1x3d_conversion.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_step1x3d_checkpoint(
        args.source_directory, args.output_directory, safe_serialization=not args.no_safe_serialization
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STEP1X3D_CHECKPOINT_REVISION",
    "STEP1X3D_REFERENCE_REVISION",
    "convert_step1x3d_checkpoint",
    "convert_step1x3d_transformer_state_dict",
    "convert_step1x3d_vae_state_dict",
    "main",
]
