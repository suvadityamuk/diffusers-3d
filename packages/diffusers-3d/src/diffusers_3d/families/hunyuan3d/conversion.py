"""Convert the released Hunyuan3D-2.1 shape checkpoint (``hunyuan3d-dit-v2-1``) into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Dinov2Config, Dinov2Model

from .models import Hunyuan3DDiTModel
from .pipeline import Hunyuan3DImageTo3DPipeline
from .scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from .vae import Hunyuan3DShapeVAE

HUNYUAN3D_REFERENCE_REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
HUNYUAN3D_HUB_REPOSITORY = "tencent/Hunyuan3D-2.1"
HUNYUAN3D_HUB_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
HUNYUAN3D_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"

# Released config values for options this port does not implement, and the options it maps.
_TRANSFORMER_FIXED = {
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": False,
    "with_decoupled_ca": False,
    "use_attention_pooling": False,
    "use_pos_emb": False,
}
_TRANSFORMER_MAPPED = {
    "in_channels",
    "hidden_size",
    "context_dim",
    "depth",
    "num_heads",
    "num_moe_layers",
    "num_experts",
    "moe_top_k",
}
# ``input_size`` and ``text_len`` only size the unused positional embedding and attention pooling.
_TRANSFORMER_IGNORED = {"input_size", "text_len"}
_VAE_FIXED = {
    "include_pi": False,
    "qkv_bias": False,
    "qk_norm": True,
    "geo_decoder_mlp_expand_ratio": 4,
    "geo_decoder_downsample_ratio": 1,
    "geo_decoder_ln_post": True,
}
_VAE_MAPPED = {"num_latents", "embed_dim", "width", "heads", "num_decoder_layers", "num_freqs", "scale_factor"}
# Point-cloud encoder options; generation only decodes.
_VAE_IGNORED = {"num_encoder_layers", "point_feats", "pc_size", "pc_sharpedge_size"}

_TRANSFORMER_RENAMES = (
    (r"^blocks\.", "transformer_blocks."),
    (r"^t_embedder\.mlp\.0\.", "time_embedding.linear_1."),
    (r"^t_embedder\.mlp\.2\.", "time_embedding.linear_2."),
    (r"^x_embedder\.", "proj_in."),
    (r"^final_layer\.norm_final\.", "norm_out."),
    (r"^final_layer\.linear\.", "proj_out."),
    (r"\.q_norm\.", ".norm_q."),
    (r"\.k_norm\.", ".norm_k."),
    (r"\.out_proj\.", ".to_out."),
    (r"\.mlp\.fc1\.", ".ff.net.0.proj."),
    (r"\.mlp\.fc2\.", ".ff.net.2."),
    (r"\.moe\.", ".ff."),
)
_VAE_RENAMES = (
    (r"^transformer\.resblocks\.", "transformer_blocks."),
    (r"^geo_decoder\.cross_attn_decoder\.", "decoder_block."),
    (r"^geo_decoder\.ln_post\.", "norm_out."),
    (r"^geo_decoder\.output_proj\.", "proj_out."),
    (r"^geo_decoder\.", ""),
    (r"\.ln_(\d)\.", lambda match: f".norm{match.group(1)}."),
    (r"\.attn\.attention\.q_norm\.", ".attn.norm_q."),
    (r"\.attn\.attention\.k_norm\.", ".attn.norm_k."),
    (r"\.attn\.c_qkv\.", ".attn.to_qkv."),
    (r"\.attn\.c_kv\.", ".attn.to_kv."),
    (r"\.attn\.c_q\.", ".attn.to_q."),
    (r"\.attn\.c_proj\.", ".attn.to_out."),
    (r"\.mlp\.c_fc\.", ".ff.net.0.proj."),
    (r"\.mlp\.c_proj\.", ".ff.net.2."),
)


def _rename(state_dict: Mapping[str, torch.Tensor], renames: tuple) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state_dict.items():
        for pattern, replacement in renames:
            key = re.sub(pattern, replacement, key)
        renamed[key] = value
    return renamed


def _split_fused_heads(state_dict: dict[str, torch.Tensor], heads: int) -> dict[str, torch.Tensor]:
    """Split fused ``to_qkv`` / ``to_kv`` rows the way the released attention does: by head first, then q/k/v."""

    for key in [key for key in state_dict if ".to_qkv." in key or ".to_kv." in key]:
        fused_name = ".to_qkv." if ".to_qkv." in key else ".to_kv."
        names = ("to_q", "to_k", "to_v") if fused_name == ".to_qkv." else ("to_k", "to_v")
        parts = state_dict.pop(key).unflatten(0, (heads, len(names), -1)).unbind(1)
        for name, part in zip(names, parts):
            state_dict[key.replace(fused_name, f".{name}.")] = part.flatten(0, 1)
    return state_dict


def convert_transformer_state_dict(state_dict: Mapping[str, torch.Tensor], num_heads: int) -> dict[str, torch.Tensor]:
    """Map a released ``HunYuanDiTPlain`` state dict onto :class:`Hunyuan3DDiTModel`."""

    converted = _rename(state_dict, _TRANSFORMER_RENAMES)
    for key in [key for key in converted if ".attn1.to_q." in key or ".attn2.to_k." in key]:
        prefix, suffix = key.split(".to_q." if ".attn1." in key else ".to_k.")
        names = ("to_q", "to_k", "to_v") if ".attn1." in key else ("to_k", "to_v")
        fused = ".to_qkv." if len(names) == 3 else ".to_kv."
        converted[f"{prefix}{fused}{suffix}"] = torch.cat(
            [converted.pop(f"{prefix}.{name}.{suffix}") for name in names]
        )
    return _split_fused_heads(converted, num_heads)


def convert_vae_state_dict(state_dict: Mapping[str, torch.Tensor], heads: int) -> dict[str, torch.Tensor]:
    """Map the decoder half of a released ``ShapeVAE`` state dict onto :class:`Hunyuan3DShapeVAE`."""

    decoder = {key: value for key, value in state_dict.items() if not key.startswith(("encoder.", "pre_kl."))}
    return _split_fused_heads(_rename(decoder, _VAE_RENAMES), heads)


def _model_config(section: Mapping[str, Any], fixed: Mapping[str, Any], mapped: set[str], ignored: set[str]):
    params = dict(section["params"])
    unknown = set(params).difference(fixed, mapped, ignored)
    unsupported = {key: params.get(key) for key, value in fixed.items() if params.get(key) != value}
    if unknown or unsupported:
        raise ValueError(f"unsupported configuration: unknown={sorted(unknown)}, unsupported={unsupported}")
    return {key: params[key] for key in mapped}


def _load_checkpoint(folder: Path) -> dict[str, dict[str, torch.Tensor]]:
    paths = sorted(folder.glob("model*.safetensors")) or sorted(folder.glob("model*.ckpt"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one model*.safetensors or model*.ckpt file in {folder}, found {paths}")
    if paths[0].suffix == ".ckpt":
        return torch.load(paths[0], map_location="cpu", weights_only=True, mmap=True)
    checkpoint: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(paths[0]).items():
        component, _, name = key.partition(".")
        checkpoint.setdefault(component, {})[name] = value
    return checkpoint


def _materialize(model_type: type[Any], config: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]):
    with init_empty_weights():
        model = model_type(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_hunyuan3d_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str = HUNYUAN3D_HUB_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Convert the released shape model into a :class:`Hunyuan3DImageTo3DPipeline` folder.

    ``source`` is a local folder holding ``config.yaml`` and the aggregate ``model.fp16.ckpt`` (the release's
    ``hunyuan3d-dit-v2-1`` folder or its parent), or a Hub repository id downloaded at ``revision``. Weights keep
    their stored dtype. The conditioner's DINOv2 is Meta's ``facebook/dinov2-large`` cast to float16, so it is not
    written: the pipeline loads it from Meta's repository. The point-cloud encoder of the ShapeVAE is dropped.
    """

    root = Path(source)
    if not root.is_dir():
        root = Path(
            snapshot_download(str(source), revision=revision, allow_patterns=[f"{HUNYUAN3D_SHAPE_SUBFOLDER}/*"])
        )
    folder = root / HUNYUAN3D_SHAPE_SUBFOLDER if (root / HUNYUAN3D_SHAPE_SUBFOLDER).is_dir() else root
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    if config["pipeline"]["target"] != "hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline":
        raise ValueError("only the released Hunyuan3DDiTFlowMatchingPipeline configuration is supported")
    encoder = config["conditioner"]["params"]["main_image_encoder"]
    if encoder["type"] != "DinoImageEncoder" or not encoder["kwargs"].get("use_cls_token", True):
        raise ValueError("only the released DINOv2 image encoder with its CLS token is supported")
    scheduler_params = config["scheduler"]["params"]
    if set(scheduler_params) != {"num_train_timesteps"}:
        raise ValueError(f"unsupported scheduler parameters: {sorted(scheduler_params)}")

    transformer_config = _model_config(config["model"], _TRANSFORMER_FIXED, _TRANSFORMER_MAPPED, _TRANSFORMER_IGNORED)
    vae_config = _model_config(config["vae"], _VAE_FIXED, _VAE_MAPPED, _VAE_IGNORED)
    dinov2_config = Dinov2Config.from_dict(
        {key: value for key, value in encoder["kwargs"]["config"].items() if key != "torch_dtype"}
    )
    if encoder["kwargs"]["image_size"] != dinov2_config.image_size:
        raise ValueError("the DINOv2 input size must match its configured image_size")
    image_processor = config["image_processor"]["params"]

    checkpoint = _load_checkpoint(folder)
    with init_empty_weights():
        image_encoder = Dinov2Model(dinov2_config)
    pipeline = Hunyuan3DImageTo3DPipeline(
        image_encoder=image_encoder.eval(),
        transformer=_materialize(
            Hunyuan3DDiTModel,
            transformer_config,
            convert_transformer_state_dict(checkpoint["model"], transformer_config["num_heads"]),
        ),
        vae=_materialize(
            Hunyuan3DShapeVAE, vae_config, convert_vae_state_dict(checkpoint["vae"], vae_config["heads"])
        ),
        scheduler=Hunyuan3DFlowMatchEulerDiscreteScheduler(**scheduler_params),
        image_size=image_processor["size"],
        border_ratio=image_processor["border_ratio"],
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    report = {
        "reference_revision": HUNYUAN3D_REFERENCE_REVISION,
        "source": str(folder.resolve()),
        "hub_revision": revision if not Path(source).is_dir() else None,
        "dropped_vae_encoder_keys": sorted(
            key for key in checkpoint["vae"] if key.startswith(("encoder.", "pre_kl."))
        ),
    }
    (destination / "hunyuan3d_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help=f"local release folder or Hub repository id, e.g. {HUNYUAN3D_HUB_REPOSITORY}")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=HUNYUAN3D_HUB_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_hunyuan3d_checkpoint(
        args.source, args.output_directory, revision=args.revision, safe_serialization=not args.no_safe_serialization
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HUNYUAN3D_HUB_REVISION",
    "HUNYUAN3D_REFERENCE_REVISION",
    "convert_hunyuan3d_checkpoint",
    "convert_transformer_state_dict",
    "convert_vae_state_dict",
    "main",
]
