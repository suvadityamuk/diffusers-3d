"""Convert a released MoGe ``model.pt`` (``Ruicheng/moge-vitl`` or ``Ruicheng/moge-2-vitl``) into a Diffusers folder."""

from __future__ import annotations

import argparse
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from .models import DINOV2_VITL14_CONFIG, MoGe2Model, MoGeModel, match_dinov2_attention_names

# MoGe code each variant reproduces: SAM 3D Objects pins a8c3734 (before MoGe-2); Pixal3D installs unpinned HEAD,
# which was 74fbce0 (2026-08-19) at Pixal3D's revision f7cf384 (2026-09-01).
MOGE_REFERENCE_REVISION = "a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
MOGE2_REFERENCE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"
MOGE_WEIGHTS_REPOSITORY = "Ruicheng/moge-vitl"
MOGE_WEIGHTS_REVISION = "ad326bfb61facd6c52b5a825bc1e34d7c97d9672"
MOGE2_WEIGHTS_REPOSITORY = "Ruicheng/moge-2-vitl"
MOGE2_WEIGHTS_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
_WEIGHTS_REVISIONS = {MOGE_WEIGHTS_REPOSITORY: MOGE_WEIGHTS_REVISION, MOGE2_WEIGHTS_REPOSITORY: MOGE2_WEIGHTS_REVISION}

# torch-hub DINOv2 names: ``dinov2_vit{s,b,l}14`` (MoGe vendors the loader; ViT-g uses SwiGLU and is not supported).
_DINOV2_CONFIGS = {
    "dinov2_vits14": {**DINOV2_VITL14_CONFIG, "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "dinov2_vitb14": {**DINOV2_VITL14_CONFIG, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "dinov2_vitl14": dict(DINOV2_VITL14_CONFIG),
}
_DINOV2_RULES = (
    (r"^cls_token$", "embeddings.cls_token"),
    (r"^mask_token$", "embeddings.mask_token"),
    (r"^pos_embed$", "embeddings.position_embeddings"),
    (r"^patch_embed\.proj\.", "embeddings.patch_embeddings.projection."),
    (r"^norm\.", "layernorm."),
    (r"^blocks\.(\d+)\.attn\.proj\.", r"encoder.layer.\1.attention.output.dense."),
    (r"^blocks\.(\d+)\.ls(\d)\.gamma$", r"encoder.layer.\1.layer_scale\2.lambda1"),
    (r"^blocks\.", "encoder.layer."),
)
_RES_BLOCK_RULES = (
    (r"\.layers\.0\.", ".norm1."),
    (r"\.layers\.2\.", ".conv1."),
    (r"\.layers\.3\.", ".norm2."),
    (r"\.layers\.5\.", ".conv2."),
)
_MOGE_RULES = (
    (r"^head\.projects\.", "projections."),
    (r"^head\.upsample_blocks\.(\d+)\.0\.0\.", r"upsamplers.\1.conv_transpose."),
    (r"^head\.upsample_blocks\.(\d+)\.0\.1\.", r"upsamplers.\1.conv."),
    # Residual blocks follow the upsampler inside each upstream stage.
    (r"^head\.upsample_blocks\.(\d+)\.(\d+)\.", lambda match: f"res_blocks.{match[1]}.{int(match[2]) - 1}."),
    (r"^head\.output_block\.(\d+)\.0\.", r"output_convs_in.\1."),
    (r"^head\.output_block\.(\d+)\.2\.", r"output_convs_out.\1."),
    *_RES_BLOCK_RULES,
)
_MOGE2_RULES = (
    (r"^encoder\.output_projections\.", "output_projections."),
    (r"\.resamplers\.(\d+)\.0\.", r".resamplers.\1.conv_transpose."),
    (r"\.resamplers\.(\d+)\.1\.", r".resamplers.\1.conv."),
    (r"^(points_head|mask_head)\.output_blocks\.\d+\.", r"\1.output_projection."),
    (r"^scale_head\.(\d+)\.", lambda match: f"scale_head.{int(match[1]) // 2}."),
    *_RES_BLOCK_RULES,
)
_NORMALIZATION_BUFFERS = {
    "image_mean": torch.tensor([0.485, 0.456, 0.406]),
    "image_std": torch.tensor([0.229, 0.224, 0.225]),
}


def _rename(key: str, rules) -> str:
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def convert_dinov2_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map torch-hub DINOv2 keys onto ``transformers.Dinov2Model``, splitting the fused ``qkv`` projection."""

    converted = {}
    for key, value in state_dict.items():
        if match := re.fullmatch(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", key):
            for part, tensor in zip(("query", "key", "value"), value.chunk(3, dim=0)):
                converted[f"encoder.layer.{match[1]}.attention.attention.{part}.{match[2]}"] = tensor
        else:
            converted[_rename(key, _DINOV2_RULES)] = value
    return converted


def _split_checkpoint(
    state_dict: Mapping[str, torch.Tensor], backbone_prefix: str, buffer_prefix: str, rules
) -> dict[str, torch.Tensor]:
    backbone, converted = {}, {}
    for key, value in state_dict.items():
        if key.startswith(backbone_prefix):
            backbone[key[len(backbone_prefix) :]] = value
        elif key.startswith(buffer_prefix) and key[len(buffer_prefix) :] in _NORMALIZATION_BUFFERS:
            expected = _NORMALIZATION_BUFFERS[key[len(buffer_prefix) :]].view_as(value)
            if not torch.equal(value, expected):
                raise ValueError(f"{key} is {value.flatten().tolist()}; expected the ImageNet statistics")
        else:
            converted[_rename(key, rules)] = value
    converted.update({f"backbone.{key}": value for key, value in convert_dinov2_state_dict(backbone).items()})
    return converted


def convert_moge_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename ``moge.model.v1.MoGeModel`` keys to :class:`MoGeModel` keys."""

    return _split_checkpoint(state_dict, "backbone.", "", _MOGE_RULES)


def convert_moge2_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename ``moge.model.v2.MoGeModel`` keys to :class:`MoGe2Model` keys."""

    return _split_checkpoint(state_dict, "encoder.backbone.", "encoder.", _MOGE2_RULES)


def _backbone_config(name: str) -> dict:
    if name not in _DINOV2_CONFIGS:
        raise ValueError(f"unsupported MoGe backbone {name!r}; expected one of {sorted(_DINOV2_CONFIGS)}")
    return _DINOV2_CONFIGS[name]


def moge_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """:class:`MoGeModel` arguments for a v1 ``model_config``, as ``moge.model.v1.MoGeModel.__init__`` reads it."""

    unsupported = {
        key: model_config[key]
        for key, default in (("res_block_norm", "group_norm"), ("last_res_blocks", 0), ("last_conv_size", 1))
        if model_config.get(key, default) != default
    }
    if unsupported:
        raise ValueError(f"unsupported MoGe settings {unsupported}")
    if "trained_area_range" in model_config:  # upstream's legacy argument
        num_tokens_range = [area // 14**2 for area in model_config["trained_area_range"]]
    else:
        num_tokens_range = list(model_config.get("num_tokens_range", [1200, 2500]))
    return {
        "backbone_config": _backbone_config(model_config.get("encoder", "dinov2_vitb14")),
        "intermediate_layers": model_config.get("intermediate_layers", 4),
        "dim_proj": model_config.get("dim_proj", 512),
        "dim_upsample": list(model_config.get("dim_upsample", [256, 128, 128])),
        "dim_times_res_block_hidden": model_config.get("dim_times_res_block_hidden", 1),
        "num_res_blocks": model_config.get("num_res_blocks", 1),
        "last_conv_channels": model_config.get("last_conv_channels", 32),
        "remap_output": model_config.get("remap_output", "linear"),
        "num_tokens_range": num_tokens_range,
        "mask_threshold": model_config.get("mask_threshold", 0.5),
    }


def moge2_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """:class:`MoGe2Model` arguments for a v2 ``model_config`` with point and mask heads and a scale head."""

    encoder, neck, points_head, mask_head = (
        model_config[key] for key in ("encoder", "neck", "points_head", "mask_head")
    )
    if model_config.get("normal_head") is not None or model_config.get("scale_head") is None:
        raise ValueError("only MoGe-2 checkpoints with point, mask and scale heads and no normal head are supported")
    dim_res_blocks, resamplers = list(neck["dim_res_blocks"]), neck["resamplers"]
    resamplers = (
        list(resamplers) if isinstance(resamplers, (list, tuple)) else [resamplers] * (len(dim_res_blocks) - 1)
    )
    for name, stack in (("neck", neck), ("points_head", points_head), ("mask_head", mask_head)):
        # ``ConvStack`` defaults: layer_norm / group_norm residual norms, ReLU, hidden width 1x.
        settings = (
            stack.get("res_block_in_norm", "layer_norm"),
            stack.get("res_block_hidden_norm", "group_norm"),
            stack.get("activation", "relu"),
            stack.get("dim_times_res_block_hidden", 1),
        )
        if settings != ("none", "none", "relu", 1):
            raise ValueError(f"unsupported MoGe-2 {name} settings {stack}")
        if list(stack["dim_res_blocks"]) != dim_res_blocks:
            raise ValueError(f"MoGe-2 {name} dims {stack['dim_res_blocks']} differ from the neck's {dim_res_blocks}")
    if list(points_head["num_res_blocks"]) != list(mask_head["num_res_blocks"]):
        raise ValueError("MoGe-2 point and mask heads must share their block counts")
    return {
        "backbone_config": _backbone_config(encoder["backbone"]),
        "intermediate_layers": list(encoder["intermediate_layers"]),
        "dim_features": encoder["dim_out"],
        "dim_res_blocks": dim_res_blocks,
        "neck_num_res_blocks": list(neck["num_res_blocks"]),
        "head_num_res_blocks": list(points_head["num_res_blocks"]),
        "resamplers": resamplers,
        "scale_head_dims": list(model_config["scale_head"]["dims"]),
        "remap_output": model_config.get("remap_output", "linear"),
        "num_tokens_range": list(model_config.get("num_tokens_range", [1200, 3600])),
    }


def load_moge_checkpoint(source: str | Path, *, revision: str | None = None) -> dict[str, Any]:
    """Load ``model.pt`` from a local file or a Hub repository, with ``torch.load(weights_only=True)``."""

    path = Path(source)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    repository = str(source)
    revision = revision or _WEIGHTS_REVISIONS.get(repository)
    # Download into an empty directory of its own and remove it after loading; the file is a pickle.
    with tempfile.TemporaryDirectory(prefix="moge-") as directory:
        path = hf_hub_download(repository, "model.pt", revision=revision, local_dir=directory)
        return torch.load(path, map_location="cpu", weights_only=True)


def convert_moge_checkpoint(
    source: str | Path,
    output_directory: str | Path | None = None,
    *,
    revision: str | None = None,
    safe_serialization: bool = True,
) -> MoGeModel | MoGe2Model:
    """Convert a MoGe or MoGe-2 ``model.pt`` (local, or a Hub repository at ``revision``) and optionally save it.

    The version follows the checkpoint's ``model_config``: a backbone name string is MoGe (v1), an encoder dict is
    MoGe-2. Tensors keep their released float32 precision, which is what upstream ``from_pretrained`` loads.
    """

    checkpoint = load_moge_checkpoint(source, revision=revision)
    model_config = checkpoint["model_config"]
    if isinstance(model_config["encoder"], str):
        model_type, config, state_dict = (
            MoGeModel,
            moge_config(model_config),
            convert_moge_state_dict(checkpoint["model"]),
        )
    else:
        model_type, config = MoGe2Model, moge2_config(model_config)
        state_dict = convert_moge2_state_dict(checkpoint["model"])
    with init_empty_weights():
        model = model_type(**config)
    match_dinov2_attention_names(state_dict, model.state_dict().keys())
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    if output_directory is not None:
        model.save_pretrained(output_directory, safe_serialization=safe_serialization)
    return model


def load_moge_estimator(
    repository: str, *, revision: str | None = None, dtype: torch.dtype | None = None, **hub_kwargs
):
    """Download a MoGe or MoGe-2 release ``model.pt`` (``Ruicheng/moge-vitl``, ``Ruicheng/moge-2-vitl``) and convert it
    on load with :func:`convert_moge_checkpoint`; float32 unless ``dtype``."""

    names = ("token", "cache_dir", "local_files_only", "force_download")
    download = {key: hub_kwargs[key] for key in names if hub_kwargs.get(key) is not None}
    # The release is a pickle; ``convert_moge_checkpoint`` reads it with ``torch.load(weights_only=True)``.
    model = convert_moge_checkpoint(hf_hub_download(repository, "model.pt", revision=revision, **download))
    return model if dtype is None else model.to(dtype)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="local model.pt or Hub repository id (Ruicheng/moge-vitl, Ruicheng/moge-2-vitl)"
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=None, help="Hub revision; defaults to the pinned release revision")
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_moge_checkpoint(
        args.source, args.output_directory, revision=args.revision, safe_serialization=not args.no_safe_serialization
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MOGE2_REFERENCE_REVISION",
    "MOGE2_WEIGHTS_REPOSITORY",
    "MOGE2_WEIGHTS_REVISION",
    "MOGE_REFERENCE_REVISION",
    "MOGE_WEIGHTS_REPOSITORY",
    "MOGE_WEIGHTS_REVISION",
    "convert_dinov2_state_dict",
    "convert_moge2_state_dict",
    "convert_moge_checkpoint",
    "convert_moge_state_dict",
    "load_moge_checkpoint",
    "load_moge_estimator",
    "main",
    "moge2_config",
    "moge_config",
]
