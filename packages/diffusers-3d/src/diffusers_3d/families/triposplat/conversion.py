"""Convert the released VAST-AI/TripoSplat checkpoints into a diffusers-3d pipeline folder."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from .decoder import TripoSplatGaussianDecoder
from .models import TripoSplatImageVAE, TripoSplatTransformerModel, convert_triposplat_vae_state_dict
from .pipeline import TripoSplatImageTo3DPipeline

TRIPOSPLAT_REFERENCE_REVISION = "d8db9e018b413dd9c4a9fe22463781bf98e8e68d"
TRIPOSPLAT_WEIGHTS_REPOSITORY = "VAST-AI/TripoSplat"
TRIPOSPLAT_WEIGHTS_REVISION = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
# The release also bundles copies of DINOv3 (``clip_vision/``) and the Flux.2 VAE (``vae/flux2-vae.safetensors``),
# which are gated third-party weights; the pipeline loads them from their own repositories instead.
_FILES = {
    "transformer": "diffusion_models/triposplat_fp16.safetensors",
    "decoder": "vae/triposplat_vae_decoder_fp16.safetensors",
}
# DINOv3 ViT-H/16+, as upstream's ``DinoV3ViT`` defaults.
DINOV3_VITH16_CONFIG = {
    "hidden_size": 1280,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
    "intermediate_size": 5120,
    "patch_size": 16,
    "num_register_tokens": 4,
    "use_gated_mlp": True,
    "hidden_act": "silu",
    "key_bias": False,
    "layerscale_value": 1.0,
    "rope_theta": 100.0,
    "layer_norm_eps": 1e-5,
    "image_size": 1024,
}

_BLOCK_RULES = (
    (r"\.attn\.qkv\.", ".attn.to_qkv."),
    (r"\.attn\.out\.", ".attn.to_out."),
    (r"\.(self_attn|cross_attn|attn)\.(q|k)_(rms_)?norm\.", r".\1.norm_\2."),
    (r"\.mlp\.mlp\.0\.", ".ff.net.0.proj."),
    (r"\.mlp\.mlp\.2\.", ".ff.net.2."),
)
_TRANSFORMER_RULES = (
    (r"^t_embedder\.mlp\.0\.", "time_embed.linear_1."),
    (r"^t_embedder\.mlp\.2\.", "time_embed.linear_2."),
    (r"^adaLN_modulation\.", "time_modulation."),
    (r"^input_layer\.", "proj_in."),
    (r"^cond_embedder\.", "context_proj."),
    (r"^cond_embedder2\.", "context_proj_2."),
    (r"^noise_repo_layers\.", "noise_rope."),
    (r"^context_repo_layers\.", "context_rope."),
    (r"^repo_layers\.", "rope."),
    (r"^out_layer\.", "proj_out."),
    (r"^cam_out_layer\.", "camera_proj_out."),
    *_BLOCK_RULES,
)
_DECODER_RULES = (
    (r"^octree\.in_proj\.", "octree_proj_in."),
    (r"^octree\.input_layer\.", "octree_input."),
    (r"^octree\.l_embedder\.mlp\.0\.", "octree_level_embed.linear_1."),
    (r"^octree\.l_embedder\.mlp\.2\.", "octree_level_embed.linear_2."),
    (r"^octree\.adaLN_modulation\.", "octree_modulation."),
    (r"^octree\.blocks\.(\d+)\.cross_attn\.", r"octree_blocks.\1.attn."),
    (r"^octree\.blocks\.", "octree_blocks."),
    (r"^octree\.out_proj\.", "octree_proj_out."),
    (r"^gs\.in_proj\.", "gaussian_proj_in."),
    (r"^gs\.input_layer\.", "gaussian_input."),
    (r"^gs\.blocks\.(\d+)\.self_attn\.", r"gaussian_blocks.\1.attn1."),
    (r"^gs\.blocks\.(\d+)\.cross_attn\.", r"gaussian_blocks.\1.attn2."),
    (r"^gs\.blocks\.", "gaussian_blocks."),
    (r"^gs\.out_proj\.", "gaussian_proj_out."),
    (r"^gs\.points_offset_perturbation$", "offset_perturbation"),
    (r"^gs\.base_offset_scale$", "base_offset_scale"),
    (r"\.(attn\d?)\.(q|k)_rms_norm\.", r".\1.norm_\2."),
    *_BLOCK_RULES[3:],
)


def _rename(state_dict: dict[str, torch.Tensor], rules) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state_dict.items():
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        renamed[key] = value
    return renamed


def convert_triposplat_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename upstream ``LatentSeqMMFlowModel`` keys to :class:`TripoSplatTransformerModel` keys."""

    # The camera MLP's linear layers sit at ``mlp.0, 2, ...``; all but the last are ``net.<i>.proj`` and the last is
    # ``net.<num_layers>`` (see ``TripoSplatTransformerModel``).
    num_layers = len({key.split(".")[2] for key in state_dict if key.startswith("cam_refiner.mlp.")})
    renamed = {}
    for key, value in state_dict.items():
        if match := re.match(r"^cam_refiner\.mlp\.(\d+)\.", key):
            index = int(match[1]) // 2
            prefix = f"camera_proj.net.{index}.proj." if index < num_layers - 1 else f"camera_proj.net.{num_layers}."
            renamed[prefix + key[match.end() :]] = value
        else:
            renamed[key] = value
    return _rename(renamed, _TRANSFORMER_RULES)


def convert_triposplat_decoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename upstream ``OctreeGaussianDecoder`` keys to :class:`TripoSplatGaussianDecoder` keys."""

    return _rename(state_dict, _DECODER_RULES)


def convert_dinov3_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move the HF-layout DINOv3 blocks under ``model.``, where Transformers 5 keeps them."""

    return {re.sub(r"^layer\.", "model.layer.", key): value for key, value in state_dict.items()}


def _load(model_type: type[torch.nn.Module], config: dict, state_dict: dict[str, torch.Tensor]):
    with init_empty_weights():
        model = model_type(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_triposplat_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = TRIPOSPLAT_WEIGHTS_REVISION,
    configs: dict[str, dict] | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Convert a local TripoSplat snapshot, or download ``source`` from the Hub at ``revision``.

    The release ships bare state dicts; ``configs`` overrides the released architecture per component
    (``image_encoder``, ``vae``, ``transformer``, ``decoder``). The flow transformer and decoder keep their released float16, and upstream's
    shifted Euler sampler becomes ``FlowMatchEulerDiscreteScheduler(shift=3.0)``.

    The bundled DINOv3 and Flux.2 VAE are not read: the pipeline loads them from their gated repositories at run
    time, so they are built here as empty shells that ``model_index.json`` names but whose weights are never saved.
    The BiRefNet background remover and the TripoSplat VAE encoder (unused at inference) are not converted either.
    """

    root = Path(source)
    if not root.is_dir():
        root = Path(snapshot_download(str(source), revision=revision, allow_patterns=list(_FILES.values())))
    state_dicts = {name: load_file(root / path) for name, path in _FILES.items()}
    configs = {"image_encoder": DINOV3_VITH16_CONFIG, "vae": {}, "transformer": {}, "decoder": {}, **(configs or {})}

    with init_empty_weights():
        image_encoder = DINOv3ViTModel(DINOv3ViTConfig(**configs["image_encoder"]))
        vae = TripoSplatImageVAE(**configs["vae"])
    pipeline = TripoSplatImageTo3DPipeline(
        image_encoder=image_encoder.eval(),
        vae=vae.eval(),
        transformer=_load(
            TripoSplatTransformerModel,
            configs["transformer"],
            convert_triposplat_transformer_state_dict(state_dicts["transformer"]),
        ),
        decoder=_load(
            TripoSplatGaussianDecoder,
            configs["decoder"],
            convert_triposplat_decoder_state_dict(state_dicts["decoder"]),
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local snapshot directory or Hub repository id (e.g. VAST-AI/TripoSplat)")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=TRIPOSPLAT_WEIGHTS_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_triposplat_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DINOV3_VITH16_CONFIG",
    "TRIPOSPLAT_REFERENCE_REVISION",
    "TRIPOSPLAT_WEIGHTS_REPOSITORY",
    "TRIPOSPLAT_WEIGHTS_REVISION",
    "convert_dinov3_state_dict",
    "convert_triposplat_checkpoint",
    "convert_triposplat_decoder_state_dict",
    "convert_triposplat_transformer_state_dict",
    "convert_triposplat_vae_state_dict",
    "main",
]
