"""Convert the released SAM 3D Objects ``checkpoints/`` folder (``facebook/sam-3d-objects``) into a Diffusers folder."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from ..trellis.decoders import TrellisSLatGaussianDecoder, TrellisSLatMeshDecoder, TrellisSparseStructureDecoder
from ..trellis.models import TrellisSLatFlowModel
from ..trellis.scheduler import TrellisFlowEulerScheduler
from .conditioner import Sam3DConditionEmbedder
from .models import Sam3DSparseStructureFlowModel
from .pipeline import Sam3DObjectsPipeline

SAM3D_REFERENCE_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"

_GENERATOR = "_base_models.generator.reverse_fn.backbone."
_EMBEDDER = "_base_models.condition_embedder."
# Upstream's merged pose stream is named after its first modality.
_UPSTREAM_ROTATION = "6drotation_normalized"


def _dinov2_config() -> Dinov2WithRegistersConfig:
    """``dinov2_vitl14_reg``, the configuration of ``facebook/dinov2-with-registers-large``."""

    return Dinov2WithRegistersConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        patch_size=14,
        image_size=518,
        num_register_tokens=4,
        layerscale_value=1.0,
        layer_norm_eps=1e-6,
    )


def _dinov2_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a torch-hub DINOv2 state dict onto the Hub layout of ``transformers.Dinov2WithRegistersModel``."""

    renames = {
        "cls_token": "embeddings.cls_token",
        "mask_token": "embeddings.mask_token",
        "register_tokens": "embeddings.register_tokens",
        "pos_embed": "embeddings.position_embeddings",
        "patch_embed.proj.": "embeddings.patch_embeddings.projection.",
        "norm.": "layernorm.",
    }
    block_renames = {
        "attn.proj.": "attention.output.dense.",
        "ls1.gamma": "layer_scale1.lambda1",
        "ls2.gamma": "layer_scale2.lambda1",
    }
    converted = {}
    for key, value in state.items():
        block = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
        if block is None:
            prefix = next(name for name in renames if key.startswith(name))
            converted[renames[prefix] + key[len(prefix) :]] = value
            continue
        target, name = f"encoder.layer.{block[1]}.", block[2]
        if name.startswith("attn.qkv."):
            suffix = name.removeprefix("attn.qkv.")
            for part, tensor in zip(("query", "key", "value"), value.chunk(3, dim=0)):
                converted[f"{target}attention.attention.{part}.{suffix}"] = tensor
            continue
        prefix = next((old for old in block_renames if name.startswith(old)), None)
        converted[target + (name if prefix is None else block_renames[prefix] + name[len(prefix) :])] = value
    return converted


def _pointmap_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the timm window block of ``PointPatchEmbed`` onto ``diffusers.BasicTransformerBlock``."""

    renames = {
        "blocks.0.attn.proj.": "block.attn1.to_out.0.",
        "blocks.0.norm1.": "block.norm1.",
        "blocks.0.norm2.": "block.norm3.",
        "blocks.0.mlp.fc1.": "block.ff.net.0.proj.",
        "blocks.0.mlp.fc2.": "block.ff.net.2.",
    }
    converted = {}
    for key, value in state.items():
        if key.startswith("blocks.0.attn.qkv."):
            suffix = key.removeprefix("blocks.0.attn.qkv.")
            for part, tensor in zip("qkv", value.chunk(3, dim=0)):
                converted[f"block.attn1.to_{part}.{suffix}"] = tensor
            continue
        prefix = next((old for old in renames if key.startswith(old)), None)
        converted[key if prefix is None else renames[prefix] + key[len(prefix) :]] = value
    return converted


def _children(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def convert_dinov2_encoder(
    state: Mapping[str, torch.Tensor], config: Dinov2WithRegistersConfig
) -> Dinov2WithRegistersModel:
    """Build ``Dinov2WithRegistersModel`` from a torch-hub DINOv2 state dict, requiring every key to match."""

    model, info = Dinov2WithRegistersModel.from_pretrained(
        None, config=config, state_dict=_dinov2_state(state), output_loading_info=True
    )
    if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
        raise ValueError(f"DINOv2 state mismatch: {info}")
    return model.eval()


def convert_condition_embedder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map upstream ``EmbedderFuser`` keys (``condition_embedder.`` prefix removed) onto :class:`Sam3DConditionEmbedder`.

    The DINOv2 backbones (``module_list.{0,1}.backbone``) are separate pipeline components; see
    :func:`convert_dinov2_encoder`.
    """

    converted = {"idx_emb": state["idx_emb"]}
    for key, value in _pointmap_state(_children(state, "module_list.2.")).items():
        converted[f"pointmap_encoder.{key}"] = value
    for key, value in _children(state, "projection_nets.").items():
        index, name = key.split(".", 1)
        if name.startswith("1.w"):
            continue
        converted[f"projections.{index}.{name}"] = value
    for index in sorted({key.split(".", 1)[0] for key in _children(state, "projection_nets.")}):
        # Upstream gates with ``silu(w1 x) * w3 x``; Diffusers' SwiGLU computes ``hidden, gate = proj(x).chunk(2)``.
        weights = _children(state, f"projection_nets.{index}.1.")
        converted[f"projections.{index}.1.net.0.proj.weight"] = torch.cat([weights["w3.weight"], weights["w1.weight"]])
        converted[f"projections.{index}.1.net.2.weight"] = weights["w2.weight"]
    return converted


def convert_sparse_structure_flow_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the upstream mixture-of-transformers keys (backbone prefix removed) onto the package model."""

    converted = {}
    for key, value in state.items():
        if key.startswith("latent_mapping."):
            key = key.replace(f".{_UPSTREAM_ROTATION}.", ".rotation.")
        else:
            key = key.replace(f".{_UPSTREAM_ROTATION}", ".pose")
            key = re.sub(
                r"self_attn\.(to_qkv|q_rms_norm|k_rms_norm|to_out)\.(shape|pose)\.", r"self_attn.streams.\2.\1.", key
            )
        converted[key] = value
    return converted


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    """Read ``<stem>.safetensors`` when present (and non-empty), otherwise the released ``<stem>.ckpt``."""

    safetensors_path = path.with_suffix(".safetensors")
    if safetensors_path.is_file() and safetensors_path.stat().st_size:
        return load_file(safetensors_path, device="cpu")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path} (or {safetensors_path})")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return state.get("state_dict", state)


def _build(model_type: type[Any], config: dict[str, Any], state: Mapping[str, torch.Tensor]) -> Any:
    with init_empty_weights():
        model = model_type(**config)
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"{model_type.__name__} state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    # Released half-precision tensors load into the dtype the upstream module was built with, as ``load_state_dict`` does.
    model.load_state_dict(
        {key: value.to(expected[key].dtype) for key, value in state.items()}, strict=True, assign=True
    )
    return model.eval()


def _require(actual: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    mismatched = {key: actual.get(key) for key, value in expected.items() if actual.get(key) != value}
    if mismatched:
        raise ValueError(f"{name} differs from the released configuration this converter supports: {mismatched}")


def _check_configs(source: Path) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        return yaml.safe_load((source / name).read_text(encoding="utf-8"))

    pipeline = load("pipeline.yaml")
    _require(
        pipeline,
        {
            "pose_decoder_name": "ScaleShiftInvariant",
            "slat_cfg_strength": 1,
            "slat_rescale_t": 1,
            "downsample_ss_dist": 1,
        },
        "pipeline.yaml",
    )
    if any(
        key.startswith("ss_") and key.split("_", 1)[1].startswith(("cfg", "inference", "rescale")) for key in pipeline
    ):
        raise ValueError(
            "pipeline.yaml overrides the sparse-structure sampler; only the released defaults are supported"
        )
    ss = load("ss_generator.yaml")["module"]
    flow = Sam3DSparseStructureFlowModel.production_config()
    ss_flow = ss["generator"]["backbone"]["reverse_fn"]["backbone"]
    _require(
        ss_flow, {key: flow[key] for key in ("resolution", "in_channels", "model_channels", "num_blocks")}, "ss flow"
    )
    (pose_stream,) = ss_flow["latent_share_transformer"].values()
    pose_latents = [
        ("rotation" if name == _UPSTREAM_ROTATION else name, ss_flow["latent_mapping"][name]["in_channels"])
        for name in pose_stream
    ]
    if pose_latents != [tuple(item) for item in flow["pose_latents"]]:
        raise ValueError(f"unexpected layout modalities: {pose_latents}")
    pointmap = Sam3DConditionEmbedder.production_pointmap_config()
    ss_embedders = ss["condition_embedder"]["backbone"]["embedder_list"]
    _require(
        ss_embedders[2][0],
        {**{key: pointmap[key] for key in ("embed_dim", "input_size", "patch_size")}, "remap_output": "linear"},
        "pointmap",
    )
    slat = load("slat_generator.yaml")["module"]
    _require(
        slat["generator"]["backbone"]["reverse_fn"]["backbone"], TrellisSLatFlowModel.production_config(), "slat flow"
    )
    image_size = _dinov2_config().image_size
    for embedders, prenorm in ((ss_embedders, False), (slat["condition_embedder"]["backbone"]["embedder_list"], True)):
        for embedder, _ in embedders[:2]:
            _require(embedder, {"dino_model": "dinov2_vitl14_reg", "input_size": image_size}, "DINOv2 embedder")
            if embedder.get("prenorm_features", False) != prenorm:
                raise ValueError("unexpected DINOv2 prenorm_features setting")
    decoder = TrellisSparseStructureDecoder.production_config()
    _require(
        load("ss_decoder.yaml"), {key: decoder[key] for key in ("channels", "latent_channels")}, "ss_decoder.yaml"
    )
    gaussian = load("slat_decoder_gs.yaml")
    _require(
        gaussian["representation_config"], TrellisSLatGaussianDecoder.production_representation_config(), "gaussian"
    )
    mesh = TrellisSLatMeshDecoder.production_config()
    _require(
        load("slat_decoder_mesh.yaml"), {key: mesh[key] for key in ("model_channels", "representation_config")}, "mesh"
    )
    return pipeline


def _shared_dinov2(*states: Mapping[str, torch.Tensor]) -> Dinov2WithRegistersModel:
    # The image and mask embedders of both generators are copies of the same frozen DINOv2; convert it once.
    state, *copies = (
        _children(state, f"{_EMBEDDER}module_list.{index}.backbone.") for state in states for index in (0, 1)
    )
    for copy in copies:
        if copy.keys() != state.keys() or not all(torch.equal(value, copy[key]) for key, value in state.items()):
            raise ValueError("the DINOv2 image and mask backbones of ss_generator and slat_generator differ")
    return convert_dinov2_encoder(state, _dinov2_config())


def convert_sam3d_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    safe_serialization: bool = True,
) -> Path:
    """Convert the released ``checkpoints/`` folder.

    Writes the DINOv2 image encoder, the sparse-structure conditioner, flow model, decoder and scheduler, the
    structured-latent conditioner, flow model and scheduler, and the 32-splat Gaussian and FlexiCubes mesh decoders.
    The 4-splat Gaussian decoder and the sparse-structure decoder copy embedded in ``ss_generator`` are not used by
    the released sampler and are skipped. MoGe depth estimation is not part of the checkpoint; pass pointmaps to the
    pipeline instead.
    """

    source = Path(source_directory)
    pipeline_config = _check_configs(source)
    ss_state = _load_state(source / "ss_generator.ckpt")
    slat_state = _load_state(source / "slat_generator.ckpt")
    components: dict[str, Callable[[], Any]] = {
        "image_encoder": lambda: _shared_dinov2(ss_state, slat_state),
        "sparse_structure_conditioner": lambda: _build(
            Sam3DConditionEmbedder,
            Sam3DConditionEmbedder.production_config(),
            convert_condition_embedder_state(_children(ss_state, _EMBEDDER)),
        ),
        "sparse_structure_flow_model": lambda: _build(
            Sam3DSparseStructureFlowModel,
            Sam3DSparseStructureFlowModel.production_config(),
            convert_sparse_structure_flow_state(_children(ss_state, _GENERATOR)),
        ),
        "sparse_structure_decoder": lambda: _build(
            TrellisSparseStructureDecoder,
            {**TrellisSparseStructureDecoder.production_config(), "use_fp16": False},
            _load_state(source / "ss_decoder.ckpt"),
        ),
        "slat_conditioner": lambda: _build(
            Sam3DConditionEmbedder,
            Sam3DConditionEmbedder.production_config(prenorm_features=True, pointmap=False),
            convert_condition_embedder_state(_children(slat_state, _EMBEDDER)),
        ),
        "slat_flow_model": lambda: _build(
            TrellisSLatFlowModel, TrellisSLatFlowModel.production_config(), _children(slat_state, _GENERATOR)
        ),
        "gaussian_decoder": lambda: _build(
            TrellisSLatGaussianDecoder,
            TrellisSLatGaussianDecoder.production_config(),
            _load_state(source / "slat_decoder_gs.ckpt"),
        ),
        "mesh_decoder": lambda: _build(
            TrellisSLatMeshDecoder,
            TrellisSLatMeshDecoder.production_config(),
            _load_state(source / "slat_decoder_mesh.ckpt"),
        ),
    }
    pipeline = Sam3DObjectsPipeline(
        **{name: build() for name, build in components.items()},
        sparse_structure_scheduler=TrellisFlowEulerScheduler(sigma_min=0.0),
        slat_scheduler=TrellisFlowEulerScheduler(sigma_min=0.0),
        slat_mean=pipeline_config["slat_mean"],
        slat_std=pipeline_config["slat_std"],
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path, help="the released checkpoints/ folder with pipeline.yaml")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_sam3d_checkpoint(
        args.source_directory, args.output_directory, safe_serialization=not args.no_safe_serialization
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SAM3D_REFERENCE_REVISION",
    "convert_condition_embedder_state",
    "convert_dinov2_encoder",
    "convert_sam3d_checkpoint",
    "convert_sparse_structure_flow_state",
    "main",
]
