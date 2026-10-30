"""Convert the released Hunyuan3D-Omni checkpoint (``tencent/Hunyuan3D-Omni``) into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2Model

from ..hunyuan3d.conversion import (
    _TRANSFORMER_FIXED,
    _TRANSFORMER_IGNORED,
    _TRANSFORMER_MAPPED,
    _materialize,
    _model_config,
    convert_transformer_state_dict,
    convert_vae_state_dict,
)
from ..hunyuan3d.models import DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, Hunyuan3DDiTModel
from ..hunyuan3d.scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from ..hunyuan3d.vae import Hunyuan3DShapeVAE
from .models import Hunyuan3DOmniControlEncoder
from .pipeline import Hunyuan3DOmniImageTo3DPipeline

HUNYUAN3D_OMNI_REFERENCE_REVISION = "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8"
HUNYUAN3D_OMNI_HUB_REPOSITORY = "tencent/Hunyuan3D-Omni"
HUNYUAN3D_OMNI_HUB_REVISION = "70e803bfb4e127d534049d8ab8c8cb511780d485"
# Upstream loads DINOv2 from the Hub at run time; the checkpoint does not contain it, and neither does the conversion.
DINOV2_REPOSITORY = DINOV2_LARGE_REPOSITORY
DINOV2_REVISION = DINOV2_LARGE_REVISION

_VAE_FIXED = {"include_pi": False, "qkv_bias": False, "qk_norm": True}
_VAE_MAPPED = {"num_latents", "embed_dim", "width", "heads", "num_decoder_layers", "num_freqs"}
# Point-cloud encoder, training, and kernel options; generation only decodes.
_VAE_IGNORED = {
    "num_encoder_layers",
    "point_feats",
    "sharpedge_label",
    "pc_size",
    "pc_sharpedge_size",
    "downsample_ratio",
    "train_num_latents_range",
    "train_num_latents_probs",
    "use_ln_post",
    "init_scale",
    "flash",
    "use_checkpoint",
}
_CONTROL_FIXED = {"include_pi": False}
_CONTROL_MAPPED = {"width", "num_freqs", "voxel_resolution"}
# Training-time point dropout and noise, and an unused image size.
_CONTROL_IGNORED = {
    "image_encoder",
    "image_size",
    "resolutions",
    "re_sample",
    "random_noise",
    "noise_ratio",
    "noise_scales",
    "drop_point",
    "drop_ratio",
}
_CONTROL_RENAMES = {
    "cond_signal_embedding.": "type_embedding.",
    "cond_signal_linear.": "type_proj.",
    "linear.0.": "proj_in.",
    "linear.1.": "norm.",
}


def _config(folder: Path, name: str, target: str) -> dict[str, Any]:
    config = json.loads((folder / name / "config.json").read_text(encoding="utf-8"))
    if config.get("target") != target:
        raise ValueError(f"only the released {target} {name} is supported, got {config.get('target')}")
    return config


def convert_control_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the released ``OmniEncoder`` control weights onto :class:`Hunyuan3DOmniControlEncoder`."""

    converted = {}
    for key, value in state_dict.items():
        prefix = next(prefix for prefix in _CONTROL_RENAMES if key.startswith(prefix))
        converted[_CONTROL_RENAMES[prefix] + key.removeprefix(prefix)] = value
    return converted


def convert_hunyuan3d_omni_checkpoint(
    source: str | Path,
    output_directory: str | Path,
    *,
    revision: str = HUNYUAN3D_OMNI_HUB_REVISION,
    ema: bool = False,
    image_encoder: str | Path = DINOV2_REPOSITORY,
    image_encoder_revision: str | None = DINOV2_REVISION,
    safe_serialization: bool = True,
) -> Path:
    """Convert the release into a :class:`Hunyuan3DOmniImageTo3DPipeline` folder.

    ``source`` is a local release folder (``config.json`` plus the ``model``, ``vae``, ``cond_encoder``,
    ``scheduler``, and ``image_processor`` folders) or a Hub repository id downloaded at ``revision``. ``ema``
    converts ``pytorch_model_ema.bin`` instead of ``pytorch_model.bin`` for the DiT. ``image_encoder`` is the
    DINOv2 checkpoint upstream fetches at run time; only its config is read, since the pipeline loads it from Meta's
    repository and the folder never holds it. Weights keep their stored dtype; the ShapeVAE encoder is dropped.
    """

    model_file = "pytorch_model_ema.bin" if ema else "pytorch_model.bin"
    root = Path(source)
    if not root.is_dir():
        patterns = ["config.json", "*/config.json", f"model/{model_file}", "vae/*.bin", "cond_encoder/*.bin"]
        root = Path(snapshot_download(str(source), revision=revision, allow_patterns=patterns))
    pipeline_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if pipeline_config["target"] != "hy3dshape.pipelines.Hunyuan3DOmniSiTFlowMatchingPipeline":
        raise ValueError("only the released Hunyuan3DOmniSiTFlowMatchingPipeline configuration is supported")
    model = _config(root, "model", "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain")
    vae = _config(root, "vae", "hy3dshape.models.autoencoders.shape_vae_pl.ShapeVAEPL")
    if vae["params"]["module_cfg"]["target"] != "hy3dshape.models.autoencoders.shape_vae.ShapeVAE":
        raise ValueError("only the released ShapeVAE is supported")
    control = _config(root, "cond_encoder", "hy3dshape.models.conditioners.omni_encoder.OmniEncoder")
    dino = control["params"]["image_encoder"]
    if dino["target"] != "hy3dshape.models.conditioners.omni_encoder.DinoImageEncoder" or not dino["params"].get(
        "use_cls_token", True
    ):
        raise ValueError("only the released DINOv2 image encoder with its CLS token is supported")
    if _config(root, "image_processor", "hy3dshape.preprocessors.ImageProcessorV2")["params"]:
        raise ValueError("only the default ImageProcessorV2 (512 px, border ratio 0.15) is supported")
    scheduler = json.loads((root / "scheduler" / "config.json").read_text(encoding="utf-8"))
    transport, sampler = scheduler["transport"]["params"], scheduler["sampler"]
    if (transport.get("path_type"), transport.get("prediction"), sampler["ode_params"]["sampling_method"]) != (
        "Linear",
        "velocity",
        "euler",
    ):
        raise ValueError("only the released linear-path velocity transport with Euler sampling is supported")

    transformer_config = _model_config(model, _TRANSFORMER_FIXED, _TRANSFORMER_MAPPED, _TRANSFORMER_IGNORED)
    vae_config = {
        **_model_config(vae["params"]["module_cfg"], _VAE_FIXED, _VAE_MAPPED, _VAE_IGNORED),
        "scale_factor": vae["scale_factor"],
    }
    control_config = _model_config(control, _CONTROL_FIXED, _CONTROL_MAPPED & set(control["params"]), _CONTROL_IGNORED)

    with init_empty_weights():
        image_encoder_model = Dinov2Model(
            Dinov2Config.from_pretrained(str(image_encoder), revision=image_encoder_revision)
        )
    if dino["params"]["image_size"] != image_encoder_model.config.image_size:
        raise ValueError("the DINOv2 input size must match its configured image_size")

    def load(name: str, filename: str = "pytorch_model.bin") -> dict[str, torch.Tensor]:
        return torch.load(root / name / filename, map_location="cpu", weights_only=True, mmap=True)

    vae_state = {key.removeprefix("sal."): value for key, value in load("vae").items()}
    pipeline = Hunyuan3DOmniImageTo3DPipeline(
        image_encoder=image_encoder_model.eval(),
        control_encoder=_materialize(
            Hunyuan3DOmniControlEncoder, control_config, convert_control_encoder_state_dict(load("cond_encoder"))
        ),
        transformer=_materialize(
            Hunyuan3DDiTModel,
            transformer_config,
            convert_transformer_state_dict(load("model", model_file), transformer_config["num_heads"]),
        ),
        vae=_materialize(Hunyuan3DShapeVAE, vae_config, convert_vae_state_dict(vae_state, vae_config["heads"])),
        scheduler=Hunyuan3DFlowMatchEulerDiscreteScheduler(),
    )
    destination = Path(output_directory)
    pipeline.save_pretrained(destination, safe_serialization=safe_serialization)
    report = {
        "reference_revision": HUNYUAN3D_OMNI_REFERENCE_REVISION,
        "source": str(root.resolve()),
        "hub_revision": revision if not Path(source).is_dir() else None,
        "transformer_weights": model_file,
        "image_encoder": str(image_encoder),
        "image_encoder_revision": image_encoder_revision,
        "dropped_vae_encoder_keys": sorted(key for key in vae_state if key.startswith(("encoder.", "pre_kl."))),
    }
    (destination / "hunyuan3d_omni_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help=f"local release folder or Hub repository id, e.g. {HUNYUAN3D_OMNI_HUB_REPOSITORY}"
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision", default=HUNYUAN3D_OMNI_HUB_REVISION)
    parser.add_argument("--ema", action="store_true", help="convert the EMA DiT weights")
    parser.add_argument("--image-encoder", default=DINOV2_REPOSITORY)
    parser.add_argument("--image-encoder-revision", default=DINOV2_REVISION)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_hunyuan3d_omni_checkpoint(
        args.source,
        args.output_directory,
        revision=args.revision,
        ema=args.ema,
        image_encoder=args.image_encoder,
        image_encoder_revision=args.image_encoder_revision or None,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HUNYUAN3D_OMNI_HUB_REVISION",
    "HUNYUAN3D_OMNI_REFERENCE_REVISION",
    "convert_control_encoder_state_dict",
    "convert_hunyuan3d_omni_checkpoint",
    "main",
]
