"""Convert official Hunyuan3D-2.1 aggregate checkpoints to Diffusers folders."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers import __version__ as diffusers_version
from safetensors import safe_open

from .conditioner import Hunyuan3DDinov2Conditioner
from .models import Hunyuan3DShapeDiTModel
from .pipeline import Hunyuan3DImageToShapePipeline
from .scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from .vae import Hunyuan3DShapeVAE

HUNYUAN3D_REFERENCE_REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
_VAE_UNSUPPORTED_PREFIXES = ("encoder.", "pre_kl.")


def _load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    payload = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(payload)
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("PyYAML is required to convert an official config.yaml") from error
        data = yaml.safe_load(payload)
    if not isinstance(data, dict):
        raise ValueError("configuration must contain a mapping")
    return data


def _component_config(config: Mapping[str, Any], component: str) -> dict[str, Any]:
    value = config.get(component, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {component!r} must be a mapping")
    params = value.get("params", value)
    if not isinstance(params, Mapping):
        raise ValueError(f"config parameters for {component!r} must be a mapping")
    return dict(params)


def _conditioner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    params = _component_config(config, "conditioner")
    main = params.get("main_image_encoder", params)
    if not isinstance(main, Mapping):
        raise ValueError("conditioner main_image_encoder must be a mapping")
    kwargs = main.get("kwargs", main)
    if not isinstance(kwargs, Mapping):
        raise ValueError("conditioner kwargs must be a mapping")
    dinov2_config = kwargs.get("config")
    if dinov2_config is None:
        return Hunyuan3DDinov2Conditioner.production_config()
    if not isinstance(dinov2_config, Mapping):
        raise ValueError("conditioner DINOv2 config must be a mapping")
    normalized_config = dict(dinov2_config)
    normalized_config.pop("torch_dtype", None)
    return {
        "dinov2_config": normalized_config,
        "image_size": int(kwargs.get("image_size", normalized_config.get("image_size", 518))),
        "use_cls_token": bool(kwargs.get("use_cls_token", True)),
    }


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("PyTorch checkpoint must contain a mapping")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict must contain a mapping")
    return state_dict


def _load_component_state(
    path: Path,
    component: str,
    checkpoint_data: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        prefix = f"{component}."
        state_dict: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            for key in keys:
                if key.startswith(prefix):
                    state_dict[key.removeprefix(prefix)] = handle.get_tensor(key)
            if not state_dict and component == "model":
                # Standalone official denoiser safetensors do not use a prefix.
                state_dict = {key: handle.get_tensor(key) for key in keys}
        return state_dict

    checkpoint = _load_torch_checkpoint(path) if checkpoint_data is None else checkpoint_data
    nested = checkpoint.get(component)
    if isinstance(nested, Mapping):
        state_dict = dict(nested)
    else:
        prefix = f"{component}."
        state_dict = {
            key.removeprefix(prefix): value
            for key, value in checkpoint.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if (
            not state_dict
            and component == "model"
            and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items())
        ):
            state_dict = dict(checkpoint)
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise ValueError(f"checkpoint component {component!r} must be a tensor state dict")
    return state_dict


def _meta_model(model_type: type[Any], config: Mapping[str, Any]):
    with torch.device("meta"):
        return model_type(**dict(config))


def _save_strict_component(
    model_type: type[Any],
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    output_directory: Path,
    *,
    safe_serialization: bool,
) -> None:
    model = _meta_model(model_type, config)
    incompatible = model.load_state_dict(dict(state_dict), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"{model_type.__name__} state dict mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.save_pretrained(output_directory, safe_serialization=safe_serialization)
    del model
    gc.collect()


def _save_vae_decoder(
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    output_directory: Path,
    *,
    safe_serialization: bool,
) -> tuple[str, ...]:
    model = _meta_model(Hunyuan3DShapeVAE, config)
    expected = set(model.state_dict())
    provided = set(state_dict)
    missing = expected.difference(provided)
    unexpected = provided.difference(expected)
    unsupported = tuple(sorted(key for key in unexpected if key.startswith(_VAE_UNSUPPORTED_PREFIXES)))
    invalid = sorted(unexpected.difference(unsupported))
    if missing or invalid:
        raise ValueError(f"Hunyuan3DShapeVAE decoder mismatch: missing={sorted(missing)}, unexpected={invalid}")
    decoder_state = {key: state_dict[key] for key in expected}
    incompatible = model.load_state_dict(decoder_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("internal VAE decoder strict loading failed")
    model.save_pretrained(output_directory, safe_serialization=safe_serialization)
    del model
    gc.collect()
    return unsupported


def _map_conditioner_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith("main_image_encoder.model."):
            key = f"model.{key.removeprefix('main_image_encoder.model.')}"
        elif key.startswith("main_image_encoder."):
            key = key.removeprefix("main_image_encoder.")
        mapped[key] = value
    return mapped


def _write_pipeline_index(
    output_directory: Path,
    *,
    image_processor_size: int,
    image_processor_border_ratio: float,
) -> None:
    model_index = {
        "_class_name": Hunyuan3DImageToShapePipeline.__name__,
        "_diffusers_version": diffusers_version,
        "conditioner": [
            "diffusers_3d.families.hunyuan3d.conditioner",
            Hunyuan3DDinov2Conditioner.__name__,
        ],
        "denoiser": [
            "diffusers_3d.families.hunyuan3d.models",
            Hunyuan3DShapeDiTModel.__name__,
        ],
        "image_processor_border_ratio": image_processor_border_ratio,
        "image_processor_size": image_processor_size,
        "scheduler": [
            "diffusers_3d.families.hunyuan3d.scheduler",
            Hunyuan3DFlowMatchEulerDiscreteScheduler.__name__,
        ],
        "vae": [
            "diffusers_3d.families.hunyuan3d.vae",
            Hunyuan3DShapeVAE.__name__,
        ],
    }
    (output_directory / "model_index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Hunyuan3DImageToShapePipeline.object3d_model_index().save_pretrained(output_directory)


def convert_hunyuan3d_checkpoint(
    checkpoint_path: str | Path,
    output_directory: str | Path,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    conditioner_path: str | Path | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Convert one official aggregate checkpoint without tensor cloning.

    Safetensors components are read one at a time. PyTorch ``.ckpt`` files use
    memory-mapped loading when supported by their serialization format. Models
    are instantiated on the meta device and assigned source tensors, avoiding a
    second initialized copy of multi-gigabyte parameters.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    loaded_config = _load_config(config)
    checkpoint_data = None if checkpoint.suffix.lower() == ".safetensors" else _load_torch_checkpoint(checkpoint)

    denoiser_config = _component_config(loaded_config, "model") or Hunyuan3DShapeDiTModel.production_config()
    vae_config = _component_config(loaded_config, "vae") or Hunyuan3DShapeVAE.production_config()
    conditioner_config = (
        _conditioner_config(loaded_config)
        if "conditioner" in loaded_config
        else Hunyuan3DDinov2Conditioner.production_config()
    )

    denoiser_state = _load_component_state(checkpoint, "model", checkpoint_data)
    if not denoiser_state:
        raise ValueError("checkpoint has no model component")
    _save_strict_component(
        Hunyuan3DShapeDiTModel,
        denoiser_config,
        denoiser_state,
        destination / "denoiser",
        safe_serialization=safe_serialization,
    )
    del denoiser_state
    gc.collect()

    vae_state = _load_component_state(checkpoint, "vae", checkpoint_data)
    if not vae_state:
        raise ValueError("checkpoint has no vae component")
    unsupported_vae_keys = _save_vae_decoder(
        vae_config,
        vae_state,
        destination / "vae",
        safe_serialization=safe_serialization,
    )
    del vae_state
    gc.collect()

    conditioner_state = _load_component_state(checkpoint, "conditioner", checkpoint_data)
    if conditioner_state:
        _save_strict_component(
            Hunyuan3DDinov2Conditioner,
            conditioner_config,
            _map_conditioner_state(conditioner_state),
            destination / "conditioner",
            safe_serialization=safe_serialization,
        )
    elif conditioner_path is not None:
        conditioner = Hunyuan3DDinov2Conditioner.from_dinov2_pretrained(
            str(conditioner_path),
            image_size=conditioner_config["image_size"],
            use_cls_token=conditioner_config["use_cls_token"],
            local_files_only=True,
        )
        conditioner.save_pretrained(destination / "conditioner", safe_serialization=safe_serialization)
        del conditioner
    else:
        raise ValueError(
            "checkpoint has no conditioner weights; pass conditioner_path pointing to a local DINOv2 model"
        )
    del conditioner_state
    del checkpoint_data
    gc.collect()

    scheduler_config = _component_config(loaded_config, "scheduler")
    scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler(**(scheduler_config or {"num_train_timesteps": 1000}))
    scheduler.save_pretrained(destination / "scheduler")
    image_processor_config = _component_config(loaded_config, "image_processor")
    image_processor_size = int(image_processor_config.get("size", 512))
    image_processor_border_ratio = float(image_processor_config.get("border_ratio", 0.15))
    _write_pipeline_index(
        destination,
        image_processor_size=image_processor_size,
        image_processor_border_ratio=image_processor_border_ratio,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "decode_only_vae": True,
        "reference_revision": HUNYUAN3D_REFERENCE_REVISION,
        "unsupported_vae_keys": list(unsupported_vae_keys),
    }
    (destination / "hunyuan3d_conversion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--conditioner-path", type=Path)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_hunyuan3d_checkpoint(
        args.checkpoint,
        args.output_directory,
        config=args.config,
        conditioner_path=args.conditioner_path,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HUNYUAN3D_REFERENCE_REVISION",
    "convert_hunyuan3d_checkpoint",
    "main",
]
