"""Convert local TRELLIS component JSON/safetensors pairs into Diffusers folders."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers import __version__ as diffusers_version
from safetensors.torch import load_file

from .conditioner import TrellisDinov2Conditioner
from .decoders import TrellisSLatGaussianDecoder, TrellisSparseStructureDecoder
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .pipeline import TrellisImageTo3DPipeline
from .scheduler import TrellisFlowEulerScheduler

TRELLIS_REFERENCE_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"

_COMPONENT_TYPES = {
    "sparse_structure_flow_model": {
        "SparseStructureFlowModel": TrellisSparseStructureFlowModel,
    },
    "sparse_structure_decoder": {
        "SparseStructureDecoder": TrellisSparseStructureDecoder,
    },
    "slat_flow_model": {
        "SLatFlowModel": TrellisSLatFlowModel,
        "ElasticSLatFlowModel": TrellisSLatFlowModel,
    },
    "slat_decoder_gs": {
        "SLatGaussianDecoder": TrellisSLatGaussianDecoder,
        "ElasticSLatGaussianDecoder": TrellisSLatGaussianDecoder,
    },
}
_EXPERIMENTAL_COMPONENTS = {"slat_flow_model", "slat_decoder_gs"}
_UNSUPPORTED_COMPONENTS = {"slat_decoder_mesh", "slat_decoder_rf"}
_UPSTREAM_COMPONENTS = set(_COMPONENT_TYPES) | _UNSUPPORTED_COMPONENTS
_PIPELINE_ARGUMENTS = {
    "image_cond_model",
    "models",
    "slat_normalization",
    "slat_sampler",
    "sparse_structure_sampler",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return value


def _load_component_descriptor(path: Path) -> tuple[str, dict[str, Any]]:
    descriptor = _load_json(path)
    if set(descriptor) != {"name", "args"}:
        raise ValueError(f"component descriptor {path} must contain exactly 'name' and 'args'")
    if not isinstance(descriptor["name"], str) or not isinstance(descriptor["args"], Mapping):
        raise ValueError(f"component descriptor {path} has invalid name or args")
    return descriptor["name"], dict(descriptor["args"])


def _save_component(
    component_key: str,
    source_stem: Path,
    destination: Path,
    *,
    safe_serialization: bool,
) -> tuple[type[Any], dict[str, Any]]:
    model_name, model_config = _load_component_descriptor(source_stem.with_suffix(".json"))
    model_types = _COMPONENT_TYPES[component_key]
    if model_name not in model_types:
        raise ValueError(
            f"unsupported TRELLIS model {model_name!r} for {component_key!r}; expected {sorted(model_types)}"
        )
    weights_path = source_stem.with_suffix(".safetensors")
    if not weights_path.is_file():
        raise FileNotFoundError(f"component weights do not exist: {weights_path}")
    model_type = model_types[model_name]
    with torch.device("meta"):
        model = model_type(**model_config)
    state_dict = load_file(weights_path, device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"{component_key} state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.save_pretrained(destination, safe_serialization=safe_serialization)
    return model_type, model_config


def _component_source(root: Path, reference: str) -> Path:
    candidate = root / reference
    if candidate.with_suffix(".json").is_file() and candidate.with_suffix(".safetensors").is_file():
        return candidate
    direct = Path(reference)
    if direct.with_suffix(".json").is_file() and direct.with_suffix(".safetensors").is_file():
        return direct
    raise FileNotFoundError(f"TRELLIS component pair does not exist for reference {reference!r}")


def _validate_sampler(value: object, *, name: str) -> tuple[float, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"name", "args", "params"}:
        raise ValueError(f"{name} must contain exactly 'name', 'args', and 'params'")
    if value["name"] != "FlowEulerGuidanceIntervalSampler":
        raise ValueError(f"{name} must use FlowEulerGuidanceIntervalSampler")
    args = value["args"]
    params = value["params"]
    if not isinstance(args, Mapping) or set(args) != {"sigma_min"}:
        raise ValueError(f"{name}.args must contain exactly 'sigma_min'")
    if not isinstance(params, Mapping) or set(params) != {"steps", "cfg_strength", "cfg_interval", "rescale_t"}:
        raise ValueError(
            f"{name}.params must contain exactly 'steps', 'cfg_strength', 'cfg_interval', and 'rescale_t'"
        )
    steps = params["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError(f"{name}.params.steps must be a positive integer")
    try:
        cfg_interval = [float(item) for item in params["cfg_interval"]]
        cfg_strength = float(params["cfg_strength"])
        rescale_t = float(params["rescale_t"])
        sigma_min = float(args["sigma_min"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains non-numeric sampler values") from error
    normalized = {
        "steps": steps,
        "cfg_strength": cfg_strength,
        "cfg_interval": cfg_interval,
        "rescale_t": rescale_t,
    }
    if (
        not math.isfinite(sigma_min)
        or not 0 <= sigma_min < 1
        or normalized["steps"] <= 0
        or not math.isfinite(normalized["cfg_strength"])
        or len(normalized["cfg_interval"]) != 2
        or not all(math.isfinite(item) for item in normalized["cfg_interval"])
        or not 0 <= normalized["cfg_interval"][0] <= normalized["cfg_interval"][1] <= 1
        or not math.isfinite(normalized["rescale_t"])
        or normalized["rescale_t"] <= 0
    ):
        raise ValueError(f"{name}.params contains invalid sampler values")
    return sigma_min, normalized


def _validate_normalization(value: object) -> tuple[list[float], list[float]]:
    if not isinstance(value, Mapping) or set(value) != {"mean", "std"}:
        raise ValueError("slat_normalization must contain exact mean/std arrays")
    try:
        mean = [float(item) for item in value["mean"]]
        std = [float(item) for item in value["std"]]
    except (TypeError, ValueError) as error:
        raise ValueError("slat_normalization mean/std must be numeric arrays") from error
    if (
        not mean
        or len(mean) != len(std)
        or not all(math.isfinite(item) for item in mean)
        or not all(math.isfinite(item) and item > 0 for item in std)
    ):
        raise ValueError("slat_normalization must contain matching finite means and positive standard deviations")
    return mean, std


def convert_trellis_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    conditioner_path: str | Path,
    safe_serialization: bool = True,
    include_experimental: bool = False,
) -> Path:
    """Convert a local upstream pipeline without importing TRELLIS at runtime."""

    source = Path(source_directory)
    pipeline_path = source / "pipeline.json"
    pipeline_config = _load_json(pipeline_path)
    if set(pipeline_config) != {"name", "args"}:
        raise ValueError("pipeline.json must contain exactly 'name' and 'args'")
    if pipeline_config["name"] != "TrellisImageTo3DPipeline":
        raise ValueError("pipeline.json must describe TrellisImageTo3DPipeline")
    args = pipeline_config["args"]
    if not isinstance(args, Mapping):
        raise ValueError("pipeline args must be a mapping")
    if set(args) != _PIPELINE_ARGUMENTS:
        raise ValueError(f"pipeline args must contain exactly {sorted(_PIPELINE_ARGUMENTS)}")
    if args["image_cond_model"] != "dinov2_vitl14_reg":
        raise ValueError("only the released dinov2_vitl14_reg image conditioner is supported")
    models = args.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("pipeline args.models must be a mapping")
    missing = _UPSTREAM_COMPONENTS.difference(models)
    unknown = set(models).difference(_UPSTREAM_COMPONENTS)
    if missing or unknown:
        raise ValueError(f"pipeline component mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    component_index: dict[str, list[str | None]] = {}
    component_report: dict[str, dict[str, Any]] = {}
    skipped_components: dict[str, str] = {}
    output_names = {
        "sparse_structure_flow_model": "sparse_structure_flow_model",
        "sparse_structure_decoder": "sparse_structure_decoder",
        "slat_flow_model": "slat_flow_model",
        "slat_decoder_gs": "gaussian_decoder",
    }
    for component_key, reference in models.items():
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"pipeline model reference {component_key!r} must be a non-empty string")
        if component_key in _UNSUPPORTED_COMPONENTS:
            skipped_components[component_key] = (
                "the package has no reviewed object-native decoder for this upstream component"
            )
            continue
        if component_key in _EXPERIMENTAL_COMPONENTS and not include_experimental:
            skipped_components[component_key] = "experimental sparse SLAT conversion was not explicitly requested"
            continue
        model_type, model_config = _save_component(
            component_key,
            _component_source(source, reference),
            destination / output_names[component_key],
            safe_serialization=safe_serialization,
        )
        component_index[output_names[component_key]] = [model_type.__module__, model_type.__name__]
        component_report[component_key] = {
            "source": reference,
            "target": output_names[component_key],
            "class": model_type.__name__,
            "config": model_config,
        }

    conditioner = TrellisDinov2Conditioner.from_pretrained(
        conditioner_path,
        local_files_only=True,
    )
    conditioner.save_pretrained(destination / "conditioner", safe_serialization=safe_serialization)
    component_index["conditioner"] = [TrellisDinov2Conditioner.__module__, TrellisDinov2Conditioner.__name__]

    sigma_min, sparse_sampler_params = _validate_sampler(
        args["sparse_structure_sampler"],
        name="sparse_structure_sampler",
    )
    scheduler = TrellisFlowEulerScheduler(sigma_min=sigma_min)
    scheduler.save_pretrained(destination / "sparse_structure_scheduler")
    component_index["sparse_structure_scheduler"] = [
        TrellisFlowEulerScheduler.__module__,
        TrellisFlowEulerScheduler.__name__,
    ]

    slat_sigma_min, slat_sampler_params = _validate_sampler(args["slat_sampler"], name="slat_sampler")
    converted_slat = "slat_flow_model" in component_index
    if converted_slat:
        slat_scheduler = TrellisFlowEulerScheduler(sigma_min=slat_sigma_min)
        slat_scheduler.save_pretrained(destination / "slat_scheduler")
        component_index["slat_scheduler"] = [
            TrellisFlowEulerScheduler.__module__,
            TrellisFlowEulerScheduler.__name__,
        ]
    else:
        component_index["slat_flow_model"] = [None, None]
        component_index["slat_scheduler"] = [None, None]
    if "gaussian_decoder" not in component_index:
        component_index["gaussian_decoder"] = [None, None]

    normalization_mean, normalization_std = _validate_normalization(args["slat_normalization"])
    if converted_slat:
        slat_channels = int(component_report["slat_flow_model"]["config"]["out_channels"])
        if len(normalization_mean) != slat_channels:
            raise ValueError("SLAT normalization must contain one value per SLAT flow output channel")
        slat_mean = normalization_mean
        slat_std = normalization_std
    else:
        slat_mean = None
        slat_std = None

    model_index: dict[str, Any] = {
        "_class_name": TrellisImageTo3DPipeline.__name__,
        "_diffusers_version": diffusers_version,
        **component_index,
        "slat_mean": slat_mean,
        "slat_std": slat_std,
    }
    (destination / "model_index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    TrellisImageTo3DPipeline.object3d_model_index().save_pretrained(destination)
    report = {
        "components": component_report,
        "reference_revision": TRELLIS_REFERENCE_REVISION,
        "samplers": {
            "slat": slat_sampler_params,
            "sparse_structure": sparse_sampler_params,
        },
        "skipped_components": skipped_components,
        "source_pipeline": str(pipeline_path.resolve()),
        "production_slat_checkpoint_parity": False,
        "production_gaussian_checkpoint_parity": False,
    }
    (destination / "trellis_conversion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--conditioner-path", type=Path, required=True)
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_trellis_checkpoint(
        args.source_directory,
        args.output_directory,
        conditioner_path=args.conditioner_path,
        safe_serialization=not args.no_safe_serialization,
        include_experimental=args.include_experimental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TRELLIS_REFERENCE_REVISION", "convert_trellis_checkpoint", "main"]
