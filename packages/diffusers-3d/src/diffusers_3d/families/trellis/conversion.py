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

from .decoders import (
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
)
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .pipeline import (
    CLIP_REPO,
    CLIP_REVISION,
    DINOV2_REPO,
    DINOV2_REVISION,
    TrellisImageTo3DPipeline,
    TrellisTextTo3DPipeline,
)
from .scheduler import TrellisFlowEulerScheduler

TRELLIS_REFERENCE_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"

_COMPONENT_TYPES = {
    "sparse_structure_flow_model": {"SparseStructureFlowModel": TrellisSparseStructureFlowModel},
    "sparse_structure_decoder": {"SparseStructureDecoder": TrellisSparseStructureDecoder},
    "slat_flow_model": {"SLatFlowModel": TrellisSLatFlowModel, "ElasticSLatFlowModel": TrellisSLatFlowModel},
    "slat_decoder_gs": {
        "SLatGaussianDecoder": TrellisSLatGaussianDecoder,
        "ElasticSLatGaussianDecoder": TrellisSLatGaussianDecoder,
    },
    "slat_decoder_mesh": {"SLatMeshDecoder": TrellisSLatMeshDecoder, "ElasticSLatMeshDecoder": TrellisSLatMeshDecoder},
    "slat_decoder_rf": {
        "SLatRadianceFieldDecoder": TrellisSLatRadianceFieldDecoder,
        "ElasticSLatRadianceFieldDecoder": TrellisSLatRadianceFieldDecoder,
    },
}
_UPSTREAM_COMPONENTS = set(_COMPONENT_TYPES)
# Upstream ``pipeline.json`` model keys -> pipeline component subfolders.
_OUTPUT_NAMES = {
    "sparse_structure_flow_model": "sparse_structure_flow_model",
    "sparse_structure_decoder": "sparse_structure_decoder",
    "slat_flow_model": "slat_flow_model",
    "slat_decoder_gs": "gaussian_decoder",
    "slat_decoder_mesh": "mesh_decoder",
    "slat_decoder_rf": "radiance_field_decoder",
}
_SHARED_PIPELINE_ARGUMENTS = {"models", "slat_normalization", "slat_sampler", "sparse_structure_sampler"}
# Upstream pipeline name -> (package pipeline, upstream encoder argument, released encoder id)
_PIPELINES = {
    "TrellisImageTo3DPipeline": (TrellisImageTo3DPipeline, "image_cond_model", "dinov2_vitl14_reg"),
    "TrellisTextTo3DPipeline": (TrellisTextTo3DPipeline, "text_cond_model", "openai/clip-vit-large-patch14"),
}
_ENCODER_CLASSES = {
    "image_encoder": "Dinov2WithRegistersModel",
    "text_encoder": "CLIPTextModel",
    "tokenizer": "CLIPTokenizer",
}
# Encoder checkpoints (Hub id, revision) the pipelines load at run time; conversion never writes them.
TRELLIS_ENCODER_CHECKPOINTS = {
    TrellisImageTo3DPipeline: (DINOV2_REPO, DINOV2_REVISION),
    TrellisTextTo3DPipeline: (CLIP_REPO, CLIP_REVISION),
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
    model_types: Mapping[str, type[Any]],
    family: str,
    safe_serialization: bool,
) -> tuple[type[Any], dict[str, Any]]:
    """Build the upstream component on the meta device, load its weights strictly, and save it in Diffusers layout."""

    model_name, model_config = _load_component_descriptor(source_stem.with_suffix(".json"))
    if model_name not in model_types:
        raise ValueError(
            f"unsupported {family} model {model_name!r} for {component_key!r}; expected {sorted(model_types)}"
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
    raise FileNotFoundError(f"component pair does not exist for reference {reference!r}")


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
    normalized = {"steps": steps, "cfg_strength": cfg_strength, "cfg_interval": cfg_interval, "rescale_t": rescale_t}
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


def _validate_normalization(value: object, *, name: str = "slat_normalization") -> tuple[list[float], list[float]]:
    if not isinstance(value, Mapping) or set(value) != {"mean", "std"}:
        raise ValueError(f"{name} must contain exact mean/std arrays")
    try:
        mean = [float(item) for item in value["mean"]]
        std = [float(item) for item in value["std"]]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} mean/std must be numeric arrays") from error
    if (
        not mean
        or len(mean) != len(std)
        or not all(math.isfinite(item) for item in mean)
        or not all(math.isfinite(item) and item > 0 for item in std)
    ):
        raise ValueError(f"{name} must contain matching finite means and positive standard deviations")
    return mean, std


def convert_trellis_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    safe_serialization: bool = True,
) -> Path:
    """Convert a local upstream image or text pipeline without importing TRELLIS at runtime.

    ``pipeline.json`` selects the target: ``TrellisImageTo3DPipeline`` or ``TrellisTextTo3DPipeline``. Every released
    TRELLIS network is converted: the sparse-structure flow and decoder, the SLAT flow model, and the Gaussian, mesh,
    and radiance-field decoders. The DINOv2 or CLIP encoders are not: ``model_index.json`` names them, and the
    pipeline loads them from their own repositories at the pinned ``TRELLIS_ENCODER_CHECKPOINTS`` revisions.
    """

    source = Path(source_directory)
    pipeline_path = source / "pipeline.json"
    pipeline_config = _load_json(pipeline_path)
    if set(pipeline_config) != {"name", "args"}:
        raise ValueError("pipeline.json must contain exactly 'name' and 'args'")
    if pipeline_config["name"] not in _PIPELINES:
        raise ValueError(f"pipeline.json must describe one of {sorted(_PIPELINES)}")
    pipeline_type, encoder_argument, released_encoder = _PIPELINES[pipeline_config["name"]]
    args = pipeline_config["args"]
    if not isinstance(args, Mapping):
        raise ValueError("pipeline args must be a mapping")
    if set(args) != _SHARED_PIPELINE_ARGUMENTS | {encoder_argument}:
        raise ValueError(
            f"pipeline args must contain exactly {sorted(_SHARED_PIPELINE_ARGUMENTS | {encoder_argument})}"
        )
    if args[encoder_argument] != released_encoder:
        raise ValueError(f"only the released {released_encoder!r} encoder is supported")
    models = args.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("pipeline args.models must be a mapping")
    missing = _UPSTREAM_COMPONENTS.difference(models)
    unknown = set(models).difference(_UPSTREAM_COMPONENTS)
    if missing or unknown:
        raise ValueError(f"pipeline component mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    component_index: dict[str, list[str]] = {}
    component_report: dict[str, dict[str, Any]] = {}
    for component_key, reference in models.items():
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"pipeline model reference {component_key!r} must be a non-empty string")
        output_name = _OUTPUT_NAMES[component_key]
        model_type, model_config = _save_component(
            component_key,
            _component_source(source, reference),
            destination / output_name,
            model_types=_COMPONENT_TYPES[component_key],
            family="TRELLIS",
            safe_serialization=safe_serialization,
        )
        component_index[output_name] = [model_type.__module__, model_type.__name__]
        component_report[component_key] = {
            "source": reference,
            "target": output_name,
            "class": model_type.__name__,
            "config": model_config,
        }

    for name in pipeline_type.external_components:
        component_index[name] = ["transformers", _ENCODER_CLASSES[name]]

    sigma_min, sparse_sampler_params = _validate_sampler(
        args["sparse_structure_sampler"], name="sparse_structure_sampler"
    )
    TrellisFlowEulerScheduler(sigma_min=sigma_min).save_pretrained(destination / "sparse_structure_scheduler")
    scheduler_entry = [TrellisFlowEulerScheduler.__module__, TrellisFlowEulerScheduler.__name__]
    component_index["sparse_structure_scheduler"] = scheduler_entry

    slat_sigma_min, slat_sampler_params = _validate_sampler(args["slat_sampler"], name="slat_sampler")
    TrellisFlowEulerScheduler(sigma_min=slat_sigma_min).save_pretrained(destination / "slat_scheduler")
    component_index["slat_scheduler"] = scheduler_entry

    slat_mean, slat_std = _validate_normalization(args["slat_normalization"])
    if len(slat_mean) != int(component_report["slat_flow_model"]["config"]["out_channels"]):
        raise ValueError("SLAT normalization must contain one value per SLAT flow output channel")

    model_index: dict[str, Any] = {
        "_class_name": pipeline_type.__name__,
        "_diffusers_version": diffusers_version,
        **component_index,
        "slat_mean": slat_mean,
        "slat_std": slat_std,
    }
    (destination / "model_index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "components": component_report,
        "reference_revision": TRELLIS_REFERENCE_REVISION,
        "samplers": {"slat": slat_sampler_params, "sparse_structure": sparse_sampler_params},
        "skipped_components": {},
        "source_pipeline": str(pipeline_path.resolve()),
        "production_gpu_quality_verified": False,
    }
    (destination / "trellis_conversion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_trellis_checkpoint(
        args.source_directory,
        args.output_directory,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TRELLIS_ENCODER_CHECKPOINTS", "TRELLIS_REFERENCE_REVISION", "convert_trellis_checkpoint", "main"]
