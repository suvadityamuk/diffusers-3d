"""Convert local TRELLIS.2 component JSON/safetensors pairs into Diffusers folders."""

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

from .conditioner import Trellis2Dinov3Conditioner
from .decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
from .models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
from .pipeline import Trellis2ImageTo3DPipeline
from .scheduler import Trellis2FlowEulerScheduler

TRELLIS2_REFERENCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"

_COMPONENT_TYPES = {
    "sparse_structure_flow_model": {
        "SparseStructureFlowModel": Trellis2SparseStructureFlowModel,
    },
    "sparse_structure_decoder": {
        "SparseStructureDecoder": Trellis2SparseStructureDecoder,
    },
    "shape_slat_flow_model_512": {
        "SLatFlowModel": Trellis2SLatFlowModel,
        "ElasticSLatFlowModel": Trellis2SLatFlowModel,
    },
    "tex_slat_flow_model_512": {
        "SLatFlowModel": Trellis2SLatFlowModel,
        "ElasticSLatFlowModel": Trellis2SLatFlowModel,
    },
    "shape_slat_decoder": {
        "FlexiDualGridVaeDecoder": Trellis2ShapeDualGridDecoder,
    },
    "tex_slat_decoder": {
        "SparseUnetVaeDecoder": Trellis2PBRSparseDecoder,
    },
}
_EXPERIMENTAL_COMPONENTS = {
    "shape_slat_flow_model_512",
    "tex_slat_flow_model_512",
    "shape_slat_decoder",
    "tex_slat_decoder",
}
_UNSUPPORTED_CASCADE_COMPONENTS = {
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_1024",
}
_UPSTREAM_COMPONENTS = set(_COMPONENT_TYPES) | _UNSUPPORTED_CASCADE_COMPONENTS
_PIPELINE_ARGUMENTS = {
    "default_pipeline_type",
    "image_cond_model",
    "models",
    "rembg_model",
    "shape_slat_normalization",
    "shape_slat_sampler",
    "sparse_structure_sampler",
    "tex_slat_normalization",
    "tex_slat_sampler",
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


def _component_source(root: Path, reference: str) -> Path:
    candidate = root / reference
    if candidate.with_suffix(".json").is_file() and candidate.with_suffix(".safetensors").is_file():
        return candidate
    direct = Path(reference)
    if direct.with_suffix(".json").is_file() and direct.with_suffix(".safetensors").is_file():
        return direct
    raise FileNotFoundError(f"TRELLIS.2 component pair does not exist for reference {reference!r}")


def _portable_experimental_config(component_key: str, config: Mapping[str, Any]) -> bool:
    if component_key.endswith("flow_model_512"):
        return config.get("require_flex_gemm") is False
    return config.get("portable_tiny") is True


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
            f"unsupported TRELLIS.2 model {model_name!r} for {component_key!r}; expected {sorted(model_types)}"
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


def _validate_sampler(value: object, *, name: str) -> tuple[float, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"name", "args", "params"}:
        raise ValueError(f"{name} must contain exactly 'name', 'args', and 'params'")
    if value["name"] != "FlowEulerGuidanceIntervalSampler":
        raise ValueError(f"{name} must use FlowEulerGuidanceIntervalSampler")
    args = value["args"]
    params = value["params"]
    expected_params = {"steps", "guidance_strength", "guidance_rescale", "guidance_interval", "rescale_t"}
    if not isinstance(args, Mapping) or set(args) != {"sigma_min"}:
        raise ValueError(f"{name}.args must contain exactly 'sigma_min'")
    if not isinstance(params, Mapping) or set(params) != expected_params:
        raise ValueError(f"{name}.params must contain exactly {sorted(expected_params)}")
    try:
        normalized = {
            "steps": int(params["steps"]),
            "guidance_strength": float(params["guidance_strength"]),
            "guidance_rescale": float(params["guidance_rescale"]),
            "guidance_interval": [float(value) for value in params["guidance_interval"]],
            "rescale_t": float(params["rescale_t"]),
        }
        sigma_min = float(args["sigma_min"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains non-numeric sampler values") from error
    if (
        isinstance(params["steps"], bool)
        or normalized["steps"] <= 0
        or not math.isfinite(sigma_min)
        or not 0 <= sigma_min < 1
        or not math.isfinite(normalized["guidance_strength"])
        or not 0 <= normalized["guidance_rescale"] <= 1
        or len(normalized["guidance_interval"]) != 2
        or not 0 <= normalized["guidance_interval"][0] <= normalized["guidance_interval"][1] <= 1
        or normalized["rescale_t"] <= 0
    ):
        raise ValueError(f"{name} contains invalid sampler values")
    return sigma_min, normalized


def _validate_normalization(value: object, *, name: str) -> tuple[list[float], list[float]]:
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


def convert_trellis2_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    conditioner_path: str | Path,
    safe_serialization: bool = True,
    include_experimental: bool = False,
) -> Path:
    """Convert reviewed components and explicitly compatible tiny experimental components."""

    source = Path(source_directory)
    pipeline_path = source / "pipeline.json"
    pipeline_config = _load_json(pipeline_path)
    if set(pipeline_config) != {"name", "args"} or pipeline_config["name"] != "Trellis2ImageTo3DPipeline":
        raise ValueError("pipeline.json must describe Trellis2ImageTo3DPipeline with exact name/args keys")
    args = pipeline_config["args"]
    if not isinstance(args, Mapping) or set(args) != _PIPELINE_ARGUMENTS:
        raise ValueError(f"pipeline args must contain exactly {sorted(_PIPELINE_ARGUMENTS)}")
    image_model = args["image_cond_model"]
    if (
        not isinstance(image_model, Mapping)
        or image_model.get("name") != "DinoV3FeatureExtractor"
        or not isinstance(image_model.get("args"), Mapping)
    ):
        raise ValueError("only the released DinoV3FeatureExtractor image conditioner is supported")
    models = args["models"]
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
        "shape_slat_flow_model_512": "shape_slat_flow_model",
        "shape_slat_decoder": "shape_slat_decoder",
        "tex_slat_flow_model_512": "texture_slat_flow_model",
        "tex_slat_decoder": "pbr_decoder",
    }
    for component_key, reference in models.items():
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"pipeline model reference {component_key!r} must be a non-empty string")
        if component_key in _UNSUPPORTED_CASCADE_COMPONENTS:
            skipped_components[component_key] = (
                "the full 1024 cascade is unsupported until production sparse/GPU parity is measured"
            )
            continue
        if component_key in _EXPERIMENTAL_COMPONENTS:
            if not include_experimental:
                skipped_components[component_key] = "experimental O-Voxel conversion was not explicitly requested"
                continue
            source_stem = _component_source(source, reference)
            _, component_config = _load_component_descriptor(source_stem.with_suffix(".json"))
            if not _portable_experimental_config(component_key, component_config):
                skipped_components[component_key] = (
                    "official production sparse weights are intentionally not loaded by the backend-free tiny class"
                )
                continue
        else:
            source_stem = _component_source(source, reference)
        model_type, model_config = _save_component(
            component_key,
            source_stem,
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

    conditioner = Trellis2Dinov3Conditioner.from_pretrained(conditioner_path, local_files_only=True)
    conditioner.save_pretrained(destination / "conditioner", safe_serialization=safe_serialization)
    component_index["conditioner"] = [Trellis2Dinov3Conditioner.__module__, Trellis2Dinov3Conditioner.__name__]

    sampler_values = {}
    for source_name, output_name in (
        ("sparse_structure_sampler", "sparse_structure_scheduler"),
        ("shape_slat_sampler", "shape_slat_scheduler"),
        ("tex_slat_sampler", "texture_slat_scheduler"),
    ):
        sigma_min, parameters = _validate_sampler(args[source_name], name=source_name)
        sampler_values[source_name] = parameters
        required_component = {
            "sparse_structure_scheduler": "sparse_structure_flow_model",
            "shape_slat_scheduler": "shape_slat_flow_model",
            "texture_slat_scheduler": "texture_slat_flow_model",
        }[output_name]
        if required_component in component_index:
            scheduler = Trellis2FlowEulerScheduler(sigma_min=sigma_min)
            scheduler.save_pretrained(destination / output_name)
            component_index[output_name] = [
                Trellis2FlowEulerScheduler.__module__,
                Trellis2FlowEulerScheduler.__name__,
            ]
        else:
            component_index[output_name] = [None, None]

    for optional in ("shape_slat_flow_model", "shape_slat_decoder", "texture_slat_flow_model", "pbr_decoder"):
        component_index.setdefault(optional, [None, None])

    shape_mean, shape_std = _validate_normalization(
        args["shape_slat_normalization"],
        name="shape_slat_normalization",
    )
    texture_mean, texture_std = _validate_normalization(
        args["tex_slat_normalization"],
        name="tex_slat_normalization",
    )
    if "shape_slat_flow_model" in component_report:
        channels = int(component_report["shape_slat_flow_model"]["config"]["out_channels"])
        if len(shape_mean) != channels:
            raise ValueError("shape SLAT normalization must match converted flow output channels")
    else:
        shape_mean = shape_std = None
    if "texture_slat_flow_model" in component_report:
        channels = int(component_report["texture_slat_flow_model"]["config"]["out_channels"])
        if len(texture_mean) != channels:
            raise ValueError("texture SLAT normalization must match converted flow output channels")
    else:
        texture_mean = texture_std = None

    limitations = {
        "reviewed_formats": ["sparse_structure"],
        "experimental_formats": ["shape_slat", "texture_slat", "o_voxel", "mesh"],
        "production_1024_cascade": "unsupported_until_flex_gemm_ovoxel_gpu_parity",
        "official_full_checkpoint_parity": False,
        "production_gpu_quality_verified": False,
    }
    model_index: dict[str, Any] = {
        "_class_name": Trellis2ImageTo3DPipeline.__name__,
        "_diffusers_version": diffusers_version,
        **component_index,
        "shape_slat_mean": shape_mean,
        "shape_slat_std": shape_std,
        "texture_slat_mean": texture_mean,
        "texture_slat_std": texture_std,
        "default_pipeline_type": args["default_pipeline_type"],
        "sparse_structure_sampler_defaults": sampler_values["sparse_structure_sampler"],
        "shape_slat_sampler_defaults": sampler_values["shape_slat_sampler"],
        "texture_slat_sampler_defaults": sampler_values["tex_slat_sampler"],
        "capability_limitations": limitations,
    }
    (destination / "model_index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Trellis2ImageTo3DPipeline.object3d_model_index().save_pretrained(destination)
    report = {
        "components": component_report,
        "reference_revision": TRELLIS2_REFERENCE_REVISION,
        "samplers": sampler_values,
        "skipped_components": skipped_components,
        "source_pipeline": str(pipeline_path.resolve()),
        "reviewed_sparse_structure_conversion": True,
        "production_sparse_ovoxel_checkpoint_parity": False,
        "production_gpu_quality_verified": False,
    }
    (destination / "trellis2_conversion.json").write_text(
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
    convert_trellis2_checkpoint(
        args.source_directory,
        args.output_directory,
        conditioner_path=args.conditioner_path,
        safe_serialization=not args.no_safe_serialization,
        include_experimental=args.include_experimental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TRELLIS2_REFERENCE_REVISION", "convert_trellis2_checkpoint", "main"]
