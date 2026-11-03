"""Convert a released Pixal3D checkpoint (``pipeline.json`` or ``pipeline_mv.json``) into a Diffusers pipeline folder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers import __version__ as diffusers_version
from safetensors.torch import load_file

from ..moge.models import MoGe2Model
from ..trellis.conversion import _component_source, _load_component_descriptor, _load_json, _validate_normalization
from ..trellis2.conversion import _OUTPUT_NAMES, _PIPELINE_ARGUMENTS, _validate_sampler
from ..trellis2.decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
from ..trellis2.models import Trellis2RotaryPositionEmbedder
from ..trellis2.pipeline import RELEASED_STEMS
from ..trellis2.scheduler import Trellis2FlowEulerScheduler
from .conditioner import NAF_CHECKPOINT_SHA256, Pixal3DNAFUpsampler, _naf_block_key, _naf_state_dict  # noqa: F401
from .models import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from .pipeline import Pixal3DImageTo3DPipeline

_EXTERNAL_STEMS = {
    name: RELEASED_STEMS[name] for name in ("sparse_structure_decoder", "shape_slat_decoder", "pbr_decoder")
}

PIXAL3D_REFERENCE_REVISION = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"
PIXAL3D_WEIGHTS_REVISION = "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"
NAF_REFERENCE_REVISION = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"

# ``pipeline.json`` keeps TRELLIS.2's pipeline name; ``pipeline_mv.json`` names the multi-view variant.
_PIPELINE_NAMES = {"Trellis2ImageTo3DPipeline": False, "Pixal3DMVImageTo3DPipeline": True}
_SLAT_FLOW_TYPES = {"SLatFlowModel": Pixal3DSLatFlowModel, "ElasticSLatFlowModel": Pixal3DSLatFlowModel}
_COMPONENT_TYPES = {
    "sparse_structure_flow_model": {"SparseStructureFlowModel": Pixal3DSparseStructureFlowModel},
    "sparse_structure_decoder": {"SparseStructureDecoder": Trellis2SparseStructureDecoder},
    "shape_slat_flow_model_512": _SLAT_FLOW_TYPES,
    "shape_slat_flow_model_1024": _SLAT_FLOW_TYPES,
    "tex_slat_flow_model_512": _SLAT_FLOW_TYPES,
    "tex_slat_flow_model_1024": _SLAT_FLOW_TYPES,
    "shape_slat_decoder": {"FlexiDualGridVaeDecoder": Trellis2ShapeDualGridDecoder},
    "tex_slat_decoder": {"SparseUnetVaeDecoder": Trellis2PBRSparseDecoder},
}
_REQUIRED_COMPONENTS = {"sparse_structure_flow_model", "sparse_structure_decoder"}
_FLOW_TYPES = (Pixal3DSparseStructureFlowModel, Pixal3DSLatFlowModel)


def _rope_phases(model: Pixal3DSparseStructureFlowModel) -> torch.Tensor:
    axis = torch.arange(model.resolution)
    coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    rope_freq = tuple(float(value) for value in model.config.rope_freq)
    return Trellis2RotaryPositionEmbedder(model.model_channels // model.num_heads, rope_freq=rope_freq)(coordinates)


def _save_component(
    component_key: str, source_stem: Path, destination: Path, *, safe_serialization: bool
) -> tuple[type[Any], dict[str, Any]]:
    model_name, model_config = _load_component_descriptor(source_stem.with_suffix(".json"))
    model_types = _COMPONENT_TYPES[component_key]
    if model_name not in model_types:
        raise ValueError(
            f"unsupported Pixal3D model {model_name!r} for {component_key!r}; expected {sorted(model_types)}"
        )
    model_type = model_types[model_name]
    state_dict = load_file(source_stem.with_suffix(".safetensors"), device="cpu")
    if model_type in _FLOW_TYPES:
        if model_config.pop("image_attn_mode", None) != "proj":
            raise ValueError(f"{component_key} must be a projection-conditioned (image_attn_mode='proj') flow")
        model_config.setdefault("proj_in_channels", model_config["cond_channels"])
    with torch.device("meta"):
        model = model_type(**model_config)
    if "rope_phases" in state_dict:
        # The dense flow checkpoint stores its RoPE phases; the port derives them from the config.
        stored = state_dict.pop("rope_phases")
        if model_type is not Pixal3DSparseStructureFlowModel or not torch.allclose(
            stored, _rope_phases(model), atol=1e-5, rtol=0.0
        ):
            raise ValueError(f"{component_key} stores RoPE phases that differ from its configuration")
    # The release stores every tensor in float32; keep the configured bfloat16 torso instead of doubling the size.
    dtypes = {key: value.dtype for key, value in model.state_dict().items()}
    state_dict = {key: value.to(dtypes.get(key, value.dtype)) for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(destination, safe_serialization=safe_serialization)
    return model_type, model_config


def convert_pixal3d_checkpoint(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    config_file: str = "pipeline.json",
    depth_estimator: bool = True,
    safe_serialization: bool = True,
) -> Path:
    """Convert a released Pixal3D folder into a Diffusers pipeline folder.

    ``config_file`` picks the single-view (``pipeline.json``) or multi-view (``pipeline_mv.json``) flows; both share
    the TRELLIS.2 decoders. Only Pixal3D's own networks (the flows) are converted. The third-party networks it uses
    unchanged are named in ``model_index.json`` but never written, and the pipeline loads them from their original
    sources: the gated DINOv3 ``image_encoder`` (``facebook/dinov3-vitl16-pretrain-lvd1689m``), valeo.ai's NAF
    ``feature_upsampler`` (the ``naf_release.pth`` release, checked against its SHA-256), for single-view checkpoints
    with ``depth_estimator`` the MoGe-2 field-of-view estimator (``Ruicheng/moge-2-vitl``), and the release's three
    decoders, byte-identical copies of ``microsoft/TRELLIS-image-large`` (sparse structure) and
    ``microsoft/TRELLIS.2-4B`` (shape and PBR) files.
    """

    source = Path(source_directory)
    pipeline_path = source / config_file
    pipeline_config = _load_json(pipeline_path)
    if set(pipeline_config) != {"name", "args"} or pipeline_config["name"] not in _PIPELINE_NAMES:
        raise ValueError(f"{config_file} must describe one of {sorted(_PIPELINE_NAMES)} with exact name/args keys")
    multiview = _PIPELINE_NAMES[pipeline_config["name"]]
    args = pipeline_config["args"]
    if not isinstance(args, Mapping) or set(args) != _PIPELINE_ARGUMENTS:
        raise ValueError(f"pipeline args must contain exactly {sorted(_PIPELINE_ARGUMENTS)}")
    models = args["models"]
    if not isinstance(models, Mapping):
        raise ValueError("pipeline args.models must be a mapping")
    missing = _REQUIRED_COMPONENTS.difference(models)
    unknown = set(models).difference(_COMPONENT_TYPES)
    if missing or unknown:
        raise ValueError(f"pipeline component mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    component_index: dict[str, list[str | None]] = {}
    component_report: dict[str, dict[str, Any]] = {}
    for component_key, reference in models.items():
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"pipeline model reference {component_key!r} must be a non-empty string")
        output_name = _OUTPUT_NAMES[component_key]
        if output_name in _EXTERNAL_STEMS:
            # A byte-identical copy of a TRELLIS release file, which the pipeline loads from that release.
            if Path(reference).name != Path(_EXTERNAL_STEMS[output_name]).name:
                raise ValueError(
                    f"{component_key} must be the TRELLIS release's {_EXTERNAL_STEMS[output_name]}, which the pipeline "
                    f"loads; got {reference!r}"
                )
            model_type = next(iter(_COMPONENT_TYPES[component_key].values()))
            component_index[output_name] = [model_type.__module__, model_type.__name__]
            continue
        model_type, model_config = _save_component(
            component_key,
            _component_source(source, reference),
            destination / output_name,
            safe_serialization=safe_serialization,
        )
        component_index[output_name] = [model_type.__module__, model_type.__name__]
        component_report[output_name] = {"source": reference, "class": model_type.__name__, "config": model_config}

    component_index["image_encoder"] = ["transformers", "DINOv3ViTModel"]
    component_index["feature_upsampler"] = [Pixal3DNAFUpsampler.__module__, Pixal3DNAFUpsampler.__name__]
    if depth_estimator and not multiview:
        component_index["depth_estimator"] = [MoGe2Model.__module__, MoGe2Model.__name__]

    sampler_values = {}
    for source_name, output_name, required_component in (
        ("sparse_structure_sampler", "sparse_structure_scheduler", "sparse_structure_flow_model"),
        ("shape_slat_sampler", "shape_slat_scheduler", "shape_slat_flow_model"),
        ("tex_slat_sampler", "texture_slat_scheduler", "texture_slat_flow_model"),
    ):
        sigma_min, sampler_values[source_name] = _validate_sampler(args[source_name], name=source_name)
        if required_component in component_index or f"{required_component}_1024" in component_index:
            Trellis2FlowEulerScheduler(sigma_min=sigma_min).save_pretrained(destination / output_name)
            component_index[output_name] = [Trellis2FlowEulerScheduler.__module__, Trellis2FlowEulerScheduler.__name__]
    for component in Pixal3DImageTo3DPipeline._optional_components:
        component_index.setdefault(component, [None, None])

    def normalization(name: str, *flow_names: str) -> tuple[list[float] | None, list[float] | None]:
        mean, std = _validate_normalization(args[name], name=name)
        converted = [component_report[flow] for flow in flow_names if flow in component_report]
        if not converted:
            return None, None
        if any(len(mean) != int(flow["config"]["out_channels"]) for flow in converted):
            raise ValueError(f"{name} must match the converted flow output channels")
        return mean, std

    shape_mean, shape_std = normalization(
        "shape_slat_normalization", "shape_slat_flow_model", "shape_slat_flow_model_1024"
    )
    texture_mean, texture_std = normalization(
        "tex_slat_normalization", "texture_slat_flow_model", "texture_slat_flow_model_1024"
    )
    model_index: dict[str, Any] = {
        "_class_name": Pixal3DImageTo3DPipeline.__name__,
        "_diffusers_version": diffusers_version,
        **component_index,
        "shape_slat_mean": shape_mean,
        "shape_slat_std": shape_std,
        "texture_slat_mean": texture_mean,
        "texture_slat_std": texture_std,
        "default_pipeline_type": Pixal3DImageTo3DPipeline._validate_pipeline_type(args["default_pipeline_type"]),
        "image_size": int(args["image_cond_model"]["args"].get("image_size", 512)),
        "sparse_structure_sampler_defaults": sampler_values["sparse_structure_sampler"],
        "shape_slat_sampler_defaults": sampler_values["shape_slat_sampler"],
        "texture_slat_sampler_defaults": sampler_values["tex_slat_sampler"],
        "multiview": multiview,
        "capability_limitations": {
            "official_full_checkpoint_parity": False,
            "production_gpu_quality_verified": False,
        },
    }
    (destination / "model_index.json").write_text(json.dumps(model_index, indent=2, sort_keys=True) + "\n")
    report = {
        "components": component_report,
        "reference_revision": PIXAL3D_REFERENCE_REVISION,
        "naf_reference_revision": NAF_REFERENCE_REVISION,
        "samplers": sampler_values,
        "source_pipeline": str(pipeline_path.resolve()),
        "multiview": multiview,
        "production_gpu_quality_verified": False,
    }
    (destination / "pixal3d_conversion.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--config-file", default="pipeline.json", help="pipeline.json (single view) or pipeline_mv.json"
    )
    parser.add_argument("--no-depth-estimator", action="store_true")
    parser.add_argument("--no-safe-serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_pixal3d_checkpoint(
        args.source_directory,
        args.output_directory,
        config_file=args.config_file,
        depth_estimator=not args.no_depth_estimator,
        safe_serialization=not args.no_safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NAF_CHECKPOINT_SHA256",
    "NAF_REFERENCE_REVISION",
    "PIXAL3D_REFERENCE_REVISION",
    "PIXAL3D_WEIGHTS_REVISION",
    "convert_pixal3d_checkpoint",
    "main",
]
