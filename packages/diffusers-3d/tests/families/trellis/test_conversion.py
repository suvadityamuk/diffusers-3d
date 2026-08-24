from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
)
from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint


def _write_component(root, stem, upstream_name, model):
    (root / f"{stem}.json").write_text(
        json.dumps({"name": upstream_name, "args": dict(model.config)}),
        encoding="utf-8",
    )
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()}, root / f"{stem}.safetensors"
    )


def _pipeline_config(*, normalization_channels: int = 4):
    sampler = {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 25,
            "cfg_strength": 5.0,
            "cfg_interval": [0.5, 1.0],
            "rescale_t": 3.0,
        },
    }
    return {
        "name": "TrellisImageTo3DPipeline",
        "args": {
            "image_cond_model": "dinov2_vitl14_reg",
            "models": {
                "sparse_structure_flow_model": "ss_flow",
                "sparse_structure_decoder": "ss_decoder",
                "slat_flow_model": "slat_flow",
                "slat_decoder_gs": "slat_gs",
                "slat_decoder_mesh": "slat_mesh",
                "slat_decoder_rf": "slat_rf",
            },
            "slat_normalization": {
                "mean": [0.0] * normalization_channels,
                "std": [1.0] * normalization_channels,
            },
            "slat_sampler": sampler,
            "sparse_structure_sampler": sampler,
        },
    }


def test_synthetic_portable_component_conversion_and_auto_load(
    tmp_path,
    tiny_trellis_components,
):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _ = tiny_trellis_components()
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    output = convert_trellis_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "conditioner",
    )
    report = json.loads((output / "trellis_conversion.json").read_text(encoding="utf-8"))
    assert set(report["components"]) == {"sparse_structure_decoder", "sparse_structure_flow_model"}
    assert set(report["skipped_components"]) == {
        "slat_decoder_gs",
        "slat_decoder_mesh",
        "slat_decoder_rf",
        "slat_flow_model",
    }
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is TrellisImageTo3DPipeline
    assert loaded.slat_flow_model is None
    assert loaded.gaussian_decoder is None


def test_synthetic_experimental_slat_conversion_is_opt_in(tmp_path, tiny_trellis_components):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _, slat_flow, _, gaussian_decoder = tiny_trellis_components(include_slat=True)
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    _write_component(source, "slat_flow", "SLatFlowModel", slat_flow)
    _write_component(source, "slat_gs", "SLatGaussianDecoder", gaussian_decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    output = convert_trellis_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "conditioner",
        include_experimental=True,
    )
    report = json.loads((output / "trellis_conversion.json").read_text(encoding="utf-8"))
    assert report["components"]["slat_flow_model"]["class"] == TrellisSLatFlowModel.__name__
    assert report["components"]["slat_decoder_gs"]["class"] == TrellisSLatGaussianDecoder.__name__
    loaded = TrellisImageTo3DPipeline.from_pretrained(output, local_files_only=True)
    assert type(loaded.slat_flow_model) is TrellisSLatFlowModel
    assert type(loaded.gaussian_decoder) is TrellisSLatGaussianDecoder


def test_converter_strictly_rejects_pipeline_component_drift(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _pipeline_config()
    del config["args"]["models"]["slat_decoder_rf"]
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*slat_decoder_rf"):
        convert_trellis_checkpoint(
            source,
            tmp_path / "converted",
            conditioner_path=tmp_path / "conditioner",
        )
