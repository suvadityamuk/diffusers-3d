from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Dinov2Config, Dinov2Model

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
)
from diffusers_3d.families.trellis.conversion import _load_encoders, convert_trellis_checkpoint

pytestmark = pytest.mark.integration


def _write_component(root, stem, upstream_name, model):
    (root / f"{stem}.json").write_text(
        json.dumps({"name": upstream_name, "args": dict(model.config)}), encoding="utf-8"
    )
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()}, root / f"{stem}.safetensors"
    )


def _pipeline_config(*, normalization_channels: int = 4):
    sampler = {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {"steps": 25, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
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
            "slat_normalization": {"mean": [0.0] * normalization_channels, "std": [1.0] * normalization_channels},
            "slat_sampler": sampler,
            "sparse_structure_sampler": sampler,
        },
    }


def test_synthetic_conversion_converts_slat_components_and_auto_loads(tmp_path, tiny_trellis_components):
    source = tmp_path / "source"
    source.mkdir()
    dinov2, flow, decoder, _, slat_flow, _, gaussian_decoder, mesh_decoder, rf_decoder = tiny_trellis_components(
        include_slat=True
    )
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    _write_component(source, "slat_flow", "SLatFlowModel", slat_flow)
    _write_component(source, "slat_gs", "SLatGaussianDecoder", gaussian_decoder)
    _write_component(source, "slat_mesh", "SLatMeshDecoder", mesh_decoder)
    _write_component(source, "slat_rf", "SLatRadianceFieldDecoder", rf_decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    # The released ``dinov2_vitl14_reg`` lives on the Hub as a Transformers ``Dinov2WithRegistersModel``.
    dinov2.save_pretrained(tmp_path / "dinov2")

    output = convert_trellis_checkpoint(source, tmp_path / "converted", encoder_path=tmp_path / "dinov2")
    report = json.loads((output / "trellis_conversion.json").read_text(encoding="utf-8"))
    model_index = json.loads((output / "model_index.json").read_text(encoding="utf-8"))
    sidecar = json.loads((output / "object3d_model_index.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 2
    assert model_index["image_encoder"] == ["transformers", "Dinov2WithRegistersModel"]
    assert {component["name"] for component in sidecar["components"]} == {
        "image_encoder",
        "gaussian_decoder",
        "mesh_decoder",
        "radiance_field_decoder",
        "slat_flow_model",
        "slat_scheduler",
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "sparse_structure_scheduler",
    }
    for component in sidecar["components"]:
        value = model_index.get(component["name"], [None, None])
        if value != [None, None] and component["name"] != "image_encoder":
            assert value == component["expected_class"].rsplit(".", 1)
    assert set(report["components"]) == {
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "slat_flow_model",
        "slat_decoder_gs",
        "slat_decoder_mesh",
        "slat_decoder_rf",
    }
    assert report["components"]["slat_flow_model"]["class"] == TrellisSLatFlowModel.__name__
    assert report["components"]["slat_decoder_gs"]["class"] == TrellisSLatGaussianDecoder.__name__
    assert report["components"]["slat_decoder_mesh"]["class"] == TrellisSLatMeshDecoder.__name__
    assert report["components"]["slat_decoder_rf"]["class"] == TrellisSLatRadianceFieldDecoder.__name__
    assert report["skipped_components"] == {}
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is TrellisImageTo3DPipeline
    assert type(loaded.slat_flow_model) is TrellisSLatFlowModel
    assert type(loaded.gaussian_decoder) is TrellisSLatGaussianDecoder
    assert type(loaded.mesh_decoder) is TrellisSLatMeshDecoder
    assert type(loaded.radiance_field_decoder) is TrellisSLatRadianceFieldDecoder
    assert loaded.config.slat_mean == [0.0] * slat_flow.config.out_channels
    for name, parameter in dinov2.state_dict().items():
        assert torch.equal(loaded.image_encoder.state_dict()[name], parameter), name


def test_converter_rejects_an_encoder_without_register_tokens(tmp_path):
    config = Dinov2Config(image_size=8, patch_size=4, hidden_size=12, num_hidden_layers=1, num_attention_heads=3)
    Dinov2Model(config).save_pretrained(tmp_path / "dinov2")
    with pytest.raises(ValueError, match="does not match Dinov2WithRegistersModel"):
        _load_encoders(str(tmp_path / "dinov2"), TrellisImageTo3DPipeline)


def test_converter_requires_every_referenced_component_pair(tmp_path, tiny_trellis_components):
    source = tmp_path / "source"
    source.mkdir()
    image_encoder, flow, decoder, _ = tiny_trellis_components()
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    image_encoder.save_pretrained(tmp_path / "dinov2")

    with pytest.raises(FileNotFoundError, match="slat_flow"):
        convert_trellis_checkpoint(source, tmp_path / "converted", encoder_path=tmp_path / "dinov2")


def test_converter_strictly_rejects_pipeline_component_drift(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _pipeline_config()
    del config["args"]["models"]["slat_decoder_rf"]
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*slat_decoder_rf"):
        convert_trellis_checkpoint(source, tmp_path / "converted", encoder_path=tmp_path / "dinov2")
