from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    TrellisDinov2Conditioner,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
)
from diffusers_3d.families.trellis.conversion import convert_trellis_checkpoint

pytestmark = pytest.mark.integration


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


def test_synthetic_conversion_converts_slat_components_and_auto_loads(tmp_path, tiny_trellis_components):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _, slat_flow, _, gaussian_decoder, mesh_decoder = tiny_trellis_components(
        include_slat=True
    )
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    _write_component(source, "slat_flow", "SLatFlowModel", slat_flow)
    _write_component(source, "slat_gs", "SLatGaussianDecoder", gaussian_decoder)
    _write_component(source, "slat_mesh", "SLatMeshDecoder", mesh_decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    # The released ``dinov2_vitl14_reg`` lives on the Hub as a Transformers ``Dinov2WithRegistersModel``; the
    # converter accepts that folder directly and folds its register tokens into the conditioner.
    dinov2 = Dinov2WithRegistersModel(
        Dinov2WithRegistersConfig(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            mlp_ratio=2,
            image_size=8,
            patch_size=4,
            num_register_tokens=conditioner.num_register_tokens,
        )
    ).eval()
    dinov2.save_pretrained(tmp_path / "dinov2")

    output = convert_trellis_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "dinov2",
    )
    report = json.loads((output / "trellis_conversion.json").read_text(encoding="utf-8"))
    model_index = json.loads((output / "model_index.json").read_text(encoding="utf-8"))
    sidecar = json.loads((output / "object3d_model_index.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 2
    converted_conditioner = TrellisDinov2Conditioner.from_pretrained(output / "conditioner")
    torch.testing.assert_close(converted_conditioner.register_tokens, dinov2.embeddings.register_tokens)
    images = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        ours = converted_conditioner(images, value_range=None).embeddings
        normalized = (images - converted_conditioner.image_mean) / converted_conditioner.image_std
        # TRELLIS applies an unparameterized final norm instead of DINOv2's learned one: compare pre-norm tokens.
        reference = dinov2.encoder(dinov2.embeddings(normalized)).last_hidden_state
    torch.testing.assert_close(ours, torch.nn.functional.layer_norm(reference, (12,)))
    assert {component["name"] for component in sidecar["components"]} == {
        "conditioner",
        "gaussian_decoder",
        "mesh_decoder",
        "slat_flow_model",
        "slat_scheduler",
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "sparse_structure_scheduler",
    }
    for component in sidecar["components"]:
        value = model_index.get(component["name"], [None, None])
        if value != [None, None]:
            assert value == component["expected_class"].rsplit(".", 1)
    assert set(report["components"]) == {
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "slat_flow_model",
        "slat_decoder_gs",
        "slat_decoder_mesh",
    }
    assert report["components"]["slat_flow_model"]["class"] == TrellisSLatFlowModel.__name__
    assert report["components"]["slat_decoder_gs"]["class"] == TrellisSLatGaussianDecoder.__name__
    assert report["components"]["slat_decoder_mesh"]["class"] == TrellisSLatMeshDecoder.__name__
    assert set(report["skipped_components"]) == {"slat_decoder_rf"}
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is TrellisImageTo3DPipeline
    assert type(loaded.slat_flow_model) is TrellisSLatFlowModel
    assert type(loaded.gaussian_decoder) is TrellisSLatGaussianDecoder
    assert type(loaded.mesh_decoder) is TrellisSLatMeshDecoder
    assert loaded.config.slat_mean == [0.0] * slat_flow.config.out_channels


def test_converter_requires_every_referenced_component_pair(tmp_path, tiny_trellis_components):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _ = tiny_trellis_components()
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    with pytest.raises(FileNotFoundError, match="slat_flow"):
        convert_trellis_checkpoint(
            source,
            tmp_path / "converted",
            conditioner_path=tmp_path / "conditioner",
        )


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
