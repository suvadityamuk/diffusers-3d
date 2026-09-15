from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    OVoxelAsset,
    SparseVoxelAsset,
    Trellis2ImageTo3DPipeline,
    Trellis2SLatFlowModel,
)
from diffusers_3d.families.trellis2.conversion import convert_trellis2_checkpoint

pytestmark = pytest.mark.integration


def _write_component(root, stem, upstream_name, model):
    (root / f"{stem}.json").write_text(
        json.dumps({"name": upstream_name, "args": dict(model.config)}),
        encoding="utf-8",
    )
    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()},
        root / f"{stem}.safetensors",
    )


def _sampler(*, guidance_rescale: float):
    return {
        "name": "FlowEulerGuidanceIntervalSampler",
        "args": {"sigma_min": 1e-5},
        "params": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": [0.6, 1.0],
            "rescale_t": 5.0,
        },
    }


def _pipeline_config():
    return {
        "name": "Trellis2ImageTo3DPipeline",
        "args": {
            "default_pipeline_type": "1024_cascade",
            "image_cond_model": {
                "name": "DinoV3FeatureExtractor",
                "args": {
                    "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "image_size": 512,
                },
            },
            "models": {
                "sparse_structure_flow_model": "ss_flow",
                "sparse_structure_decoder": "ss_decoder",
                "shape_slat_flow_model_512": "shape_flow",
                "shape_slat_flow_model_1024": "shape_flow_1024",
                "shape_slat_decoder": "shape_decoder",
                "tex_slat_flow_model_512": "texture_flow",
                "tex_slat_flow_model_1024": "texture_flow_1024",
                "tex_slat_decoder": "pbr_decoder",
            },
            "rembg_model": "briaai/RMBG-2.0",
            "shape_slat_normalization": {"mean": [0.0] * 4, "std": [1.0] * 4},
            "shape_slat_sampler": _sampler(guidance_rescale=0.5),
            "sparse_structure_sampler": _sampler(guidance_rescale=0.7),
            "tex_slat_normalization": {"mean": [0.0] * 4, "std": [1.0] * 4},
            "tex_slat_sampler": _sampler(guidance_rescale=0.0),
        },
    }


def test_synthetic_full_conversion_auto_loads_and_runs_every_stage(tmp_path, tiny_trellis2_components):
    source = tmp_path / "source"
    source.mkdir()
    (
        conditioner,
        flow,
        decoder,
        _,
        shape_flow,
        _,
        shape_decoder,
        texture_flow,
        _,
        pbr_decoder,
    ) = tiny_trellis2_components(include_slat=True)
    torch.manual_seed(7)
    shape_flow_1024 = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config())
    torch.manual_seed(8)
    texture_flow_1024 = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config(texture=True))
    for stem, upstream_name, model in (
        ("ss_flow", "SparseStructureFlowModel", flow),
        ("ss_decoder", "SparseStructureDecoder", decoder),
        ("shape_flow", "SLatFlowModel", shape_flow),
        ("shape_flow_1024", "ElasticSLatFlowModel", shape_flow_1024),
        ("shape_decoder", "FlexiDualGridVaeDecoder", shape_decoder),
        ("texture_flow", "SLatFlowModel", texture_flow),
        ("texture_flow_1024", "ElasticSLatFlowModel", texture_flow_1024),
        ("pbr_decoder", "SparseUnetVaeDecoder", pbr_decoder),
    ):
        _write_component(source, stem, upstream_name, model)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    output = convert_trellis2_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "conditioner",
    )
    report = json.loads((output / "trellis2_conversion.json").read_text(encoding="utf-8"))
    model_index = json.loads((output / "model_index.json").read_text(encoding="utf-8"))
    sidecar = json.loads((output / "object3d_model_index.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 2
    expected_components = {
        "conditioner",
        "pbr_decoder",
        "shape_slat_decoder",
        "shape_slat_flow_model",
        "shape_slat_flow_model_1024",
        "shape_slat_scheduler",
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "sparse_structure_scheduler",
        "texture_slat_flow_model",
        "texture_slat_flow_model_1024",
        "texture_slat_scheduler",
    }
    assert {component["name"] for component in sidecar["components"]} == expected_components
    for component in sidecar["components"]:
        assert model_index[component["name"]] == component["expected_class"].rsplit(".", 1)
    assert set(report["components"]) == expected_components - {
        "conditioner",
        "shape_slat_scheduler",
        "sparse_structure_scheduler",
        "texture_slat_scheduler",
    }
    assert report["components"]["shape_slat_flow_model_1024"]["source"] == "shape_flow_1024"
    assert report["samplers"]["sparse_structure_sampler"]["guidance_rescale"] == 0.7
    assert not report["production_gpu_quality_verified"]

    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is Trellis2ImageTo3DPipeline
    assert loaded.config.default_pipeline_type == "1024_cascade"
    assert loaded.config.shape_slat_mean == [0.0] * 4
    assert loaded.config.sparse_structure_sampler_defaults["guidance_rescale"] == 0.7
    assert loaded.config.capability_limitations == {
        "official_full_checkpoint_parity": False,
        "production_gpu_quality_verified": False,
    }
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.shape_slat_flow_model_1024.out_layer.weight,
            shape_flow_1024.out_layer.weight,
        )
    result = loaded(
        torch.zeros(3, 8, 8),
        formats=("shape_slat", "o_voxel"),
        sparse_structure_sampler_params={"steps": 1, "guidance_rescale": 0.0},
        shape_slat_sampler_params={"steps": 1},
        texture_slat_sampler_params={"steps": 1},
        sparse_structure_latents=torch.zeros(1, 2, 2, 2, 2),
        generator=torch.Generator().manual_seed(5),
    )
    assert [type(value) for value in result.objects] == [SparseVoxelAsset, OVoxelAsset]


def test_synthetic_sparse_structure_only_conversion_is_allowed(tmp_path, tiny_trellis2_components):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _ = tiny_trellis2_components()
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    config = _pipeline_config()
    config["args"]["models"] = {
        "sparse_structure_flow_model": "ss_flow",
        "sparse_structure_decoder": "ss_decoder",
    }
    config["args"]["default_pipeline_type"] = "512"
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    # A raw Transformers DINOv3 folder (what ``hf download facebook/dinov3-...`` produces) is accepted directly.
    conditioner.model.save_pretrained(tmp_path / "dinov3")

    output = convert_trellis2_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "dinov3",
    )
    report = json.loads((output / "trellis2_conversion.json").read_text(encoding="utf-8"))
    model_index = json.loads((output / "model_index.json").read_text(encoding="utf-8"))
    assert set(report["components"]) == {"sparse_structure_decoder", "sparse_structure_flow_model"}
    assert model_index["shape_slat_flow_model"] == [None, None]
    assert model_index["shape_slat_scheduler"] == [None, None]
    assert model_index["shape_slat_mean"] is None
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert loaded.shape_slat_flow_model is None and loaded.pbr_decoder is None
    assert [
        type(value) for value in loaded(torch.zeros(3, 8, 8), sparse_structure_sampler_params={"steps": 1}).objects
    ] == [SparseVoxelAsset]


def test_converter_rejects_component_drift(tmp_path, tiny_trellis2_components):
    source = tmp_path / "source"
    source.mkdir()
    config = _pipeline_config()
    config["args"]["models"]["unknown_model"] = "nope"
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*unknown_model"):
        convert_trellis2_checkpoint(source, tmp_path / "converted", conditioner_path=tmp_path / "conditioner")

    config = _pipeline_config()
    del config["args"]["models"]["sparse_structure_decoder"]
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*sparse_structure_decoder"):
        convert_trellis2_checkpoint(source, tmp_path / "converted", conditioner_path=tmp_path / "conditioner")

    # A weight file whose layout does not match its descriptor is rejected instead of partially loaded.
    conditioner, flow, decoder, _ = tiny_trellis2_components()
    config = _pipeline_config()
    config["args"]["models"] = {"sparse_structure_flow_model": "ss_flow", "sparse_structure_decoder": "ss_decoder"}
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    (source / "ss_decoder.json").write_text(json.dumps({"name": "SLatFlowModel", "args": dict(decoder.config)}))
    with pytest.raises(ValueError, match="unsupported TRELLIS.2 model 'SLatFlowModel'"):
        convert_trellis2_checkpoint(source, tmp_path / "converted", conditioner_path=tmp_path / "conditioner")
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    state = {key: value for key, value in flow.state_dict().items() if "blocks.1" not in key}
    save_file({key: value.detach().contiguous() for key, value in state.items()}, source / "ss_flow.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        convert_trellis2_checkpoint(source, tmp_path / "converted", conditioner_path=tmp_path / "conditioner")
