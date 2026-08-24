from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file

from diffusers_3d import AutoPipelineForImageTo3D, Trellis2ImageTo3DPipeline
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


def test_synthetic_reviewed_conversion_skips_production_experimental_weights_and_auto_loads(
    tmp_path,
    tiny_trellis2_components,
):
    source = tmp_path / "source"
    source.mkdir()
    conditioner, flow, decoder, _ = tiny_trellis2_components()
    _write_component(source, "ss_flow", "SparseStructureFlowModel", flow)
    _write_component(source, "ss_decoder", "SparseStructureDecoder", decoder)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    output = convert_trellis2_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "conditioner",
    )
    report = json.loads((output / "trellis2_conversion.json").read_text(encoding="utf-8"))
    assert set(report["components"]) == {"sparse_structure_decoder", "sparse_structure_flow_model"}
    assert set(report["skipped_components"]) == {
        "shape_slat_decoder",
        "shape_slat_flow_model_1024",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
        "tex_slat_flow_model_512",
    }
    assert not report["production_sparse_ovoxel_checkpoint_parity"]
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is Trellis2ImageTo3DPipeline
    assert loaded.shape_slat_flow_model is None
    assert loaded.texture_slat_flow_model is None
    assert loaded.pbr_decoder is None
    assert loaded.config.default_pipeline_type == "1024_cascade"
    assert loaded._sparse_target_resolution(loaded.config.default_pipeline_type) == 32
    assert loaded.config.capability_limitations["experimental_formats"] == [
        "shape_slat",
        "texture_slat",
        "o_voxel",
        "mesh",
    ]


def test_synthetic_tiny_experimental_conversion_is_opt_in(tmp_path, tiny_trellis2_components):
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
    ) = tiny_trellis2_components(include_experimental=True)
    for stem, upstream_name, model in (
        ("ss_flow", "SparseStructureFlowModel", flow),
        ("ss_decoder", "SparseStructureDecoder", decoder),
        ("shape_flow", "SLatFlowModel", shape_flow),
        ("shape_decoder", "FlexiDualGridVaeDecoder", shape_decoder),
        ("texture_flow", "SLatFlowModel", texture_flow),
        ("pbr_decoder", "SparseUnetVaeDecoder", pbr_decoder),
    ):
        _write_component(source, stem, upstream_name, model)
    (source / "pipeline.json").write_text(json.dumps(_pipeline_config()), encoding="utf-8")
    conditioner.save_pretrained(tmp_path / "conditioner")

    output = convert_trellis2_checkpoint(
        source,
        tmp_path / "converted",
        conditioner_path=tmp_path / "conditioner",
        include_experimental=True,
    )
    report = json.loads((output / "trellis2_conversion.json").read_text(encoding="utf-8"))
    assert set(report["components"]) == {
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    }
    loaded = Trellis2ImageTo3DPipeline.from_pretrained(output, local_files_only=True)
    assert loaded.shape_slat_flow_model is not None
    assert loaded.texture_slat_flow_model is not None
    assert loaded.pbr_decoder is not None


def test_converter_rejects_component_drift_and_never_misloads_production_sparse_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = _pipeline_config()
    del config["args"]["models"]["tex_slat_flow_model_1024"]
    (source / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*tex_slat_flow_model_1024"):
        convert_trellis2_checkpoint(
            source,
            tmp_path / "converted",
            conditioner_path=tmp_path / "conditioner",
        )
