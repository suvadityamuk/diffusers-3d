from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    ImageCondition,
    OVoxelAsset,
    Pixal3DImageTo3DPipeline,
    Pixal3DSLatFlowModel,
)
from diffusers_3d.families.pixal3d import conversion
from diffusers_3d.families.pixal3d.conversion import _naf_block_key, _rope_phases, convert_pixal3d_checkpoint

from .conftest import TINY_IMAGE_SIZE, tiny_pipeline

pytestmark = pytest.mark.integration


def _write_component(root, stem, upstream_name, model, *, projected=False):
    config = {key: value for key, value in model.config.items() if not key.startswith("_")}
    state = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
    if projected:
        config["image_attn_mode"] = "proj"
        if "rope_phases" in model._buffers:
            # The released dense flow stores its RoPE phases and leaves proj_in_channels implicit.
            state["rope_phases"] = _rope_phases(model)
            del config["proj_in_channels"]
    (root / f"{stem}.json").write_text(json.dumps({"name": upstream_name, "args": config}))
    save_file(state, root / f"{stem}.safetensors")


def _sampler(guidance_rescale: float) -> dict:
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


@pytest.mark.parametrize(
    ("config_file", "name", "multiview"),
    [
        ("pipeline.json", "Trellis2ImageTo3DPipeline", False),
        ("pipeline_mv.json", "Pixal3DMVImageTo3DPipeline", True),
    ],
)
def test_synthetic_release_converts_auto_loads_and_runs(tmp_path, config_file, name, multiview):
    reference = tiny_pipeline()
    source = tmp_path / "source"
    (source / "ckpts").mkdir(parents=True)
    for stem, upstream_name, model, projected in (
        ("ss_flow", "SparseStructureFlowModel", reference.sparse_structure_flow_model, True),
        ("ss_dec", "SparseStructureDecoder", reference.sparse_structure_decoder, False),
        ("shape_512", "ElasticSLatFlowModel", reference.shape_slat_flow_model, True),
        ("shape_1024", "ElasticSLatFlowModel", reference.shape_slat_flow_model_1024, True),
        ("shape_dec", "FlexiDualGridVaeDecoder", reference.shape_slat_decoder, False),
        ("tex_1024", "ElasticSLatFlowModel", reference.texture_slat_flow_model_1024, True),
        ("tex_dec", "SparseUnetVaeDecoder", reference.pbr_decoder, False),
    ):
        _write_component(source / "ckpts", stem, upstream_name, model, projected=projected)
    models = {
        "sparse_structure_flow_model": "ckpts/ss_flow",
        "sparse_structure_decoder": "ckpts/ss_dec",
        "shape_slat_flow_model_512": "ckpts/shape_512",
        "shape_slat_flow_model_1024": "ckpts/shape_1024",
        "shape_slat_decoder": "ckpts/shape_dec",
        "tex_slat_flow_model_1024": "ckpts/tex_1024",
        "tex_slat_decoder": "ckpts/tex_dec",
    }
    args = {
        "models": models,
        "sparse_structure_sampler": _sampler(0.7),
        "shape_slat_sampler": _sampler(0.5),
        "shape_slat_normalization": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "tex_slat_sampler": _sampler(0.0),
        "tex_slat_normalization": {"mean": [0.0] * 4, "std": [1.0] * 4},
        "image_cond_model": {
            "name": "DinoV3FeatureExtractor",
            "args": {"model_name": "facebook/dinov3", "image_size": TINY_IMAGE_SIZE},
        },
        "rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}},
        "default_pipeline_type": "1536_cascade",
    }
    (source / config_file).write_text(json.dumps({"name": name, "args": args}))
    reference.image_encoder.save_pretrained(tmp_path / "dinov3")
    reference.feature_upsampler.save_pretrained(tmp_path / "naf")

    output = convert_pixal3d_checkpoint(
        source,
        tmp_path / "converted",
        image_encoder_path=tmp_path / "dinov3",
        naf_checkpoint_path=tmp_path / "naf",
        config_file=config_file,
    )
    report = json.loads((output / "pixal3d_conversion.json").read_text())
    assert report["reference_revision"] == conversion.PIXAL3D_REFERENCE_REVISION
    assert report["multiview"] is multiview
    pipeline = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(pipeline) is Pixal3DImageTo3DPipeline
    assert pipeline.config.multiview is multiview
    assert pipeline.texture_slat_flow_model is None
    assert pipeline.config.image_size == TINY_IMAGE_SIZE
    for component in (
        "image_encoder",
        "sparse_structure_flow_model",
        "shape_slat_flow_model_1024",
        "feature_upsampler",
    ):
        torch.testing.assert_close(
            getattr(pipeline, component).state_dict(), getattr(reference, component).state_dict(), atol=0, rtol=0
        )
    if not multiview:
        image = torch.rand(4, 20, 24)
        image[3] = 1
        output = pipeline(
            ImageCondition(image),
            camera_angle_x=0.4,
            sparse_structure_sampler_params={"steps": 1},
            shape_slat_sampler_params={"steps": 1},
            texture_slat_sampler_params={"steps": 1},
        )
        assert [type(value) for value in output.objects] == [OVoxelAsset]


def test_projection_mode_and_rope_phases_are_checked(tmp_path):
    reference = tiny_pipeline(texture=False)
    _write_component(tmp_path, "ss_flow", "SparseStructureFlowModel", reference.sparse_structure_flow_model)
    with pytest.raises(ValueError, match="image_attn_mode='proj'"):
        conversion._save_component(
            "sparse_structure_flow_model", tmp_path / "ss_flow", tmp_path / "out", safe_serialization=True
        )
    _write_component(
        tmp_path, "ss_flow", "SparseStructureFlowModel", reference.sparse_structure_flow_model, projected=True
    )
    state = dict(reference.sparse_structure_flow_model.state_dict())
    save_file(
        {**state, "rope_phases": _rope_phases(reference.sparse_structure_flow_model) * 1j},
        tmp_path / "ss_flow.safetensors",
    )
    with pytest.raises(ValueError, match="RoPE phases"):
        conversion._save_component(
            "sparse_structure_flow_model", tmp_path / "ss_flow", tmp_path / "out", safe_serialization=True
        )


def test_float32_release_weights_keep_the_configured_bfloat16_torso(tmp_path):
    model = Pixal3DSLatFlowModel(**{**Pixal3DSLatFlowModel.tiny_config(), "dtype": "bfloat16"})
    config = {key: value for key, value in model.config.items() if not key.startswith("_")}
    (tmp_path / "flow.json").write_text(
        json.dumps({"name": "ElasticSLatFlowModel", "args": {**config, "image_attn_mode": "proj"}})
    )
    save_file(
        {key: value.float().contiguous() for key, value in model.state_dict().items()}, tmp_path / "flow.safetensors"
    )
    conversion._save_component(
        "shape_slat_flow_model_512", tmp_path / "flow", tmp_path / "out", safe_serialization=True
    )
    saved = load_file(tmp_path / "out" / "diffusion_pytorch_model.safetensors")
    assert saved["blocks.0.cross_attn.proj_linear.weight"].dtype is torch.bfloat16
    assert saved["blocks.0.modulation"].dtype is saved["t_embedder.mlp.0.weight"].dtype is torch.float32


def test_naf_release_keys_map_onto_the_flat_encoders(tmp_path, monkeypatch):
    model = tiny_pipeline(texture=False).feature_upsampler
    released = {
        "image_encoder.encoder.0.weight": model.encoder[0].weight,
        "image_encoder.encoder.1.norm1.weight": model.encoder[1].weight,
        "image_encoder.encoder.1.conv1.weight": model.encoder[3].weight,
        "image_encoder.encoder.1.norm2.bias": model.encoder[4].bias,
        "image_encoder.sem_encoder.1.conv2.bias": model.semantic_encoder[6].bias,
        "image_encoder.rope.periods": 100.0 ** (2 * torch.arange(16, dtype=torch.float32) / 32),
    }
    path = tmp_path / "naf_release.pth"
    torch.save({key: value.detach().clone() for key, value in released.items()}, path)
    with pytest.raises(ValueError, match="pinned NAF release"):
        conversion._naf_state_dict(path)
    monkeypatch.setattr(conversion, "NAF_CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    state = conversion._naf_state_dict(path)
    for key, value in released.items():
        if "rope" not in key:
            assert torch.equal(state[_naf_block_key(key)], value)
            assert torch.equal(model.state_dict()[_naf_block_key(key)], value)
