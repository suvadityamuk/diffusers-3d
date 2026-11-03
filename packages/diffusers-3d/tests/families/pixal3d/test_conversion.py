from __future__ import annotations

import hashlib
import json

import pytest
import torch
from _configs import tiny_config
from moge.configs import tiny_moge2_config
from safetensors.torch import load_file, save_file

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    ImageCondition,
    OVoxelAsset,
    Pixal3DImageTo3DPipeline,
    Pixal3DSLatFlowModel,
)
from diffusers_3d.families.moge import MoGe2Model
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
def test_synthetic_release_converts_auto_loads_and_runs(
    tmp_path, monkeypatch, config_file, name, multiview, local_external_components
):
    reference = tiny_pipeline()
    source = tmp_path / "source"
    (source / "ckpts").mkdir(parents=True)
    for stem, upstream_name, model, projected in (
        ("ss_flow", "SparseStructureFlowModel", reference.sparse_structure_flow_model, True),
        ("shape_512", "ElasticSLatFlowModel", reference.shape_slat_flow_model, True),
        ("shape_1024", "ElasticSLatFlowModel", reference.shape_slat_flow_model_1024, True),
        ("tex_1024", "ElasticSLatFlowModel", reference.texture_slat_flow_model_1024, True),
    ):
        _write_component(source / "ckpts", stem, upstream_name, model, projected=projected)
    models = {
        "sparse_structure_flow_model": "ckpts/ss_flow",
        # Byte-identical copies of TRELLIS release files: the pipeline loads them from those releases.
        "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
        "shape_slat_flow_model_512": "ckpts/shape_512",
        "shape_slat_flow_model_1024": "ckpts/shape_1024",
        "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
        "tex_slat_flow_model_1024": "ckpts/tex_1024",
        "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
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
    # The released MoGe-2 is 1.3 GB; a tiny one stands in for it.
    estimator = MoGe2Model(**tiny_moge2_config()).eval()
    calls = local_external_components(
        Pixal3DImageTo3DPipeline,
        image_encoder=reference.image_encoder,
        feature_upsampler=reference.feature_upsampler,
        depth_estimator=estimator,
        sparse_structure_decoder=reference.sparse_structure_decoder,
        shape_slat_decoder=reference.shape_slat_decoder,
        pbr_decoder=reference.pbr_decoder,
    )
    output = convert_pixal3d_checkpoint(source, tmp_path / "converted", config_file=config_file)
    report = json.loads((output / "pixal3d_conversion.json").read_text())
    assert report["reference_revision"] == conversion.PIXAL3D_REFERENCE_REVISION
    assert report["multiview"] is multiview
    # Third-party networks (DINOv3, NAF, MoGe-2, TRELLIS decoders) are named but never converted; loading fetches them.
    index = json.loads((output / "model_index.json").read_text())
    assert index["image_encoder"] == ["transformers", "DINOv3ViTModel"]
    assert index["feature_upsampler"][1] == "Pixal3DNAFUpsampler"
    assert index["depth_estimator"] == ([None, None] if multiview else [MoGe2Model.__module__, "MoGe2Model"])
    external = ("image_encoder", "feature_upsampler", "depth_estimator", "sparse_structure_decoder")
    assert not any((output / name).exists() for name in (*external, "shape_slat_decoder", "pbr_decoder"))
    pipeline = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    expected = {
        "image_encoder": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "feature_upsampler": "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth",
        "sparse_structure_decoder": "microsoft/TRELLIS-image-large",
        "shape_slat_decoder": "microsoft/TRELLIS.2-4B",
        "pbr_decoder": "microsoft/TRELLIS.2-4B",
    } | ({} if multiview else {"depth_estimator": "Ruicheng/moge-2-vitl"})
    assert {call[0]: call[1] for call in calls} == expected and len(calls) == len(expected)
    assert type(pipeline) is Pixal3DImageTo3DPipeline
    pipeline.disable_background_removal()
    assert pipeline.config.multiview is multiview
    assert pipeline.texture_slat_flow_model is None
    assert pipeline.config.image_size == TINY_IMAGE_SIZE
    # Only single-view checkpoints estimate their field of view.
    assert (pipeline.depth_estimator is None) is multiview
    if not multiview:
        assert pipeline.depth_estimator.dtype is torch.float32
        torch.testing.assert_close(pipeline.depth_estimator.state_dict(), estimator.state_dict(), atol=0, rtol=0)
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
    model = Pixal3DSLatFlowModel(**{**tiny_config(Pixal3DSLatFlowModel), "dtype": "bfloat16"})
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
    state = conversion._naf_state_dict(path, hashlib.sha256(path.read_bytes()).hexdigest())
    for key, value in released.items():
        if "rope" not in key:
            assert torch.equal(state[_naf_block_key(key)], value)
            assert torch.equal(model.state_dict()[_naf_block_key(key)], value)


def test_naf_loads_from_its_release_url_and_never_from_an_uncached_offline_path(tmp_path):
    from diffusers_3d.families.pixal3d.conditioner import NAF_CHECKPOINT_URL, load_naf_upsampler

    component = Pixal3DImageTo3DPipeline.external_components["feature_upsampler"]
    assert (component.repository, component.revision) == (NAF_CHECKPOINT_URL, conversion.NAF_CHECKPOINT_SHA256)
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_naf_upsampler(cache_dir=tmp_path, local_files_only=True)
