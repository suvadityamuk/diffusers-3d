from __future__ import annotations

import json
import warnings

import pytest
import torch
from hunyuan3d.test_conversion import _released_transformer, _released_vae

from diffusers_3d import AutoPipelineForImageTo3D
from diffusers_3d.families.hunyuan3d_omni.conversion import convert_hunyuan3d_omni_checkpoint, main
from diffusers_3d.families.hunyuan3d_omni.pipeline import Hunyuan3DOmniImageTo3DPipeline

from .conftest import make_tiny_omni_pipeline, sample_latents

pytestmark = pytest.mark.integration

# Inverse of the converter's control-encoder key mapping.
_CONTROL_NAMES = {
    "type_embedding.": "cond_signal_embedding.",
    "type_proj.": "cond_signal_linear.",
    "proj_in.": "linear.0.",
    "norm.": "linear.1.",
}


def _released_control_encoder(state_dict):
    renamed = {}
    for key, value in state_dict.items():
        prefix = next(prefix for prefix in _CONTROL_NAMES if key.startswith(prefix))
        renamed[_CONTROL_NAMES[prefix] + key.removeprefix(prefix)] = value
    return renamed


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_release(folder, pipeline):
    """Write the ``tencent/Hunyuan3D-Omni`` layout: per-component ``config.json`` and ``pytorch_model.bin``."""

    transformer, vae, control_encoder = pipeline.transformer, pipeline.vae, pipeline.control_encoder
    pipeline.image_encoder.save_pretrained(folder / "dinov2")
    _write(
        folder / "config.json", {"target": "hy3dshape.pipelines.Hunyuan3DOmniSiTFlowMatchingPipeline", "params": {}}
    )
    _write(
        folder / "model" / "config.json",
        {
            "target": "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain",
            "params": {
                **{key: transformer.config[key] for key in ("in_channels", "hidden_size", "context_dim", "depth")},
                **{
                    key: transformer.config[key] for key in ("num_heads", "num_moe_layers", "num_experts", "moe_top_k")
                },
                "input_size": 6,
                "text_len": 5,
                "qk_norm": True,
                "qk_norm_type": "rms",
                "qkv_bias": False,
                "with_decoupled_ca": False,
                "use_attention_pooling": False,
                "use_pos_emb": False,
            },
        },
    )
    torch.save(
        _released_transformer(transformer.state_dict(), transformer.config.num_heads),
        folder / "model" / "pytorch_model.bin",
    )
    vae_params = {key: vae.config[key] for key in ("num_latents", "embed_dim", "width", "heads", "num_decoder_layers")}
    _write(
        folder / "vae" / "config.json",
        {
            "target": "hy3dshape.models.autoencoders.shape_vae_pl.ShapeVAEPL",
            "params": {
                "module_cfg": {
                    "target": "hy3dshape.models.autoencoders.shape_vae.ShapeVAE",
                    "params": {
                        **vae_params,
                        "num_freqs": vae.config.num_freqs,
                        "include_pi": False,
                        "qkv_bias": False,
                        "qk_norm": True,
                        "num_encoder_layers": 1,
                        "point_feats": 3,
                        "sharpedge_label": True,
                        "downsample_ratio": 20,
                        "flash": True,
                    },
                },
                "loss_cfg": {"target": "torch.nn.Identity"},
            },
            "scale_factor": vae.config.scale_factor,
        },
    )
    released_vae = {**_released_vae(vae.state_dict(), vae.config.heads), "pre_kl.weight": torch.zeros(16, 32)}
    torch.save({f"sal.{key}": value for key, value in released_vae.items()}, folder / "vae" / "pytorch_model.bin")
    _write(
        folder / "cond_encoder" / "config.json",
        {
            "target": "hy3dshape.models.conditioners.omni_encoder.OmniEncoder",
            "params": {
                "image_encoder": {
                    "target": "hy3dshape.models.conditioners.omni_encoder.DinoImageEncoder",
                    "params": {"version": "facebook/dinov2-large", "image_size": 28, "use_cls_token": True},
                },
                **{key: control_encoder.config[key] for key in ("width", "num_freqs", "voxel_resolution")},
                "include_pi": False,
                "drop_point": True,
                "drop_ratio": 0,
                "random_noise": True,
                "noise_ratio": 0.2,
                "noise_scales": [0.01, 0.02],
            },
        },
    )
    torch.save(_released_control_encoder(control_encoder.state_dict()), folder / "cond_encoder" / "pytorch_model.bin")
    _write(
        folder / "scheduler" / "config.json",
        {
            "transport": {"params": {"path_type": "Linear", "prediction": "velocity", "shift_scale": 0.125}},
            "sampler": {"params": {}, "ode_params": {"sampling_method": "euler", "num_steps": 50}},
        },
    )
    _write(
        folder / "image_processor" / "config.json",
        {"target": "hy3dshape.preprocessors.ImageProcessorV2", "params": {}},
    )


def test_released_layout_converts_to_an_equivalent_auto_loadable_pipeline(tmp_path, local_external_components):
    pipeline = make_tiny_omni_pipeline()
    _write_release(tmp_path / "release", pipeline)
    arguments = ["--image-encoder", str(tmp_path / "release" / "dinov2"), "--image-encoder-revision", ""]
    local_external_components(Hunyuan3DOmniImageTo3DPipeline, image_encoder=pipeline.image_encoder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert main([str(tmp_path / "release"), str(tmp_path / "converted"), *arguments]) == 0
        assert not (tmp_path / "converted" / "image_encoder").exists()
        loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "converted", local_files_only=True)

    assert type(loaded) is Hunyuan3DOmniImageTo3DPipeline
    report = json.loads((tmp_path / "converted" / "hunyuan3d_omni_conversion.json").read_text())
    assert report["dropped_vae_encoder_keys"] == ["pre_kl.weight"]
    assert loaded.config.image_size == 512 and loaded.vae.config.scale_factor == pipeline.vae.config.scale_factor
    for name in ("image_encoder", "control_encoder", "transformer", "vae"):
        expected, actual = getattr(pipeline, name).state_dict(), getattr(loaded, name).state_dict()
        assert expected.keys() == actual.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected), name
    loaded.register_to_config(image_size=16)
    torch.testing.assert_close(sample_latents(loaded), sample_latents(pipeline))


def test_converter_rejects_unported_released_options(tmp_path):
    _write_release(tmp_path / "release", make_tiny_omni_pipeline())
    path = tmp_path / "release" / "cond_encoder" / "config.json"
    config = json.loads(path.read_text())
    config["params"]["include_pi"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="include_pi"):
        convert_hunyuan3d_omni_checkpoint(tmp_path / "release", tmp_path / "converted")
