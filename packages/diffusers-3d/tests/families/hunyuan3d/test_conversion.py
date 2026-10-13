from __future__ import annotations

import json
import re
import warnings

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from diffusers_3d import AutoPipelineForImageTo3D, Hunyuan3DImageTo3DPipeline, Object3DModelIndex
from diffusers_3d.families.hunyuan3d.conversion import convert_hunyuan3d_checkpoint, main

from .conftest import make_tiny_hunyuan3d_pipeline, sample_latents

pytestmark = pytest.mark.integration

# Inverse of the converter's key mapping, used to write a synthetic release in the upstream layout.
_RELEASED_TRANSFORMER_NAMES = (
    (r"^transformer_blocks\.", "blocks."),
    (r"^time_embedding\.linear_1\.", "t_embedder.mlp.0."),
    (r"^time_embedding\.linear_2\.", "t_embedder.mlp.2."),
    (r"^proj_in\.", "x_embedder."),
    (r"^norm_out\.", "final_layer.norm_final."),
    (r"^proj_out\.", "final_layer.linear."),
    (r"\.norm_q\.", ".q_norm."),
    (r"\.norm_k\.", ".k_norm."),
    (r"\.to_out\.", ".out_proj."),
    (r"\.ff\.(experts|gate|shared_experts)", r".moe.\1"),
    (r"\.ff\.net\.0\.proj\.", ".mlp.fc1."),
    (r"\.ff\.net\.2\.", ".mlp.fc2."),
)
_RELEASED_VAE_NAMES = (
    (r"^transformer_blocks\.", "transformer.resblocks."),
    (r"^decoder_block\.", "geo_decoder.cross_attn_decoder."),
    (r"^query_proj\.", "geo_decoder.query_proj."),
    (r"^norm_out\.", "geo_decoder.ln_post."),
    (r"^proj_out\.", "geo_decoder.output_proj."),
    (r"\.norm(\d)\.", r".ln_\1."),
    (r"\.attn\.norm_q\.", ".attn.attention.q_norm."),
    (r"\.attn\.norm_k\.", ".attn.attention.k_norm."),
    (r"\.attn\.to_qkv\.", ".attn.c_qkv."),
    (r"\.attn\.to_kv\.", ".attn.c_kv."),
    (r"\.attn\.to_q\.", ".attn.c_q."),
    (r"\.attn\.to_out\.", ".attn.c_proj."),
    (r"\.ff\.net\.0\.proj\.", ".mlp.c_fc."),
    (r"\.ff\.net\.2\.", ".mlp.c_proj."),
)
_TRANSFORMER_KEYS = ("in_channels", "hidden_size", "context_dim", "depth", "num_heads") + (
    "num_moe_layers",
    "num_experts",
    "moe_top_k",
)
_VAE_KEYS = ("num_latents", "embed_dim", "width", "heads", "num_decoder_layers", "num_freqs", "scale_factor")


def _rename(state_dict, names):
    renamed = {}
    for key, value in state_dict.items():
        for pattern, replacement in names:
            key = re.sub(pattern, replacement, key)
        renamed[key] = value
    return renamed


def _fuse_heads(parts, heads):
    """Concatenate projection rows head by head: the layout the released attention splits per head."""

    return torch.stack([part.unflatten(0, (heads, -1)) for part in parts], dim=1).flatten(0, 2)


def _released_transformer(state_dict, heads):
    # The released DiT keeps separate q/k/v layers and fuses their outputs in its forward.
    state_dict = dict(state_dict)
    for key in [key for key in state_dict if key.endswith((".attn1.to_q.weight", ".attn2.to_k.weight"))]:
        prefix = key.rsplit(".", 2)[0]
        names = ("to_q", "to_k", "to_v") if ".attn1." in key else ("to_k", "to_v")
        fused = _fuse_heads([state_dict[f"{prefix}.{name}.weight"] for name in names], heads)
        state_dict.update({f"{prefix}.{name}.weight": part for name, part in zip(names, fused.chunk(len(names)))})
    return _rename(state_dict, _RELEASED_TRANSFORMER_NAMES)


def _released_vae(state_dict, heads):
    state_dict = dict(state_dict)
    for key in [key for key in state_dict if key.endswith(".attn.to_k.weight")]:
        prefix = key.rsplit(".", 2)[0]
        names = ("to_q", "to_k", "to_v") if prefix.startswith("transformer_blocks") else ("to_k", "to_v")
        fused = _fuse_heads([state_dict.pop(f"{prefix}.{name}.weight") for name in names], heads)
        state_dict[f"{prefix}.{'to_qkv' if len(names) == 3 else 'to_kv'}.weight"] = fused
    return _rename(state_dict, _RELEASED_VAE_NAMES)


def _write_release(folder, pipeline, weights_format="safetensors"):
    """Write ``config.yaml`` plus the aggregate checkpoint: a nested fp16 pickle like the release, or safetensors."""

    transformer, vae, image_encoder = pipeline.transformer, pipeline.vae, pipeline.image_encoder
    folder.mkdir()
    # The release stores DINOv2 in the Hub layout, which ``save_pretrained`` writes on every Transformers version.
    image_encoder.save_pretrained(folder / "dinov2")
    components = {
        "model": _released_transformer(transformer.state_dict(), transformer.config.num_heads),
        "vae": {**_released_vae(vae.state_dict(), vae.config.heads), "pre_kl.weight": torch.zeros(16, 32)},
        "conditioner": {
            f"main_image_encoder.model.{key}": value
            for key, value in load_file(folder / "dinov2" / "model.safetensors").items()
        },
    }
    if weights_format == "ckpt":
        torch.save(
            {name: {key: value.half() for key, value in state.items()} for name, state in components.items()},
            folder / "model.fp16.ckpt",
        )
    else:
        flat = {
            f"{name}.{key}": value.contiguous() for name, state in components.items() for key, value in state.items()
        }
        save_file(flat, folder / "model.fp16.safetensors")
    config = {
        "model": {
            "target": "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain",
            "params": {
                **{key: transformer.config[key] for key in _TRANSFORMER_KEYS},
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
        "vae": {
            "target": "hy3dshape.models.autoencoders.ShapeVAE",
            "params": {
                **{key: vae.config[key] for key in _VAE_KEYS},
                "include_pi": False,
                "qkv_bias": False,
                "qk_norm": True,
                "geo_decoder_mlp_expand_ratio": 4,
                "geo_decoder_downsample_ratio": 1,
                "geo_decoder_ln_post": True,
                "num_encoder_layers": 1,
                "point_feats": 4,
                "pc_size": 81920,
                "pc_sharpedge_size": 0,
            },
        },
        "conditioner": {
            "target": "hy3dshape.models.conditioner.SingleImageEncoder",
            "params": {
                "main_image_encoder": {
                    "type": "DinoImageEncoder",
                    "kwargs": {
                        "config": {**image_encoder.config.to_dict(), "torch_dtype": "float32"},
                        "image_size": image_encoder.config.image_size,
                        "use_cls_token": True,
                    },
                }
            },
        },
        "scheduler": {
            "target": "hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler",
            "params": {"num_train_timesteps": 1000},
        },
        "image_processor": {
            "target": "hy3dshape.preprocessors.ImageProcessorV2",
            "params": {"size": 16, "border_ratio": 0.15},
        },
        "pipeline": {"target": "hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline"},
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


@pytest.mark.parametrize("weights_format", ["ckpt", "safetensors"])
def test_released_layout_converts_to_an_equivalent_auto_loadable_pipeline(tmp_path, weights_format):
    pipeline = make_tiny_hunyuan3d_pipeline()
    # Round the tiny weights through fp16 so the fp16 release pickle holds them exactly.
    with torch.no_grad():
        for name in ("image_encoder", "transformer", "vae"):
            for parameter in getattr(pipeline, name).parameters():
                parameter.copy_(parameter.half())
    _write_release(tmp_path / "hunyuan3d-dit-v2-1", pipeline, weights_format)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert main([str(tmp_path), str(tmp_path / "converted")]) == 0
        loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path / "converted", local_files_only=True)

    assert type(loaded) is Hunyuan3DImageTo3DPipeline
    Object3DModelIndex.from_pretrained(tmp_path / "converted", local_files_only=True)
    report = json.loads((tmp_path / "converted" / "hunyuan3d_conversion.json").read_text())
    assert report["dropped_vae_encoder_keys"] == ["pre_kl.weight"]
    stored = load_file(tmp_path / "converted" / "transformer" / "diffusion_pytorch_model.safetensors")
    assert stored["proj_in.weight"].dtype is (torch.float16 if weights_format == "ckpt" else torch.float32)
    for name in ("image_encoder", "transformer", "vae"):
        expected, actual = getattr(pipeline, name).state_dict(), getattr(loaded, name).state_dict()
        assert expected.keys() == actual.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected), name
    torch.testing.assert_close(sample_latents(loaded), sample_latents(pipeline))


def test_converter_rejects_unported_released_options(tmp_path):
    config = _write_release(tmp_path / "release", make_tiny_hunyuan3d_pipeline())
    config["model"]["params"]["use_attention_pooling"] = True
    (tmp_path / "release" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="use_attention_pooling"):
        convert_hunyuan3d_checkpoint(tmp_path / "release", tmp_path / "converted")
