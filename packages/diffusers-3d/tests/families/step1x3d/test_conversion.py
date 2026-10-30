from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from diffusers_3d import AutoPipelineForImageTo3D, Step1X3DImageTo3DPipeline
from diffusers_3d.families.step1x3d.conversion import (
    STEP1X3D_REFERENCE_REVISION,
    convert_step1x3d_checkpoint,
    convert_step1x3d_vae_state_dict,
    main,
)

pytestmark = pytest.mark.integration

_UPSTREAM_MLP = {"ff.net.0.proj": "mlp.c_fc", "ff.net.2": "mlp.c_proj"}
_UPSTREAM_VAE = {
    "query_proj": "decoder.query_proj",
    "norm1": "decoder.cross_attn_decoder.ln_1",
    "norm_context": "decoder.cross_attn_decoder.ln_2",
    "norm2": "decoder.cross_attn_decoder.ln_3",
    "attn.to_q": "decoder.cross_attn_decoder.attn.c_q",
    "attn.to_out.0": "decoder.cross_attn_decoder.attn.c_proj",
    "ff.net.0.proj": "decoder.cross_attn_decoder.mlp.c_fc",
    "ff.net.2": "decoder.cross_attn_decoder.mlp.c_proj",
    "norm_out": "decoder.ln_post",
    "proj_out": "decoder.output_proj",
    "post_kl": "post_kl",
}
_UPSTREAM_BLOCK = {"norm1": "ln_1", "norm3": "ln_2", "attn1.to_out.0": "attn.c_proj", **_UPSTREAM_MLP}


def _fuse_heads(heads: int, *parts: torch.Tensor) -> torch.Tensor:
    """Upstream layout: output rows grouped per head as ``[q_h, k_h, v_h]``."""

    return torch.stack([part.view(heads, -1, part.shape[-1]) for part in parts], dim=1).reshape(-1, parts[0].shape[-1])


def _upstream_vae_state(vae) -> dict[str, torch.Tensor]:
    heads = vae.config.num_attention_heads
    state = vae.state_dict()
    upstream = {"encoder.input_proj.weight": torch.zeros(2, 2), "pre_kl.bias": torch.zeros(2)}
    for key, value in state.items():
        module, parameter = key.rsplit(".", 1)
        if module.startswith("transformer_blocks."):
            _, index, name = module.split(".", 2)
            if name in ("attn1.to_k", "attn1.to_v"):
                continue
            prefix = f"transformer.resblocks.{index}"
            if name == "attn1.to_q":
                upstream[f"{prefix}.attn.c_qkv.{parameter}"] = _fuse_heads(
                    heads, value, state[f"{module[:-4]}to_k.{parameter}"], state[f"{module[:-4]}to_v.{parameter}"]
                )
            else:
                upstream[f"{prefix}.{_UPSTREAM_BLOCK[name]}.{parameter}"] = value
        elif module == "attn.to_k":
            upstream[f"decoder.cross_attn_decoder.attn.c_kv.{parameter}"] = _fuse_heads(
                heads, value, state[f"attn.to_v.{parameter}"]
            )
        elif module != "attn.to_v":
            upstream[f"{_UPSTREAM_VAE[module]}.{parameter}"] = value
    return upstream


def _write_release(root, components, *, label: bool = False):
    transformer, vae, image_encoder = components["transformer"], components["vae"], components["image_encoder"]
    folders = {}
    for name in ("transformer", "vae", "visual_encoder", "scheduler"):
        folders[name] = root / name
        folders[name].mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Step1X3DGeometryPipeline",
                "caption_encoder": [None, None],
                "label_encoder": ["step1x3d_geometry.models", "LabelEncoder"] if label else [None, None],
                "visual_encoder": ["step1x3d_geometry.models.conditional_encoders.dinov2_encoder", "Dinov2Encoder"],
            }
        )
    )
    config = transformer.config
    (folders["transformer"] / "config.json").write_text(
        json.dumps(
            {
                "input_channels": config.in_channels,
                "width": config.width,
                "layers": config.num_layers,
                "num_single_layers": config.num_single_layers,
                "num_heads": config.num_attention_heads,
                "condition_dim": config.cross_attention_dim,
                "multi_condition_type": "in_context",
                "use_visual_condition": True,
                "visual_condition_dim": config.cross_attention_dim,
                "n_views": 1,
                "use_caption_condition": False,
                "use_label_condition": False,
            }
        )
    )
    save_file(
        {
            f"dit_model.{key.replace('time_embedding.', 'time_proj.')}": value.contiguous()
            for key, value in transformer.state_dict().items()
        },
        folders["transformer"] / "diffusion_pytorch_model.safetensors",
    )
    config = vae.config
    (folders["vae"] / "config.json").write_text(
        json.dumps(
            {
                "num_latents": config.num_latents,
                "embed_dim": config.latent_channels,
                "width": config.width,
                "heads": config.num_attention_heads,
                "num_decoder_layers": config.num_layers,
                "num_freqs": config.num_freqs,
                "include_pi": config.include_pi,
                "out_dim": config.out_channels,
                "z_scale_factor": config.scaling_factor,
                "embed_type": "fourier",
                "qkv_bias": False,
                "qk_norm": False,
                "use_flash": True,
                "volume_decoder_type": "hierarchical",
                "surface_extractor_type": "mc",
            }
        )
    )
    save_file(_upstream_vae_state(vae), folders["vae"] / "diffusion_pytorch_model.safetensors")
    (folders["visual_encoder"] / "config.json").write_text(
        json.dumps(
            {
                "dino_type": "facebook/dinov2-with-registers-large",
                "image_size": image_encoder.config.image_size,
                "encode_camera": False,
                "zero_uncond_embeds": True,
                "normalize_embeds": False,
                "n_views": 1,
            }
        )
    )
    # The release stores DINOv2 in the Hub layout, which ``save_pretrained`` writes on every Transformers version.
    image_encoder.save_pretrained(root / "dinov2")
    save_file(
        {f"dino_model.{key}": value for key, value in load_file(root / "dinov2" / "model.safetensors").items()},
        folders["visual_encoder"] / "diffusion_pytorch_model.safetensors",
    )
    scheduler_config = {**components["scheduler"].config, "_class_name": "FlowMatchEulerDiscreteScheduler"}
    (folders["scheduler"] / "scheduler_config.json").write_text(json.dumps(scheduler_config))


def test_fused_head_split_matches_upstream_attention_layout():
    torch.manual_seed(0)
    heads, head_dim, width = 2, 3, 6
    fused = torch.randn(3 * width, width)
    hidden_states = torch.randn(1, 4, width)
    query, key, value = torch.nn.functional.linear(hidden_states, fused).view(1, 4, heads, -1).split(head_dim, dim=-1)
    converted, dropped = convert_step1x3d_vae_state_dict(
        {"transformer.resblocks.0.attn.c_qkv.weight": fused, "encoder.ln_post.weight": torch.ones(1)},
        num_attention_heads=heads,
    )
    for name, expected in (("to_q", query), ("to_k", key), ("to_v", value)):
        actual = torch.nn.functional.linear(hidden_states, converted[f"transformer_blocks.0.attn1.{name}.weight"])
        torch.testing.assert_close(actual.view(1, 4, heads, head_dim), expected)
    assert dropped == ["encoder.ln_post.weight"]
    with pytest.raises(ValueError, match="unexpected Step1X-3D VAE key"):
        convert_step1x3d_vae_state_dict({"decoder.unknown.weight": torch.ones(1)}, num_attention_heads=heads)


def test_converted_release_round_trips_through_auto_pipeline(tmp_path, tiny_step1x3d_components):
    source = tmp_path / "Step1X-3D-Geometry-1300m"
    _write_release(source, tiny_step1x3d_components)
    dinov2_config = tiny_step1x3d_components["image_encoder"].config.to_dict()
    output = convert_step1x3d_checkpoint(source, tmp_path / "converted", dinov2_config=dinov2_config)

    pipeline = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(pipeline) is Step1X3DImageTo3DPipeline
    for name in ("image_encoder", "transformer", "vae"):
        expected = tiny_step1x3d_components[name].state_dict()
        actual = getattr(pipeline, name).state_dict()
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, atol=0.0, rtol=0.0)
    for name in ("transformer", "vae"):
        converted = {key: value for key, value in getattr(pipeline, name).config.items() if not key.startswith("_")}
        assert converted.items() <= dict(tiny_step1x3d_components[name].config).items()
    assert pipeline.scheduler.config.shift == 1.0
    report = json.loads((output / "step1x3d_conversion.json").read_text())
    assert report["reference_revision"] == STEP1X3D_REFERENCE_REVISION
    assert report["dropped_vae_keys"] == ["encoder.input_proj.weight", "pre_kl.bias"]


def test_label_variant_is_rejected_and_cli_converts(tmp_path, tiny_step1x3d_components):
    label = tmp_path / "label"
    _write_release(label, tiny_step1x3d_components, label=True)
    with pytest.raises(ValueError, match="unsupported model_index"):
        convert_step1x3d_checkpoint(label, tmp_path / "unused")

    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
