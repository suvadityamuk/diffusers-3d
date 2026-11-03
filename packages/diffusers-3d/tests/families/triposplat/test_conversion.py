from __future__ import annotations

import dataclasses
import json

import torch
from _configs import tiny_config
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
from transformers import DINOv3ViTModel

from diffusers_3d.families.triposplat import (
    TripoSplatGaussianDecoder,
    TripoSplatImageTo3DPipeline,
    TripoSplatImageVAE,
    TripoSplatTransformerModel,
    convert_triposplat_checkpoint,
)
from diffusers_3d.families.triposplat.conversion import main
from diffusers_3d.families.triposplat.pipeline import UPSTREAM_DTYPES

from .configs import tiny_dinov3_config

# Inverse of the converter's renames, applied to tiny components to rebuild the released key layout.
_TRANSFORMER_PREFIXES = (
    ("time_embed.linear_1.", "t_embedder.mlp.0."),
    ("time_embed.linear_2.", "t_embedder.mlp.2."),
    ("time_modulation.", "adaLN_modulation."),
    ("context_proj_2.", "cond_embedder2."),
    ("context_proj.", "cond_embedder."),
    ("camera_proj_out.", "cam_out_layer."),
    ("camera_proj.net.0.proj.", "cam_refiner.mlp.0."),
    ("camera_proj.net.2.", "cam_refiner.mlp.2."),
    ("noise_rope.", "noise_repo_layers."),
    ("context_rope.", "context_repo_layers."),
    ("rope.", "repo_layers."),
    ("proj_in.", "input_layer."),
    ("proj_out.", "out_layer."),
)
_DECODER_PREFIXES = (
    ("octree_proj_in.", "octree.in_proj."),
    ("octree_input.", "octree.input_layer."),
    ("octree_level_embed.linear_1.", "octree.l_embedder.mlp.0."),
    ("octree_level_embed.linear_2.", "octree.l_embedder.mlp.2."),
    ("octree_modulation.", "octree.adaLN_modulation."),
    ("octree_proj_out.", "octree.out_proj."),
    ("octree_blocks.", "octree.blocks."),
    ("gaussian_proj_in.", "gs.in_proj."),
    ("gaussian_input.", "gs.input_layer."),
    ("gaussian_proj_out.", "gs.out_proj."),
    ("gaussian_blocks.", "gs.blocks."),
    ("offset_perturbation", "gs.points_offset_perturbation"),
    ("base_offset_scale", "gs.base_offset_scale"),
)
_FF = ((".ff.net.0.proj.", ".mlp.mlp.0."), (".ff.net.2.", ".mlp.mlp.2."))


def _released_key(key: str, prefixes, infixes) -> str:
    for ours, theirs in prefixes:
        if key.startswith(ours):
            key = theirs + key.removeprefix(ours)
            break
    for ours, theirs in infixes:
        key = key.replace(ours, theirs)
    return key


def _released_transformer_key(key: str) -> str:
    attention = ((".attn.to_qkv.", ".attn.qkv."), (".attn.to_out.", ".attn.out."), (".norm_q.", ".q_norm."))
    return _released_key(key, _TRANSFORMER_PREFIXES, (*attention, (".norm_k.", ".k_norm."), *_FF))


def _released_decoder_key(key: str) -> str:
    attention = ((".attn.", ".cross_attn."), (".attn1.", ".self_attn."), (".attn2.", ".cross_attn."))
    norms = ((".norm_q.", ".q_rms_norm."), (".norm_k.", ".k_rms_norm."))
    return _released_key(key, _DECODER_PREFIXES, (*attention, *norms, *_FF))


def test_conversion_renames_released_keys_and_writes_a_loadable_pipeline(
    tmp_path, tiny_triposplat_components, local_external_components, monkeypatch
):
    # The release's bundled DINOv3 (clip_vision/) and Flux.2 VAE copies are gated third-party weights: the converter
    # never reads them, so the tiny release leaves them out.
    root = tmp_path / "TripoSplat"
    for folder in ("vae", "diffusion_models"):
        (root / folder).mkdir(parents=True)
    components = tiny_triposplat_components
    save_file(
        {_released_transformer_key(k): v for k, v in components["transformer"].state_dict().items()},
        root / "diffusion_models" / "triposplat_fp16.safetensors",
    )
    save_file(
        {_released_decoder_key(k): v.contiguous() for k, v in components["decoder"].state_dict().items()},
        root / "vae" / "triposplat_vae_decoder_fp16.safetensors",
    )

    output = tmp_path / "converted"
    convert_triposplat_checkpoint(
        root,
        output,
        configs={
            "image_encoder": tiny_dinov3_config(),
            "vae": tiny_config(TripoSplatImageVAE),
            "transformer": tiny_config(TripoSplatTransformerModel),
            "decoder": tiny_config(TripoSplatGaussianDecoder),
        },
    )
    index = json.loads((output / "model_index.json").read_text())
    assert index["image_encoder"] == ["transformers", "DINOv3ViTModel"] and index["vae"][1] == "TripoSplatImageVAE"
    assert not (output / "image_encoder").exists() and not (output / "vae").exists()

    # Loading fetches them from their repositories; the VAE goes through the real loader from a local folder in the
    # FLUX.2-dev layout, with TripoSplat's batch-norm epsilon whatever the VAE config says.
    flux2 = tmp_path / "flux2" / "vae"
    flux2.mkdir(parents=True)
    vae_config = {**tiny_config(TripoSplatImageVAE), "batch_norm_eps": 1e-4, "_class_name": "AutoencoderKLFlux2"}
    (flux2 / "config.json").write_text(json.dumps(vae_config))
    vae_state = dict(components["vae"].state_dict())
    vae_state.update({"decoder.conv_in.weight": torch.zeros(1), "post_quant_conv.weight": torch.zeros(1)})
    save_file(vae_state, flux2 / "diffusion_pytorch_model.safetensors")
    vae = TripoSplatImageTo3DPipeline.external_components["vae"]
    calls = local_external_components(TripoSplatImageTo3DPipeline, image_encoder=components["image_encoder"])
    external = TripoSplatImageTo3DPipeline.external_components
    monkeypatch.setitem(external, "vae", dataclasses.replace(vae, repository=str(tmp_path / "flux2")))
    pipeline = TripoSplatImageTo3DPipeline.from_pretrained(output, local_files_only=True)
    assert [(call[1], call[3]) for call in calls] == [("facebook/dinov3-vith16plus-pretrain-lvd1689m", torch.bfloat16)]
    assert pipeline.vae.bn.eps == 1e-5 and not pipeline.vae.config.clamp_logvar
    assert type(pipeline.image_encoder) is DINOv3ViTModel
    assert type(pipeline.scheduler) is FlowMatchEulerDiscreteScheduler and pipeline.scheduler.config.shift == 3.0
    # Loading without a dtype gives each component upstream's precision.
    for name in ("image_encoder", "vae", "transformer", "decoder"):
        dtype = UPSTREAM_DTYPES.get(name, UPSTREAM_DTYPES["default"])
        assert getattr(pipeline, name).dtype == dtype
        expected = {
            key: value.to(dtype) if value.is_floating_point() else value
            for key, value in components[name].state_dict().items()
        }
        actual = getattr(pipeline, name).state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected), name


def test_main_parses_arguments(monkeypatch, tmp_path):
    import diffusers_3d.families.triposplat.conversion as conversion

    calls = {}
    monkeypatch.setattr(
        conversion, "convert_triposplat_checkpoint", lambda *args, **kwargs: calls.update(a=args, k=kwargs)
    )
    assert main(["VAST-AI/TripoSplat", str(tmp_path), "--revision", "main"]) == 0
    assert calls["a"] == ("VAST-AI/TripoSplat", tmp_path)
    assert calls["k"] == {"revision": "main", "safe_serialization": True}
