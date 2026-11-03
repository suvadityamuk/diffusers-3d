from __future__ import annotations

import copy
import dataclasses
import json

import pytest
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
from transformers import DINOv3ViTModel
from triposplat.test_conversion import _released_decoder_key, _released_transformer_key

from diffusers_3d.families.idol import IdolImageTo3DPipeline, convert_idol_checkpoint
from diffusers_3d.families.idol.conversion import (
    convert_idol_transformer_state_dict,
    idol_component_configs,
    main,
)
from diffusers_3d.families.idol.pipeline import UPSTREAM_DTYPES, float16_blocks
from diffusers_3d.families.triposplat import TripoSplatGaussianDecoder, TripoSplatTransformerModel
from diffusers_3d.families.triposplat.models import load_flux2_vae_encoder

from .configs import (
    production_decoder_config,
    production_transformer_config,
    tiny_decoder_config,
    tiny_dinov3_config,
    tiny_transformer_config,
    tiny_upstream_config,
    tiny_vae_config,
)


def _sobol(count: int) -> torch.Tensor:
    return torch.quasirandom.SobolEngine(3, scramble=True, seed=123).draw(count, dtype=torch.float32)[None]


def _write_release(root, components) -> None:
    """Write tiny components in the released speridlabs/idol layout."""

    root.mkdir()
    (root / "config.json").write_text(json.dumps(tiny_upstream_config()))
    denoiser = {_released_transformer_key(k): v for k, v in components["transformer"].state_dict().items()}
    denoiser["pos_pe"] = _sobol(tiny_transformer_config()["num_latent_tokens"])
    save_file(denoiser, root / "denoiser.safetensors")
    decoder = {_released_decoder_key(k): v.contiguous() for k, v in components["decoder"].state_dict().items()}
    for name, prefix in (("decoder", "octree."), ("decoder_gs", "gs.")):
        part = {k.removeprefix(prefix): v for k, v in decoder.items() if k.startswith(prefix)}
        save_file(part, root / f"{name}.safetensors")


def _write_flux2_vae(root, vae) -> None:
    """A Diffusers-layout Flux.2 VAE folder: ``vae/config.json`` and the full state dict, decoder half included."""

    (root / "vae").mkdir(parents=True)
    config = tiny_vae_config()
    (root / "vae" / "config.json").write_text(json.dumps({**config, "_class_name": "AutoencoderKLFlux2"}))
    state_dict = dict(vae.state_dict())
    state_dict.update({"decoder.conv_in.weight": torch.zeros(1), "post_quant_conv.weight": torch.zeros(1)})
    save_file(state_dict, root / "vae" / "diffusion_pytorch_model.safetensors")


def test_conversion_writes_a_loadable_pipeline_without_the_gated_encoders(
    tmp_path, tiny_idol_components, local_external_components, monkeypatch
):
    components = tiny_idol_components
    _write_release(tmp_path / "idol", components)
    _write_flux2_vae(tmp_path / "flux2", components["vae"])
    vae = IdolImageTo3DPipeline.external_components["vae"]
    calls = local_external_components(IdolImageTo3DPipeline, image_encoder=components["image_encoder"])
    # The VAE goes through the real loader, from a local folder in the FLUX.2-dev layout.
    external = IdolImageTo3DPipeline.external_components
    monkeypatch.setitem(external, "vae", dataclasses.replace(vae, repository=str(tmp_path / "flux2")))

    shells = {"image_encoder": tiny_dinov3_config(), "vae": tiny_vae_config()}
    output = convert_idol_checkpoint(tmp_path / "idol", tmp_path / "converted", encoder_configs=shells)
    # DINOv3 and the Flux.2 VAE are never downloaded or saved; model_index.json still names them.
    index = json.loads((output / "model_index.json").read_text())
    assert index["image_encoder"] == ["transformers", "DINOv3ViTModel"] and index["vae"][1] == "TripoSplatImageVAE"
    assert not (output / "image_encoder").exists() and not (output / "vae").exists()
    pipeline = IdolImageTo3DPipeline.from_pretrained(output, local_files_only=True)
    assert calls == [
        (
            "image_encoder",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "c807c9eeea853df70aec4069e6f56b28ddc82acc",
            torch.bfloat16,
        )
    ]
    assert pipeline.vae.config.clamp_logvar
    assert type(pipeline.image_encoder) is DINOv3ViTModel
    assert type(pipeline.scheduler) is FlowMatchEulerDiscreteScheduler and pipeline.scheduler.config.shift == 3.0
    assert pipeline.vae.bn.eps == tiny_vae_config()["batch_norm_eps"]
    # Loading without a dtype gives each component upstream's precision, with float16 transformer and decoder blocks.
    reference = {name: copy.deepcopy(module) for name, module in components.items() if name != "scheduler"}
    float16_blocks(reference["transformer"]), float16_blocks(reference["decoder"])
    for name in ("image_encoder", "vae", "transformer", "decoder"):
        dtype = UPSTREAM_DTYPES.get(name, UPSTREAM_DTYPES["default"])
        assert getattr(pipeline, name).dtype == dtype, name
        expected = {
            key: value.to(dtype) if value.dtype == torch.float32 and dtype != torch.float32 else value
            for key, value in reference[name].state_dict().items()
        }
        actual = getattr(pipeline, name).state_dict()
        assert all(actual[key].dtype == expected[key].dtype for key in expected), name
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected), name


def test_flux2_vae_encoder_drops_the_decoder_half(tmp_path, tiny_idol_components):
    _write_flux2_vae(tmp_path, tiny_idol_components["vae"])
    expected = tiny_idol_components["vae"].state_dict()
    vae = load_flux2_vae_encoder(tmp_path)
    # Float32 as stored, the config's batch-norm epsilon, and no clamp unless asked (TripoSplat).
    assert vae.dtype is torch.float32 and not vae.config.clamp_logvar and vae.bn.eps == 1e-4
    assert all(torch.equal(vae.state_dict()[key], value) for key, value in expected.items())
    vae = load_flux2_vae_encoder(tmp_path, dtype=torch.bfloat16, batch_norm_eps=1e-5, clamp_logvar=True)
    assert vae.dtype is torch.bfloat16 and vae.config.clamp_logvar and vae.bn.eps == 1e-5
    assert all(torch.equal(vae.state_dict()[key], value.to(torch.bfloat16)) for key, value in expected.items())


def test_released_config_maps_to_the_production_triposplat_configs():
    # speridlabs/idol@2731c229 config.json, abridged to the fields the mapping reads.
    released = tiny_upstream_config()
    released["denoiser"]["args"].update(
        q_token_length=8192, in_channels=16, model_channels=1024, cond_channels=1280, cond2_channels=128,
        num_blocks=24, num_heads=16, num_head_channels=64, mlp_ratio=4,
    )  # fmt: skip
    shared = {"model_channels": 1024, "cond_channels": 16, "num_heads": 16, "mlp_ratio": 4}
    released["decoder"]["args"].update(shared, num_blocks=4)
    released["decoder_gs"]["args"].update(shared, num_blocks=16)
    released["decoder_gs"]["args"]["representation_config"]["num_gaussians"] = 32
    released["max_voxel_level"] = 8
    configs = idol_component_configs(released)
    assert configs["transformer"] == production_transformer_config()
    assert configs["decoder"] == production_decoder_config()

    released["decoder_gs"]["args"]["representation_config"]["lr"]["_xyz"] = 0.5
    with pytest.raises(ValueError, match="learning-rate"):
        idol_component_configs(released)


def test_production_layouts_hold_the_released_tensor_counts():
    # speridlabs/idol@2731c229 safetensors headers: 560 denoiser tensors (pos_pe included), 60 octree decoder and
    # 328 Gaussian decoder tensors.
    with init_empty_weights():
        transformer = TripoSplatTransformerModel(**production_transformer_config()).state_dict()
        decoder = TripoSplatGaussianDecoder(**production_decoder_config()).state_dict()
    assert (len(transformer) + 1, len(decoder)) == (560, 60 + 328)


def test_transformer_conversion_rejects_positions_that_are_not_the_sobol_points(tiny_idol_components):
    state_dict = {_released_transformer_key(k): v for k, v in tiny_idol_components["transformer"].state_dict().items()}
    count = tiny_transformer_config()["num_latent_tokens"]
    assert convert_idol_transformer_state_dict({**state_dict, "pos_pe": _sobol(count)}).keys() == (
        tiny_idol_components["transformer"].state_dict().keys()
    )
    with pytest.raises(ValueError, match="Sobol"):
        convert_idol_transformer_state_dict({**state_dict, "pos_pe": torch.zeros(1, count, 3)})


def test_main_parses_arguments(monkeypatch, tmp_path):
    import diffusers_3d.families.idol.conversion as conversion

    calls = {}
    monkeypatch.setattr(conversion, "convert_idol_checkpoint", lambda *args, **kwargs: calls.update(a=args, k=kwargs))
    assert main(["speridlabs/idol", str(tmp_path)]) == 0
    assert calls["a"] == ("speridlabs/idol", tmp_path)
    assert calls["k"] == {"revision": conversion.IDOL_WEIGHTS_REVISION, "safe_serialization": True}


def test_conversion_rejects_other_image_encoders(tmp_path, tiny_idol_components):
    _write_release(tmp_path / "idol", tiny_idol_components)
    config = json.loads((tmp_path / "idol" / "config.json").read_text())
    config["image_encoders"]["dinov3"] = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    (tmp_path / "idol" / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="must name the image encoders the pipeline loads"):
        convert_idol_checkpoint(tmp_path / "idol", tmp_path / "converted")


def test_tiny_configs_round_trip_through_the_upstream_config(tiny_idol_components):
    configs = idol_component_configs(tiny_upstream_config())
    assert configs["transformer"] == tiny_transformer_config()
    assert {k: configs["decoder"][k] for k in tiny_decoder_config()} == tiny_decoder_config()
