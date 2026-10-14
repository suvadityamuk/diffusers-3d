from __future__ import annotations

import json
import re
import warnings

import pytest
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

from diffusers_3d import Hunyuan3DPaintPipeline, Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
from diffusers_3d.families.hunyuan3d import paint_conversion
from diffusers_3d.families.hunyuan3d.paint_conversion import convert_unet_state_dict, unet_config

RELEASED_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.10.0.dev0",
    "act_fn": "silu",
    "attention_head_dim": [1, 2],
    "block_out_channels": [16, 32],
    "center_input_sample": False,
    "cross_attention_dim": 12,
    "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 8,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "sample_size": 8,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
    "use_linear_projection": True,
}
_WRAPPED = r"\.transformer_blocks\.(\d+)\.(norm1|norm2|norm3|attn1|attn2|ff)\."


def _released_state_dict(model: Hunyuan3DPaintUNet) -> dict[str, torch.Tensor]:
    """The upstream checkpoint layout: blocks wrapped as `.transformer`, learned tokens on the UNet, `unet_dual`."""

    state = {}
    for key, value in model.state_dict().items():
        if key == "material_embeds":
            state["unet.learned_text_clip_albedo"], state["unet.learned_text_clip_mr"] = value.clone()
            continue
        if key == "reference_embeds":
            state["unet.learned_text_clip_ref"] = value.clone()
            continue
        key = re.sub(r"^reference_unet\.", "unet_dual.", key)
        key = re.sub(r"^image_projection\.image_embeds\.", "unet.image_proj_model_dino.proj.", key)
        key = re.sub(r"^image_projection\.norm\.", "unet.image_proj_model_dino.norm.", key)
        key = re.sub(r"\.processor\.to_out_mr\.", ".processor.to_out_mr.0.", key)
        state[re.sub(_WRAPPED, r".transformer_blocks.\1.transformer.\2.", key)] = value.clone()
    return state


@pytest.fixture
def tiny_release(tmp_path):
    torch.manual_seed(0)
    unet = Hunyuan3DPaintUNet(**unet_config(RELEASED_CONFIG))
    with torch.no_grad():
        for parameter in unet.parameters():
            parameter.normal_()
    folder = tmp_path / "release" / "hunyuan3d-paintpbr-v2-1"
    (folder / "unet").mkdir(parents=True)
    (folder / "unet" / "config.json").write_text(json.dumps(RELEASED_CONFIG))
    torch.save(_released_state_dict(unet), folder / "unet" / "diffusion_pytorch_model.bin")
    vae = AutoencoderKL(
        block_out_channels=(8, 16),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        norm_num_groups=8,
    )
    vae.save_pretrained(folder / "vae", safe_serialization=False)
    DDIMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        steps_offset=1,
        clip_sample=False,
    ).save_pretrained(folder / "scheduler")
    return tmp_path / "release", unet, vae


def test_converts_a_tiny_release_into_a_loadable_pipeline(tiny_release, tmp_path, monkeypatch):
    source, unet, vae = tiny_release
    dinov2 = Dinov2Model(
        Dinov2Config(
            hidden_size=1536, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=1, image_size=28, patch_size=14
        )
    )
    processor = BitImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28})
    monkeypatch.setattr(paint_conversion.Dinov2Model, "from_pretrained", lambda *args, **kwargs: dinov2)
    monkeypatch.setattr(paint_conversion.BitImageProcessorPil, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(
        paint_conversion, "_load_upscaler", lambda: Hunyuan3DPaintRRDBNet(**Hunyuan3DPaintRRDBNet.tiny_config())
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        destination = paint_conversion.convert_hunyuan3d_paint_checkpoint(source, tmp_path / "converted")
        monkeypatch.undo()
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(destination)
    for name, value in unet.state_dict().items():
        assert torch.equal(pipeline.unet.state_dict()[name], value), name
    for name, value in vae.state_dict().items():
        assert torch.equal(pipeline.vae.state_dict()[name], value), name
    assert type(pipeline.scheduler) is UniPCMultistepScheduler
    config = pipeline.scheduler.config
    assert (config.prediction_type, config.timestep_spacing, config.rescale_betas_zero_snr) == (
        "v_prediction",
        "trailing",
        True,
    )
    assert pipeline.upscaler is not None and (destination / "unet" / "diffusion_pytorch_model.safetensors").exists()
    report = json.loads((destination / "hunyuan3d_paint_conversion.json").read_text())
    assert report["image_encoder"]["revision"] == paint_conversion.DINOV2_REVISION
    assert "text_encoder" in report["dropped_components"]


def test_unet_config_rejects_options_the_port_does_not_implement():
    with pytest.raises(ValueError, match="unsupported"):
        unet_config({**RELEASED_CONFIG, "use_linear_projection": False})
    with pytest.raises(ValueError, match="unknown"):
        unet_config({**RELEASED_CONFIG, "class_embed_type": "timestep"})


def test_released_keys_map_onto_the_model():
    unet = Hunyuan3DPaintUNet(**unet_config(RELEASED_CONFIG))
    converted = convert_unet_state_dict(_released_state_dict(unet))
    assert converted.keys() == unet.state_dict().keys()
