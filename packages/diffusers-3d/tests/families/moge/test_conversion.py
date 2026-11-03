from __future__ import annotations

import json

import pytest
import torch

from diffusers_3d.families.moge import MoGe2Model, MoGeModel
from diffusers_3d.families.moge.conversion import convert_dinov2_state_dict, moge2_config, moge_config
from diffusers_3d.families.moge.models import DINOV2_VITL14_CONFIG

from .configs import RELEASED_MOGE2_CONFIG, RELEASED_MOGE_CONFIG


@pytest.mark.parametrize(
    ("model_type", "translate", "released"),
    [(MoGeModel, moge_config, RELEASED_MOGE_CONFIG), (MoGe2Model, moge2_config, RELEASED_MOGE2_CONFIG)],
)
def test_released_configs_are_the_defaults(model_type, translate, released):
    config = translate(released)
    with torch.device("meta"):
        defaults, converted = model_type().config, model_type(**config).config
    # Compare as saved to ``config.json`` (tuples become lists); ``backbone_config=None`` means DINOv2 ViT-L/14.
    defaults = {key: value for key, value in defaults.items() if not key.startswith("_")}
    defaults["backbone_config"] = DINOV2_VITL14_CONFIG
    converted = {key: value for key, value in converted.items() if not key.startswith("_")}
    assert json.loads(json.dumps(converted)) == json.loads(json.dumps(defaults))


def test_released_moge_config_translates_the_legacy_area_range():
    config = moge_config(RELEASED_MOGE_CONFIG)
    # ``trained_area_range`` [250000, 500000] pixels at 14-pixel patches.
    assert config["num_tokens_range"] == [1275, 2551]
    assert config["intermediate_layers"] == 4
    assert config["backbone_config"]["hidden_size"] == 1024


def test_unsupported_configs_are_rejected():
    with pytest.raises(ValueError, match="last_res_blocks"):
        moge_config({**RELEASED_MOGE_CONFIG, "last_res_blocks": 1})
    with pytest.raises(ValueError, match="backbone"):
        moge_config({**RELEASED_MOGE_CONFIG, "encoder": "dinov2_vitg14"})
    with pytest.raises(ValueError, match="normal head"):
        moge2_config({**RELEASED_MOGE2_CONFIG, "normal_head": RELEASED_MOGE2_CONFIG["mask_head"]})
    neck = {**RELEASED_MOGE2_CONFIG["neck"], "res_block_hidden_norm": "group_norm"}
    with pytest.raises(ValueError, match="neck"):
        moge2_config({**RELEASED_MOGE2_CONFIG, "neck": neck})


def test_dinov2_keys_map_onto_transformers():
    qkv = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    state = {
        "pos_embed": torch.zeros(1),
        "blocks.3.attn.qkv.weight": qkv,
        "blocks.3.attn.proj.bias": torch.zeros(1),
        "blocks.3.ls2.gamma": torch.zeros(1),
        "blocks.3.mlp.fc1.weight": torch.zeros(1),
        "norm.weight": torch.zeros(1),
    }
    converted = convert_dinov2_state_dict(state)
    assert sorted(converted) == [
        "embeddings.position_embeddings",
        "encoder.layer.3.attention.attention.key.weight",
        "encoder.layer.3.attention.attention.query.weight",
        "encoder.layer.3.attention.attention.value.weight",
        "encoder.layer.3.attention.output.dense.bias",
        "encoder.layer.3.layer_scale2.lambda1",
        "encoder.layer.3.mlp.fc1.weight",
        "layernorm.weight",
    ]
    torch.testing.assert_close(converted["encoder.layer.3.attention.attention.key.weight"], qkv[2:4])
