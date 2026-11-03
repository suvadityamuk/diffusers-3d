"""Tiny MoGe configs, and the ``model_config`` dicts stored in the released checkpoints."""

from __future__ import annotations

from typing import Any

from _configs import MOGE_TINY_BACKBONE as TINY_BACKBONE
from _configs import tiny_config

from diffusers_3d.families.moge import MoGe2Model, MoGeModel


def tiny_moge_config() -> dict[str, Any]:
    return tiny_config(MoGeModel)


def tiny_moge2_config() -> dict[str, Any]:
    return tiny_config(MoGe2Model)


# ``checkpoint["model_config"]`` of Ruicheng/moge-vitl and Ruicheng/moge-2-vitl at the pinned revisions.
RELEASED_MOGE_CONFIG = {
    "encoder": "dinov2_vitl14",
    "remap_output": "exp",
    "output_mask": True,
    "split_head": True,
    "intermediate_layers": 4,
    "dim_upsample": [256, 128, 64],
    "dim_times_res_block_hidden": 2,
    "num_res_blocks": 2,
    "trained_area_range": [250000, 500000],
    "last_conv_channels": 32,
    "last_conv_size": 1,
}
_RELEASED_HEAD = {
    "dim_in": [1024, 256, 128, 64, 32],
    "dim_res_blocks": [1024, 256, 128, 64, 32],
    "num_res_blocks": [0, 1, 1, 1, 0],
    "res_block_in_norm": "none",
    "res_block_hidden_norm": "none",
    "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
}
RELEASED_MOGE2_CONFIG = {
    "encoder": {"backbone": "dinov2_vitl14", "intermediate_layers": [5, 11, 17, 23], "dim_out": 1024},
    "neck": {
        "dim_in": [1026, 2, 2, 2, 2],
        "dim_out": None,
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 2, 2, 2, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "points_head": {**_RELEASED_HEAD, "dim_out": [None, None, None, None, 3]},
    "mask_head": {**_RELEASED_HEAD, "dim_out": [None, None, None, None, 1]},
    "scale_head": {"dims": [1024, 1024, 1024, 1]},
    "remap_output": "exp",
    "num_tokens_range": [1200, 3600],
}

__all__ = ["RELEASED_MOGE2_CONFIG", "RELEASED_MOGE_CONFIG", "TINY_BACKBONE", "tiny_moge2_config", "tiny_moge_config"]
