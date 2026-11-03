"""Tiny and production configs for the TripoSplat components, as plain dicts."""

from __future__ import annotations

from typing import Any


def tiny_dinov3_config() -> dict[str, Any]:
    return {
        "hidden_size": 24,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "patch_size": 16,
        "num_register_tokens": 4,
        "use_gated_mlp": True,
        "hidden_act": "silu",
        "key_bias": False,
        "image_size": 32,
    }
