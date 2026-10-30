"""Tiny and published configs for TripoSR components."""

from __future__ import annotations

from typing import Any

from _configs import tiny_config

from diffusers_3d.families.triposr.models import TripoSRTransformerModel, TripoSRTriplaneDecoder


def tiny_vit_config() -> dict[str, Any]:
    # A 4x4 patch grid at 32 px; ``hidden_size`` matches the transformer's ``cross_attention_dim``.
    return {
        "hidden_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "intermediate_size": 24,
        "image_size": 32,
        "patch_size": 8,
    }


def upstream_config(image_tokenizer_path: str) -> dict[str, Any]:
    """The upstream ``config.yaml`` layout at tiny size; the tokenizer interpolates a 2x2 position grid to 4x4."""

    transformer = tiny_config(TripoSRTransformerModel)
    decoder = tiny_config(TripoSRTriplaneDecoder)
    return {
        "cond_image_size": 32,
        "image_tokenizer_cls": "tsr.models.tokenizers.image.DINOSingleImageTokenizer",
        "image_tokenizer": {"pretrained_model_name_or_path": image_tokenizer_path},
        "tokenizer_cls": "tsr.models.tokenizers.triplane.Triplane1DTokenizer",
        "tokenizer": {"plane_size": transformer["triplane_size"], "num_channels": transformer["num_channels"]},
        "backbone_cls": "tsr.models.transformer.transformer_1d.Transformer1D",
        "backbone": {
            "in_channels": transformer["num_channels"],
            "num_attention_heads": transformer["num_attention_heads"],
            "attention_head_dim": transformer["attention_head_dim"],
            "num_layers": transformer["num_layers"],
            "cross_attention_dim": transformer["cross_attention_dim"],
            "norm_num_groups": transformer["norm_num_groups"],
        },
        "post_processor_cls": "tsr.models.network_utils.TriplaneUpsampleNetwork",
        "post_processor": {"in_channels": transformer["num_channels"], "out_channels": transformer["out_channels"]},
        "decoder_cls": "tsr.models.network_utils.NeRFMLP",
        "decoder": {
            "in_channels": 3 * decoder["triplane_channels"],
            "n_neurons": decoder["hidden_size"],
            "n_hidden_layers": decoder["num_hidden_layers"],
            "activation": "silu",
        },
        "renderer_cls": "tsr.models.nerf_renderer.TriplaneNeRFRenderer",
        "renderer": {
            "radius": 0.87,
            "feature_reduction": "concat",
            "density_activation": "exp",
            "density_bias": -1.0,
            "num_samples_per_ray": 8,
        },
    }


def upstream_vit_config() -> dict[str, Any]:
    """The tiny ViT before conversion: a 2x2 position grid at 16 px that TripoSR interpolates to its 32 px input."""

    return {**tiny_vit_config(), "image_size": 16}
