"""Tiny and production configs for the IDOL components (TripoSplat classes), as plain dicts."""

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


def production_dinov3_config() -> dict[str, Any]:
    """facebook/dinov3-vith16plus-pretrain-lvd1689m."""

    return {
        "hidden_size": 1280,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 16,
        "num_register_tokens": 4,
        "use_gated_mlp": True,
        "hidden_act": "silu",
        "key_bias": False,
        "layerscale_value": 1.0,
        "rope_theta": 100.0,
        "layer_norm_eps": 1e-5,
    }


def tiny_vae_config() -> dict[str, Any]:
    # Four blocks keep the 8x downsampling (then 2x2 patchify) that aligns VAE and 16-pixel DINOv3 patches.
    return {
        "latent_channels": 4,
        "block_out_channels": (8, 8, 8, 8),
        "layers_per_block": 1,
        "norm_num_groups": 4,
        "batch_norm_eps": 1e-4,
    }


def production_vae_config() -> dict[str, Any]:
    """The encoder half of the black-forest-labs/FLUX.2-dev VAE."""

    return {
        "latent_channels": 32,
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,
        "batch_norm_eps": 1e-4,
    }


def tiny_transformer_config() -> dict[str, Any]:
    # Upstream sizes its camera MLP by ``num_refiner_blocks``; the released (and only supported) value is 2.
    return {
        "in_channels": 4,
        "camera_channels": 5,
        "num_latent_tokens": 10,
        "width": 36,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "num_attention_heads": 2,
        "mlp_ratio": 2.0,
        "cross_attention_dim": 24,
        "cross_attention_dim_2": 16,
    }


def production_transformer_config() -> dict[str, Any]:
    return {
        "in_channels": 16,
        "camera_channels": 5,
        "num_latent_tokens": 8192,
        "width": 1024,
        "num_layers": 24,
        "num_refiner_layers": 2,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "cross_attention_dim": 1280,
        "cross_attention_dim_2": 128,
    }


def tiny_decoder_config() -> dict[str, Any]:
    return {
        "latent_channels": 4,
        "width": 32,
        "num_attention_heads": 2,
        "num_octree_layers": 1,
        "num_gaussian_layers": 2,
        "mlp_ratio": 2.0,
        "octree_depth": 3,
        "gaussians_per_token": 4,
    }


def production_decoder_config() -> dict[str, Any]:
    return {
        "latent_channels": 16,
        "width": 1024,
        "num_attention_heads": 16,
        "num_octree_layers": 4,
        "num_gaussian_layers": 16,
        "mlp_ratio": 4.0,
        "octree_depth": 8,
        "gaussians_per_token": 32,
        "offset_extent": 1.5,
        "scaling_bias": 0.004,
        "opacity_bias": 0.1,
        "rotation_scale": 0.1,
        "filter_kernel_size": 0.0009,
    }


def tiny_upstream_config() -> dict[str, Any]:
    """An upstream ``config.json`` describing the tiny transformer and decoder."""

    transformer, decoder = tiny_transformer_config(), tiny_decoder_config()
    shared = {
        "model_channels": decoder["width"],
        "cond_channels": decoder["latent_channels"],
        "num_heads": decoder["num_attention_heads"],
        "mlp_ratio": decoder["mlp_ratio"],
    }
    return {
        "denoiser": {
            "name": "LatentSeqMMFlowModel",
            "args": {
                "q_token_length": transformer["num_latent_tokens"],
                "in_channels": transformer["in_channels"],
                "cam_channels": transformer["camera_channels"],
                "model_channels": transformer["width"],
                "cond_channels": transformer["cross_attention_dim"],
                "cond2_channels": transformer["cross_attention_dim_2"],
                "num_blocks": transformer["num_layers"],
                "num_refiner_blocks": transformer["num_refiner_layers"],
                "num_heads": transformer["num_attention_heads"],
                "num_head_channels": transformer["width"] // transformer["num_attention_heads"],
                "mlp_ratio": transformer["mlp_ratio"],
            },
        },
        "decoder": {
            "name": "OctreeProbabilityFixedlenDecoder",
            "args": {**shared, "num_blocks": decoder["num_octree_layers"]},
        },
        "decoder_gs": {
            "name": "GaussianFixedlenDecoder",
            "args": {
                **shared,
                "in_channels": 3,
                "num_blocks": decoder["num_gaussian_layers"],
                "representation_config": {
                    "num_gaussians": decoder["gaussians_per_token"],
                    "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
                    "perturbe_size": 1.5,
                    "3d_filter_kernel_size": 0.0009,
                    "scaling_bias": 0.004,
                    "opacity_bias": 0.1,
                },
            },
        },
        "image_size": 32,
        "max_voxel_level": decoder["octree_depth"],
        "image_encoders": {
            "dinov3": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "flux2_vae": "black-forest-labs/FLUX.2-dev",
        },
    }
