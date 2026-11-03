"""Tiny configs for building every model at test size, and published-size configs used only by tests."""

from __future__ import annotations

from typing import Any

from diffusers_3d.families.hunyuan3d.models import Hunyuan3DDiTModel
from diffusers_3d.families.hunyuan3d.paint_models import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
from diffusers_3d.families.hunyuan3d.vae import Hunyuan3DShapeVAE
from diffusers_3d.families.hunyuan3d_omni.models import Hunyuan3DOmniControlEncoder
from diffusers_3d.families.moge import MoGe2Model, MoGeModel
from diffusers_3d.families.pixal3d.conditioner import Pixal3DNAFUpsampler
from diffusers_3d.families.pixal3d.models import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from diffusers_3d.families.sam3d_objects.conditioner import Sam3DConditionEmbedder
from diffusers_3d.families.sam3d_objects.models import Sam3DSparseStructureFlowModel
from diffusers_3d.families.step1x3d.models import Step1X3DTransformer1DModel
from diffusers_3d.families.step1x3d.texture_attention import Step1X3DMultiViewAdapter
from diffusers_3d.families.step1x3d.vae import Step1X3DShapeDecoder
from diffusers_3d.families.trellis.decoders import (
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
)
from diffusers_3d.families.trellis.models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from diffusers_3d.families.trellis2.decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder
from diffusers_3d.families.trellis2.models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
from diffusers_3d.families.triposg.models import TripoSGTransformerModel
from diffusers_3d.families.triposg.vae import TripoSGVAEDecoder
from diffusers_3d.families.triposplat.decoder import TripoSplatGaussianDecoder
from diffusers_3d.families.triposplat.models import TripoSplatImageVAE, TripoSplatTransformerModel
from diffusers_3d.families.triposr.models import TripoSRTransformerModel, TripoSRTriplaneDecoder


def _production_config_Hunyuan3DDiTModel() -> dict[str, Any]:
    return {
        "in_channels": 64,
        "hidden_size": 2048,
        "context_dim": 1024,
        "depth": 21,
        "num_heads": 16,
        "num_moe_layers": 6,
        "num_experts": 8,
        "moe_top_k": 2,
    }


# A 112-pixel native grid (8x8 positions), so every MoGe token grid is interpolated.
MOGE_TINY_BACKBONE = {
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "mlp_ratio": 4,
    "patch_size": 14,
    "image_size": 112,
    "layerscale_value": 1.0,
    "layer_norm_eps": 1e-6,
}


def _tiny_config_MoGeModel() -> dict[str, Any]:
    return {
        "backbone_config": MOGE_TINY_BACKBONE,
        "intermediate_layers": 2,
        "dim_proj": 16,
        "dim_upsample": [32, 16, 16],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 1,
        "last_conv_channels": 8,
        "num_tokens_range": [40, 80],
    }


def _tiny_config_MoGe2Model() -> dict[str, Any]:
    return {
        "backbone_config": MOGE_TINY_BACKBONE,
        "intermediate_layers": [0, 1, 2, 3],
        "dim_features": 32,
        "dim_res_blocks": [32, 16, 16, 8, 8],
        "neck_num_res_blocks": [0, 1, 1, 1, 0],
        "head_num_res_blocks": [0, 1, 1, 1, 0],
        "scale_head_dims": [32, 16, 1],
        "num_tokens_range": [40, 80],
    }


def _tiny_config_TripoSplatTransformerModel() -> dict[str, Any]:
    # The camera MLP follows ``num_refiner_layers``, which must be at least 2, as upstream.
    return {
        "in_channels": 4,
        "camera_channels": 5,
        "num_latent_tokens": 10,
        "width": 32,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "num_attention_heads": 2,
        "mlp_ratio": 2.0,
        "cross_attention_dim": 24,
        "cross_attention_dim_2": 16,
    }


def _production_config_TripoSplatTransformerModel() -> dict[str, Any]:
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


def _tiny_config_TripoSplatGaussianDecoder() -> dict[str, Any]:
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


def _production_config_TripoSplatGaussianDecoder() -> dict[str, Any]:
    return {
        "latent_channels": 16,
        "width": 1024,
        "num_attention_heads": 16,
        "num_octree_layers": 4,
        "num_gaussian_layers": 16,
        "mlp_ratio": 4.0,
        "octree_depth": 8,
        "gaussians_per_token": 32,
    }


def _tiny_config_TripoSplatImageVAE() -> dict[str, Any]:
    # Four blocks keep the 8x downsampling (then 2x2 patchify) that aligns VAE and 16-pixel DINOv3 patches.
    return {"latent_channels": 4, "block_out_channels": (8, 8, 8, 8), "layers_per_block": 1, "norm_num_groups": 4}


def _production_config_TripoSplatImageVAE() -> dict[str, Any]:
    return {"latent_channels": 32, "block_out_channels": (128, 256, 512, 512), "layers_per_block": 2}


def _tiny_config_TripoSRTransformerModel() -> dict[str, Any]:
    return {
        "triplane_size": 4,
        "num_channels": 16,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "num_layers": 2,
        "cross_attention_dim": 12,
        "norm_num_groups": 4,
        "out_channels": 6,
    }


def _tiny_config_TripoSRTriplaneDecoder() -> dict[str, Any]:
    return {"triplane_channels": 6, "hidden_size": 8, "num_hidden_layers": 2}


def _production_config_TripoSRTransformerModel() -> dict[str, Any]:
    # stabilityai/TripoSR@5b521936 config.yaml: tokenizer, backbone, and post_processor.
    return {
        "triplane_size": 32,
        "num_channels": 1024,
        "num_attention_heads": 16,
        "attention_head_dim": 64,
        "num_layers": 16,
        "cross_attention_dim": 768,
        "norm_num_groups": 32,
        "out_channels": 40,
    }


def _production_config_TripoSRTriplaneDecoder() -> dict[str, Any]:
    return {"triplane_channels": 40, "hidden_size": 64, "num_hidden_layers": 9, "radius": 0.87, "density_bias": -1.0}


def _tiny_config_Hunyuan3DOmniControlEncoder() -> dict[str, Any]:
    # ``width`` matches the tiny DiT's ``context_dim`` and DINOv2's ``hidden_size``.
    return {"width": 24, "num_freqs": 2, "voxel_resolution": 4}


def _production_config_Hunyuan3DOmniControlEncoder() -> dict[str, Any]:
    return {"width": 1024, "num_freqs": 8, "voxel_resolution": 16, "num_type_tokens": 10}


def _tiny_config_Hunyuan3DDiTModel() -> dict[str, Any]:
    return {
        "in_channels": 8,
        "hidden_size": 32,
        "context_dim": 24,
        "depth": 4,
        "num_heads": 4,
        "num_moe_layers": 1,
        "num_experts": 3,
        "moe_top_k": 2,
    }


def _production_config_Hunyuan3DPaintUNet() -> dict[str, Any]:
    return {}


def _tiny_config_Hunyuan3DPaintUNet() -> dict[str, Any]:
    return {
        "sample_size": 8,
        "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
        "block_out_channels": (16, 32),
        "layers_per_block": 1,
        "attention_head_dim": (1, 2),
        "cross_attention_dim": 12,
        "norm_num_groups": 8,
        "num_text_tokens": 3,
        "image_embed_dim": 10,
        "num_image_text_embeds": 2,
    }


def _production_config_Hunyuan3DPaintRRDBNet() -> dict[str, Any]:
    return {}


def _tiny_config_Hunyuan3DPaintRRDBNet() -> dict[str, Any]:
    return {"num_features": 8, "num_blocks": 1, "growth_channels": 4}


def _production_config_Hunyuan3DShapeVAE() -> dict[str, Any]:
    return {
        "num_latents": 4096,
        "embed_dim": 64,
        "width": 1024,
        "heads": 16,
        "num_decoder_layers": 16,
        "num_freqs": 8,
        "scale_factor": 1.0039506158752403,
    }


def _tiny_config_Hunyuan3DShapeVAE() -> dict[str, Any]:
    return {
        "num_latents": 6,
        "embed_dim": 8,
        "width": 32,
        "heads": 4,
        "num_decoder_layers": 2,
        "num_freqs": 2,
        "scale_factor": 1.25,
    }


def _production_config_Pixal3DNAFUpsampler() -> dict[str, Any]:
    return {}


def _tiny_config_Pixal3DNAFUpsampler() -> dict[str, Any]:
    return {"dim": 16, "num_heads": 2, "rope_heads": 2, "kernel_size": 3, "encoder_layers": 1}


def _production_config_Pixal3DSparseStructureFlowModel() -> dict[str, Any]:
    return {**production_config(_parent(Pixal3DSparseStructureFlowModel)), "proj_in_channels": 1024}


def _tiny_config_Pixal3DSparseStructureFlowModel() -> dict[str, Any]:
    return {**tiny_config(_parent(Pixal3DSparseStructureFlowModel)), "proj_in_channels": 12}


def _production_config_Pixal3DSLatFlowModel(*, texture=False) -> dict[str, Any]:
    return {**production_config(_parent(Pixal3DSLatFlowModel), texture=texture), "proj_in_channels": 2048}


def _tiny_config_Pixal3DSLatFlowModel(*, texture=False) -> dict[str, Any]:
    return {**tiny_config(_parent(Pixal3DSLatFlowModel), texture=texture), "proj_in_channels": 24}


def _tiny_config_Sam3DConditionEmbedder() -> dict[str, Any]:
    return {
        "image_embed_dim": 12,
        "prenorm_features": False,
        "projection_hidden_multiplier": 1.0,
        "pointmap_config": {"embed_dim": 8, "input_size": 8, "patch_size": 4, "num_heads": 2, "mlp_ratio": 2.0},
    }


def _tiny_config_Sam3DSparseStructureFlowModel() -> dict[str, Any]:
    return {
        "resolution": 2,
        "in_channels": 2,
        "pose_latents": Sam3DSparseStructureFlowModel.production_pose_latents(),
        "model_channels": 16,
        "cond_channels": 12,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "qk_rms_norm": True,
    }


def _tiny_config_Step1X3DTransformer1DModel() -> dict[str, Any]:
    return {
        "in_channels": 4,
        "width": 16,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_attention_heads": 2,
        "cross_attention_dim": 12,
    }


def _tiny_config_Step1X3DMultiViewAdapter() -> dict[str, Any]:
    return {"hidden_sizes": {"down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": 8}}


def _tiny_config_Step1X3DShapeDecoder() -> dict[str, Any]:
    return {
        "num_latents": 8,
        "latent_channels": 4,
        "width": 16,
        "num_attention_heads": 2,
        "num_layers": 1,
        "num_freqs": 2,
    }


def _tiny_config_TrellisSparseStructureDecoder() -> dict[str, Any]:
    return {
        "out_channels": 1,
        "latent_channels": 2,
        "num_res_blocks": 1,
        "num_res_blocks_middle": 1,
        "channels": [8, 4],
        "norm_type": "layer",
        "use_fp16": False,
    }


def _tiny_config_TrellisSLatGaussianDecoder() -> dict[str, Any]:
    representation_config = TrellisSLatGaussianDecoder.production_representation_config()
    representation_config["num_gaussians"] = 2
    representation_config["perturb_offset"] = False
    return {
        "resolution": 8,
        "model_channels": 16,
        "latent_channels": 4,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "attn_mode": "full",
        "window_size": 2,
        "use_fp16": False,
        "representation_config": representation_config,
    }


def _tiny_config_TrellisSLatMeshDecoder() -> dict[str, Any]:
    return {
        "resolution": 8,
        "model_channels": 32,
        "latent_channels": 4,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "attn_mode": "full",
        "window_size": 2,
        "use_fp16": False,
        "representation_config": {"use_color": True},
        "num_groups": 4,
    }


def _production_config_TrellisSLatRadianceFieldDecoder() -> dict[str, Any]:
    return {
        "resolution": 64,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "use_fp16": True,
        "representation_config": {"rank": 16, "dim": 8},
    }


def _tiny_config_TrellisSLatRadianceFieldDecoder() -> dict[str, Any]:
    return {
        "resolution": 8,
        "model_channels": 32,
        "latent_channels": 4,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "attn_mode": "full",
        "window_size": 2,
        "use_fp16": False,
        "representation_config": {"rank": 2, "dim": 4},
    }


def _production_config_TrellisSparseStructureFlowModel() -> dict[str, Any]:
    return {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 1024,
        "cond_channels": 1024,
        "num_blocks": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 1,
        "pe_mode": "ape",
        "qk_rms_norm": True,
        "use_fp16": True,
    }


def _tiny_config_TrellisSparseStructureFlowModel() -> dict[str, Any]:
    return {
        "resolution": 4,
        "in_channels": 2,
        "out_channels": 2,
        "model_channels": 16,
        "cond_channels": 12,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "patch_size": 2,
        "pe_mode": "ape",
        "qk_rms_norm": True,
        "use_fp16": False,
    }


def _tiny_config_TrellisSLatFlowModel() -> dict[str, Any]:
    return {
        "resolution": 8,
        "in_channels": 4,
        "out_channels": 4,
        "model_channels": 16,
        "cond_channels": 12,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "patch_size": 1,
        "num_io_res_blocks": 0,
        "io_block_channels": None,
        "pe_mode": "ape",
        "qk_rms_norm": True,
        "use_fp16": False,
    }


def _production_config_Trellis2ShapeDualGridDecoder() -> dict[str, Any]:
    return {
        "resolution": 256,
        "model_channels": [1024, 512, 256, 128, 64],
        "latent_channels": 32,
        "num_blocks": [4, 16, 8, 4, 0],
        "block_type": ["SparseConvNeXtBlock3d"] * 5,
        "up_block_type": ["SparseResBlockC2S3d"] * 4,
        "block_args": [{}, {}, {}, {}, {}],
        "voxel_margin": 0.5,
        "use_fp16": True,
    }


def _tiny_config_Trellis2ShapeDualGridDecoder() -> dict[str, Any]:
    return {
        "resolution": 16,
        "model_channels": [8, 8],
        "latent_channels": 4,
        "num_blocks": [1, 0],
        "block_type": ["SparseConvNeXtBlock3d"] * 2,
        "up_block_type": ["SparseResBlockC2S3d"],
        "block_args": [{}, {}],
        "voxel_margin": 0.5,
        "use_fp16": False,
    }


def _production_config_Trellis2PBRSparseDecoder() -> dict[str, Any]:
    return {
        "out_channels": 6,
        "model_channels": [1024, 512, 256, 128, 64],
        "latent_channels": 32,
        "num_blocks": [4, 16, 8, 4, 0],
        "block_type": ["SparseConvNeXtBlock3d"] * 5,
        "up_block_type": ["SparseResBlockC2S3d"] * 4,
        "block_args": [{}, {}, {}, {}, {}],
        "pred_subdiv": False,
        "use_fp16": True,
        "channel_layout": ["base_color", "metallic", "roughness", "alpha"],
    }


def _tiny_config_Trellis2PBRSparseDecoder() -> dict[str, Any]:
    return {
        "out_channels": 6,
        "model_channels": [8, 8],
        "latent_channels": 4,
        "num_blocks": [1, 0],
        "block_type": ["SparseConvNeXtBlock3d"] * 2,
        "up_block_type": ["SparseResBlockC2S3d"],
        "block_args": [{}, {}],
        "pred_subdiv": False,
        "use_fp16": False,
        "channel_layout": ["base_color", "metallic", "roughness", "alpha"],
    }


def _production_config_Trellis2SparseStructureFlowModel() -> dict[str, Any]:
    return {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 1536,
        "cond_channels": 1024,
        "num_blocks": 30,
        "num_heads": 12,
        "mlp_ratio": 5.3334,
        "pe_mode": "rope",
        "share_mod": True,
        "initialization": "scaled",
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
        "dtype": "bfloat16",
    }


def _tiny_config_Trellis2SparseStructureFlowModel() -> dict[str, Any]:
    return {
        "resolution": 2,
        "in_channels": 2,
        "out_channels": 2,
        "model_channels": 12,
        "cond_channels": 12,
        "num_blocks": 2,
        "num_heads": 3,
        "mlp_ratio": 2,
        "pe_mode": "rope",
        "share_mod": True,
        "initialization": "scaled",
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
        "dtype": "float32",
    }


def _production_config_Trellis2SLatFlowModel(*, texture=False) -> dict[str, Any]:
    return {
        "resolution": 32,
        "in_channels": 64 if texture else 32,
        "out_channels": 32,
        "model_channels": 1536,
        "cond_channels": 1024,
        "num_blocks": 30,
        "num_heads": 12,
        "mlp_ratio": 5.3334,
        "pe_mode": "rope",
        "share_mod": True,
        "initialization": "scaled",
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
        "dtype": "bfloat16",
    }


def _tiny_config_Trellis2SLatFlowModel(*, texture=False) -> dict[str, Any]:
    return {
        "resolution": 8,
        "in_channels": 8 if texture else 4,
        "out_channels": 4,
        "model_channels": 12,
        "cond_channels": 12,
        "num_blocks": 2,
        "num_heads": 3,
        "mlp_ratio": 2,
        "pe_mode": "rope",
        "share_mod": True,
        "initialization": "scaled",
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
        "dtype": "float32",
    }


def _production_config_TripoSGTransformerModel() -> dict[str, Any]:
    return {
        "in_channels": 64,
        "width": 2048,
        "num_layers": 21,
        "num_attention_heads": 16,
        "cross_attention_dim": 1024,
    }


def _tiny_config_TripoSGTransformerModel() -> dict[str, Any]:
    return {"in_channels": 4, "width": 16, "num_layers": 3, "num_attention_heads": 2, "cross_attention_dim": 12}


def _production_config_TripoSGVAEDecoder() -> dict[str, Any]:
    return {
        "in_channels": 3,
        "latent_channels": 64,
        "num_attention_heads": 8,
        "width_decoder": 1024,
        "num_layers_decoder": 16,
        "embed_frequency": 8,
        "embed_include_pi": False,
    }


def _tiny_config_TripoSGVAEDecoder() -> dict[str, Any]:
    return {
        "in_channels": 3,
        "latent_channels": 4,
        "num_attention_heads": 2,
        "width_decoder": 16,
        "num_layers_decoder": 2,
        "embed_frequency": 2,
        "embed_include_pi": False,
    }


_TINY_CONFIGS = {
    MoGeModel: _tiny_config_MoGeModel,
    MoGe2Model: _tiny_config_MoGe2Model,
    TripoSplatTransformerModel: _tiny_config_TripoSplatTransformerModel,
    TripoSplatGaussianDecoder: _tiny_config_TripoSplatGaussianDecoder,
    TripoSplatImageVAE: _tiny_config_TripoSplatImageVAE,
    TripoSRTransformerModel: _tiny_config_TripoSRTransformerModel,
    TripoSRTriplaneDecoder: _tiny_config_TripoSRTriplaneDecoder,
    Hunyuan3DOmniControlEncoder: _tiny_config_Hunyuan3DOmniControlEncoder,
    Hunyuan3DDiTModel: _tiny_config_Hunyuan3DDiTModel,
    Hunyuan3DPaintUNet: _tiny_config_Hunyuan3DPaintUNet,
    Hunyuan3DPaintRRDBNet: _tiny_config_Hunyuan3DPaintRRDBNet,
    Hunyuan3DShapeVAE: _tiny_config_Hunyuan3DShapeVAE,
    Pixal3DNAFUpsampler: _tiny_config_Pixal3DNAFUpsampler,
    Pixal3DSparseStructureFlowModel: _tiny_config_Pixal3DSparseStructureFlowModel,
    Pixal3DSLatFlowModel: _tiny_config_Pixal3DSLatFlowModel,
    Sam3DConditionEmbedder: _tiny_config_Sam3DConditionEmbedder,
    Sam3DSparseStructureFlowModel: _tiny_config_Sam3DSparseStructureFlowModel,
    Step1X3DTransformer1DModel: _tiny_config_Step1X3DTransformer1DModel,
    Step1X3DMultiViewAdapter: _tiny_config_Step1X3DMultiViewAdapter,
    Step1X3DShapeDecoder: _tiny_config_Step1X3DShapeDecoder,
    TrellisSparseStructureDecoder: _tiny_config_TrellisSparseStructureDecoder,
    TrellisSLatGaussianDecoder: _tiny_config_TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder: _tiny_config_TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder: _tiny_config_TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureFlowModel: _tiny_config_TrellisSparseStructureFlowModel,
    TrellisSLatFlowModel: _tiny_config_TrellisSLatFlowModel,
    Trellis2ShapeDualGridDecoder: _tiny_config_Trellis2ShapeDualGridDecoder,
    Trellis2PBRSparseDecoder: _tiny_config_Trellis2PBRSparseDecoder,
    Trellis2SparseStructureFlowModel: _tiny_config_Trellis2SparseStructureFlowModel,
    Trellis2SLatFlowModel: _tiny_config_Trellis2SLatFlowModel,
    TripoSGTransformerModel: _tiny_config_TripoSGTransformerModel,
    TripoSGVAEDecoder: _tiny_config_TripoSGVAEDecoder,
}


_PRODUCTION_CONFIGS = {
    TripoSplatTransformerModel: _production_config_TripoSplatTransformerModel,
    TripoSplatGaussianDecoder: _production_config_TripoSplatGaussianDecoder,
    TripoSplatImageVAE: _production_config_TripoSplatImageVAE,
    TripoSRTransformerModel: _production_config_TripoSRTransformerModel,
    TripoSRTriplaneDecoder: _production_config_TripoSRTriplaneDecoder,
    Hunyuan3DOmniControlEncoder: _production_config_Hunyuan3DOmniControlEncoder,
    Hunyuan3DDiTModel: _production_config_Hunyuan3DDiTModel,
    Hunyuan3DPaintUNet: _production_config_Hunyuan3DPaintUNet,
    Hunyuan3DPaintRRDBNet: _production_config_Hunyuan3DPaintRRDBNet,
    Hunyuan3DShapeVAE: _production_config_Hunyuan3DShapeVAE,
    Pixal3DNAFUpsampler: _production_config_Pixal3DNAFUpsampler,
    Pixal3DSparseStructureFlowModel: _production_config_Pixal3DSparseStructureFlowModel,
    Pixal3DSLatFlowModel: _production_config_Pixal3DSLatFlowModel,
    TrellisSLatRadianceFieldDecoder: _production_config_TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureFlowModel: _production_config_TrellisSparseStructureFlowModel,
    Trellis2ShapeDualGridDecoder: _production_config_Trellis2ShapeDualGridDecoder,
    Trellis2PBRSparseDecoder: _production_config_Trellis2PBRSparseDecoder,
    Trellis2SparseStructureFlowModel: _production_config_Trellis2SparseStructureFlowModel,
    Trellis2SLatFlowModel: _production_config_Trellis2SLatFlowModel,
    TripoSGTransformerModel: _production_config_TripoSGTransformerModel,
    TripoSGVAEDecoder: _production_config_TripoSGVAEDecoder,
}


def _parent(model_type: type) -> type:
    return model_type.__mro__[1]


def tiny_config(model_type: type, **kwargs: Any) -> dict[str, Any]:
    """The smallest config each model's tests build it at, looked up along the class hierarchy."""

    for klass in model_type.__mro__:
        if klass in _TINY_CONFIGS:
            return _TINY_CONFIGS[klass](**kwargs)
    raise KeyError(f"no tiny config for {model_type.__name__}")


def small_trellis_slat_flow_config() -> dict[str, Any]:
    """TRELLIS's SLAT flow layout at toy width: exercises the IO stages, pooling, and skip connections on CPU."""

    return {
        **tiny_config(TrellisSLatFlowModel),
        "resolution": 8,
        "model_channels": 16,
        "patch_size": 2,
        "num_io_res_blocks": 2,
        "io_block_channels": [8],
    }


def production_config(model_type: type, **kwargs: Any) -> dict[str, Any]:
    """The published checkpoint's config, for tests that build a model at full size on the meta device."""

    for klass in model_type.__mro__:
        if klass in _PRODUCTION_CONFIGS:
            return _PRODUCTION_CONFIGS[klass](**kwargs)
        if "production_config" in vars(klass):
            return klass.production_config(**kwargs)
    raise KeyError(f"no production config for {model_type.__name__}")
