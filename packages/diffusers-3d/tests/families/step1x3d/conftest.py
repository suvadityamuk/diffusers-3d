from __future__ import annotations

import pytest
import torch
from _configs import tiny_config
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from diffusers_3d import (
    Step1X3DFlowMatchEulerScheduler,
    Step1X3DImageTo3DPipeline,
    Step1X3DShapeDecoder,
    Step1X3DTransformer1DModel,
)


@pytest.fixture
def tiny_step1x3d_components():
    torch.manual_seed(0)
    image_encoder = Dinov2WithRegistersModel(
        Dinov2WithRegistersConfig(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            mlp_ratio=2,
            image_size=8,
            patch_size=4,
            num_register_tokens=2,
        )
    )
    torch.manual_seed(1)
    transformer = Step1X3DTransformer1DModel(**tiny_config(Step1X3DTransformer1DModel))
    torch.manual_seed(2)
    vae = Step1X3DShapeDecoder(**tiny_config(Step1X3DShapeDecoder))
    return {
        "image_encoder": image_encoder.eval(),
        "transformer": transformer.eval(),
        "vae": vae.eval(),
        "scheduler": Step1X3DFlowMatchEulerScheduler(),
    }


@pytest.fixture
def tiny_step1x3d_pipeline(tiny_step1x3d_components):
    pipeline = Step1X3DImageTo3DPipeline(**tiny_step1x3d_components)
    pipeline.disable_background_removal()  # upstream's default rembg would download u2net
    return pipeline


@pytest.fixture
def rgba_condition():
    """A 4x20x16 RGBA image whose opaque foreground is an off-centre 11x9 rectangle."""

    from diffusers_3d import ImageCondition

    image = torch.linspace(0.0, 1.0, 4 * 20 * 16).reshape(4, 20, 16)
    image[3] = 0
    image[3, 4:15, 3:12] = 1
    return ImageCondition(image)


@pytest.fixture
def sphere_field():
    """Build an occupancy-style field (positive inside a sphere) on the released grid."""

    from diffusers_3d.backends.occupancy import dense_grid_points

    def make(resolution: int, *, radius: float = 0.6, bounds: float = 1.05) -> torch.Tensor:
        points = dense_grid_points((-bounds,) * 3 + (bounds,) * 3, resolution)
        return (radius - points.norm(dim=-1)).view(1, *([resolution + 1] * 3))

    return make


def make_tiny_step1x3d_texture_components(tokenizer) -> dict:
    """SDXL-shaped tiny components: three UNet levels, a two-level VAE, and a matching T2I-Adapter."""

    from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    from diffusers_3d import Step1X3DMultiViewAdapter

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(8, 8, 16),
        layers_per_block=1,
        sample_size=8,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=2,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=6 * 4 + 8,
        cross_attention_dim=16,
        norm_num_groups=4,
    )
    text = {
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "max_position_embeddings": 8,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    return {
        "vae": AutoencoderKL(
            block_out_channels=(4, 4),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1,
            norm_num_groups=2,
        ),
        "text_encoder": CLIPTextModel(CLIPTextConfig(**text)),
        "text_encoder_2": CLIPTextModelWithProjection(CLIPTextConfig(**text, projection_dim=8)),
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer,
        "unet": unet,
        "adapter": T2IAdapter(
            in_channels=6,
            channels=(8, 8, 16, 16),
            num_res_blocks=1,
            downscale_factor=4,
            adapter_type="full_adapter_xl",
        ),
        "mv_adapter": Step1X3DMultiViewAdapter(hidden_sizes=Step1X3DMultiViewAdapter.hidden_sizes_for(unet)),
        "scheduler": DDPMScheduler(
            beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, clip_sample=False
        ),
    }


@pytest.fixture
def tiny_step1x3d_texture_pipeline(tiny_clip_tokenizer):
    from diffusers_3d import Step1X3DMeshTexturePipeline

    pipeline = Step1X3DMeshTexturePipeline(**make_tiny_step1x3d_texture_components(tiny_clip_tokenizer))
    pipeline.disable_background_removal()  # upstream's default BiRefNet would download its weights
    return pipeline


@pytest.fixture
def icosphere_mesh():
    trimesh = pytest.importorskip("trimesh")
    from diffusers_3d import MeshAsset

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
    return MeshAsset(vertices=torch.tensor(sphere.vertices, dtype=torch.float32), faces=torch.tensor(sphere.faces))
