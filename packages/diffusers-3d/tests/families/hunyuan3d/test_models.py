from __future__ import annotations

import numpy as np
import torch
from _configs import production_config, tiny_config
from transformers import Dinov2Config, Dinov2Model

from diffusers_3d import (
    Hunyuan3DDiTModel,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DShapeVAE,
)


def test_transformer_forward_backward_checkpointing_and_save_load(tmp_path):
    model = Hunyuan3DDiTModel(**tiny_config(Hunyuan3DDiTModel))
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(2, 6, 8, generator=generator)
    timestep = torch.tensor([0.1, 0.9])
    context = torch.randn(2, 5, 24, generator=generator)

    model.enable_gradient_checkpointing()
    output = model(latents, timestep, context).sample
    assert output.shape == latents.shape
    output.square().mean().backward()
    assert model.transformer_blocks[-1].ff.experts[0].net[0].proj.weight.grad is not None
    assert model.transformer_blocks[-1].skip_linear.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = Hunyuan3DDiTModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(latents, timestep, context).sample, model(latents, timestep, context).sample)


def test_moe_routes_each_token_to_its_top_k_experts():
    moe = Hunyuan3DDiTModel(**tiny_config(Hunyuan3DDiTModel)).transformer_blocks[-1].ff
    tokens = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        weights, selected = moe.gate(tokens[0]).softmax(-1).topk(moe.top_k, dim=-1)
        expected = torch.stack(
            [
                sum(weight * moe.experts[index](token) for weight, index in zip(token_weights, token_experts))
                for token, token_weights, token_experts in zip(tokens[0], weights, selected)
            ]
        )
        torch.testing.assert_close(moe(tokens)[0], expected + moe.shared_experts(tokens[0]))


def test_vae_decodes_tokens_and_queries_occupancy():
    vae = Hunyuan3DShapeVAE(**tiny_config(Hunyuan3DShapeVAE)).eval()
    latents = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(2))
    points = torch.rand(2, 11, 3, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        tokens = vae(latents).sample
        logits = vae.query_occupancy(tokens, points)
        single = vae.query_occupancy(tokens[:1], points[:1, :4])
    assert tokens.shape == (2, 6, 32)
    assert logits.shape == (2, 11)
    torch.testing.assert_close(single, logits[:1, :4])


def test_production_layouts_match_the_released_checkpoint():
    # Key counts read from the published model.fp16.ckpt (the VAE's fused q/k/v rows are split into three keys).
    with torch.device("meta"):
        transformer = Hunyuan3DDiTModel(**production_config(Hunyuan3DDiTModel))
        vae = Hunyuan3DShapeVAE(**production_config(Hunyuan3DShapeVAE))
        image_encoder = Dinov2Model(
            Dinov2Config(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, image_size=518)
        )
    assert len(transformer.state_dict()) == 752
    assert len(vae.state_dict()) == 266 + 2 * 16 + 1
    assert len(image_encoder.state_dict()) == 439
    assert transformer.time_embedding.linear_1.weight.shape == (8192, 2048)
    assert transformer.transformer_blocks[20].ff.gate.weight.shape == (8, 2048)
    assert transformer.transformer_blocks[0].attn2.to_k.weight.shape == (2048, 1024)
    assert vae.query_proj.weight.shape == (1024, 51)


def test_scheduler_integrates_from_noise_at_zero_to_data_at_one():
    scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([0.0, 500.0, 1000.0]))
    torch.testing.assert_close(scheduler.sigmas, torch.tensor([0.0, 0.5, 1.0, 1.0]))
    sample, velocity = torch.tensor([1.0]), torch.tensor([2.0])
    for timestep in scheduler.timesteps:
        sample = scheduler.step(velocity, timestep, sample).prev_sample
    torch.testing.assert_close(sample, torch.tensor([3.0]))
    scheduler.set_timesteps(50)
    assert torch.equal(scheduler.sigmas[:-1], torch.from_numpy(np.linspace(0, 1, 50)).float())
