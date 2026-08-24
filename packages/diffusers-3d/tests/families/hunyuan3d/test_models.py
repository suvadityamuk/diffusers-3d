from __future__ import annotations

import torch

from diffusers_3d import Hunyuan3DShapeDiTModel


def _inputs(model: Hunyuan3DShapeDiTModel):
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(2, model.config.input_size, model.config.in_channels, generator=generator)
    timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(2, model.config.text_len, model.config.context_dim, generator=generator)
    return hidden_states, timestep, context


def test_tiny_forward_config_and_save_load(tmp_path):
    model = Hunyuan3DShapeDiTModel(**Hunyuan3DShapeDiTModel.tiny_config()).eval()
    hidden_states, timestep, context = _inputs(model)
    with torch.no_grad():
        output = model(hidden_states, timestep, context).sample
        tuple_output = model(hidden_states, timestep, context, return_dict=False)[0]
    assert output.shape == hidden_states.shape
    torch.testing.assert_close(output, tuple_output)
    assert model.blocks[-1].use_moe
    assert not model.blocks[0].use_moe

    model.save_pretrained(tmp_path)
    loaded = Hunyuan3DShapeDiTModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        reloaded = loaded(hidden_states, timestep, context).sample
    torch.testing.assert_close(output, reloaded)


def test_backward_and_gradient_checkpointing():
    model = Hunyuan3DShapeDiTModel(**Hunyuan3DShapeDiTModel.tiny_config())
    model.enable_gradient_checkpointing()
    hidden_states, timestep, context = _inputs(model)
    output = model(hidden_states, timestep, context).sample
    output.square().mean().backward()
    assert model.gradient_checkpointing
    assert model.x_embedder.weight.grad is not None
    assert model.blocks[-1].moe.gate.weight.grad is not None


def test_state_dict_names_match_released_denoiser_layout():
    model = Hunyuan3DShapeDiTModel(**Hunyuan3DShapeDiTModel.tiny_config())
    keys = set(model.state_dict())
    assert "x_embedder.weight" in keys
    assert "t_embedder.mlp.0.weight" in keys
    assert "blocks.0.attn1.to_q.weight" in keys
    assert "blocks.0.attn2.to_k.weight" in keys
    assert "blocks.2.moe.experts.0.net.0.proj.weight" in keys
    assert "blocks.2.moe.shared_experts.net.2.weight" in keys
    assert "final_layer.linear.weight" in keys
