from __future__ import annotations

import torch
from accelerate import init_empty_weights

from diffusers_3d import TripoSGTransformerModel, TripoSGVAEDecoder


def test_transformer_forward_backward_checkpointing_and_save_load(tmp_path):
    model = TripoSGTransformerModel(**TripoSGTransformerModel.tiny_config())
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(2, 5, 4, generator=generator)
    context = torch.randn(2, 3, 12, generator=generator)
    timestep = torch.tensor([900.0, 100.0])

    output = model(latents, timestep, context).sample
    assert output.shape == latents.shape
    torch.testing.assert_close(output, model(latents, timestep, context, return_dict=False)[0])
    model.enable_gradient_checkpointing()
    checkpointed = model(latents, timestep, context).sample
    checkpointed.square().mean().backward()
    torch.testing.assert_close(checkpointed, output)
    assert model.blocks[-1].skip_linear.weight.grad is not None
    assert model.blocks[0].attn2.norm_q.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TripoSGTransformerModel.from_pretrained(tmp_path, local_files_only=True)
    with torch.no_grad():
        torch.testing.assert_close(loaded(latents, timestep, context).sample, output.detach())


def test_vae_decoder_chunking_does_not_change_the_field(tmp_path):
    model = TripoSGVAEDecoder(**TripoSGVAEDecoder.tiny_config()).eval()
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 6, 4, generator=generator)
    points = torch.rand(1, 37, 3, generator=generator) * 2 - 1
    with torch.no_grad():
        whole = model(latents, points).sample
        torch.testing.assert_close(model(latents, points, num_chunks=10).sample, whole)
        assert whole.shape == (1, 37, 1)
        model.save_pretrained(tmp_path)
        loaded = TripoSGVAEDecoder.from_pretrained(tmp_path, local_files_only=True)
        torch.testing.assert_close(loaded(latents, points).sample, whole)
    assert "frequencies" not in model.state_dict()


def test_production_layouts_match_the_published_checkpoint_headers():
    # VAST-AI/TripoSG@2c1c516d safetensors headers: 554 transformer tensors; the VAE's 356 minus 125 encoder/quant.
    with init_empty_weights():
        transformer = TripoSGTransformerModel(**TripoSGTransformerModel.production_config())
        vae = TripoSGVAEDecoder(**TripoSGVAEDecoder.production_config())
    transformer_state = transformer.state_dict()
    vae_state = vae.state_dict()
    assert len(transformer_state) == 554
    assert len(vae_state) == 231
    assert sum(1 for key in transformer_state if key.endswith("skip_linear.weight")) == 10
    assert transformer_state["blocks.0.attn2.to_k.weight"].shape == (2048, 1024)
    assert transformer_state["blocks.0.attn1.norm_q.weight"].shape == (128,)
    assert transformer_state["time_proj.linear_1.weight"].shape == (8192, 2048)
    assert vae_state["proj_query.weight"].shape == (1024, 51)
    assert vae_state["blocks.16.attn2.norm_cross.weight"].shape == (1024,)
    assert vae_state["post_quant.weight"].shape == (1024, 64)
