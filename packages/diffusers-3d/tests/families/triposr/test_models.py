from __future__ import annotations

import torch
from _configs import production_config, tiny_config
from accelerate import init_empty_weights

from diffusers_3d.families.triposr.models import TripoSRTransformerModel, TripoSRTriplaneDecoder


def test_transformer_forward_backward_checkpointing_and_save_load(tmp_path):
    model = TripoSRTransformerModel(**tiny_config(TripoSRTransformerModel))
    context = torch.randn(2, 5, 12, generator=torch.Generator().manual_seed(0))

    output = model(context).sample
    assert output.shape == (2, 3, 6, 8, 8)
    torch.testing.assert_close(output, model(context, return_dict=False)[0])
    # Each batch item only sees its own image tokens.
    torch.testing.assert_close(model(context[1:]).sample[0], output[1])
    model.enable_gradient_checkpointing()
    checkpointed = model(context).sample
    checkpointed.square().mean().backward()
    torch.testing.assert_close(checkpointed, output)
    assert model.embeddings.grad is not None
    assert model.backbone.transformer_blocks[0].attn2.to_k.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TripoSRTransformerModel.from_pretrained(tmp_path, local_files_only=True)
    with torch.no_grad():
        torch.testing.assert_close(loaded(context).sample, output.detach())


def test_decoder_chunking_and_batching_do_not_change_the_field(tmp_path):
    model = TripoSRTriplaneDecoder(**tiny_config(TripoSRTriplaneDecoder)).eval()
    generator = torch.Generator().manual_seed(1)
    triplanes = torch.randn(2, 3, 6, 8, 8, generator=generator)
    points = torch.rand(2, 37, 3, generator=generator) * 1.74 - 0.87
    with torch.no_grad():
        whole = model(triplanes, points)
        chunked = model(triplanes, points, chunk_size=10)
        torch.testing.assert_close(chunked.density, whole.density)
        torch.testing.assert_close(chunked.color, whole.color)
        torch.testing.assert_close(model(triplanes[1:], points[1:]).density[0], whole.density[1])
        assert whole.density.shape == (2, 37, 1) and whole.color.shape == (2, 37, 3)
        assert (whole.density > 0).all() and ((whole.color >= 0) & (whole.color <= 1)).all()
        model.save_pretrained(tmp_path)
        loaded = TripoSRTriplaneDecoder.from_pretrained(tmp_path, local_files_only=True)
        torch.testing.assert_close(loaded(triplanes, points).density, whole.density)


def test_production_layouts_match_the_released_checkpoint():
    # stabilityai/TripoSR@5b521936 model.ckpt: 326 backbone + 1 tokenizer + 2 post_processor tensors, 20 decoder.
    with init_empty_weights():
        transformer = TripoSRTransformerModel(**production_config(TripoSRTransformerModel))
        decoder = TripoSRTriplaneDecoder(**production_config(TripoSRTriplaneDecoder))
    transformer_state = transformer.state_dict()
    decoder_state = decoder.state_dict()
    assert len(transformer_state) == 329
    assert len(decoder_state) == 20
    assert transformer_state["embeddings"].shape == (3, 1024, 32, 32)
    assert transformer_state["backbone.transformer_blocks.15.attn2.to_k.weight"].shape == (1024, 768)
    assert transformer_state["backbone.transformer_blocks.0.ff.net.0.proj.weight"].shape == (8192, 1024)
    assert transformer_state["upsample.weight"].shape == (1024, 40, 2, 2)
    assert decoder_state["layers.0.weight"].shape == (64, 120)
    assert decoder_state["layers.18.weight"].shape == (4, 64)
