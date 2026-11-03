from __future__ import annotations

import copy

import torch

from diffusers_3d import Step1X3DShapeDecoder, Step1X3DTransformer1DModel


def test_transformer_forward_metadata_and_save_load(tmp_path, tiny_step1x3d_components):
    transformer = tiny_step1x3d_components["transformer"]
    config = transformer.config
    torch.manual_seed(3)
    latents = torch.randn(2, 5, config.in_channels)
    context = torch.randn(2, 7, config.cross_attention_dim)
    with torch.no_grad():
        sample = transformer(latents, torch.tensor([1000.0, 500.0]), context).sample
        scalar = transformer(latents, torch.tensor(500.0), context).sample
        batched = transformer(latents, torch.tensor([500.0, 500.0]), context).sample
        shifted = transformer(latents, torch.tensor([500.0, 500.0]), context + 1).sample

    assert sample.shape == latents.shape
    torch.testing.assert_close(scalar, batched, atol=0.0, rtol=0.0)
    assert not torch.allclose(shifted, batched)

    transformer.save_pretrained(tmp_path)
    loaded = Step1X3DTransformer1DModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(latents, torch.tensor([1000.0, 500.0]), context).sample, sample)


def test_shape_decoder_fourier_query_and_save_load(tmp_path, tiny_step1x3d_components):
    vae = tiny_step1x3d_components["vae"]
    config = vae.config
    torch.manual_seed(4)
    latents = torch.randn(2, config.num_latents, config.latent_channels)
    points = torch.rand(2, 6, 3) * 2 - 1
    with torch.no_grad():
        tokens = vae.decode(latents)
        logits = vae.query(points, tokens)
        frequencies = 2.0 ** torch.arange(config.num_freqs, dtype=torch.float32)
        embedding = (points[..., None] * frequencies).reshape(2, 6, -1)
        query = vae.query_proj(torch.cat([points, embedding.sin(), embedding.cos()], dim=-1))
        query = query + vae.attn(vae.norm1(query), vae.norm_context(tokens))
        query = query + vae.ff(vae.norm2(query))
        expected = vae.proj_out(vae.norm_out(query))
        forward = vae(latents, points).sample

    assert tokens.shape == (2, config.num_latents, config.width)
    torch.testing.assert_close(logits, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(forward, logits, atol=0.0, rtol=0.0)

    vae.save_pretrained(tmp_path)
    loaded = Step1X3DShapeDecoder.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded.query(points, loaded.decode(latents)), logits)


def test_shape_decoder_embeds_queries_in_their_own_dtype(tiny_step1x3d_components):
    vae = tiny_step1x3d_components["vae"]
    config = vae.config
    torch.manual_seed(5)
    latents = torch.randn(1, config.num_latents, config.latent_channels).to(torch.bfloat16)
    points = torch.tensor([[[0.995, 0.2, -0.3], [0.9975, 0.2, -0.3]]]).to(torch.bfloat16)
    with torch.no_grad():
        half = copy.deepcopy(vae).to(torch.bfloat16)
        tokens = half.decode(latents)
        logits = half.query(points, tokens)
        frequencies = 2.0 ** torch.arange(config.num_freqs, dtype=torch.bfloat16)
        embedding = (points[..., None] * frequencies).flatten(-2)
        embedding = torch.cat([points, embedding.sin(), embedding.cos()], dim=-1)
        expected = half.query_proj(embedding)
        expected = expected + half.attn(half.norm1(expected), half.norm_context(tokens))
        expected = half.proj_out(half.norm_out(expected + half.ff(half.norm2(expected))))

    # Upstream casts query points to the model dtype; 2.5e-3 apart near 1.0 they merge in bfloat16.
    assert logits.dtype is torch.bfloat16
    assert torch.equal(logits[0, 0], logits[0, 1])
    torch.testing.assert_close(logits, expected, atol=0.0, rtol=0.0)
