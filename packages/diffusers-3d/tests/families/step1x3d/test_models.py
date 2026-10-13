from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from diffusers_3d import Object3DKind, Step1X3DShapeDecoder, Step1X3DTransformer1DModel
from diffusers_3d.families.step1x3d.vae import step1x3d_grid_points


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
    metadata = Step1X3DTransformer1DModel.object3d_metadata()
    assert (metadata.family_id, metadata.component_role) == ("step1x3d", "transformer")
    assert metadata.supported_object_kinds == (Object3DKind.MESH,)

    transformer.save_pretrained(tmp_path)
    loaded = Step1X3DTransformer1DModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(latents, torch.tensor([1000.0, 500.0]), context).sample, sample)


def test_grid_points_match_numpy_linspace_bitwise():
    for bounds, resolution in ((1.05, 384), (1.05, 16), (1.01, 7)):
        axis = np.linspace(-bounds, bounds, resolution + 1, dtype=np.float32)
        expected = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        actual = step1x3d_grid_points(bounds, resolution).numpy()
        assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match="positive"):
        step1x3d_grid_points(1.0, 0)


def test_shape_decoder_fourier_query_dense_grid_and_save_load(tmp_path, tiny_step1x3d_components):
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
        grid = vae.decode_grid(latents, bounds=1.05, resolution=4, chunk_size=7)
        unchunked = vae.decode_grid(latents, bounds=1.05, resolution=4, chunk_size=10**6)
        dense = vae(latents, step1x3d_grid_points(1.05, 4).expand(2, -1, -1)).sample

    assert tokens.shape == (2, config.num_latents, config.width)
    torch.testing.assert_close(logits, expected, atol=0.0, rtol=0.0)
    assert grid.shape == (2, 5, 5, 5) and grid.dtype == torch.float32
    torch.testing.assert_close(grid, unchunked)
    torch.testing.assert_close(grid, dense[..., 0].view(2, 5, 5, 5))

    vae.save_pretrained(tmp_path)
    loaded = Step1X3DShapeDecoder.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded.query(points, loaded.decode(latents)), logits)


def test_shape_decoder_embeds_half_precision_queries_in_float32(tiny_step1x3d_components):
    vae = tiny_step1x3d_components["vae"]
    config = vae.config
    torch.manual_seed(5)
    latents = torch.randn(1, config.num_latents, config.latent_channels)
    # 2.5e-3 apart near 1.0: one bfloat16 step there is 7.8e-3, so casting the points first would merge them.
    points = torch.tensor([[[0.995, 0.2, -0.3], [0.9975, 0.2, -0.3]]])
    with torch.no_grad():
        reference = vae.query(points, vae.decode(latents))
        half = copy.deepcopy(vae).to(torch.bfloat16)
        logits = half.query(points, half.decode(latents.to(torch.bfloat16)))
        grid = half.decode_grid(latents.to(torch.bfloat16), resolution=4)

    assert logits.dtype is torch.bfloat16
    assert logits[0, 0] != logits[0, 1]
    torch.testing.assert_close(logits.float(), reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(grid, vae.decode_grid(latents, resolution=4), atol=5e-2, rtol=5e-2)
