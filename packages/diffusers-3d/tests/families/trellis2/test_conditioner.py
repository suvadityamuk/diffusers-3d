from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers_3d import Trellis2Dinov3Conditioner


def test_dinov3_conditioner_manual_token_normalization_tiny_no_download_and_save_load(tmp_path):
    torch.manual_seed(0)
    conditioner = Trellis2Dinov3Conditioner(**Trellis2Dinov3Conditioner.tiny_config()).eval()
    images = torch.linspace(0.0, 1.0, 2 * 3 * 6 * 10).reshape(2, 3, 6, 10)
    with torch.no_grad():
        actual = conditioner(images).embeddings
        resized = F.interpolate(
            images,
            size=(conditioner.image_size, conditioner.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        normalized = (resized - conditioner.image_mean) / conditioner.image_std
        hidden_states = conditioner.model.embeddings(normalized, bool_masked_pos=None)
        phases = conditioner.model.rope_embeddings(normalized)
        for layer in conditioner.model.model.layer:
            hidden_states = layer(hidden_states, position_embeddings=phases)
        expected = F.layer_norm(hidden_states, hidden_states.shape[-1:])

    assert actual.shape == (2, conditioner.num_tokens, 12)
    assert conditioner.num_tokens == 7
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.equal(conditioner.unconditional_embedding(2), torch.zeros_like(actual))

    conditioner.save_pretrained(tmp_path)
    loaded = Trellis2Dinov3Conditioner.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(images).embeddings, actual)
