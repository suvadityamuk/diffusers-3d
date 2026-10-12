from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers_3d import TrellisDinov2Conditioner


def test_trellis_dinov2_normalization_register_tokens_and_save_load(tmp_path):
    torch.manual_seed(0)
    conditioner = TrellisDinov2Conditioner(**TrellisDinov2Conditioner.tiny_config()).eval()
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    with torch.no_grad():
        actual = conditioner(images).embeddings
        normalized = (images - conditioner.image_mean) / conditioner.image_std
        expected = conditioner.model.embeddings(normalized)
        expected = torch.cat(
            [
                expected[:, :1],
                conditioner.register_tokens.expand(images.shape[0], -1, -1),
                expected[:, 1:],
            ],
            dim=1,
        )
        expected = conditioner.model.encoder(expected).last_hidden_state
        expected = F.layer_norm(expected, expected.shape[-1:])
    assert actual.shape == (2, 7, 12)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(conditioner.unconditional_embedding(2), torch.zeros_like(actual))
    torch.testing.assert_close(conditioner.image_mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    torch.testing.assert_close(conditioner.image_std.flatten(), torch.tensor([0.229, 0.224, 0.225]))

    conditioner.save_pretrained(tmp_path)
    loaded = TrellisDinov2Conditioner.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(images).embeddings, actual)
