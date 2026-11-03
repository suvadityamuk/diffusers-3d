from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def test_encode_conditioning_layer_norms_prenorm_tokens_with_registers(tiny_trellis_pipeline):
    pipeline = tiny_trellis_pipeline
    encoder = pipeline.image_encoder
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    with torch.no_grad():
        conditional, unconditional = pipeline.encode_conditioning(images)
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        expected = encoder.encoder(encoder.embeddings((images - mean) / std)).last_hidden_state
        resized, _ = pipeline.encode_conditioning(torch.rand(1, 3, 6, 10))
    # Class, two register, and four patch tokens.
    assert conditional.shape == (2, 7, 12)
    torch.testing.assert_close(conditional, F.layer_norm(expected, (12,)), atol=0.0, rtol=0.0)
    assert torch.equal(unconditional, torch.zeros_like(conditional))
    assert resized.shape == (1, 7, 12)
