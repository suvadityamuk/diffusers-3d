from __future__ import annotations

import torch
import torch.nn.functional as F


def test_image_encoder_tokens_use_affine_free_final_norm(tiny_trellis2_pipeline):
    pipeline = tiny_trellis2_pipeline
    model = pipeline.image_encoder.eval()
    images = torch.linspace(0.0, 1.0, 2 * 3 * 6 * 10).reshape(2, 3, 6, 10)
    with torch.no_grad():
        conditional, negative = pipeline.encode_conditioning(images)
        resized = F.interpolate(images, size=(8, 8), mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        normalized = (resized - mean) / torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        hidden_states = model.embeddings(normalized, bool_masked_pos=None)
        phases = model.rope_embeddings(normalized)
        for layer in model.model.layer:
            hidden_states = layer(hidden_states, position_embeddings=phases)
        expected = F.layer_norm(hidden_states, hidden_states.shape[-1:])

    assert conditional.shape == (2, 7, 12)
    assert torch.equal(conditional, expected)
    assert torch.equal(negative, torch.zeros_like(expected))
