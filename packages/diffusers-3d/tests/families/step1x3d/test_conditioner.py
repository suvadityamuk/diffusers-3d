from __future__ import annotations

import torch
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def test_encode_conditioning_drops_registers_and_returns_zero_null_condition(tiny_step1x3d_pipeline):
    pipeline = tiny_step1x3d_pipeline
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    pixel_values = (images - torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)) / torch.tensor(
        IMAGENET_DEFAULT_STD
    ).view(1, 3, 1, 1)
    with torch.no_grad():
        conditional, negative = pipeline.encode_conditioning(images)
        hidden_states = pipeline.image_encoder(pixel_values).last_hidden_state
        resized, _ = pipeline.encode_conditioning(torch.rand(1, 3, 6, 10))
    registers = pipeline.image_encoder.config.num_register_tokens

    assert conditional.shape == (2, 5, 12)
    torch.testing.assert_close(conditional[:, :1], hidden_states[:, :1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(conditional[:, 1:], hidden_states[:, 1 + registers :], atol=0.0, rtol=0.0)
    assert torch.equal(negative, torch.zeros_like(conditional))
    assert resized.shape == (1, 5, 12)
