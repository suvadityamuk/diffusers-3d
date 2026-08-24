from __future__ import annotations

import torch

from diffusers_3d import Hunyuan3DDinov2Conditioner


def test_tiny_dinov2_tokens_unconditional_and_save_load(tmp_path):
    torch.manual_seed(0)
    conditioner = Hunyuan3DDinov2Conditioner(**Hunyuan3DDinov2Conditioner.tiny_config()).eval()
    images = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    with torch.no_grad():
        output = conditioner(images).embeddings
    assert output.shape == (2, 5, 32)
    assert not any(parameter.requires_grad for parameter in conditioner.parameters())
    torch.testing.assert_close(
        conditioner.unconditional_embedding(2),
        torch.zeros_like(output),
    )

    conditioner.save_pretrained(tmp_path)
    loaded = Hunyuan3DDinov2Conditioner.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        reloaded = loaded(images).embeddings
    torch.testing.assert_close(output, reloaded)


def test_released_normalization_constants_and_cls_drop():
    config = Hunyuan3DDinov2Conditioner.tiny_config()
    config["use_cls_token"] = False
    conditioner = Hunyuan3DDinov2Conditioner(**config)
    torch.testing.assert_close(
        conditioner.image_mean.flatten(),
        torch.tensor([0.485, 0.456, 0.406]),
    )
    torch.testing.assert_close(
        conditioner.image_std.flatten(),
        torch.tensor([0.229, 0.224, 0.225]),
    )
    assert conditioner(torch.zeros(1, 3, 8, 8)).embeddings.shape == (1, 4, 32)
    assert conditioner.unconditional_embedding(1).shape == (1, 4, 32)
