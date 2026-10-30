from __future__ import annotations

import torch
from _configs import tiny_config

from diffusers_3d import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet
from diffusers_3d.families.hunyuan3d.paint_models import Hunyuan3DPaintTransformerBlock, position_voxel_indices


def _unet() -> Hunyuan3DPaintUNet:
    torch.manual_seed(0)
    model = Hunyuan3DPaintUNet(**tiny_config(Hunyuan3DPaintUNet)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    return model


def _inputs(batch: int = 2, views: int = 3):
    generator = torch.Generator().manual_seed(0)
    position_maps = torch.rand(views, 3, 32, 32, generator=generator)
    position_maps[:, :, :8] = 1.0
    return {
        "sample": torch.randn(batch, 2, views, 4, 8, 8, generator=generator),
        "timestep": 300,
        "condition_latents": torch.randn(views, 8, 8, 8, generator=generator),
        "position_maps": position_maps,
        "image_embeds": torch.randn(batch, 3, 10, generator=generator),
        "reference_scale": torch.tensor([0.0, 1.0])[:batch],
    }


def test_unet_wraps_two_stock_unets_and_replaces_every_main_block():
    model = _unet()
    assert type(model.unet).__name__ == type(model.reference_unet).__name__ == "UNet2DConditionModel"
    assert model.unet.config.in_channels == 12 and model.reference_unet.config.in_channels == 4
    blocks = [module for module in model.unet.modules() if isinstance(module, Hunyuan3DPaintTransformerBlock)]
    assert blocks and all(block.reference_key for block in blocks)
    assert not any(isinstance(module, Hunyuan3DPaintTransformerBlock) for module in model.reference_unet.modules())


def test_reference_features_match_the_main_blocks_and_scale_zero_ignores_them():
    model = _unet()
    inputs = _inputs()
    features = model.encode_reference(torch.randn(1, 4, 8, 8))
    keys = {block.reference_key for block in model.unet.modules() if isinstance(block, Hunyuan3DPaintTransformerBlock)}
    assert set(features) == keys
    with torch.no_grad():
        output = model(**inputs, reference_features=features).sample
        perturbed = {name: value + 1 for name, value in features.items()}
        changed = model(**inputs, reference_features=perturbed).sample
    assert output.shape == (2, 2, 3, 4, 8, 8)
    torch.testing.assert_close(changed[0], output[0])  # reference scale 0
    assert not torch.allclose(changed[1], output[1])


def test_views_attend_to_each_other():
    model = _unet()
    inputs = _inputs(batch=1)
    features = model.encode_reference(torch.randn(1, 4, 8, 8))
    with torch.no_grad():
        output = model(**inputs, reference_features=features).sample
        inputs["sample"][:, :, 2] += 1
        changed = model(**inputs, reference_features=features).sample
    assert not torch.allclose(changed[:, :, 0], output[:, :, 0])


def test_position_voxel_indices_cover_every_resolution_and_zero_the_background():
    maps = torch.ones(2, 3, 32, 32)
    maps[:, :, 8:, 8:] = 0.5
    indices = position_voxel_indices(maps, 8)
    assert {length: resolution for length, (_, resolution) in indices.items()} == {128: 64, 32: 32, 8: 16, 2: 8}
    finest = indices[128][0].view(2, 8, 8, 3)
    assert (finest[:, :2, :2] == 0).all() and (finest[:, 2:, 2:] == round(0.5 * 63)).all()


def test_rrdbnet_upscales_four_times_with_the_released_layer_names():
    model = Hunyuan3DPaintRRDBNet(**tiny_config(Hunyuan3DPaintRRDBNet)).eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 5, 7)).shape == (2, 3, 20, 28)
    names = set(model.state_dict())
    assert {"conv_first.weight", "body.0.rdb3.conv5.bias", "conv_up2.weight", "conv_last.bias"} <= names
