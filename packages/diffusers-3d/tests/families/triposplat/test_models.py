from __future__ import annotations

import pytest
import torch
from _configs import production_config, tiny_config
from accelerate import init_empty_weights

from diffusers_3d import CoordinateSystem, GaussianSplatAsset
from diffusers_3d.families.triposplat import TripoSplatGaussianDecoder, TripoSplatImageVAE, TripoSplatTransformerModel


def _transformer_inputs(config, batch_size=2):
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch_size, config["num_latent_tokens"], config["in_channels"], generator=generator),
        torch.randn(batch_size, 1, config["camera_channels"], generator=generator),
        torch.tensor([900.0, 100.0][:batch_size]),
        torch.randn(batch_size, 7, config["cross_attention_dim"], generator=generator),
        torch.randn(batch_size, 7, config["cross_attention_dim_2"], generator=generator),
    )


def test_transformer_forward_backward_checkpointing_and_save_load(tmp_path):
    config = tiny_config(TripoSplatTransformerModel)
    model = TripoSplatTransformerModel(**config)
    inputs = _transformer_inputs(config)
    output = model(*inputs)
    assert output.sample.shape == inputs[0].shape and output.camera.shape == inputs[1].shape
    torch.testing.assert_close(output.sample, model(*inputs, return_dict=False)[0])
    model.enable_gradient_checkpointing()
    checkpointed = model(*inputs)
    (checkpointed.sample.square().mean() + checkpointed.camera.square().mean()).backward()
    torch.testing.assert_close(checkpointed.sample, output.sample)
    assert model.rope[0].freqs_2.grad is not None and model.blocks[0].attn.norm_q.gamma.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TripoSplatTransformerModel.from_pretrained(tmp_path, local_files_only=True)
    with torch.no_grad():
        torch.testing.assert_close(loaded(*inputs).sample, output.sample.detach())


def test_decoder_returns_z_up_assets_inside_the_unit_cube_and_round_trips(tmp_path):
    config = tiny_config(TripoSplatGaussianDecoder)
    model = TripoSplatGaussianDecoder(**config).eval()
    latents = torch.randn(2, 6, config["latent_channels"], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assets = model(latents, 20, generator=torch.Generator().manual_seed(2)).assets
        again = model(latents, 20, generator=torch.Generator().manual_seed(2), return_dict=False)[0]
        model.save_pretrained(tmp_path)
        loaded = TripoSplatGaussianDecoder.from_pretrained(tmp_path, local_files_only=True)
        reloaded = loaded(latents, 20, generator=torch.Generator().manual_seed(2)).assets
    assert len(assets) == 2
    for asset, repeat, reload in zip(assets, again, reloaded):
        assert type(asset) is GaussianSplatAsset
        assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
        assert asset.means.shape == (20, 3) and asset.sh_coefficients.shape == (20, 1, 3)
        assert asset.active_sh_degree == 0
        # Points lie in [0, 1]^3 and offsets are bounded by 0.75 * softplus(...), so means stay near [-0.5, 0.5].
        assert asset.means.abs().max() < 1.0
        assert torch.equal(asset.means, repeat.means) and torch.equal(asset.means, reload.means)
    assert "offset_perturbation" in model.state_dict() and "base_offset_scale" in model.state_dict()


def test_decoder_rejects_counts_that_are_not_a_multiple_of_the_gaussians_per_token():
    model = TripoSplatGaussianDecoder(**tiny_config(TripoSplatGaussianDecoder))
    with pytest.raises(ValueError, match="multiple of 4"):
        model(torch.zeros(1, 6, 4), 10)
    with pytest.raises(ValueError, match="at least 4"):
        model(torch.zeros(1, 6, 4), 0)


def test_vae_samples_the_posterior_with_the_generator():
    model = TripoSplatImageVAE(**tiny_config(TripoSplatImageVAE)).eval()
    images = torch.rand(1, 3, 32, 48, generator=torch.Generator().manual_seed(3)) * 2 - 1
    with torch.no_grad():
        first = model(images, generator=torch.Generator().manual_seed(4))
        second = model(images, generator=torch.Generator().manual_seed(4))
        other = model(images, generator=torch.Generator().manual_seed(5))
    assert first.shape == (1, 2 * 3, 4 * 4)
    assert torch.equal(first, second) and not torch.equal(first, other)


def test_production_layouts_match_the_published_checkpoint_headers():
    # VAST-AI/TripoSplat@56a96e60 safetensors headers: 559 flow-model tensors, 388 decoder tensors, and the
    # 111 encoder-half tensors of the 251 in flux2-vae.safetensors.
    with init_empty_weights():
        transformer = TripoSplatTransformerModel(**production_config(TripoSplatTransformerModel)).state_dict()
        decoder = TripoSplatGaussianDecoder(**production_config(TripoSplatGaussianDecoder)).state_dict()
        vae = TripoSplatImageVAE(**production_config(TripoSplatImageVAE)).state_dict()
    assert (len(transformer), len(decoder), len(vae)) == (559, 388, 111)
    assert transformer["rope.0.freqs_2"].shape == (12,)
    assert transformer["blocks.0.attn.norm_q.gamma"].shape == (16, 64)
    assert transformer["camera_proj.net.0.proj.weight"].shape == (1024, 5)
    assert decoder["gaussian_proj_out.weight"].shape == (480, 1024)
    assert decoder["octree_blocks.0.attn.to_kv.weight"].shape == (2048, 16)
    assert vae["bn.running_mean"].shape == (128,)


def test_camera_mlp_depth_follows_num_refiner_layers():
    # Upstream's camera MLP has ``num_refiner_blocks`` linear layers; it cannot be built with fewer than two.
    config = tiny_config(TripoSplatTransformerModel)
    model = TripoSplatTransformerModel(**{**config, "num_refiner_layers": 4})
    linears = [name for name, module in model.camera_proj.named_modules() if isinstance(module, torch.nn.Linear)]
    assert linears == ["net.0.proj", "net.1.proj", "net.2.proj", "net.4"]
    with pytest.raises(ValueError, match="num_refiner_layers must be at least 2"):
        TripoSplatTransformerModel(**{**config, "num_refiner_layers": 1})
