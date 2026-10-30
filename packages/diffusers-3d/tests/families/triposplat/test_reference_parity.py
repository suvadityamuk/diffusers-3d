from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from _configs import tiny_config
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers_3d.families.triposplat import (
    TRIPOSPLAT_REFERENCE_REVISION,
    TripoSplatGaussianDecoder,
    TripoSplatImageTo3DPipeline,
    TripoSplatImageVAE,
    TripoSplatTransformerModel,
)
from diffusers_3d.families.triposplat.conversion import (
    convert_dinov3_state_dict,
    convert_triposplat_decoder_state_dict,
    convert_triposplat_transformer_state_dict,
)

from .configs import tiny_dinov3_config

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRIPOSPLAT_REFERENCE_ROOT", "/tmp/TripoSplat"))
REFERENCE_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSplat.git"
_REFERENCE = None


def _load_reference():
    """Import the pinned upstream ``model.py`` under a private name (it imports ``torchvision.ops``)."""

    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=TRIPOSPLAT_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=("model.py",),
        )
        import_reference_dependency("torchvision.ops")
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
    spec = importlib.util.spec_from_file_location("triposplat_reference_model", REFERENCE_ROOT / "model.py")
    _REFERENCE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_REFERENCE)
    return _REFERENCE


def _randomize(module: torch.nn.Module, seed: int, std: float = 0.2) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return module.eval()


def _upstream_transformer(reference, config=None):
    config = config or tiny_config(TripoSplatTransformerModel)
    return _randomize(
        reference.LatentSeqMMFlowModel(
            q_token_length=config["num_latent_tokens"],
            in_channels=config["in_channels"],
            model_channels=config["width"],
            cond_channels=config["cross_attention_dim"],
            out_channels=config["in_channels"],
            num_blocks=config["num_layers"],
            num_refiner_blocks=config["num_refiner_layers"],
            num_head_channels=config["width"] // config["num_attention_heads"],
            cam_channels=config["camera_channels"],
            cond2_channels=config["cross_attention_dim_2"],
            mlp_ratio=config["mlp_ratio"],
            share_mod=True,
            qk_rms_norm=True,
            use_shift_table=True,
        ),
        seed=10,
    )


def _upstream_decoder(reference):
    config = tiny_config(TripoSplatGaussianDecoder)
    shared = {
        "model_channels": config["width"],
        "cond_channels": config["latent_channels"],
        "num_heads": config["num_attention_heads"],
        "mlp_ratio": config["mlp_ratio"],
    }
    decoder = reference.OctreeGaussianDecoder(
        {**shared, "num_blocks": config["num_octree_layers"], "share_mod": True},
        {
            **shared,
            "in_channels": 3,
            "attn_mode": "full",
            "num_blocks": config["num_gaussian_layers"],
            "use_learned_offset_scale": True,
            "use_per_offset": True,
            # Upstream ``GS_DECODER_ARGS`` with the tiny Gaussian count.
            "representation_config": {
                "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
                "perturb_offset": True,
                "perturbe_size": 1.5,
                "offset_scale": 0.05,
                "num_gaussians": config["gaussians_per_token"],
                "filter_kernel_size_3d": 0.0009,
                "scaling_bias": 0.004,
                "opacity_bias": 0.1,
                "scaling_activation": "softplus",
            },
        },
    )
    initial_buffers = {name: buffer.clone() for name, buffer in decoder.named_buffers()}
    return _randomize(decoder, seed=11), initial_buffers


def _converted(model_type, config, state_dict):
    model = model_type(**config).eval()
    model.load_state_dict(state_dict, strict=True)
    return model


def test_dinov3_from_transformers_matches_upstream_vit(tmp_path):
    reference = _load_reference()
    config = tiny_dinov3_config()
    upstream = reference.DinoV3ViT(
        hidden_size=config["hidden_size"],
        num_heads=config["num_attention_heads"],
        num_layers=config["num_hidden_layers"],
        intermediate_size=config["intermediate_size"],
        num_register_tokens=config["num_register_tokens"],
    )
    # Upstream reads the released HF-layout file, which keeps blocks at ``layer.N``.
    model = _randomize(DINOv3ViTModel(DINOv3ViTConfig(**config)), seed=12)
    released = {key.removeprefix("model."): value.contiguous() for key, value in model.state_dict().items()}
    save_file(released, tmp_path / "dinov3.safetensors")
    upstream.load_safetensors(str(tmp_path / "dinov3.safetensors"))
    model.load_state_dict(convert_dinov3_state_dict(released), strict=True)
    pixels = torch.randn(2, 3, 32, 48, generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        expected = upstream.eval()(pixels)
        actual = model(pixel_values=pixels).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_flux2_vae_encoder_on_diffusers_blocks_matches_upstream(tmp_path):
    reference = _load_reference()
    upstream = reference.Flux2VAEEncoder()
    model = TripoSplatImageVAE()
    _randomize(model, seed=14, std=0.02)
    with torch.no_grad():
        model.bn.running_mean.normal_(generator=torch.Generator().manual_seed(15))
        model.bn.running_var.uniform_(0.5, 2.0, generator=torch.Generator().manual_seed(16))
    # The released file is a full Diffusers-layout Flux.2 VAE; upstream drops its decoder half.
    state_dict = {key: value.contiguous() for key, value in model.state_dict().items()}
    state_dict["decoder.conv_in.weight"] = torch.zeros(1)
    save_file(state_dict, tmp_path / "flux2-vae.safetensors")
    upstream.load_safetensors(str(tmp_path / "flux2-vae.safetensors"))
    images = torch.rand(1, 3, 32, 48, generator=torch.Generator().manual_seed(17)) * 2 - 1
    with torch.no_grad():
        expected = upstream.eval().encode(images, deterministic=False, generator=torch.Generator().manual_seed(18))
        actual = model.eval()(images, generator=torch.Generator().manual_seed(18))
    assert actual.shape == (1, 2 * 3, 128)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(("num_refiner_layers", "dtype"), [(2, torch.float32), (3, torch.float32), (2, torch.float16)])
def test_transformer_matches_upstream_flow_model(num_refiner_layers, dtype):
    """Upstream sizes the camera MLP by ``num_refiner_blocks``; it runs the released model in float16."""

    reference = _load_reference()
    config = {**tiny_config(TripoSplatTransformerModel), "num_refiner_layers": num_refiner_layers}
    upstream = _upstream_transformer(reference, config)
    model = _converted(
        TripoSplatTransformerModel, config, convert_triposplat_transformer_state_dict(upstream.state_dict())
    )
    upstream, model = upstream.to(dtype), model.to(dtype)
    generator = torch.Generator().manual_seed(19)
    latents = torch.randn(2, config["num_latent_tokens"], config["in_channels"], generator=generator)
    camera = torch.randn(2, 1, config["camera_channels"], generator=generator)
    context = torch.randn(2, 7, config["cross_attention_dim"], generator=generator)
    context_2 = torch.randn(2, 7, config["cross_attention_dim_2"], generator=generator)
    timestep = torch.tensor([1000.0, 250.0])
    with torch.no_grad():
        expected = upstream(
            {"latent": latents, "camera": camera}, timestep, {"feature1": context, "feature2": context_2}
        )
        actual = model(latents, camera, timestep, context, context_2)
    torch.testing.assert_close(actual.sample, expected["latent"], atol=0, rtol=0)
    torch.testing.assert_close(actual.camera, expected["camera"], atol=0, rtol=0)


def test_octree_sampler_and_gaussian_decoder_match_upstream():
    reference = _load_reference()
    upstream, initial_buffers = _upstream_decoder(reference)
    config = tiny_config(TripoSplatGaussianDecoder)
    model = TripoSplatGaussianDecoder(**config).eval()
    assert torch.equal(model.offset_perturbation, initial_buffers["gs.points_offset_perturbation"])
    assert torch.equal(model.base_offset_scale, initial_buffers["gs.base_offset_scale"])
    model.load_state_dict(convert_triposplat_decoder_state_dict(upstream.state_dict()), strict=True)
    latents = torch.randn(1, 6, config["latent_channels"], generator=torch.Generator().manual_seed(20))
    num_points = 7

    with torch.no_grad():
        # Upstream draws from the global RNG; a CPU generator seeded alike yields the same stream.
        torch.manual_seed(21)
        expected_points = reference.OctreeProbabilityFixedlenDecoder.sample(
            upstream.octree, latents, num_points=num_points, level=config["octree_depth"]
        )["points"]
        points = model.sample_points(latents, num_points, torch.Generator().manual_seed(21))
        expected_features = upstream.gs(x={"points": points}, cond=latents)["features"]
        features = model.gaussian_features(points, latents)
        offsets = upstream.gs._get_offset(expected_features)[0]
        asset = model._assets(points, features)[0]
    torch.testing.assert_close(points, expected_points, atol=0, rtol=0)
    torch.testing.assert_close(features, expected_features, atol=0, rtol=0)

    # Upstream ``triposplat.Gaussian`` activations for one decoded set (aabb [-0.5, 1]).
    layout = upstream.gs.layout
    raw = {name: features[0][:, slice(*spec["range"])].reshape(-1, *spec["shape"]) for name, spec in layout.items()}
    scaling_bias = torch.tensor(0.004)
    scaling_bias = scaling_bias + torch.log(-torch.expm1(-scaling_bias))
    scales = torch.sqrt(F.softplus(raw["_scaling"] + scaling_bias).square() + 0.0009**2)
    rotation = raw["_rotation"] * 0.1 + torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(asset.means, ((offsets + points[0, :, None]) * 1.0 - 0.5).flatten(0, 1))
    torch.testing.assert_close(asset.log_scales, scales.log().flatten(0, 1))
    torch.testing.assert_close(asset.quaternions_wxyz, F.normalize(rotation, dim=-1).flatten(0, 1))
    torch.testing.assert_close(
        torch.sigmoid(asset.opacity_logits),
        torch.sigmoid(raw["_opacity"] + torch.log(torch.tensor(0.1) / (1 - torch.tensor(0.1)))).flatten(0, 1),
    )
    torch.testing.assert_close(asset.sh_coefficients, raw["_features_dc"].flatten(0, 1))
    assert asset.means.shape == (num_points * config["gaussians_per_token"], 3)


def test_stock_flow_match_scheduler_reproduces_upstream_shifted_euler_schedule():
    """Upstream ``FlowEulerCfgSampler``: ``t = s * u / (1 + (s - 1) * u)`` over ``u = linspace(1, 0, steps + 1)``."""

    steps, shift = 7, 3.0
    uniform = np.linspace(1, 0, steps + 1)
    expected = shift * uniform / (1 + (shift - 1) * uniform)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(sigmas=uniform[:-1])
    torch.testing.assert_close(scheduler.sigmas.double(), torch.from_numpy(expected), atol=1e-7, rtol=0)
    # Upstream feeds the model ``float32(1000 * t)``; the scheduler scales float32 sigmas, within float32 rounding.
    torch.testing.assert_close(scheduler.timesteps, torch.tensor(1000 * expected[:-1], dtype=torch.float32))
    sample = torch.randn(3, generator=torch.Generator().manual_seed(22))
    velocity = torch.randn(3, generator=torch.Generator().manual_seed(23))
    stepped = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    torch.testing.assert_close(stepped, sample - velocity * (expected[0] - expected[1]), atol=1e-6, rtol=0)
    assert math.isclose(float(scheduler.sigmas[-1]), 0.0)


def _load_reference_pipeline():
    """Import the pinned upstream ``triposplat.py``, which imports ``model`` as a top-level module."""

    model = _load_reference()
    import_reference_dependency("torchvision.transforms")
    previous = sys.modules.get("model")
    sys.modules["model"] = model
    try:
        spec = importlib.util.spec_from_file_location("triposplat_reference", REFERENCE_ROOT / "triposplat.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("model")
        else:
            sys.modules["model"] = previous
    return module


@pytest.mark.parametrize("steps", [1, 3])
def test_pipeline_sampling_matches_upstream_sequential_guidance(tiny_triposplat_components, rgba_condition, steps):
    """Upstream ``FlowEulerCfgSampler`` with ``run``'s defaults (shift 3, CFG 3) and its sequential guidance passes."""

    reference = _load_reference_pipeline()
    upstream = _upstream_transformer(_load_reference())
    config = tiny_config(TripoSplatTransformerModel)
    state_dict = convert_triposplat_transformer_state_dict(upstream.state_dict())
    transformer = _converted(TripoSplatTransformerModel, config, state_dict)
    pipeline = TripoSplatImageTo3DPipeline(**{**tiny_triposplat_components, "transformer": transformer})
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator().manual_seed(25)
    latents = torch.randn(1, config["num_latent_tokens"], config["in_channels"], generator=generator)
    camera = torch.randn(1, 1, config["camera_channels"], generator=generator)
    image_embeds, latent_embeds = pipeline.encode_image(
        rgba_condition, image_size=32, generator=torch.Generator().manual_seed(26)
    )
    cond = {"feature1": image_embeds, "feature2": latent_embeds}
    with torch.no_grad():
        expected = reference.FlowEulerCfgSampler().sample(
            upstream,
            {"latent": latents.clone(), "camera": camera.clone()},
            cond,
            {key: torch.zeros_like(value) for key, value in cond.items()},
            steps=steps,
            shift=3.0,
            guidance_scale=3.0,
        )
    kwargs = {"latents": latents, "camera_latents": camera, "num_inference_steps": steps, "image_size": 32}
    sequential = pipeline(rgba_condition, generator=torch.Generator().manual_seed(26), **kwargs).latents.latents
    batched = pipeline(rgba_condition, batch_guidance=True, generator=torch.Generator().manual_seed(26), **kwargs)
    if steps == 1:
        torch.testing.assert_close(sequential, expected["latent"], atol=0, rtol=0)
        assert not torch.equal(batched.latents.latents, expected["latent"])
    # Over several steps the stock scheduler's float32 sigmas (upstream steps float64 ones) drift about 6e-5 on this
    # random model, guided or not; batching the guidance passes adds a little more.
    torch.testing.assert_close(sequential, expected["latent"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(batched.latents.latents, expected["latent"], atol=2e-4, rtol=1e-4)
