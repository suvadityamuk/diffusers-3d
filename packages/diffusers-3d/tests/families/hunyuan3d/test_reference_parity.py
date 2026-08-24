from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from diffusers_3d import Hunyuan3DShapeDiTModel, Hunyuan3DShapeVAE

REFERENCE_ROOT = Path("/tmp/Hunyuan3D-2.1/hy3dshape/hy3dshape")


def _load_pinned_reference():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("the pinned Hunyuan3D-2.1 reference checkout is unavailable")
    package = "_hunyuan3d_test_reference"
    for name in (package, f"{package}.models", f"{package}.models.denoisers"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    utils = types.ModuleType(f"{package}.utils")
    utils.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    utils.synchronize_timer = lambda *args, **kwargs: lambda function: function
    utils.smart_load_model = lambda *args, **kwargs: None
    sys.modules[utils.__name__] = utils

    try:
        for module_name, relative_path in (
            (f"{package}.models.denoisers.moe_layers", "models/denoisers/moe_layers.py"),
            (f"{package}.models.denoisers.hunyuandit", "models/denoisers/hunyuandit.py"),
        ):
            spec = importlib.util.spec_from_file_location(module_name, REFERENCE_ROOT / relative_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except ImportError as error:
        pytest.skip(f"optional reference dependency unavailable: {error}")
    return sys.modules[f"{package}.models.denoisers.hunyuandit"].HunYuanDiTPlain


def _load_pinned_vae_reference():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("the pinned Hunyuan3D-2.1 reference checkout is unavailable")
    package = "_hunyuan3d_vae_test_reference"
    for name in (package, f"{package}.models", f"{package}.models.autoencoders"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    utils = types.ModuleType(f"{package}.utils")
    utils.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    utils.synchronize_timer = lambda *args, **kwargs: lambda function: function
    utils.smart_load_model = lambda *args, **kwargs: None
    sys.modules[utils.__name__] = utils

    try:
        for module_name, relative_path in (
            (
                f"{package}.models.autoencoders.attention_processors",
                "models/autoencoders/attention_processors.py",
            ),
            (
                f"{package}.models.autoencoders.attention_blocks",
                "models/autoencoders/attention_blocks.py",
            ),
            (
                f"{package}.models.autoencoders.surface_extractors",
                "models/autoencoders/surface_extractors.py",
            ),
            (
                f"{package}.models.autoencoders.volume_decoders",
                "models/autoencoders/volume_decoders.py",
            ),
            (f"{package}.models.autoencoders.model", "models/autoencoders/model.py"),
        ):
            spec = importlib.util.spec_from_file_location(module_name, REFERENCE_ROOT / relative_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except ImportError as error:
        pytest.skip(f"optional VAE reference dependency unavailable: {error}")
    return sys.modules[f"{package}.models.autoencoders.model"].ShapeVAE


def test_tiny_denoiser_matches_pinned_reference():
    reference_type = _load_pinned_reference()
    config = Hunyuan3DShapeDiTModel.tiny_config()
    torch.manual_seed(0)
    reference = reference_type(**config).eval()
    model = Hunyuan3DShapeDiTModel(**config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(2, config["input_size"], config["in_channels"], generator=generator)
    timesteps = torch.rand(2, generator=generator)
    context = torch.randn(2, config["text_len"], config["context_dim"], generator=generator)
    with torch.no_grad():
        expected = reference(hidden_states, timesteps, {"main": context})
        actual = model(hidden_states, timesteps, context).sample
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_tiny_vae_decoder_matches_pinned_reference():
    reference_type = _load_pinned_vae_reference()
    config = Hunyuan3DShapeVAE.tiny_config()
    torch.manual_seed(0)
    reference = reference_type(**config).eval()
    vae = Hunyuan3DShapeVAE(**config).eval()
    reference_decoder_state = {
        key: value for key, value in reference.state_dict().items() if not key.startswith(("encoder.", "pre_kl."))
    }
    assert set(vae.state_dict()) == set(reference_decoder_state)
    assert all(vae.state_dict()[key].shape == value.shape for key, value in reference_decoder_state.items())
    vae.load_state_dict(reference_decoder_state, strict=True)

    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(2, config["num_latents"], config["embed_dim"], generator=generator)
    queries = torch.randn(2, 7, 3, generator=generator)
    with torch.no_grad():
        expected_latents = reference.decode(latents)
        actual_latents = vae.decode(latents).sample
        expected_field = reference.geo_decoder(queries=queries, latents=expected_latents).squeeze(-1)
        actual_field = vae.evaluate_field(actual_latents, queries, query_chunk_size=3)
    torch.testing.assert_close(actual_latents, expected_latents, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_field, expected_field, atol=1e-6, rtol=1e-5)
