from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from diffusers_3d import Hunyuan3DShapeDiTModel, Hunyuan3DShapeVAE
from diffusers_3d.families.hunyuan3d import HUNYUAN3D_REFERENCE_REVISION

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path("/tmp/Hunyuan3D-2.1/hy3dshape/hy3dshape")
REFERENCE_REPOSITORY_ROOT = REFERENCE_ROOT.parents[1]


def _assert_pinned_revision() -> None:
    git_directory = REFERENCE_REPOSITORY_ROOT / ".git"
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        pytest.skip("the Hunyuan3D reference checkout has no verifiable revision metadata")
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        revision_path = git_directory / reference
        if revision_path.is_file():
            head = revision_path.read_text(encoding="utf-8").strip()
        else:
            packed_refs = git_directory / "packed-refs"
            if not packed_refs.is_file():
                pytest.skip("the Hunyuan3D reference checkout revision is not resolvable")
            revisions = {
                name: revision
                for revision, name in (
                    line.split(" ", maxsplit=1)
                    for line in packed_refs.read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith(("#", "^"))
                )
            }
            head = revisions.get(reference, "")
    assert head == HUNYUAN3D_REFERENCE_REVISION


def _load_pinned_reference():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("the pinned Hunyuan3D-2.1 reference checkout is unavailable")
    _assert_pinned_revision()
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
    _assert_pinned_revision()
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


def _load_standalone_reference(relative_path: str, module_name: str, class_name: str):
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("the pinned Hunyuan3D-2.1 reference checkout is unavailable")
    _assert_pinned_revision()
    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"optional reference dependency unavailable: {error}")
    return getattr(module, class_name)


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


def test_tiny_denoiser_backward_matches_pinned_reference():
    reference_type = _load_pinned_reference()
    config = Hunyuan3DShapeDiTModel.tiny_config()
    torch.manual_seed(0)
    reference = reference_type(**config).train()
    model = Hunyuan3DShapeDiTModel(**config).train()
    model.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(7)
    base_hidden_states = torch.randn(
        2,
        config["input_size"],
        config["in_channels"],
        generator=generator,
    )
    timesteps = torch.rand(2, generator=generator)
    base_context = torch.randn(
        2,
        config["text_len"],
        config["context_dim"],
        generator=generator,
    )
    reference_hidden_states = base_hidden_states.clone().requires_grad_(True)
    model_hidden_states = base_hidden_states.clone().requires_grad_(True)
    reference_context = base_context.clone().requires_grad_(True)
    model_context = base_context.clone().requires_grad_(True)
    loss_weights = torch.linspace(-0.5, 0.75, base_hidden_states.numel()).reshape_as(base_hidden_states)

    expected = reference(reference_hidden_states, timesteps, {"main": reference_context})
    actual = model(model_hidden_states, timesteps, model_context).sample
    expected_loss = (expected * loss_weights).sum() + 0.125 * expected.square().mean()
    actual_loss = (actual * loss_weights).sum() + 0.125 * actual.square().mean()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(model_hidden_states.grad, reference_hidden_states.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model_context.grad, reference_context.grad, atol=1e-6, rtol=1e-5)
    reference_parameters = dict(reference.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name in (
        "x_embedder.weight",
        "blocks.0.attn1.to_q.weight",
        "blocks.1.attn2.to_v.weight",
        "final_layer.linear.weight",
    ):
        torch.testing.assert_close(
            model_parameters[name].grad,
            reference_parameters[name].grad,
            atol=1e-6,
            rtol=1e-5,
        )


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


def test_tiny_composed_pipeline_stages_match_pinned_reference(tiny_hunyuan_pipeline):
    reference_denoiser_type = _load_pinned_reference()
    reference_vae_type = _load_pinned_vae_reference()
    reference_conditioner_type = _load_standalone_reference(
        "models/conditioner.py",
        "_hunyuan3d_conditioner_test_reference",
        "DinoImageEncoder",
    )
    reference_scheduler_type = _load_standalone_reference(
        "schedulers.py",
        "_hunyuan3d_scheduler_test_reference",
        "FlowMatchEulerDiscreteScheduler",
    )
    pipeline = tiny_hunyuan_pipeline
    pipeline.conditioner.eval()
    pipeline.denoiser.eval()
    pipeline.vae.eval()

    reference_conditioner = reference_conditioner_type(
        config=pipeline.conditioner.model.config.to_dict(),
        image_size=pipeline.conditioner.image_size,
        use_cls_token=pipeline.conditioner.use_cls_token,
    ).eval()
    reference_conditioner.model.load_state_dict(pipeline.conditioner.model.state_dict(), strict=True)
    images = pipeline.preprocess(torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8))
    with torch.no_grad():
        actual_conditioning = pipeline.encode_conditioning(images, do_classifier_free_guidance=True)
        reference_conditional = reference_conditioner(images)
        reference_conditioning = torch.cat([reference_conditional, torch.zeros_like(reference_conditional)])
    torch.testing.assert_close(actual_conditioning, reference_conditioning, atol=1e-6, rtol=1e-5)

    reference_denoiser = reference_denoiser_type(**Hunyuan3DShapeDiTModel.tiny_config()).eval()
    reference_denoiser.load_state_dict(pipeline.denoiser.state_dict(), strict=True)
    initial_latents = torch.linspace(-1.0, 1.0, 4 * 8).reshape(1, 4, 8)
    with torch.no_grad():
        actual_latents = pipeline.denoise(
            initial_latents.clone(),
            actual_conditioning,
            num_inference_steps=3,
            guidance_scale=5.0,
        )
        reference_scheduler = reference_scheduler_type(num_train_timesteps=1000)
        reference_scheduler.set_timesteps(
            3,
            device=initial_latents.device,
            sigmas=np.linspace(0.0, 1.0, 3),
        )
        expected_latents = initial_latents.clone()
        for timestep in reference_scheduler.timesteps:
            model_input = torch.cat([expected_latents, expected_latents])
            model_timestep = timestep.expand(model_input.shape[0]).to(expected_latents.dtype) / 1000
            velocity = reference_denoiser(
                model_input,
                model_timestep,
                {"main": reference_conditioning},
            )
            conditional_velocity, unconditional_velocity = velocity.chunk(2)
            velocity = unconditional_velocity + 5.0 * (conditional_velocity - unconditional_velocity)
            expected_latents = reference_scheduler.step(
                velocity,
                timestep,
                expected_latents,
                return_dict=False,
            )[0]
    torch.testing.assert_close(actual_latents, expected_latents, atol=1e-6, rtol=1e-5)

    reference_vae = reference_vae_type(**Hunyuan3DShapeVAE.tiny_config()).eval()
    incompatible = reference_vae.load_state_dict(pipeline.vae.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(("encoder.", "pre_kl.")) for key in incompatible.missing_keys)
    queries = torch.linspace(-0.9, 0.9, 21).reshape(1, 7, 3)
    with torch.no_grad():
        actual_decoded = pipeline.vae.decode(actual_latents / pipeline.vae.scale_factor).sample
        expected_decoded = reference_vae.decode(expected_latents / pipeline.vae.scale_factor)
        actual_field = pipeline.vae.evaluate_field(actual_decoded, queries, query_chunk_size=3)
        expected_field = reference_vae.geo_decoder(queries=queries, latents=expected_decoded).squeeze(-1)
    torch.testing.assert_close(actual_decoded, expected_decoded, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_field, expected_field, atol=1e-6, rtol=1e-5)
