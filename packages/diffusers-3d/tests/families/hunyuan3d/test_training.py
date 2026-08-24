from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    Hunyuan3DShapeBatch,
    Hunyuan3DShapeFlowMatchingRecipe,
)
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY


def test_released_flow_matching_equation_and_frozen_components(tiny_hunyuan_pipeline):
    pipeline = tiny_hunyuan_pipeline
    recipe = Hunyuan3DShapeFlowMatchingRecipe(pipeline)
    clean = torch.linspace(-1, 1, 64).reshape(2, 4, 8)
    noise = torch.linspace(1, -1, 64).reshape(2, 4, 8)
    timesteps = torch.tensor([0.25, 0.75])
    batch = Hunyuan3DShapeBatch(
        images=torch.zeros(2, 3, 8, 8),
        shape_latents=clean,
        noise=noise,
        timesteps=timesteps,
    )

    captured = {}

    def capture_input(_module, args):
        captured["input"] = args[0]

    def capture_output(_module, _args, output):
        captured["output"] = output.sample

    input_hook = pipeline.denoiser.register_forward_pre_hook(capture_input)
    output_hook = pipeline.denoiser.register_forward_hook(capture_output)
    output = recipe.compute_loss(batch)
    input_hook.remove()
    output_hook.remove()

    interpolation = timesteps.reshape(-1, 1, 1)
    expected_input = interpolation * clean + (1.0 - interpolation) * noise
    expected_target = clean - noise
    expected_loss = torch.nn.functional.mse_loss(captured["output"], expected_target)
    torch.testing.assert_close(captured["input"], expected_input, atol=0.0, rtol=0.0)
    torch.testing.assert_close(output.loss, expected_loss)
    assert not any(parameter.requires_grad for parameter in pipeline.vae.parameters())
    assert not any(parameter.requires_grad for parameter in pipeline.conditioner.parameters())

    output.loss.backward()
    assert pipeline.denoiser.x_embedder.weight.grad is not None
    assert all(parameter.grad is None for parameter in pipeline.vae.parameters())
    assert all(parameter.grad is None for parameter in pipeline.conditioner.parameters())


def test_surface_training_is_explicitly_unsupported(tiny_hunyuan_pipeline):
    recipe = Hunyuan3DShapeFlowMatchingRecipe(tiny_hunyuan_pipeline)
    batch = Hunyuan3DShapeBatch(
        images=torch.zeros(1, 3, 8, 8),
        surface_samples=torch.zeros(1, 4, 6),
    )
    with pytest.raises(NotImplementedError, match="precomputed"):
        recipe.compute_loss(batch)


def test_recipe_has_exact_reviewed_registration(tiny_hunyuan_pipeline):
    recipe = Hunyuan3DShapeFlowMatchingRecipe(tiny_hunyuan_pipeline)
    registration = _TRAINING_RECIPE_REGISTRY.validate(recipe)
    assert registration.recipe_type is Hunyuan3DShapeFlowMatchingRecipe
    assert registration.component_policies[0].supported_strategies[0].value == "full"
