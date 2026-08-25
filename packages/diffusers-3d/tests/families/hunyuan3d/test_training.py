from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    FullFineTune,
    Hunyuan3DShapeBatch,
    Hunyuan3DShapeExample,
    Hunyuan3DShapeFlowMatchingRecipe,
    ImageCondition,
    Object3DTrainer,
    TensorShapeError,
    TrainingConfig3D,
    TrainingManifest3D,
    TrainingTargetError,
)
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

pytestmark = pytest.mark.integration


def test_released_flow_matching_equation_and_frozen_components(tiny_hunyuan_pipeline):
    pipeline = tiny_hunyuan_pipeline
    requires_grad_before = {
        name: parameter.requires_grad
        for name, parameter in (
            list(pipeline.denoiser.named_parameters())
            + [(f"vae.{name}", parameter) for name, parameter in pipeline.vae.named_parameters()]
            + [(f"conditioner.{name}", parameter) for name, parameter in pipeline.conditioner.named_parameters()]
        )
    }
    recipe = Hunyuan3DShapeFlowMatchingRecipe(pipeline)
    requires_grad_after = {
        name: parameter.requires_grad
        for name, parameter in (
            list(pipeline.denoiser.named_parameters())
            + [(f"vae.{name}", parameter) for name, parameter in pipeline.vae.named_parameters()]
            + [(f"conditioner.{name}", parameter) for name, parameter in pipeline.conditioner.named_parameters()]
        )
    }
    assert requires_grad_after == requires_grad_before
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
    assert registration.example_type is Hunyuan3DShapeExample
    assert registration.component_policies[0].supported_strategies[0].value == "full"


def test_recipe_collates_precomputed_latents_and_rejects_mixed_sources(tiny_hunyuan_pipeline):
    recipe = Hunyuan3DShapeFlowMatchingRecipe(tiny_hunyuan_pipeline)
    latent_example = Hunyuan3DShapeExample(
        condition=ImageCondition(torch.zeros(3, 8, 8)),
        shape_latents=torch.zeros(4, 8),
    )
    batch = recipe.collate((latent_example, latent_example))
    assert batch.shape_latents.shape == (2, 4, 8)
    assert batch.surface_samples is None
    moved = latent_example.to(dtype=torch.float64)
    assert type(moved) is Hunyuan3DShapeExample
    assert moved.condition.image.dtype is torch.float64
    assert moved.shape_latents.dtype is torch.float64

    surface_example = Hunyuan3DShapeExample(
        condition=ImageCondition(torch.zeros(3, 8, 8)),
        surface_samples=torch.zeros(4, 6),
    )
    with pytest.raises(TrainingTargetError, match="cannot mix"):
        recipe.collate((latent_example, surface_example))
    with pytest.raises(TensorShapeError, match="exactly one"):
        Hunyuan3DShapeExample(condition=latent_example.condition)
    with pytest.raises(TensorShapeError, match="exactly one"):
        Hunyuan3DShapeExample(
            condition=latent_example.condition,
            shape_latents=latent_example.shape_latents,
            surface_samples=surface_example.surface_samples,
        )


def test_object3d_trainer_full_step_and_checkpoint_roundtrip(
    tmp_path,
    tiny_hunyuan_pipeline,
    tiny_hunyuan_latent_dataset,
):
    pipeline = tiny_hunyuan_pipeline
    recipe = Hunyuan3DShapeFlowMatchingRecipe(pipeline)
    assert any(parameter.requires_grad for parameter in pipeline.vae.parameters())
    assert any(parameter.requires_grad for parameter in pipeline.conditioner.parameters())

    trainer = Object3DTrainer(
        recipe,
        tiny_hunyuan_latent_dataset,
        FullFineTune(("denoiser",)),
        TrainingConfig3D(
            base_model="tests/tiny-hunyuan3d",
            revision="tiny-reference",
            dataset_fingerprint="tests/tiny-hunyuan-latents-v1",
            output_dir=tmp_path,
            train_batch_size=2,
            max_train_steps=1,
            learning_rate=1e-3,
            shuffle=False,
            seed=7,
            cpu=True,
        ),
    ).prepare()

    assert all(parameter.requires_grad for parameter in pipeline.denoiser.parameters())
    assert not any(parameter.requires_grad for parameter in pipeline.vae.parameters())
    assert not any(parameter.requires_grad for parameter in pipeline.conditioner.parameters())
    for component in (pipeline.conditioner, pipeline.vae):
        assert not component.training
        assert all(parameter.device == trainer.accelerator.device for parameter in component.parameters())
    trainable_ids = {id(parameter) for parameter in trainer.trainable_parameters}
    assert trainable_ids == {id(parameter) for parameter in pipeline.denoiser.parameters()}
    assert {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    } == trainable_ids
    assert all(name.startswith("denoiser.") for name in trainer.trainable_parameter_names)

    summary = trainer.train()
    assert summary.final_loss is not None and trainer.optimizer_steps == 1
    checkpoint_manifest = trainer.save_checkpoint()
    assert checkpoint_manifest.is_file()
    loaded_manifest = TrainingManifest3D.load(tmp_path)
    assert loaded_manifest.example_type.endswith(".Hunyuan3DShapeExample")
    assert loaded_manifest == trainer.manifest

    saved_weight = pipeline.denoiser.x_embedder.weight.detach().clone()
    saved_conditioner_weight = next(pipeline.conditioner.parameters()).detach().clone()
    saved_vae_weight = next(pipeline.vae.parameters()).detach().clone()
    with torch.no_grad():
        pipeline.denoiser.x_embedder.weight.add_(10.0)
        next(pipeline.conditioner.parameters()).add_(10.0)
        next(pipeline.vae.parameters()).add_(10.0)
    trainer.load_checkpoint(tmp_path)
    torch.testing.assert_close(pipeline.denoiser.x_embedder.weight, saved_weight, atol=0.0, rtol=0.0)
    torch.testing.assert_close(next(pipeline.conditioner.parameters()), saved_conditioner_weight, atol=0.0, rtol=0.0)
    torch.testing.assert_close(next(pipeline.vae.parameters()), saved_vae_weight, atol=0.0, rtol=0.0)
