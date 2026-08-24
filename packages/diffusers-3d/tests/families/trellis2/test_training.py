from __future__ import annotations

import torch

from diffusers_3d import (
    FullFineTune,
    Object3DTrainer,
    TrainingConfig3D,
    TrainingManifest3D,
    Trellis2ShapeSLatFlowRecipe,
    Trellis2SLatBatch,
    Trellis2SparseStructureBatch,
    Trellis2SparseStructureExample,
    Trellis2SparseStructureFlowRecipe,
    Trellis2TextureSLatBatch,
    Trellis2TextureSLatFlowRecipe,
    TrellisSparseTensor,
)
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY


def test_released_sparse_structure_logit_normal_objective_backward_and_frozen_components(
    tiny_trellis2_pipeline,
):
    pipeline = tiny_trellis2_pipeline
    recipe = Trellis2SparseStructureFlowRecipe(pipeline)
    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 2 * 2 * 2).reshape(2, 2, 2, 2, 2)
    noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    timesteps = torch.tensor([0.25, 0.75])
    batch = Trellis2SparseStructureBatch(
        images=torch.zeros(2, 3, 8, 8),
        sparse_structure_latents=clean,
        noise=noise,
        timesteps=timesteps,
        condition_dropout_mask=torch.tensor([False, True]),
    )
    captured = {}

    def capture_input(_module, args):
        captured["input"] = args[0]
        captured["timesteps"] = args[1]

    def capture_output(_module, _args, output):
        captured["output"] = output.sample

    input_hook = pipeline.sparse_structure_flow_model.register_forward_pre_hook(capture_input)
    output_hook = pipeline.sparse_structure_flow_model.register_forward_hook(capture_output)
    output = recipe.compute_loss(batch)
    input_hook.remove()
    output_hook.remove()

    interpolation = timesteps.reshape(-1, 1, 1, 1, 1)
    expected_input = (1 - interpolation) * clean + (1e-5 + (1 - 1e-5) * interpolation) * noise
    expected_target = (1 - 1e-5) * noise - clean
    torch.testing.assert_close(captured["input"], expected_input, atol=0.0, rtol=0.0)
    torch.testing.assert_close(captured["timesteps"], timesteps * 1000, atol=0.0, rtol=0.0)
    torch.testing.assert_close(output.loss, torch.nn.functional.mse_loss(captured["output"], expected_target))
    output.loss.backward()
    assert pipeline.sparse_structure_flow_model.out_layer.bias.grad is not None
    assert all(parameter.grad is None for parameter in pipeline.conditioner.parameters())
    assert all(parameter.grad is None for parameter in pipeline.sparse_structure_decoder.parameters())
    assert recipe.timestep_mean == recipe.timestep_std == 1.0
    assert recipe.p_uncond == 0.1


def test_sparse_structure_recipe_registration_collation_full_step_and_checkpoint(
    tmp_path,
    tiny_trellis2_pipeline,
    tiny_trellis2_latent_dataset,
):
    recipe = Trellis2SparseStructureFlowRecipe(tiny_trellis2_pipeline)
    registration = _TRAINING_RECIPE_REGISTRY.validate(recipe)
    assert registration.recipe_type is Trellis2SparseStructureFlowRecipe
    assert registration.example_type is Trellis2SparseStructureExample
    examples = tuple(tiny_trellis2_latent_dataset[index] for index in range(2))
    batch = recipe.collate(examples)
    assert batch.sparse_structure_latents.shape == (2, 2, 2, 2, 2)

    trainer = Object3DTrainer(
        recipe,
        tiny_trellis2_latent_dataset,
        FullFineTune(("sparse_structure_flow_model",)),
        TrainingConfig3D(
            base_model="tests/tiny-trellis2",
            revision="tiny-reference",
            output_dir=tmp_path,
            train_batch_size=2,
            max_train_steps=1,
            learning_rate=1e-3,
            shuffle=False,
            seed=7,
            cpu=True,
        ),
    ).prepare()
    assert all(
        parameter.requires_grad for parameter in tiny_trellis2_pipeline.sparse_structure_flow_model.parameters()
    )
    assert not any(parameter.requires_grad for parameter in tiny_trellis2_pipeline.conditioner.parameters())
    assert not any(
        parameter.requires_grad for parameter in tiny_trellis2_pipeline.sparse_structure_decoder.parameters()
    )
    assert all(name.startswith("sparse_structure_flow_model.") for name in trainer.trainable_parameter_names)
    assert len(trainer.train()) == 1
    checkpoint = trainer.save_checkpoint()
    assert checkpoint.is_file()
    assert TrainingManifest3D.load(tmp_path) == trainer.manifest
    saved_weight = tiny_trellis2_pipeline.sparse_structure_flow_model.out_layer.weight.detach().clone()
    with torch.no_grad():
        tiny_trellis2_pipeline.sparse_structure_flow_model.out_layer.weight.add_(10.0)
    trainer.load_checkpoint(tmp_path)
    torch.testing.assert_close(
        tiny_trellis2_pipeline.sparse_structure_flow_model.out_layer.weight,
        saved_weight,
        atol=0.0,
        rtol=0.0,
    )


def test_experimental_shape_and_texture_uniform_t_objectives_are_unregistered(tiny_trellis2_full_pipeline):
    pipeline = tiny_trellis2_full_pipeline
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2]], dtype=torch.int64)
    clean = torch.tensor([[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5], [0.25, 0.5, 0.75, 1.0]])
    shape = TrellisSparseTensor(coordinates, clean)
    texture = TrellisSparseTensor(coordinates, clean.flip(1))
    noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    timesteps = torch.tensor([0.25, 0.75])
    controls = {
        "images": torch.zeros(2, 3, 8, 8),
        "noise": noise,
        "timesteps": timesteps,
        "condition_dropout_mask": torch.tensor([False, True]),
    }

    shape_recipe = Trellis2ShapeSLatFlowRecipe(pipeline)
    shape_batch = Trellis2SLatBatch(normalized_slat=shape, **controls)
    shape_output = shape_recipe.compute_loss(shape_batch)
    assert torch.isfinite(shape_output.loss)

    texture_recipe = Trellis2TextureSLatFlowRecipe(pipeline)
    texture_batch = Trellis2TextureSLatBatch(
        normalized_texture_slat=texture,
        normalized_shape_slat=shape,
        **controls,
    )
    texture_output = texture_recipe.compute_loss(texture_batch)
    assert torch.isfinite(texture_output.loss)
    registered = {registration.recipe_type for registration in _TRAINING_RECIPE_REGISTRY}
    assert Trellis2ShapeSLatFlowRecipe not in registered
    assert Trellis2TextureSLatFlowRecipe not in registered
