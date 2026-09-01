from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    FullFineTune,
    ImageCondition,
    Object3DTrainer,
    SparseVoxelAsset,
    TrainingConfig3D,
    TrainingManifest3D,
    TrainingTargetError,
    TrellisSLatBatch,
    TrellisSLatExample,
    TrellisSLatFlowRecipe,
    TrellisSparseStructureBatch,
    TrellisSparseStructureExample,
    TrellisSparseStructureFlowRecipe,
    TrellisSparseTensor,
    preprocess_training_image_condition,
)
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY

pytestmark = pytest.mark.integration


def test_exact_training_examples_reject_out_of_range_condition_pixels():
    condition = ImageCondition(torch.full((3, 8, 8), 1.01))
    normalized_slat = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0]], dtype=torch.int64),
        features=torch.zeros(1, 4),
        voxel_size=1.0,
    )

    with pytest.raises(TrainingTargetError, match=r"\[0, 1\]"):
        TrellisSparseStructureExample(
            condition=condition,
            sparse_structure_latents=torch.zeros(2, 4, 4, 4),
        )
    with pytest.raises(TrainingTargetError, match=r"\[0, 1\]"):
        TrellisSLatExample(condition=condition, normalized_slat=normalized_slat)


def test_recipe_collators_preprocess_rgba_and_separate_masks(tiny_trellis_full_pipeline):
    rgba = torch.zeros(4, 10, 12)
    rgba[0] = 1
    rgba[3, 2:8, :5] = 1
    conditions = (
        ImageCondition(rgba),
        ImageCondition(rgba[:3], mask=rgba[3:4]),
    )
    examples = tuple(
        TrellisSparseStructureExample(
            condition=condition,
            sparse_structure_latents=torch.zeros(2, 4, 4, 4),
        )
        for condition in conditions
    )

    sparse_batch = TrellisSparseStructureFlowRecipe(tiny_trellis_full_pipeline).collate(examples)
    normalized_slat = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 0, 0]], dtype=torch.int64),
        features=torch.zeros(1, 4),
        voxel_size=1.0,
    )
    slat_batch = TrellisSLatFlowRecipe(tiny_trellis_full_pipeline).collate(
        tuple(TrellisSLatExample(condition=condition, normalized_slat=normalized_slat) for condition in conditions)
    )
    expected = torch.stack(
        [
            preprocess_training_image_condition(condition, image_size=8, foreground_scale=1.2).image
            for condition in conditions
        ]
    )

    torch.testing.assert_close(sparse_batch.images, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(slat_batch.images, expected, atol=0.0, rtol=0.0)


def _pipeline_parameter_flags(pipeline):
    return {
        f"{component_name}.{parameter_name}": parameter.requires_grad
        for component_name in ("conditioner", "sparse_structure_flow_model", "sparse_structure_decoder")
        for parameter_name, parameter in getattr(pipeline, component_name).named_parameters()
    }


def test_released_sparse_structure_objective_and_frozen_components(tiny_trellis_pipeline):
    pipeline = tiny_trellis_pipeline
    before = _pipeline_parameter_flags(pipeline)
    recipe = TrellisSparseStructureFlowRecipe(pipeline)
    assert _pipeline_parameter_flags(pipeline) == before

    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 4 * 4).reshape(2, 2, 4, 4, 4)
    noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    timesteps = torch.tensor([0.25, 0.75])
    batch = TrellisSparseStructureBatch(
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


def test_sparse_structure_compute_loss_accepts_accelerator_style_wrapper(tiny_trellis_pipeline):
    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    pipeline = tiny_trellis_pipeline
    recipe = TrellisSparseStructureFlowRecipe(pipeline)
    recipe.validate_target()
    pipeline.sparse_structure_flow_model = Wrapper(pipeline.sparse_structure_flow_model)
    clean = torch.zeros(1, 2, 4, 4, 4)
    batch = TrellisSparseStructureBatch(
        images=torch.zeros(1, 3, 8, 8),
        sparse_structure_latents=clean,
        noise=torch.ones_like(clean),
        timesteps=torch.tensor([0.5]),
        condition_dropout_mask=torch.tensor([False]),
    )

    assert recipe.compute_loss(batch).loss.ndim == 0


def test_sparse_structure_recipe_registration_collation_and_full_trainer_checkpoint(
    tmp_path,
    tiny_trellis_pipeline,
    tiny_trellis_latent_dataset,
):
    recipe = TrellisSparseStructureFlowRecipe(tiny_trellis_pipeline)
    registration = _TRAINING_RECIPE_REGISTRY.validate(recipe)
    assert registration.recipe_type is TrellisSparseStructureFlowRecipe
    assert registration.example_type is TrellisSparseStructureExample
    examples = tuple(tiny_trellis_latent_dataset[index] for index in range(2))
    batch = recipe.collate(examples)
    assert batch.sparse_structure_latents.shape == (2, 2, 4, 4, 4)

    trainer = Object3DTrainer(
        recipe,
        tiny_trellis_latent_dataset,
        FullFineTune(("sparse_structure_flow_model",)),
        TrainingConfig3D(
            base_model="tests/tiny-trellis",
            revision="tiny-reference",
            dataset_fingerprint="tests/tiny-trellis-latents-v1",
            output_dir=tmp_path,
            train_batch_size=2,
            max_train_steps=1,
            learning_rate=1e-3,
            shuffle=False,
            seed=7,
            cpu=True,
        ),
    ).prepare()
    assert all(parameter.requires_grad for parameter in tiny_trellis_pipeline.sparse_structure_flow_model.parameters())
    assert not any(parameter.requires_grad for parameter in tiny_trellis_pipeline.conditioner.parameters())
    assert not any(
        parameter.requires_grad for parameter in tiny_trellis_pipeline.sparse_structure_decoder.parameters()
    )
    for component in (
        tiny_trellis_pipeline.conditioner,
        tiny_trellis_pipeline.sparse_structure_decoder,
    ):
        assert not component.training
        assert all(parameter.device == trainer.accelerator.device for parameter in component.parameters())
    assert all(name.startswith("sparse_structure_flow_model.") for name in trainer.trainable_parameter_names)

    summary = trainer.train()
    assert summary.final_loss is not None and trainer.optimizer_steps == 1
    checkpoint = trainer.save_checkpoint()
    assert checkpoint.is_file()
    assert TrainingManifest3D.load(tmp_path) == trainer.manifest
    saved_weight = tiny_trellis_pipeline.sparse_structure_flow_model.out_layer.weight.detach().clone()
    saved_conditioner_weight = next(tiny_trellis_pipeline.conditioner.parameters()).detach().clone()
    saved_decoder_weight = next(tiny_trellis_pipeline.sparse_structure_decoder.parameters()).detach().clone()
    with torch.no_grad():
        tiny_trellis_pipeline.sparse_structure_flow_model.out_layer.weight.add_(10.0)
        next(tiny_trellis_pipeline.conditioner.parameters()).add_(10.0)
        next(tiny_trellis_pipeline.sparse_structure_decoder.parameters()).add_(10.0)
    trainer.load_checkpoint(tmp_path)
    torch.testing.assert_close(
        tiny_trellis_pipeline.sparse_structure_flow_model.out_layer.weight,
        saved_weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        next(tiny_trellis_pipeline.conditioner.parameters()),
        saved_conditioner_weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        next(tiny_trellis_pipeline.sparse_structure_decoder.parameters()),
        saved_decoder_weight,
        atol=0.0,
        rtol=0.0,
    )


def test_experimental_slat_flow_objective_matches_released_sparse_equation(tiny_trellis_full_pipeline):
    pipeline = tiny_trellis_full_pipeline
    recipe = TrellisSLatFlowRecipe(pipeline)
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2]], dtype=torch.int64)
    clean_features = torch.tensor([[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5], [0.25, 0.5, 0.75, 1.0]])
    sparse = TrellisSparseTensor(coordinates, clean_features)
    noise = torch.linspace(1.0, -1.0, clean_features.numel()).reshape_as(clean_features)
    timesteps = torch.tensor([0.25, 0.75])
    batch = TrellisSLatBatch(
        images=torch.zeros(2, 3, 8, 8),
        normalized_slat=sparse,
        noise=noise,
        timesteps=timesteps,
        condition_dropout_mask=torch.tensor([False, True]),
    )
    captured = {}

    def capture_input(_module, args):
        captured["input"] = args[0].features
        captured["timesteps"] = args[1]

    hook = pipeline.slat_flow_model.register_forward_pre_hook(capture_input)
    output = recipe.compute_loss(batch)
    hook.remove()
    per_voxel_timestep = timesteps[coordinates[:, 0]].unsqueeze(1)
    expected_input = (1 - per_voxel_timestep) * clean_features + (1e-5 + (1 - 1e-5) * per_voxel_timestep) * noise
    expected_target = (1 - 1e-5) * noise - clean_features
    torch.testing.assert_close(captured["input"], expected_input, atol=0.0, rtol=0.0)
    torch.testing.assert_close(captured["timesteps"], timesteps * 1000, atol=0.0, rtol=0.0)
    prediction = pipeline.slat_flow_model(
        sparse.replace(expected_input),
        timesteps * 1000,
        pipeline.conditioner.unconditional_embedding(2),
    ).sample.features
    torch.testing.assert_close(output.loss, torch.nn.functional.mse_loss(prediction, expected_target))
    assert all(registration.recipe_type is not TrellisSLatFlowRecipe for registration in _TRAINING_RECIPE_REGISTRY)
