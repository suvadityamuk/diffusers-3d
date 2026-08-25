from __future__ import annotations

import random
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from torch import nn

import diffusers_3d.training.trainer as trainer_module
from diffusers_3d import (
    ACCELERATOR_STATE_DIRECTORY,
    FROZEN_COMPONENT_STATE_DIRECTORY,
    TRAINER_STATE_NAME,
    ComponentPolicy,
    ContributionStatus,
    FineTuneKind,
    FrozenComponentPolicy,
    FullFineTune,
    LoRAFineTune,
    MeshAsset,
    Object3DExample,
    Object3DModel,
    Object3DPipeline,
    Object3DTrainer,
    ReviewStatus,
    TextCondition,
    TrainableParameterError,
    TrainingCheckpointError,
    TrainingConfig3D,
    TrainingConfigurationError,
    TrainingManifestMismatchError,
    TrainingPolicyError,
    TrainingRecipe3D,
    TrainingRecipeRegistration,
    TrainingRegistrationError,
    TrainingStep3DOutput,
    TrainingTargetError,
    create_training_recipe_registry,
)


@dataclass(frozen=True, slots=True)
class TinyBatch:
    inputs: torch.Tensor
    labels: torch.Tensor

    def validate(self) -> None:
        if self.inputs.ndim != 2 or self.labels.shape != self.inputs.shape:
            raise ValueError("inputs and labels must have equal rank-two shapes")
        if not self.inputs.is_floating_point() or not self.labels.is_floating_point():
            raise ValueError("inputs and labels must be floating point")

    def to(self, device=None, dtype=None, non_blocking=False):
        return type(self)(
            self.inputs.to(device=device, dtype=dtype, non_blocking=non_blocking),
            self.labels.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight + self.bias


class OtherTinyBlock(TinyBlock):
    pass


class TinyFrozenBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale


class TinyTarget(Object3DModel):
    family_id = "tiny-training"
    component_role = "denoiser"
    supported_object_kinds = ()
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.block = TinyBlock()
        self.conditioner = TinyFrozenBlock()
        self.other = nn.Parameter(torch.tensor(3.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


FULL_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="block",
    expected_types=(TinyBlock,),
    supported_strategies=(FineTuneKind.FULL,),
    full_parameter_names=("weight",),
)
FROZEN_POLICY = FrozenComponentPolicy(
    component_path="conditioner",
    expected_types=(TinyFrozenBlock,),
)


class TinyRecipe(TrainingRecipe3D[TinyTarget, Object3DExample, TinyBatch]):
    recipe_id = "tiny-objective"
    recipe_version = "1.0"
    family_id = "tiny-training"
    target_type = TinyTarget
    example_type = Object3DExample
    batch_type = TinyBatch
    component_policies = (FULL_POLICY,)
    frozen_component_policies = (FROZEN_POLICY,)

    def collate(self, examples):
        inputs = torch.stack([example.target.vertices[0, :1] for example in examples])
        return TinyBatch(inputs=inputs, labels=inputs * 2.0)

    def validate_target(self) -> None:
        if self.target.block.weight.ndim != 0:
            raise ValueError("weight must be scalar")
        if type(self.target.conditioner) is not TinyFrozenBlock:
            raise ValueError("conditioner must be exact")

    def compute_loss(self, batch: TinyBatch) -> TrainingStep3DOutput:
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.inputs)
        prediction = self.target.block(batch.inputs) + conditioning * 0
        loss = torch.nn.functional.mse_loss(prediction, batch.labels)
        return TrainingStep3DOutput(loss=loss, metrics={"prediction_mean": prediction.mean()})

    def save_weights(self, save_directory, strategy, components) -> None:
        del strategy
        component_directory = Path(save_directory) / "denoiser"
        component_directory.mkdir(parents=True, exist_ok=True)
        torch.save(components["denoiser"].state_dict(), component_directory / "pytorch_model.bin")


class StochasticTinyRecipe(TinyRecipe):
    recipe_id = "stochastic-tiny-objective"
    recipe_version = "1.0"
    family_id = "tiny-training"
    target_type = TinyTarget
    example_type = Object3DExample
    batch_type = TinyBatch
    component_policies = (FULL_POLICY,)
    frozen_component_policies = (FROZEN_POLICY,)

    def compute_loss(self, batch: TinyBatch) -> TrainingStep3DOutput:
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.inputs)
        random_offset = random.random() + float(np.random.random())
        random_offset = batch.inputs.new_tensor(random_offset) + torch.rand((), device=batch.inputs.device)
        prediction = self.target.block(batch.inputs) + conditioning * 0 + random_offset
        loss = torch.nn.functional.mse_loss(prediction, batch.labels)
        return TrainingStep3DOutput(loss=loss, metrics={"random_offset": random_offset})


class TinyRecipeMarker(TinyRecipe):
    pass


class TinyTargetMarker(TinyTarget):
    pass


class CountingDataset:
    def __init__(self, values=(1.0, 2.0)) -> None:
        self.values = tuple(values)
        self.getitem_calls = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Object3DExample:
        self.getitem_calls += 1
        value = self.values[index]
        mesh = MeshAsset(
            vertices=torch.tensor([[value, 0.0, 0.0], [value + 1.0, 0.0, 0.0], [value, 1.0, 0.0]]),
            faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        )
        return Object3DExample(mesh, TextCondition("tiny"))


class Object3DExampleMarker(Object3DExample):
    pass


class SubclassExampleDataset(CountingDataset):
    def __getitem__(self, index: int) -> Object3DExampleMarker:
        example = super().__getitem__(index)
        return Object3DExampleMarker(
            target=example.target,
            condition=example.condition,
            example_id=example.example_id,
        )


class NeverIteratedDataset:
    def __init__(self) -> None:
        self.getitem_calls = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Object3DExample:
        self.getitem_calls += 1
        raise AssertionError(f"dataset was iterated at {index}")


def make_registration(
    recipe_type=TinyRecipe,
    target_type=TinyTarget,
    *,
    review_status=ReviewStatus.REVIEWED,
) -> TrainingRecipeRegistration:
    return TrainingRecipeRegistration(
        recipe_type=recipe_type,
        target_type=target_type,
        example_type=recipe_type.example_type,
        batch_type=recipe_type.batch_type,
        recipe_id=recipe_type.recipe_id,
        recipe_version=recipe_type.recipe_version,
        family_id=recipe_type.family_id,
        component_policies=recipe_type.component_policies,
        review_status=review_status,
        frozen_component_policies=recipe_type.frozen_component_policies,
    )


def install_registry(monkeypatch, *registrations):
    registry = create_training_recipe_registry(registrations).freeze()
    monkeypatch.setattr(Object3DTrainer, "_registry", registry)
    return registry


def make_config(**kwargs) -> TrainingConfig3D:
    kwargs.setdefault("dataset_fingerprint", "tests/tiny-dataset-v1")
    return TrainingConfig3D(base_model="tests/tiny-object-3d", cpu=True, shuffle=False, **kwargs)


def test_registry_is_reviewed_exact_and_read_only():
    registration = make_registration()
    registry = create_training_recipe_registry((registration,))

    assert registry.resolve(TinyRecipe, TinyTarget, "tiny-objective") is registration
    with pytest.raises(TrainingRegistrationError):
        registry.validate(TinyRecipeMarker(TinyTarget()))
    with pytest.raises(TrainingRegistrationError):
        registry.validate(TinyRecipe(TinyTargetMarker()))
    with pytest.raises(TrainingRegistrationError, match="reviewed"):
        create_training_recipe_registry((make_registration(review_status=ReviewStatus.UNREVIEWED),))

    registry.freeze()
    with pytest.raises(TrainingRegistrationError, match="read-only"):
        registry.register(registration)


def test_prepare_moves_declared_frozen_components_to_device_and_eval(monkeypatch):
    install_registry(monkeypatch, make_registration())
    target = TinyTarget()
    target.conditioner.train()

    trainer = Object3DTrainer(
        TinyRecipe(target),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(),
    ).prepare()

    conditioner = trainer.frozen_components["conditioner"]
    assert conditioner is target.conditioner
    assert not conditioner.training
    assert all(parameter.device == trainer.accelerator.device for parameter in conditioner.parameters())
    assert all(not parameter.requires_grad for parameter in conditioner.parameters())
    assert all(
        id(parameter) not in {id(item) for item in trainer.trainable_parameters}
        for parameter in conditioner.parameters()
    )


def test_compute_loss_accepts_a_wrapped_selected_component_after_exact_validation():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    target = TinyTarget()
    recipe = TinyRecipe(target)
    recipe.validate_target()
    target.block = Wrapper(target.block)
    batch = TinyBatch(inputs=torch.ones(1, 1), labels=torch.full((1, 1), 2.0))

    output = recipe.compute_loss(batch)

    assert output.loss.ndim == 0


def test_registration_and_collate_require_exact_non_mapping_example_type(monkeypatch):
    class MappingExample(dict):
        def validate(self):
            pass

        def to(self, *args, **kwargs):
            return self

    class MappingExampleRecipe(TinyRecipe):
        recipe_id = "mapping-example"
        example_type = MappingExample

    class ImplicitExampleRecipe(TinyRecipe):
        recipe_id = "implicit-example"

    with pytest.raises(TrainingRegistrationError, match="example_type.*not a mapping"):
        create_training_recipe_registry((make_registration(MappingExampleRecipe, TinyTarget),))
    with pytest.raises(TrainingRegistrationError, match="declare.*example_type"):
        create_training_recipe_registry((make_registration(ImplicitExampleRecipe, TinyTarget),))

    registration = make_registration()
    registry = install_registry(monkeypatch, registration)
    with monkeypatch.context() as context:
        context.setattr(TinyRecipe, "example_type", Object3DExampleMarker)
        with pytest.raises(TrainingRegistrationError, match="example_type"):
            registry.resolve(TinyRecipe, TinyTarget, TinyRecipe.recipe_id)

    trainer = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        SubclassExampleDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1),
    )
    with pytest.raises(TrainingConfigurationError, match="exact Object3DExample"):
        trainer.train()


class GenericRecipe(TrainingRecipe3D):
    recipe_id = "generic"
    recipe_version = "1"
    family_id = "generic"
    target_type = object
    example_type = Object3DExample
    batch_type = TinyBatch
    component_policies = (FULL_POLICY,)

    def collate(self, examples):
        raise AssertionError("collate must not be called")

    def validate_target(self):
        raise AssertionError("validate_target must not be called")

    def compute_loss(self, batch):
        raise AssertionError("compute_loss must not be called")


class GenericModel(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, value):
        return value * self.weight


@pytest.mark.parametrize(
    "target",
    [
        nn.Linear(1, 1),
        GenericModel(),
        DiffusionPipeline(),
        TinyTarget(),
    ],
)
def test_generic_and_unregistered_targets_reject_before_all_side_effects(monkeypatch, target):
    install_registry(monkeypatch)
    dataset = NeverIteratedDataset()
    original = {id(parameter): parameter.requires_grad for parameter in getattr(target, "parameters", lambda: ())()}
    monkeypatch.setattr(
        trainer_module,
        "Accelerator",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Accelerator was constructed")),
    )
    monkeypatch.setattr(
        trainer_module,
        "AdamW",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer was constructed")),
    )

    trainer = Object3DTrainer(GenericRecipe(target), dataset, FullFineTune(("denoiser",)), make_config())
    with pytest.raises(TrainingRegistrationError):
        trainer.prepare()

    assert dataset.getitem_calls == 0
    assert {
        id(parameter): parameter.requires_grad for parameter in getattr(target, "parameters", lambda: ())()
    } == original


def test_component_exact_type_and_unknown_key_reject_before_mutation_or_iteration(monkeypatch):
    install_registry(monkeypatch, make_registration())
    target = TinyTarget()
    target.block = OtherTinyBlock()
    dataset = NeverIteratedDataset()
    original = {name: parameter.requires_grad for name, parameter in target.named_parameters()}
    with pytest.raises(TrainingTargetError, match="exact type"):
        Object3DTrainer(TinyRecipe(target), dataset, FullFineTune(("denoiser",)), make_config()).prepare()
    assert dataset.getitem_calls == 0
    assert {name: parameter.requires_grad for name, parameter in target.named_parameters()} == original

    target = TinyTarget()
    with pytest.raises(TrainingPolicyError, match="Unknown"):
        Object3DTrainer(TinyRecipe(target), dataset, FullFineTune(("unknown",)), make_config()).prepare()
    assert dataset.getitem_calls == 0
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_full_preparation_uses_exact_approved_parameters_and_optimizer_ids(monkeypatch):
    install_registry(monkeypatch, make_registration())
    target = TinyTarget()
    dataset = CountingDataset()
    trainer = Object3DTrainer(TinyRecipe(target), dataset, FullFineTune(("denoiser",)), make_config()).prepare()

    assert dataset.getitem_calls == 0
    assert target.block.weight.requires_grad
    assert not target.block.bias.requires_grad
    assert not target.other.requires_grad
    assert trainer.trainable_parameter_names == ("block.weight",)
    assert {id(parameter) for parameter in trainer.trainable_parameters} == {id(target.block.weight)}
    assert {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]} == {
        id(target.block.weight)
    }
    assert trainer.manifest.trainable_parameter_names == ("block.weight",)


class SneakyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.extra = nn.Parameter(torch.ones(()))

    def named_parameters(self, *args, **kwargs):
        self.extra.requires_grad_(True)
        return super().named_parameters(*args, **kwargs)


class SneakyTarget(TinyTarget):
    family_id = "sneaky-training"

    @register_to_config
    def __init__(self) -> None:
        Object3DModel.__init__(self)
        self.block = SneakyBlock()


SNEAKY_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="block",
    expected_types=(SneakyBlock,),
    supported_strategies=(FineTuneKind.FULL,),
    full_parameter_names=("weight",),
)


class SneakyRecipe(TinyRecipe):
    recipe_id = "sneaky-objective"
    family_id = "sneaky-training"
    target_type = SneakyTarget
    example_type = Object3DExample
    component_policies = (SNEAKY_POLICY,)
    frozen_component_policies = ()

    def validate_target(self) -> None:
        pass


class ZeroBlock(nn.Module):
    pass


class ZeroTarget(TinyTarget):
    family_id = "zero-training"

    @register_to_config
    def __init__(self) -> None:
        Object3DModel.__init__(self)
        self.block = ZeroBlock()
        self.other = nn.Parameter(torch.ones(()))


ZERO_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="block",
    expected_types=(ZeroBlock,),
    supported_strategies=(FineTuneKind.FULL,),
)


class ZeroRecipe(TinyRecipe):
    recipe_id = "zero-objective"
    family_id = "zero-training"
    target_type = ZeroTarget
    example_type = Object3DExample
    component_policies = (ZERO_POLICY,)
    frozen_component_policies = ()

    def validate_target(self) -> None:
        pass


@pytest.mark.parametrize(
    ("recipe_type", "target_type", "message"),
    [
        (SneakyRecipe, SneakyTarget, "exactly match"),
        (ZeroRecipe, ZeroTarget, "zero trainable"),
    ],
)
def test_trainable_audit_rejects_unexpected_and_zero_parameters(
    monkeypatch,
    recipe_type,
    target_type,
    message,
):
    install_registry(monkeypatch, make_registration(recipe_type, target_type))
    target = target_type()
    original = {name: parameter.requires_grad for name, parameter in target.named_parameters()}
    dataset = NeverIteratedDataset()
    with pytest.raises(TrainableParameterError, match=message):
        Object3DTrainer(recipe_type(target), dataset, FullFineTune(("denoiser",)), make_config()).prepare()
    assert dataset.getitem_calls == 0
    assert {name: parameter.requires_grad for name, parameter in target.named_parameters()} == original


class TinyTrainingPipeline(Object3DPipeline):
    family_id = "tiny-pipeline-training"

    def __init__(self) -> None:
        super().__init__()
        self.register_modules(block=TinyBlock())
        self.inference_calls = 0

    def __call__(self, *args, **kwargs):
        self.inference_calls += 1
        raise AssertionError("inference pipeline __call__ must not be used for training")


PIPELINE_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="block",
    expected_types=(TinyBlock,),
    supported_strategies=(FineTuneKind.FULL,),
    full_parameter_names=("weight",),
)


class TinyPipelineRecipe(TinyRecipe):
    recipe_id = "tiny-pipeline-objective"
    family_id = "tiny-pipeline-training"
    target_type = TinyTrainingPipeline
    example_type = Object3DExample
    component_policies = (PIPELINE_POLICY,)
    frozen_component_policies = ()

    def validate_target(self) -> None:
        pass

    def compute_loss(self, batch: TinyBatch) -> TrainingStep3DOutput:
        prediction = self.target.block(batch.inputs)
        return TrainingStep3DOutput(torch.nn.functional.mse_loss(prediction, batch.labels))


def test_pipeline_training_uses_recipe_loss_not_inference_call(monkeypatch):
    registration = make_registration(TinyPipelineRecipe, TinyTrainingPipeline)
    install_registry(monkeypatch, registration)
    pipeline = TinyTrainingPipeline()
    trainer = Object3DTrainer(
        TinyPipelineRecipe(pipeline),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1),
    )

    summary = trainer.train()

    assert summary.optimizer_steps == 1
    assert summary.final_loss is not None
    assert pipeline.inference_calls == 0
    assert trainer.optimizer_steps == 1


def test_deterministic_full_train_step(monkeypatch):
    install_registry(monkeypatch, make_registration())
    first_target = TinyTarget()
    second_target = TinyTarget()
    first = Object3DTrainer(
        TinyRecipe(first_target),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1, learning_rate=1e-2),
    )
    second = Object3DTrainer(
        TinyRecipe(second_target),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1, learning_rate=1e-2),
    )

    first_summary = first.train()
    second_summary = second.train()

    torch.testing.assert_close(first_target.block.weight, second_target.block.weight)
    assert first_summary.final_loss == pytest.approx(second_summary.final_loss)
    assert first_target.block.weight.item() != pytest.approx(0.5)


def test_gradient_accumulation_runs_a_bounded_number_of_updates(monkeypatch):
    install_registry(monkeypatch, make_registration())
    trainer = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1, gradient_accumulation_steps=2),
    )

    summary = trainer.train()

    assert trainer.optimizer_steps == 1
    assert summary.micro_steps in (1, 2)
    assert summary.final_loss is not None


def test_checkpoint_restores_full_state_counters_and_next_data_position(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration())
    config = make_config(
        max_train_steps=2,
        learning_rate=1e-2,
        lr_scheduler="linear",
        output_dir=str(tmp_path),
    )

    uninterrupted_target = TinyTarget()
    uninterrupted = Object3DTrainer(
        TinyRecipe(uninterrupted_target),
        CountingDataset((1.0, 2.0, 3.0)),
        FullFineTune(("denoiser",)),
        config,
    )
    uninterrupted.train(max_optimizer_steps=1)
    expected_summary = uninterrupted.train(max_optimizer_steps=1)
    expected_weight = uninterrupted_target.block.weight.detach().clone()
    expected_optimizer_state = uninterrupted.optimizer.state_dict()

    interrupted_target = TinyTarget()
    interrupted = Object3DTrainer(
        TinyRecipe(interrupted_target),
        CountingDataset((1.0, 2.0, 3.0)),
        FullFineTune(("denoiser",)),
        config,
    )
    interrupted.train(max_optimizer_steps=1)
    interrupted.save_checkpoint()

    resumed_dataset = CountingDataset((1.0, 2.0, 3.0))
    resumed_target = TinyTarget()
    resumed_target.conditioner.scale.data.fill_(7.0)
    resumed = Object3DTrainer(
        TinyRecipe(resumed_target),
        resumed_dataset,
        FullFineTune(("denoiser",)),
        config,
    )
    resumed.load_checkpoint(tmp_path)

    assert resumed.micro_steps == 1
    assert resumed.optimizer_steps == 1
    assert resumed_target.conditioner.scale.item() == 1.0
    frozen_state_directory = tmp_path / ACCELERATOR_STATE_DIRECTORY / FROZEN_COMPONENT_STATE_DIRECTORY
    assert tuple(frozen_state_directory.glob("*.safetensors"))
    assert resumed_dataset.getitem_calls == 0
    assert (tmp_path / ACCELERATOR_STATE_DIRECTORY / TRAINER_STATE_NAME).stat().st_mode & 0o044 == 0o044

    resumed_summary = resumed.train(max_optimizer_steps=1)

    assert resumed_dataset.getitem_calls == 1
    assert resumed_summary.final_loss == pytest.approx(expected_summary.final_loss, abs=0.0)
    torch.testing.assert_close(resumed_target.block.weight, expected_weight, atol=0.0, rtol=0.0)
    assert resumed.optimizer.state_dict() == expected_optimizer_state


def test_checkpoint_requires_accumulation_boundary_and_resumes_after_one(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration())
    config = make_config(
        max_train_steps=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-2,
        max_grad_norm=100.0,
        output_dir=str(tmp_path),
    )
    values = (1.0, 2.0, 3.0, 4.0)

    uninterrupted_target = TinyTarget()
    uninterrupted = Object3DTrainer(
        TinyRecipe(uninterrupted_target),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    )
    uninterrupted.train(max_optimizer_steps=1)
    expected_summary = uninterrupted.train(max_optimizer_steps=1)
    expected_weight = uninterrupted_target.block.weight.detach().clone()
    expected_optimizer_state = uninterrupted.optimizer.state_dict()

    interrupted_target = TinyTarget()
    interrupted = Object3DTrainer(
        TinyRecipe(interrupted_target),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    ).prepare()
    interrupted.train_step(interrupted._next_batch())
    with pytest.raises(TrainingCheckpointError, match="synchronized optimizer-step boundary"):
        interrupted.save_checkpoint()
    interrupted.train_step(interrupted._next_batch())
    assert interrupted.optimizer_steps == 1
    interrupted.save_checkpoint()

    resumed_target = TinyTarget()
    resumed_target.conditioner.scale.data.fill_(7.0)
    resumed = Object3DTrainer(
        TinyRecipe(resumed_target),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    )
    resumed.load_checkpoint(tmp_path)
    assert resumed_target.conditioner.scale.item() == 1.0

    resumed_summary = resumed.train(max_optimizer_steps=1)

    assert resumed_summary.final_loss == pytest.approx(expected_summary.final_loss, abs=0.0)
    torch.testing.assert_close(resumed_target.block.weight, expected_weight, atol=0.0, rtol=0.0)
    assert resumed.optimizer.state_dict() == expected_optimizer_state


def test_stochastic_checkpoint_continuation_restores_all_rng_state(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration(StochasticTinyRecipe))
    config = make_config(
        max_train_steps=2,
        learning_rate=1e-2,
        max_grad_norm=100.0,
        output_dir=str(tmp_path),
    )
    values = (1.0, 2.0, 3.0)

    uninterrupted_target = TinyTarget()
    uninterrupted = Object3DTrainer(
        StochasticTinyRecipe(uninterrupted_target),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    )
    uninterrupted.train(max_optimizer_steps=1)
    expected_summary = uninterrupted.train(max_optimizer_steps=1)
    expected_weight = uninterrupted_target.block.weight.detach().clone()
    expected_optimizer_state = uninterrupted.optimizer.state_dict()

    interrupted = Object3DTrainer(
        StochasticTinyRecipe(TinyTarget()),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    )
    interrupted.train(max_optimizer_steps=1)
    interrupted.save_checkpoint()

    resumed_target = TinyTarget()
    resumed = Object3DTrainer(
        StochasticTinyRecipe(resumed_target),
        CountingDataset(values),
        FullFineTune(("denoiser",)),
        config,
    )
    resumed.load_checkpoint(tmp_path)
    resumed_summary = resumed.train(max_optimizer_steps=1)

    assert resumed_summary.final_loss == pytest.approx(expected_summary.final_loss, abs=0.0)
    assert resumed_summary.final_metrics == pytest.approx(expected_summary.final_metrics, abs=0.0)
    torch.testing.assert_close(resumed_target.block.weight, expected_weight, atol=0.0, rtol=0.0)
    assert resumed.optimizer.state_dict() == expected_optimizer_state


def _save_stochastic_checkpoint(monkeypatch, checkpoint_directory):
    install_registry(monkeypatch, make_registration(StochasticTinyRecipe))
    config = make_config(max_train_steps=2, output_dir=str(checkpoint_directory))
    trainer = Object3DTrainer(
        StochasticTinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        config,
    )
    trainer.train(max_optimizer_steps=1)
    trainer.save_checkpoint()
    return config


def test_checkpoint_rejects_deleted_rng_state(monkeypatch, tmp_path):
    config = _save_stochastic_checkpoint(monkeypatch, tmp_path)
    (tmp_path / ACCELERATOR_STATE_DIRECTORY / "random_states_0.pkl").unlink()
    resumed = Object3DTrainer(
        StochasticTinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        config,
    )

    with pytest.raises(TrainingCheckpointError, match="Required RNG state"):
        resumed.load_checkpoint(tmp_path)


def test_checkpoint_rejects_corrupt_rng_state(monkeypatch, tmp_path):
    config = _save_stochastic_checkpoint(monkeypatch, tmp_path)
    (tmp_path / ACCELERATOR_STATE_DIRECTORY / "random_states_0.pkl").write_bytes(b"not a torch checkpoint")
    resumed = Object3DTrainer(
        StochasticTinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        config,
    )

    with pytest.raises(TrainingCheckpointError, match="safely load RNG state"):
        resumed.load_checkpoint(tmp_path)


def test_checkpoint_rejects_incompatible_rng_state(monkeypatch, tmp_path):
    config = _save_stochastic_checkpoint(monkeypatch, tmp_path)
    torch.save(
        {
            "numpy_random_seed": np.random.get_state(),
            "random_state": random.getstate(),
            "step": "not-an-integer",
            "torch_manual_seed": torch.get_rng_state(),
        },
        tmp_path / ACCELERATOR_STATE_DIRECTORY / "random_states_0.pkl",
    )
    resumed = Object3DTrainer(
        StochasticTinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        config,
    )

    with pytest.raises(TrainingCheckpointError, match="step must be a non-negative integer"):
        resumed.load_checkpoint(tmp_path)


def test_exact_checkpoint_persistence_rejects_multiple_processes_before_filesystem(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration())
    checkpoint_directory = tmp_path / "not-created"
    trainer = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(max_train_steps=1, output_dir=str(checkpoint_directory)),
    ).prepare()
    require_single_process = trainer_module._require_single_process_checkpoint

    def simulate_multiple_processes(_actual_num_processes, operation):
        require_single_process(2, operation)

    monkeypatch.setattr(trainer_module, "_require_single_process_checkpoint", simulate_multiple_processes)
    with pytest.raises(TrainingCheckpointError, match="num_processes == 1"):
        trainer.save_checkpoint()
    assert not checkpoint_directory.exists()
    with pytest.raises(TrainingCheckpointError, match="num_processes == 1"):
        trainer.load_checkpoint(checkpoint_directory)
    assert not checkpoint_directory.exists()


def test_checkpoint_resume_rejects_changed_dataset_fingerprint(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration())
    config = make_config(max_train_steps=1, output_dir=str(tmp_path))
    trainer = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        config,
    )
    trainer.train()
    trainer.save_checkpoint()

    changed = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        replace(config, dataset_fingerprint="tests/different-dataset"),
    )
    with pytest.raises(TrainingManifestMismatchError, match="training_config"):
        changed.load_checkpoint(tmp_path)


def test_checkpoint_requires_a_dataset_fingerprint(monkeypatch, tmp_path):
    install_registry(monkeypatch, make_registration())
    trainer = Object3DTrainer(
        TinyRecipe(TinyTarget()),
        CountingDataset(),
        FullFineTune(("denoiser",)),
        make_config(dataset_fingerprint=None, output_dir=str(tmp_path)),
    )
    trainer.train()

    with pytest.raises(TrainingCheckpointError, match="dataset_fingerprint"):
        trainer.save_checkpoint()


class FakeAdapterWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))


class FakeProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))


class FakePeftComponent(PeftAdapterMixin, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.projection = FakeProjection()
        self.received_targets = None

    def add_adapter(self, adapter_config, adapter_name="default") -> None:
        self.received_targets = tuple(adapter_config.target_modules)
        self.projection.add_module("lora_A", nn.ModuleDict({adapter_name: FakeAdapterWeights()}))
        self.projection.add_module("lora_B", nn.ModuleDict({adapter_name: FakeAdapterWeights()}))

    def delete_adapters(self, adapter_names):
        del self.projection.lora_A
        del self.projection.lora_B

    def save_lora_adapter(self, *args, **kwargs):
        pass


class LoraTarget(Object3DModel):
    family_id = "lora-training"
    component_role = "denoiser"

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.adapter = FakePeftComponent()

    def forward(self, inputs):
        return inputs


LORA_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="adapter",
    expected_types=(FakePeftComponent,),
    supported_strategies=(FineTuneKind.LORA,),
    lora_target_modules=("projection",),
)


class LoraRecipe(TinyRecipe):
    recipe_id = "lora-objective"
    family_id = "lora-training"
    target_type = LoraTarget
    example_type = Object3DExample
    component_policies = (LORA_POLICY,)
    frozen_component_policies = ()

    def validate_target(self) -> None:
        pass


def test_lora_targets_are_recipe_owned_and_actual_adapter_params_are_audited(monkeypatch):
    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            vars(self).update(kwargs)

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(LoraConfig=FakeLoraConfig))
    install_registry(monkeypatch, make_registration(LoraRecipe, LoraTarget))
    with pytest.raises(TypeError):
        LoRAFineTune(("denoiser",), target_modules=("user.regex.*",))

    target = LoraTarget()
    trainer = Object3DTrainer(
        LoraRecipe(target),
        CountingDataset(),
        LoRAFineTune(("denoiser",), rank=2),
        make_config(),
    ).prepare()

    assert target.adapter.received_targets == ("projection",)
    assert not target.adapter.projection.weight.requires_grad
    assert trainer.trainable_parameter_names == (
        "adapter.projection.lora_A.object3d_training.weight",
        "adapter.projection.lora_B.object3d_training.weight",
    )
    assert {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]} == {
        id(target.adapter.projection.lora_A["object3d_training"].weight),
        id(target.adapter.projection.lora_B["object3d_training"].weight),
    }


def test_trainer_has_no_generic_loading_shortcut():
    assert "from_pretrained" not in vars(Object3DTrainer)
