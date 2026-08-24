# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

import json

import pytest
import torch
from diffusers import DDIMScheduler

import diffusers_3d
from diffusers_3d import (
    AutoPipelineForImageTo3D,
    FullFineTune,
    Hunyuan3DDinov2Conditioner,
    Hunyuan3DImageToShapePipeline,
    Hunyuan3DShapeDiTModel,
    Hunyuan3DShapeFlowMatchingRecipe,
    Hunyuan3DShapeVAE,
    ImageCondition,
    LoRAFineTune,
    MeshAsset,
    Object3DExample,
    Object3DRegistrationError,
    Object3DTrainer,
    TrainingConfig3D,
)
from diffusers_3d._validation_family.models import (
    ContractReferenceDenoiser,
    ContractReferenceMeshDecoder,
)
from diffusers_3d._validation_family.pipeline import ContractReferencePipeline
from diffusers_3d._validation_family.training import (
    CONTRACT_REFERENCE_DENOISER_POLICY,
    ContractReferenceBatch,
    ContractReferenceDataset,
    ContractReferenceRecipe,
)
from diffusers_3d.execution.registry import _MODEL_REGISTRY, _PIPELINE_REGISTRY
from diffusers_3d.training import TRAINING_ADAPTER_NAME
from diffusers_3d.training.registry import _TRAINING_RECIPE_REGISTRY


class UnregisteredContractReferencePipeline(ContractReferencePipeline):
    pass


def make_pipeline(seed: int = 0) -> ContractReferencePipeline:
    torch.manual_seed(seed)
    return ContractReferencePipeline(
        denoiser=ContractReferenceDenoiser(),
        mesh_decoder=ContractReferenceMeshDecoder(),
        scheduler=DDIMScheduler(num_train_timesteps=10, clip_sample=False),
    )


def make_condition() -> ImageCondition:
    image = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48
    return ImageCondition(image=image)


def make_training_config(tmp_path, **kwargs) -> TrainingConfig3D:
    return TrainingConfig3D(
        base_model="tests/contract-reference",
        output_dir=tmp_path,
        cpu=True,
        shuffle=False,
        max_train_steps=1,
        learning_rate=1e-2,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("model_type", "config_name", "config_value"),
    [
        (ContractReferenceDenoiser, "condition_dim", 3),
        (ContractReferenceMeshDecoder, "num_vertices", 3),
    ],
)
def test_model_config_and_weights_roundtrip(tmp_path, model_type, config_name, config_value):
    torch.manual_seed(0)
    model = model_type()
    model.save_pretrained(tmp_path)
    loaded = model_type.from_pretrained(tmp_path)

    assert type(loaded) is model_type
    assert getattr(loaded.config, config_name) == config_value
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(actual, expected)


def test_mesh_decoder_is_tensor_native_and_differentiable():
    decoder = ContractReferenceMeshDecoder()
    latents = torch.randn(2, 9, requires_grad=True)

    vertices = decoder(latents)
    vertices.square().mean().backward()

    assert vertices.shape == (2, 3, 3)
    assert latents.grad is not None
    assert decoder.vertex_projection.weight.grad is not None
    assert decoder.faces.dtype is torch.int64


def test_explicit_stages_and_deterministic_object_output():
    pipeline = make_pipeline()
    condition = make_condition()

    conditioning = pipeline.encode_conditioning(condition)
    initial_latents = pipeline.prepare_latents(
        conditioning,
        generator=torch.Generator().manual_seed(7),
    )
    denoised_latents = pipeline.denoise_latents(
        initial_latents.clone(),
        conditioning,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(7),
    )
    meshes = pipeline.decode_mesh(denoised_latents)
    first = pipeline(condition, num_inference_steps=2, generator=torch.Generator().manual_seed(7))
    second = pipeline(condition, num_inference_steps=2, generator=torch.Generator().manual_seed(7))

    assert conditioning.shape == (1, 3)
    assert initial_latents.shape == denoised_latents.shape == (1, 9)
    assert len(meshes) == 1 and type(meshes[0]) is MeshAsset
    assert type(first.objects[0]) is MeshAsset
    torch.testing.assert_close(first.objects[0].vertices, second.objects[0].vertices)
    torch.testing.assert_close(first.latents, second.latents)


def test_pipeline_save_load_and_config_first_auto_dispatch(tmp_path):
    pipeline = make_pipeline()
    condition = make_condition()
    expected = pipeline(condition, generator=torch.Generator().manual_seed(11))
    pipeline.save_pretrained(tmp_path)

    model_index = json.loads((tmp_path / pipeline.config_name).read_text())
    assert model_index["_class_name"] == "ContractReferencePipeline"
    loaded = ContractReferencePipeline.from_pretrained(tmp_path, local_files_only=True)
    auto_loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)

    assert type(loaded) is ContractReferencePipeline
    assert type(auto_loaded) is ContractReferencePipeline
    actual = auto_loaded(condition, generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(actual.objects[0].vertices, expected.objects[0].vertices)


def test_internal_registries_are_exact_populated_and_immutable():
    assert _MODEL_REGISTRY.frozen and _PIPELINE_REGISTRY.frozen and _TRAINING_RECIPE_REGISTRY.frozen
    assert {registration.model_class for registration in _MODEL_REGISTRY} == {
        ContractReferenceDenoiser,
        ContractReferenceMeshDecoder,
        Hunyuan3DDinov2Conditioner,
        Hunyuan3DShapeDiTModel,
        Hunyuan3DShapeVAE,
    }
    assert {registration.pipeline_class for registration in _PIPELINE_REGISTRY} == {
        ContractReferencePipeline,
        Hunyuan3DImageToShapePipeline,
    }
    assert {registration.recipe_type for registration in _TRAINING_RECIPE_REGISTRY} == {
        ContractReferenceRecipe,
        Hunyuan3DShapeFlowMatchingRecipe,
    }
    assert (
        _PIPELINE_REGISTRY.resolve(
            ContractReferencePipeline.object3d_model_index(),
            "image-to-3d",
        )
        is ContractReferencePipeline
    )
    registration = _TRAINING_RECIPE_REGISTRY.resolve(
        ContractReferenceRecipe,
        ContractReferencePipeline,
        "contract-reference",
    )
    assert registration.component_policies == (CONTRACT_REFERENCE_DENOISER_POLICY,)
    assert not hasattr(diffusers_3d, "ContractReferencePipeline")


def test_unregistered_subclass_rejection_still_occurs_config_first(tmp_path):
    UnregisteredContractReferencePipeline.object3d_model_index().save_pretrained(tmp_path)
    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_typed_dataset_and_full_one_step_checkpoint_resume(tmp_path):
    dataset = ContractReferenceDataset()
    example = dataset[0]
    assert type(example) is Object3DExample
    assert type(example.condition) is ImageCondition

    pipeline = make_pipeline()
    recipe = ContractReferenceRecipe(pipeline)
    batch = recipe.collate((dataset[0], dataset[1]))
    assert type(batch) is ContractReferenceBatch

    trainer = Object3DTrainer(
        recipe,
        dataset,
        FullFineTune(("denoiser",)),
        make_training_config(tmp_path),
    ).prepare()
    optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in trainer.trainable_parameters}
    assert all(name.startswith("denoiser.") for name in trainer.trainable_parameter_names)
    assert all(not parameter.requires_grad for parameter in pipeline.mesh_decoder.parameters())

    outputs = trainer.train()
    assert len(outputs) == 1 and trainer.optimizer_steps == 1
    checkpoint_manifest = trainer.save_checkpoint()
    assert checkpoint_manifest.is_file()
    assert (tmp_path / "denoiser" / "diffusion_pytorch_model.safetensors").is_file()
    saved_projection = pipeline.denoiser.projection.weight.detach().clone()

    with torch.no_grad():
        pipeline.denoiser.projection.weight.add_(10)
    trainer.load_checkpoint(tmp_path)
    torch.testing.assert_close(pipeline.denoiser.projection.weight, saved_projection)
    assert trainer.validate_resume(tmp_path) == trainer.manifest


def test_lora_exact_target_one_step_optimizer_audit_and_checkpoint(tmp_path):
    pytest.importorskip("peft")
    pipeline = make_pipeline()
    trainer = Object3DTrainer(
        ContractReferenceRecipe(pipeline),
        ContractReferenceDataset(),
        LoRAFineTune(("denoiser",), rank=2, alpha=2),
        make_training_config(tmp_path),
    ).prepare()

    assert set(pipeline.denoiser.peft_config[TRAINING_ADAPTER_NAME].target_modules) == {"projection"}
    assert trainer.trainable_parameter_names
    assert all(
        name.startswith("denoiser.projection.")
        and TRAINING_ADAPTER_NAME in name
        and (".lora_A." in name or ".lora_B." in name)
        for name in trainer.trainable_parameter_names
    )
    optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in trainer.trainable_parameters}

    outputs = trainer.train()
    assert len(outputs) == 1 and trainer.optimizer_steps == 1
    trainer.save_checkpoint()
    assert (tmp_path / "denoiser" / "pytorch_lora_weights.safetensors").is_file()
    saved_adapter_parameters = {
        name: parameter.detach().clone()
        for name, parameter in pipeline.denoiser.named_parameters()
        if TRAINING_ADAPTER_NAME in name
    }
    with torch.no_grad():
        for name, parameter in pipeline.denoiser.named_parameters():
            if TRAINING_ADAPTER_NAME in name:
                parameter.add_(10)
    trainer.load_checkpoint(tmp_path)
    for name, parameter in pipeline.denoiser.named_parameters():
        if name in saved_adapter_parameters:
            torch.testing.assert_close(parameter, saved_adapter_parameters[name])
