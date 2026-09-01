from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch
from torch.utils.checkpoint import checkpoint as apply_checkpoint

from diffusers_3d import (
    MeshAsset,
    Object3DModelRegistration,
    Object3DPipelineRegistration,
    Trellis2Dinov3Conditioner,
    Trellis2FlowEulerScheduler,
    Trellis2ImageTo3DPipeline,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
    TrellisDinov2Conditioner,
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
)
from diffusers_3d.families.registrations import production_execution_registrations


@dataclass(frozen=True)
class ModelContract:
    model_type: type[torch.nn.Module]
    invoke: Callable[[torch.nn.Module, int, bool], torch.Tensor]

    @property
    def name(self) -> str:
        return self.model_type.__name__

    def make(self) -> torch.nn.Module:
        return self.model_type(**self.model_type.tiny_config())


def _parameter_properties(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters())
        return parameter.device, parameter.dtype
    for component in model.components.values():
        if isinstance(component, torch.nn.Module):
            parameter = next(component.parameters())
            return parameter.device, parameter.dtype
    raise AssertionError("pipeline has no torch module component")


def _output_tensor(output: Any, *, return_dict: bool) -> torch.Tensor:
    if not return_dict:
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    return output.embeddings


def _invoke_conditioner(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    images = torch.linspace(
        0.0,
        1.0,
        batch_size * 3 * 8 * 8,
        device=device,
        dtype=dtype,
    ).reshape(batch_size, 3, 8, 8)
    return _output_tensor(model(images, return_dict=return_dict), return_dict=return_dict)


def _invoke_trellis_flow(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    resolution = model.config.resolution
    hidden_states = torch.linspace(
        -1.0,
        1.0,
        batch_size * model.config.in_channels * resolution**3,
        device=device,
        dtype=dtype,
    ).reshape(batch_size, model.config.in_channels, resolution, resolution, resolution)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    context = torch.linspace(
        -0.5,
        0.5,
        batch_size * 7 * model.config.cond_channels,
        device=device,
        dtype=dtype,
    ).reshape(batch_size, 7, model.config.cond_channels)
    return _output_tensor(
        model(hidden_states, timesteps, context, return_dict=return_dict),
        return_dict=return_dict,
    )


def _invoke_sparse_decoder(model: TrellisSparseStructureDecoder, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    resolution = 2 if isinstance(model, Trellis2SparseStructureDecoder) else 4
    hidden_states = torch.linspace(
        -1.0,
        1.0,
        batch_size * model.latent_channels * resolution**3,
        device=device,
        dtype=dtype,
    ).reshape(batch_size, model.latent_channels, resolution, resolution, resolution)
    return _output_tensor(model(hidden_states, return_dict=return_dict), return_dict=return_dict)


MODEL_CONTRACTS = (
    ModelContract(TrellisSparseStructureFlowModel, _invoke_trellis_flow),
    ModelContract(TrellisSparseStructureDecoder, _invoke_sparse_decoder),
    ModelContract(TrellisDinov2Conditioner, _invoke_conditioner),
    ModelContract(Trellis2SparseStructureFlowModel, _invoke_trellis_flow),
    ModelContract(Trellis2SparseStructureDecoder, _invoke_sparse_decoder),
    ModelContract(Trellis2Dinov3Conditioner, _invoke_conditioner),
)


def _model_contract_id(contract: ModelContract) -> str:
    return contract.name


def test_model_contracts_cover_every_reviewed_registration():
    model_registrations, _ = production_execution_registrations(
        Object3DModelRegistration,
        Object3DPipelineRegistration,
    )
    assert {contract.model_type for contract in MODEL_CONTRACTS} == {
        registration.model_class for registration in model_registrations
    }


@pytest.mark.parametrize("contract", MODEL_CONTRACTS, ids=_model_contract_id)
def test_reviewed_model_batch_dtype_device_return_dict_and_save_load(contract, tmp_path):
    model = contract.make().to(device="cpu", dtype=torch.float64).eval()
    with torch.no_grad():
        output = contract.invoke(model, 2, True)
        tuple_output = contract.invoke(model, 2, False)
    assert output.shape[0] == 2
    assert output.device.type == "cpu"
    assert output.dtype is torch.float64
    torch.testing.assert_close(output, tuple_output)

    output_directory = tmp_path / contract.name
    model.save_pretrained(output_directory)
    loaded = contract.model_type.from_pretrained(
        output_directory,
        local_files_only=True,
        torch_dtype=torch.float64,
    ).eval()
    with torch.no_grad():
        reloaded = contract.invoke(loaded, 2, True)
    torch.testing.assert_close(reloaded, output, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("contract", MODEL_CONTRACTS, ids=_model_contract_id)
def test_reviewed_models_support_torch_compile_fullgraph_eager_backend(contract):
    model = contract.make().eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    with torch.no_grad():
        output = contract.invoke(compiled, 2, True)
    assert output.shape[0] == 2


GRADIENT_CHECKPOINTING_CONTRACTS = tuple(
    contract
    for contract in MODEL_CONTRACTS
    if contract.model_type
    in {
        TrellisSparseStructureFlowModel,
        Trellis2SparseStructureFlowModel,
    }
)


def _randomize_parameters(model: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.02)


def _forward_backward(contract: ModelContract, model: torch.nn.Module):
    output = contract.invoke(model, 2, True)
    weights = torch.linspace(-0.5, 0.75, output.numel(), device=output.device, dtype=output.dtype).reshape_as(output)
    loss = (output * weights).sum() + 0.125 * output.square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert any(torch.count_nonzero(gradient) for gradient in gradients.values())
    return output.detach(), gradients


@pytest.mark.parametrize("contract", GRADIENT_CHECKPOINTING_CONTRACTS, ids=_model_contract_id)
def test_reviewed_models_gradient_checkpointed_forward_backward_matches_eager(contract):
    baseline = contract.make().train()
    _randomize_parameters(baseline, seed=0)
    checkpointed = contract.make().train()
    checkpointed.load_state_dict(baseline.state_dict(), strict=True)

    checkpoint_calls = 0

    def checkpoint(function, *args):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return apply_checkpoint(function.__call__, *args, use_reentrant=False)

    checkpointed.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint)
    assert checkpointed.is_gradient_checkpointing

    baseline_output, baseline_gradients = _forward_backward(contract, baseline)
    checkpointed_output, checkpointed_gradients = _forward_backward(contract, checkpointed)

    assert checkpoint_calls > 0
    torch.testing.assert_close(checkpointed_output, baseline_output, atol=1e-6, rtol=1e-6)
    assert checkpointed_gradients.keys() == baseline_gradients.keys()
    for name in baseline_gradients:
        torch.testing.assert_close(
            checkpointed_gradients[name],
            baseline_gradients[name],
            atol=1e-6,
            rtol=1e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_reviewed_models_expose_gradient_checkpointing_exactly_where_supported():
    supported = {contract.model_type for contract in GRADIENT_CHECKPOINTING_CONTRACTS}
    assert {
        contract.model_type for contract in MODEL_CONTRACTS if contract.model_type._supports_gradient_checkpointing
    } == supported
    for contract in MODEL_CONTRACTS:
        model = contract.make()
        if contract.model_type in supported:
            model.enable_gradient_checkpointing()
            assert model.is_gradient_checkpointing
        else:
            with pytest.raises(ValueError, match="does not support gradient checkpointing"):
                model.enable_gradient_checkpointing()


def test_reviewed_attention_models_support_processor_and_native_backend_hooks():
    expected = {
        TrellisSparseStructureFlowModel,
        Trellis2SparseStructureFlowModel,
    }
    found = set()
    for contract in MODEL_CONTRACTS:
        model = contract.make().eval()
        attention_modules = [
            module
            for module in model.modules()
            if all(hasattr(module, method) for method in ("get_processor", "set_attention_backend", "set_processor"))
        ]
        if not attention_modules:
            continue
        found.add(contract.model_type)
        for module in attention_modules:
            processor = module.get_processor()
            module.set_processor(processor)
            module.set_attention_backend("_native_math")
            assert module.get_processor()._attention_backend.value == "_native_math"
        with torch.no_grad():
            assert contract.invoke(model, 2, True).shape[0] == 2
    assert found == expected


def _trellis_pipeline():
    decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    return TrellisImageTo3DPipeline(
        conditioner=TrellisDinov2Conditioner(**TrellisDinov2Conditioner.tiny_config()),
        sparse_structure_flow_model=TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config()),
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=TrellisFlowEulerScheduler(),
    )


def _trellis2_pipeline():
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    return Trellis2ImageTo3DPipeline(
        conditioner=Trellis2Dinov3Conditioner(**Trellis2Dinov3Conditioner.tiny_config()),
        sparse_structure_flow_model=Trellis2SparseStructureFlowModel(**Trellis2SparseStructureFlowModel.tiny_config()),
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=Trellis2FlowEulerScheduler(),
        default_pipeline_type="tiny",
    )


PIPELINE_FACTORIES = (
    pytest.param(_trellis_pipeline, id="TrellisImageTo3DPipeline"),
    pytest.param(_trellis2_pipeline, id="Trellis2ImageTo3DPipeline"),
)


def _invoke_pipeline(pipeline, *, return_dict: bool):
    _, dtype = _parameter_properties(pipeline)
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8, dtype=dtype).reshape(2, 3, 8, 8)
    if isinstance(pipeline, Trellis2ImageTo3DPipeline):
        latents = torch.linspace(-1.0, 1.0, 2 * 2 * 2 * 2 * 2, dtype=dtype).reshape(2, 2, 2, 2, 2)
        return pipeline(
            images,
            sparse_structure_latents=latents,
            sparse_structure_sampler_params={"steps": 2},
            return_dict=return_dict,
        )
    latents = torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 4 * 4, dtype=dtype).reshape(2, 2, 4, 4, 4)
    return pipeline(
        images,
        sparse_structure_latents=latents,
        sparse_structure_num_inference_steps=2,
        return_dict=return_dict,
    )


def _asset_tensor(asset) -> torch.Tensor:
    return asset.vertices if isinstance(asset, MeshAsset) else asset.features


@pytest.mark.integration
@pytest.mark.parametrize("pipeline_factory", PIPELINE_FACTORIES)
def test_reviewed_pipeline_batch_dtype_device_and_tuple_contract(pipeline_factory):
    pipeline = pipeline_factory().to(device="cpu", dtype=torch.float64)
    output = _invoke_pipeline(pipeline, return_dict=True)
    tuple_output = _invoke_pipeline(pipeline, return_dict=False)

    assert len(output.objects) == 2
    assert output.latents.latents.shape[0] == 2
    assert output.latents.latents.device.type == "cpu"
    assert output.latents.latents.dtype is torch.float64
    assert len(tuple_output[0]) == 2
    for actual, expected in zip(output.objects, tuple_output[0]):
        torch.testing.assert_close(_asset_tensor(actual), _asset_tensor(expected))
    torch.testing.assert_close(output.latents.latents, tuple_output[1].latents)

    signature = inspect.signature(pipeline.__call__)
    assert "callback_on_step_end" not in signature.parameters


def test_pipeline_contracts_cover_every_reviewed_registration():
    _, pipeline_registrations = production_execution_registrations(
        Object3DModelRegistration,
        Object3DPipelineRegistration,
    )
    assert {factory().__class__ for factory in (_trellis_pipeline, _trellis2_pipeline)} == {
        registration.pipeline_class for registration in pipeline_registrations
    }
