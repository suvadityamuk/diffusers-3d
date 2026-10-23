from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from torch.utils.checkpoint import checkpoint as apply_checkpoint
from transformers import (
    BitImageProcessorPil,
    CLIPTextConfig,
    CLIPTextModel,
    Dinov2Config,
    Dinov2Model,
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
)

from diffusers_3d import (
    Hunyuan3DDiTModel,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DImageTo3DPipeline,
    Hunyuan3DPaintPipeline,
    Hunyuan3DPaintRRDBNet,
    Hunyuan3DPaintUNet,
    Hunyuan3DShapeVAE,
    ImageCondition,
    MeshAsset,
    Object3DModelRegistration,
    Object3DPipelineRegistration,
    Pixal3DImageTo3DPipeline,
    Pixal3DNAFUpsampler,
    Pixal3DSLatFlowModel,
    Pixal3DSparseStructureFlowModel,
    Sam3DConditionEmbedder,
    Sam3DObjectsPipeline,
    Sam3DSparseStructureFlowModel,
    Step1X3DFlowMatchEulerScheduler,
    Step1X3DImageTo3DPipeline,
    Step1X3DMeshTexturePipeline,
    Step1X3DMultiViewAdapter,
    Step1X3DShapeDecoder,
    Step1X3DTransformer1DModel,
    Trellis2FlowEulerScheduler,
    Trellis2ImageTo3DPipeline,
    Trellis2PBRSparseDecoder,
    Trellis2ShapeDualGridDecoder,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureDecoder,
    Trellis2SparseStructureFlowModel,
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
    TrellisSparseTensor,
    TrellisTextTo3DPipeline,
    TripoSGImageTo3DPipeline,
    TripoSGTransformerModel,
    TripoSGVAEDecoder,
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


def _invoke_pixal3d_flow(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    resolution, channels = model.config.resolution, model.config.in_channels
    hidden_states = torch.linspace(-1.0, 1.0, batch_size * channels * resolution**3, device=device, dtype=dtype)
    hidden_states = hidden_states.reshape(batch_size, channels, resolution, resolution, resolution)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    tokens = torch.linspace(-0.5, 0.5, batch_size * 5 * model.config.cond_channels, device=device, dtype=dtype)
    projected = torch.linspace(
        0.5, -0.5, batch_size * resolution**3 * model.config.proj_in_channels, device=device, dtype=dtype
    )
    context = (tokens.reshape(batch_size, 5, -1), projected.reshape(batch_size, resolution**3, -1))
    return _output_tensor(model(hidden_states, timesteps, context, return_dict=return_dict), return_dict=return_dict)


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


# Three active voxels per batch item so sparse outputs regroup as (batch, voxels, channels).
_VOXELS_PER_ITEM = 3


def _sparse_latents(model: torch.nn.Module, batch_size: int, channels: int) -> TrellisSparseTensor:
    device, dtype = _parameter_properties(model)
    cells = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 3, 1]], device=device)
    coordinates = torch.cat(
        [
            torch.cat([torch.full((_VOXELS_PER_ITEM, 1), index, device=device), cells], dim=1)
            for index in range(batch_size)
        ]
    )
    features = torch.linspace(-1.0, 1.0, batch_size * _VOXELS_PER_ITEM * channels, device=device, dtype=dtype).reshape(
        batch_size * _VOXELS_PER_ITEM, channels
    )
    return TrellisSparseTensor(coordinates, features)


def _regroup(features: torch.Tensor, batch_size: int) -> torch.Tensor:
    return features.reshape(batch_size, -1, features.shape[-1])


def _invoke_slat_flow(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    hidden_states = _sparse_latents(model, batch_size, model.config.in_channels)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    context = torch.linspace(-0.5, 0.5, batch_size * 7 * model.config.cond_channels, device=device, dtype=dtype)
    output = model(hidden_states, timesteps, context.reshape(batch_size, 7, -1), return_dict=return_dict)
    return _regroup(_output_tensor(output, return_dict=return_dict).features, batch_size)


def _invoke_texture_slat_flow(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    channels = model.config.in_channels // 2
    hidden_states = _sparse_latents(model, batch_size, channels)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    context = torch.linspace(-0.5, 0.5, batch_size * 7 * model.config.cond_channels, device=device, dtype=dtype)
    output = model(
        hidden_states,
        timesteps,
        context.reshape(batch_size, 7, -1),
        concat_cond=hidden_states.replace(hidden_states.features.flip(0)),
        return_dict=return_dict,
    )
    return _regroup(_output_tensor(output, return_dict=return_dict).features, batch_size)


def _invoke_pixal3d_texture_slat_flow(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    hidden_states = _sparse_latents(model, batch_size, model.config.in_channels // 2)
    projected = _sparse_latents(model, batch_size, model.config.proj_in_channels)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    tokens = torch.linspace(-0.5, 0.5, batch_size * 5 * model.config.cond_channels, device=device, dtype=dtype)
    output = model(
        hidden_states,
        timesteps,
        (tokens.reshape(batch_size, 5, -1), projected.replace(projected.features.flip(0))),
        concat_cond=hidden_states.replace(hidden_states.features.flip(0)),
        return_dict=return_dict,
    )
    return _regroup(_output_tensor(output, return_dict=return_dict).features, batch_size)


def _invoke_feature_upsampler(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    image = torch.linspace(0.0, 1.0, batch_size * 3 * 16 * 16, device=device, dtype=dtype).reshape(
        batch_size, 3, 16, 16
    )
    features = torch.linspace(-1.0, 1.0, batch_size * 6 * 4 * 4, device=device, dtype=dtype).reshape(
        batch_size, 6, 4, 4
    )
    return _output_tensor(model(image, features, (8, 8), return_dict=return_dict), return_dict=return_dict)


def _assets(output: Any, *, return_dict: bool):
    return output.assets if return_dict else output[0]


def _invoke_gaussian_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    assets = _assets(
        model(_sparse_latents(model, batch_size, model.latent_channels), return_dict=return_dict),
        return_dict=return_dict,
    )
    return torch.stack([torch.cat([asset.means, asset.log_scales, asset.opacity_logits], dim=1) for asset in assets])


def _invoke_mesh_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    assets = _assets(
        model(_sparse_latents(model, batch_size, model.latent_channels), return_dict=return_dict),
        return_dict=return_dict,
    )
    # Vertex counts are data-dependent, so summarize each mesh by the per-channel mean over its vertices.
    return torch.stack(
        [torch.cat([asset.vertices, asset.colors, asset.extras["normal_map"]], dim=1).mean(dim=0) for asset in assets]
    )


def _invoke_radiance_field_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    assets = _assets(
        model(_sparse_latents(model, batch_size, model.latent_channels), return_dict=return_dict),
        return_dict=return_dict,
    )
    return torch.stack(
        [
            torch.cat([asset.trivec.flatten(1), asset.density, asset.color_coefficients.flatten(1)], dim=1)
            for asset in assets
        ]
    )


def _select_all_children(model: torch.nn.Module) -> None:
    # Untrained subdivision heads sit at zero; bias them so every stage keeps all eight children and
    # each batch item decodes to the same number of voxels.
    with torch.no_grad():
        for stage in model.blocks[:-1]:
            stage[-1].to_subdiv.bias.fill_(10.0)


def _invoke_shape_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    _select_all_children(model)
    assets = _assets(
        model(_sparse_latents(model, batch_size, model.latent_channels), return_dict=return_dict),
        return_dict=return_dict,
    )
    return torch.stack([torch.cat([asset.dual_grid_vertex_offsets, asset.split_weights], dim=1) for asset in assets])


def _invoke_pbr_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    shape_decoder = Trellis2ShapeDualGridDecoder(**Trellis2ShapeDualGridDecoder.tiny_config()).to(
        *_parameter_properties(model)
    )
    _select_all_children(shape_decoder)
    latents = _sparse_latents(model, batch_size, model.latent_channels)
    with torch.no_grad():
        shape = shape_decoder(latents)
    assets = _assets(
        model(latents, shape.assets, shape.subdivisions, return_dict=return_dict), return_dict=return_dict
    )
    return torch.stack(
        [torch.cat([asset.base_color, asset.metallic, asset.roughness, asset.opacity], dim=1) for asset in assets]
    )


def _invoke_triposg_transformer(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    latents = torch.linspace(-1.0, 1.0, batch_size * 5 * model.config.in_channels, device=device, dtype=dtype)
    context = torch.linspace(-0.5, 0.5, batch_size * 3 * model.config.cross_attention_dim, device=device, dtype=dtype)
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    output = model(
        latents.reshape(batch_size, 5, -1), timesteps, context.reshape(batch_size, 3, -1), return_dict=return_dict
    )
    return _output_tensor(output, return_dict=return_dict)


def _invoke_step1x3d_transformer(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    config = model.config
    latents = torch.linspace(-1.0, 1.0, batch_size * 5 * config.in_channels, device=device, dtype=dtype)
    context = torch.linspace(1.0, -1.0, batch_size * 3 * config.cross_attention_dim, device=device, dtype=dtype)
    output = model(
        latents.reshape(batch_size, 5, -1),
        torch.full((batch_size,), 500.0, device=device, dtype=dtype),
        context.reshape(batch_size, 3, -1),
        return_dict=return_dict,
    )
    return _output_tensor(output, return_dict=return_dict)


def _invoke_triposg_vae(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    latents = torch.linspace(-1.0, 1.0, batch_size * 6 * model.config.latent_channels, device=device, dtype=dtype)
    queries = torch.linspace(-1.0, 1.0, batch_size * 7 * 3, device=device, dtype=dtype).reshape(batch_size, 7, 3)
    output = model(latents.reshape(batch_size, 6, -1), queries, num_chunks=4, return_dict=return_dict)
    return _output_tensor(output, return_dict=return_dict)


def _invoke_sam3d_flow(model: Sam3DSparseStructureFlowModel, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    latents = {
        name: torch.linspace(-1.0, 1.0, torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)
        for name, shape in model.latent_shapes(batch_size).items()
    }
    timesteps = torch.linspace(200.0, 800.0, batch_size, device=device, dtype=dtype)
    context = torch.linspace(-0.5, 0.5, batch_size * 7 * model.config.cond_channels, device=device, dtype=dtype)
    output = model(latents, timesteps, context.reshape(batch_size, 7, -1), return_dict=return_dict)
    sample = output.sample if return_dict else output[0]
    return torch.cat([value.flatten(1) for value in sample.values()], dim=1)


def _invoke_sam3d_conditioner(model: Sam3DConditionEmbedder, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    shape = (batch_size, 2, 5, model.config.image_embed_dim)
    tokens = torch.linspace(-1.0, 1.0, torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)
    pointmaps = torch.linspace(-0.5, 0.5, batch_size * 2 * 3 * 8 * 8, device=device, dtype=dtype)
    output = model(tokens, tokens.flip(-1), pointmaps.reshape(batch_size, 2, 3, 8, 8), return_dict=return_dict)
    return _output_tensor(output, return_dict=return_dict)


def _invoke_step1x3d_shape_decoder(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    config = model.config
    latents = torch.linspace(-1.0, 1.0, batch_size * config.num_latents * config.latent_channels, dtype=dtype)
    points = torch.linspace(-1.0, 1.0, batch_size * 6 * 3, dtype=dtype).reshape(batch_size, 6, 3)
    output = model(
        latents.reshape(batch_size, config.num_latents, -1).to(device),
        points.to(device),
        return_dict=return_dict,
    )
    return _output_tensor(output, return_dict=return_dict)


def _invoke_step1x3d_mv_adapter(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    from diffusers.models.attention_processor import Attention

    device, dtype = _parameter_properties(model)
    (name,) = model.config.hidden_sizes
    processor = model.get_submodule(name)
    torch.manual_seed(0)
    attention = Attention(8, heads=2, dim_head=4, processor=processor).to(device=device, dtype=dtype)
    hidden_states = torch.linspace(-1.0, 1.0, batch_size * 6 * 4 * 8, device=device, dtype=dtype)
    hidden_states = hidden_states.reshape(batch_size * 6, 4, 8)
    output = attention(hidden_states, reference_hidden_states={processor: hidden_states[:, :3].flip(-1)})
    return output.unflatten(0, (batch_size, 6))


def _invoke_hunyuan3d_transformer(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    latents = torch.linspace(-1.0, 1.0, batch_size * 6 * 8, device=device, dtype=dtype).reshape(batch_size, 6, 8)
    timesteps = torch.linspace(0.2, 0.8, batch_size, device=device, dtype=dtype)
    context = torch.linspace(-0.5, 0.5, batch_size * 5 * 24, device=device, dtype=dtype).reshape(batch_size, 5, 24)
    return _output_tensor(model(latents, timesteps, context, return_dict=return_dict), return_dict=return_dict)


def _invoke_hunyuan3d_vae(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    latents = torch.linspace(-1.0, 1.0, batch_size * 6 * 8, device=device, dtype=dtype).reshape(batch_size, 6, 8)
    return _output_tensor(model(latents, return_dict=return_dict), return_dict=return_dict)


def _invoke_hunyuan3d_paint_unet(model: Hunyuan3DPaintUNet, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    config = model.config
    size, views, channels = config.sample_size, 2, config.latent_channels

    def ramp(*shape: int, low: float = -1.0) -> torch.Tensor:
        return torch.linspace(low, 1.0, torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)

    output = model(
        ramp(batch_size, 2, views, channels, size, size),
        torch.tensor(500.0, device=device, dtype=dtype),
        ramp(views, 2 * channels, size, size),
        ramp(views, 3, 2 * size, 2 * size, low=0.0),
        ramp(batch_size, 3, config.image_embed_dim),
        model.encode_reference(ramp(1, channels, size, size)),
        ramp(batch_size, low=0.0),
        return_dict=return_dict,
    )
    return _output_tensor(output, return_dict=return_dict)


def _invoke_hunyuan3d_paint_rrdbnet(model: torch.nn.Module, batch_size: int, return_dict: bool) -> torch.Tensor:
    device, dtype = _parameter_properties(model)
    return model(
        torch.linspace(0.0, 1.0, batch_size * 3 * 5 * 6, device=device, dtype=dtype).reshape(batch_size, 3, 5, 6)
    )


MODEL_CONTRACTS = (
    ModelContract(TrellisSparseStructureFlowModel, _invoke_trellis_flow),
    ModelContract(TrellisSparseStructureDecoder, _invoke_sparse_decoder),
    ModelContract(TrellisSLatFlowModel, _invoke_slat_flow),
    ModelContract(TrellisSLatGaussianDecoder, _invoke_gaussian_decoder),
    ModelContract(TrellisSLatMeshDecoder, _invoke_mesh_decoder),
    ModelContract(TrellisSLatRadianceFieldDecoder, _invoke_radiance_field_decoder),
    ModelContract(Trellis2SparseStructureFlowModel, _invoke_trellis_flow),
    ModelContract(Trellis2SparseStructureDecoder, _invoke_sparse_decoder),
    ModelContract(Trellis2SLatFlowModel, _invoke_texture_slat_flow),
    ModelContract(Trellis2ShapeDualGridDecoder, _invoke_shape_decoder),
    ModelContract(Trellis2PBRSparseDecoder, _invoke_pbr_decoder),
    ModelContract(TripoSGTransformerModel, _invoke_triposg_transformer),
    ModelContract(TripoSGVAEDecoder, _invoke_triposg_vae),
    ModelContract(Sam3DSparseStructureFlowModel, _invoke_sam3d_flow),
    ModelContract(Sam3DConditionEmbedder, _invoke_sam3d_conditioner),
    ModelContract(Step1X3DTransformer1DModel, _invoke_step1x3d_transformer),
    ModelContract(Step1X3DShapeDecoder, _invoke_step1x3d_shape_decoder),
    ModelContract(Step1X3DMultiViewAdapter, _invoke_step1x3d_mv_adapter),
    ModelContract(Pixal3DSparseStructureFlowModel, _invoke_pixal3d_flow),
    ModelContract(Pixal3DSLatFlowModel, _invoke_pixal3d_texture_slat_flow),
    ModelContract(Pixal3DNAFUpsampler, _invoke_feature_upsampler),
    ModelContract(Hunyuan3DDiTModel, _invoke_hunyuan3d_transformer),
    ModelContract(Hunyuan3DShapeVAE, _invoke_hunyuan3d_vae),
    ModelContract(Hunyuan3DPaintUNet, _invoke_hunyuan3d_paint_unet),
    ModelContract(Hunyuan3DPaintRRDBNet, _invoke_hunyuan3d_paint_rrdbnet),
)
# Sparse voxel models index by coordinates (unique / nonzero) and mixture-of-experts models route tokens with
# nonzero, so their shapes are data-dependent and they cannot be captured as a single graph.
SPARSE_MODEL_TYPES = {
    Hunyuan3DDiTModel,
    TrellisSLatFlowModel,
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    Trellis2SLatFlowModel,
    Trellis2ShapeDualGridDecoder,
    Trellis2PBRSparseDecoder,
    Pixal3DSLatFlowModel,
}


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
    compiled = torch.compile(model, backend="eager", fullgraph=contract.model_type not in SPARSE_MODEL_TYPES)
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
        TripoSGTransformerModel,
        Sam3DSparseStructureFlowModel,
        Pixal3DSparseStructureFlowModel,
        *SPARSE_MODEL_TYPES,
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
        TrellisSLatFlowModel,
        TrellisSLatGaussianDecoder,
        TrellisSLatMeshDecoder,
        TrellisSLatRadianceFieldDecoder,
        Trellis2SLatFlowModel,
        TripoSGTransformerModel,
        TripoSGVAEDecoder,
        Sam3DSparseStructureFlowModel,
        Pixal3DSparseStructureFlowModel,
        Pixal3DSLatFlowModel,
        Hunyuan3DDiTModel,
        Hunyuan3DShapeVAE,
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


def _trellis_sparse_structure_components():
    decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    return {
        "sparse_structure_flow_model": TrellisSparseStructureFlowModel(
            **TrellisSparseStructureFlowModel.tiny_config()
        ),
        "sparse_structure_decoder": decoder,
        "sparse_structure_scheduler": TrellisFlowEulerScheduler(),
    }


def _trellis_pipeline(request=None):
    dinov2 = Dinov2WithRegistersConfig(
        hidden_size=12, num_hidden_layers=1, num_attention_heads=3, patch_size=4, image_size=8, num_register_tokens=2
    )
    return TrellisImageTo3DPipeline(
        image_encoder=Dinov2WithRegistersModel(dinov2).eval(), **_trellis_sparse_structure_components()
    )


def _trellis_text_pipeline(request):
    clip = CLIPTextConfig(
        vocab_size=64,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
        max_position_embeddings=8,
    )
    return TrellisTextTo3DPipeline(
        text_encoder=CLIPTextModel(clip).eval(),
        tokenizer=request.getfixturevalue("tiny_clip_tokenizer"),
        **_trellis_sparse_structure_components(),
    )


def _tiny_dinov3() -> DINOv3ViTModel:
    config = DINOv3ViTConfig(
        image_size=8,
        patch_size=4,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_register_tokens=2,
    )
    return DINOv3ViTModel(config).eval()


def _trellis2_pipeline(request=None):
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    return Trellis2ImageTo3DPipeline(
        image_encoder=_tiny_dinov3(),
        sparse_structure_flow_model=Trellis2SparseStructureFlowModel(**Trellis2SparseStructureFlowModel.tiny_config()),
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=Trellis2FlowEulerScheduler(),
        default_pipeline_type="512",
        image_size=8,
    )


def _triposg_pipeline(request=None):
    vae = TripoSGVAEDecoder(**TripoSGVAEDecoder.tiny_config())
    # Route the query embedding straight through so the decoded field is a smooth function of x crossing zero.
    block = vae.blocks[-1]
    with torch.no_grad():
        for layer in (block.attn2.to_out[0], block.ff.net[2], vae.proj_query, vae.proj_out):
            layer.weight.zero_()
            layer.bias.zero_()
        vae.proj_query.weight[0, 0], vae.proj_query.weight[1, 0] = 1.0, -1.0
        vae.proj_query.bias[2], vae.proj_query.bias[3] = 0.5, -0.5
        vae.proj_out.weight[0, 0] = 1.0
    return TripoSGImageTo3DPipeline(
        feature_extractor=BitImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28}),
        image_encoder=Dinov2Model(
            Dinov2Config(image_size=28, patch_size=14, hidden_size=12, num_hidden_layers=1, num_attention_heads=3)
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(),
        transformer=TripoSGTransformerModel(**TripoSGTransformerModel.tiny_config()),
        vae=vae,
    )


def _sam3d_pipeline(request=None):
    decoder = TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    dinov2 = Dinov2WithRegistersConfig(
        hidden_size=12, num_hidden_layers=1, num_attention_heads=3, patch_size=4, image_size=8, num_register_tokens=2
    )
    with pytest.warns(UserWarning, match="SAM License"):
        return Sam3DObjectsPipeline(
            image_encoder=Dinov2WithRegistersModel(dinov2),
            sparse_structure_conditioner=Sam3DConditionEmbedder(**Sam3DConditionEmbedder.tiny_config()),
            sparse_structure_flow_model=Sam3DSparseStructureFlowModel(**Sam3DSparseStructureFlowModel.tiny_config()),
            sparse_structure_decoder=decoder,
            sparse_structure_scheduler=TrellisFlowEulerScheduler(sigma_min=0.0),
        )


def _step1x3d_pipeline(request=None):
    vae = Step1X3DShapeDecoder(**Step1X3DShapeDecoder.tiny_config())
    # Pass the query embedding straight through so the occupancy field is a smooth function of x crossing zero.
    with torch.no_grad():
        for layer in (vae.attn.to_out[0], vae.ff.net[2], vae.query_proj, vae.proj_out):
            layer.weight.zero_()
            layer.bias.zero_()
        vae.query_proj.weight[0, 0], vae.query_proj.weight[1, 0] = 1.0, -1.0
        vae.query_proj.bias[2], vae.query_proj.bias[3] = 0.5, -0.5
        vae.proj_out.weight[0, 0] = 1.0
    return Step1X3DImageTo3DPipeline(
        image_encoder=Dinov2WithRegistersModel(
            Dinov2WithRegistersConfig(
                hidden_size=12,
                num_hidden_layers=1,
                num_attention_heads=3,
                mlp_ratio=2,
                image_size=8,
                patch_size=4,
                num_register_tokens=2,
            )
        ),
        transformer=Step1X3DTransformer1DModel(**Step1X3DTransformer1DModel.tiny_config()),
        vae=vae,
        scheduler=Step1X3DFlowMatchEulerScheduler(),
    )


def _step1x3d_texture_pipeline(request):
    from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
    from transformers import CLIPTextModelWithProjection

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(8, 8, 16),
        layers_per_block=1,
        sample_size=8,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=2,
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=6 * 4 + 8,
        cross_attention_dim=16,
        norm_num_groups=4,
    )
    text = CLIPTextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=1,
        projection_dim=8,
    )
    tokenizer = request.getfixturevalue("tiny_clip_tokenizer")
    return Step1X3DMeshTexturePipeline(
        vae=AutoencoderKL(
            block_out_channels=(4, 4),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            norm_num_groups=2,
        ),
        text_encoder=CLIPTextModel(text),
        text_encoder_2=CLIPTextModelWithProjection(text),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer,
        unet=unet,
        adapter=T2IAdapter(
            in_channels=6,
            channels=(8, 8, 16, 16),
            num_res_blocks=1,
            downscale_factor=4,
            adapter_type="full_adapter_xl",
        ),
        mv_adapter=Step1X3DMultiViewAdapter(hidden_sizes=Step1X3DMultiViewAdapter.hidden_sizes_for(unet)),
        scheduler=DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, clip_sample=False),
    )


def _pixal3d_pipeline(request=None):
    decoder = Trellis2SparseStructureDecoder(**Trellis2SparseStructureDecoder.tiny_config())
    with torch.no_grad():
        decoder.out_layer[-1].weight.zero_()
        decoder.out_layer[-1].bias.fill_(1.0)
    return Pixal3DImageTo3DPipeline(
        image_encoder=_tiny_dinov3(),
        feature_upsampler=Pixal3DNAFUpsampler(**Pixal3DNAFUpsampler.tiny_config()),
        sparse_structure_flow_model=Pixal3DSparseStructureFlowModel(**Pixal3DSparseStructureFlowModel.tiny_config()),
        sparse_structure_decoder=decoder,
        sparse_structure_scheduler=Trellis2FlowEulerScheduler(),
        image_size=16,
    )


def _hunyuan3d_pipeline(request=None):
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        return Hunyuan3DImageTo3DPipeline(
            image_encoder=Dinov2Model(
                Dinov2Config(
                    hidden_size=24,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    mlp_ratio=2,
                    image_size=28,
                    patch_size=14,
                )
            ),
            transformer=Hunyuan3DDiTModel(**Hunyuan3DDiTModel.tiny_config()),
            vae=Hunyuan3DShapeVAE(**Hunyuan3DShapeVAE.tiny_config()),
            scheduler=Hunyuan3DFlowMatchEulerDiscreteScheduler(),
            image_size=16,
        )


def _hunyuan3d_paint_pipeline(request=None):
    torch.manual_seed(0)
    unet = Hunyuan3DPaintUNet(**Hunyuan3DPaintUNet.tiny_config())
    with torch.no_grad():
        for parameter in unet.parameters():
            parameter.normal_(std=0.05)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        return Hunyuan3DPaintPipeline(
            feature_extractor=BitImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28}),
            image_encoder=Dinov2Model(
                Dinov2Config(hidden_size=10, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14)
            ),
            unet=unet,
            vae=AutoencoderKL(
                block_out_channels=(8, 16),
                down_block_types=("DownEncoderBlock2D",) * 2,
                up_block_types=("UpDecoderBlock2D",) * 2,
                norm_num_groups=8,
            ),
            scheduler=UniPCMultistepScheduler(prediction_type="v_prediction", timestep_spacing="trailing"),
            upscaler=Hunyuan3DPaintRRDBNet(**Hunyuan3DPaintRRDBNet.tiny_config()),
        )


PIPELINE_FACTORIES = (
    pytest.param(_trellis_pipeline, id="TrellisImageTo3DPipeline"),
    pytest.param(_trellis_text_pipeline, id="TrellisTextTo3DPipeline"),
    pytest.param(_trellis2_pipeline, id="Trellis2ImageTo3DPipeline"),
    # Meshes come from scikit-image marching cubes.
    pytest.param(_triposg_pipeline, id="TripoSGImageTo3DPipeline", marks=pytest.mark.portable),
    pytest.param(_sam3d_pipeline, id="Sam3DObjectsPipeline"),
    pytest.param(_step1x3d_pipeline, id="Step1X3DImageTo3DPipeline", marks=pytest.mark.portable),
    # Meshes are UV-unwrapped with xatlas.
    pytest.param(_step1x3d_texture_pipeline, id="Step1X3DMeshTexturePipeline", marks=pytest.mark.portable),
    pytest.param(_pixal3d_pipeline, id="Pixal3DImageTo3DPipeline"),
    # Surface extraction uses scikit-image.
    pytest.param(_hunyuan3d_pipeline, id="Hunyuan3DImageTo3DPipeline", marks=pytest.mark.portable),
)


def _invoke_pipeline(pipeline, *, return_dict: bool):
    _, dtype = _parameter_properties(pipeline)
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8, dtype=dtype).reshape(2, 3, 8, 8)
    conditioning = ["a red chair", "a chair"] if isinstance(pipeline, TrellisTextTo3DPipeline) else images
    if isinstance(pipeline, TripoSGImageTo3DPipeline):
        rgba = torch.cat([images, torch.zeros(2, 1, 8, 8, dtype=dtype)], dim=1)
        rgba[:, 3, 2:6, 3:6] = 1.0
        latents = torch.linspace(-1.0, 1.0, 2 * 6 * 4, dtype=dtype).reshape(2, 6, 4)
        return pipeline(
            rgba,
            latents=latents,
            num_inference_steps=2,
            num_tokens=6,
            dense_octree_depth=3,
            hierarchical_octree_depth=4,
            return_dict=return_dict,
        )
    if isinstance(pipeline, Sam3DObjectsPipeline):
        mask = torch.zeros(1, 8, 8, dtype=dtype)
        mask[:, 2:6, 1:7] = 1.0
        shapes = pipeline.sparse_structure_flow_model.latent_shapes(2)
        latents = {
            name: torch.linspace(-1.0, 1.0, torch.Size(shape).numel(), dtype=dtype).reshape(shape)
            for name, shape in shapes.items()
        }
        return pipeline(
            [ImageCondition(image=image, mask=mask) for image in images],
            sparse_structure_latents=latents,
            sparse_structure_num_inference_steps=2,
            return_dict=return_dict,
        )
    if isinstance(pipeline, Step1X3DMeshTexturePipeline):
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=1)
        mesh = MeshAsset(vertices=torch.tensor(sphere.vertices, dtype=dtype), faces=torch.tensor(sphere.faces))
        return pipeline(
            [mesh, mesh],
            [ImageCondition(image=image) for image in images],
            num_inference_steps=2,
            resolution=16,
            latent_texture_size=16,
            texture_size=16,
            generator=torch.Generator().manual_seed(0),
            return_dict=return_dict,
        )
    if isinstance(pipeline, Step1X3DImageTo3DPipeline):
        latents = torch.linspace(-1.0, 1.0, 2 * 8 * 4, dtype=dtype).reshape(2, 8, 4)
        return pipeline(images, latents=latents, num_inference_steps=2, octree_resolution=8, return_dict=return_dict)
    if isinstance(pipeline, Hunyuan3DImageTo3DPipeline):
        latents = torch.linspace(-1.0, 1.0, 2 * 6 * 8, dtype=dtype).reshape(2, 6, 8)
        return pipeline(images, latents=latents, num_inference_steps=2, octree_resolution=8, return_dict=return_dict)
    if isinstance(pipeline, Trellis2ImageTo3DPipeline):
        latents = torch.linspace(-1.0, 1.0, 2 * 2 * 2 * 2 * 2, dtype=dtype).reshape(2, 2, 2, 2, 2)
        camera = {"camera_angle_x": 0.5} if isinstance(pipeline, Pixal3DImageTo3DPipeline) else {}
        return pipeline(
            images,
            sparse_structure_latents=latents,
            sparse_structure_sampler_params={"steps": 2},
            return_dict=return_dict,
            **camera,
        )
    latents = torch.linspace(-1.0, 1.0, 2 * 2 * 4 * 4 * 4, dtype=dtype).reshape(2, 2, 4, 4, 4)
    return pipeline(
        conditioning,
        sparse_structure_latents=latents,
        sparse_structure_num_inference_steps=2,
        return_dict=return_dict,
    )


@pytest.mark.integration
@pytest.mark.portable
def test_reviewed_texturing_pipeline_dtype_device_and_tuple_contract():
    pipeline = _hunyuan3d_paint_pipeline().to(device="cpu", dtype=torch.float64)
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(subdivisions=1)
    mesh = MeshAsset(vertices=torch.tensor(sphere.vertices, dtype=torch.float32), faces=torch.tensor(sphere.faces))
    image = ImageCondition(image=torch.linspace(0.0, 1.0, 4 * 8 * 8, dtype=torch.float64).reshape(4, 8, 8))
    outputs = [
        pipeline(
            mesh,
            image,
            num_inference_steps=2,
            view_size=16,
            render_size=32,
            texture_size=16,
            max_faces=None,
            generator=torch.Generator().manual_seed(0),
            return_dict=return_dict,
        )
        for return_dict in (True, False)
    ]
    assert len(outputs[0].objects) == len(outputs[1][0]) == 1
    assert outputs[0].latents.latents.dtype is torch.float64 and outputs[0].latents.latents.device.type == "cpu"
    torch.testing.assert_close(outputs[0].latents.latents, outputs[1][1].latents)
    torch.testing.assert_close(outputs[0].objects[0].materials[0].base_color, outputs[1][0][0].materials[0].base_color)


def _asset_tensor(asset) -> torch.Tensor:
    return asset.vertices if isinstance(asset, MeshAsset) else asset.features


@pytest.mark.integration
@pytest.mark.parametrize("pipeline_factory", PIPELINE_FACTORIES)
def test_reviewed_pipeline_batch_dtype_device_and_tuple_contract(pipeline_factory, request):
    pipeline = pipeline_factory(request).to(device="cpu", dtype=torch.float64)
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
    assert {
        TrellisImageTo3DPipeline,
        TrellisTextTo3DPipeline,
        Trellis2ImageTo3DPipeline,
        TripoSGImageTo3DPipeline,
        Sam3DObjectsPipeline,
        Step1X3DImageTo3DPipeline,
        Step1X3DMeshTexturePipeline,
        Pixal3DImageTo3DPipeline,
        Hunyuan3DImageTo3DPipeline,
        Hunyuan3DPaintPipeline,
    } == {registration.pipeline_class for registration in pipeline_registrations}
