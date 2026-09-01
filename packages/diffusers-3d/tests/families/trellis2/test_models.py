from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    BackendUnavailableError,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureFlowModel,
    TrellisSparseTensor,
)


def test_sparse_structure_flow_forward_backward_state_layout_and_save_load(tmp_path):
    model = Trellis2SparseStructureFlowModel(**Trellis2SparseStructureFlowModel.tiny_config())
    with torch.no_grad():
        model.out_layer.weight.normal_(std=0.02)
    model.enable_gradient_checkpointing()
    generator = torch.Generator().manual_seed(10)
    hidden_states = torch.randn(2, 2, 2, 2, 2, generator=generator)
    timesteps = torch.tensor([250.0, 750.0])
    context = torch.randn(2, 7, 12, generator=generator)

    output = model(hidden_states, timesteps, context).sample
    tuple_output = model(hidden_states, timesteps, context, return_dict=False)[0]
    assert output.shape == hidden_states.shape
    torch.testing.assert_close(output, tuple_output)
    output.square().mean().backward()
    assert model.gradient_checkpointing
    assert model.input_layer.weight.grad is not None
    assert model.blocks[0].self_attn.to_qkv.weight.grad is not None
    assert {
        "rope_phases",
        "t_embedder.mlp.0.weight",
        "adaLN_modulation.1.weight",
        "blocks.0.modulation",
        "blocks.0.self_attn.q_rms_norm.gamma",
        "blocks.0.cross_attn.k_rms_norm.gamma",
        "out_layer.weight",
    }.issubset(model.state_dict())

    model.eval().save_pretrained(tmp_path)
    loaded = Trellis2SparseStructureFlowModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(hidden_states, timesteps, context).sample,
            model(hidden_states, timesteps, context).sample,
        )


def test_slat_flow_tiny_shape_and_texture_concat_forward_backward_and_save_load(tmp_path):
    shape_model = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config())
    texture_model = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.tiny_config(texture=True))
    with torch.no_grad():
        shape_model.out_layer.weight.normal_(std=0.02)
        texture_model.out_layer.weight.normal_(std=0.02)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(11)
    shape_input = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    shape_condition = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    texture_input = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    timesteps = torch.tensor([200.0, 800.0])
    context = torch.randn(2, 7, 12, generator=generator)

    shape_output = shape_model(shape_input, timesteps, context).sample
    texture_model.enable_gradient_checkpointing()
    texture_output = texture_model(
        texture_input,
        timesteps,
        context,
        concat_cond=shape_condition,
    ).sample
    assert torch.equal(shape_output.coordinates, coordinates)
    assert torch.equal(texture_output.coordinates, coordinates)
    assert shape_output.features.shape == texture_output.features.shape == (4, 4)
    texture_output.features.square().mean().backward()
    assert texture_model.blocks[0].cross_attn.to_kv.weight.grad is not None

    texture_model.eval().save_pretrained(tmp_path)
    loaded = Trellis2SLatFlowModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        restored = loaded(texture_input, timesteps, context, concat_cond=shape_condition).sample
    torch.testing.assert_close(texture_output.features.detach(), restored.features)

    misaligned = TrellisSparseTensor(coordinates.flip(0), shape_condition.features)
    with pytest.raises(ValueError, match="coordinate-aligned"):
        texture_model(texture_input, timesteps, context, concat_cond=misaligned)


def test_production_slat_path_requires_explicit_flex_gemm_support():
    with pytest.raises((BackendUnavailableError, NotImplementedError), match="flex_gemm|FlexGEMM"):
        Trellis2SLatFlowModel(**Trellis2SLatFlowModel.production_config())
