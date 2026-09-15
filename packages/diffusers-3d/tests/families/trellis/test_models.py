from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    TrellisSLatFlowModel,
    TrellisSparseStructureFlowModel,
    TrellisSparseTensor,
)


def _dense_inputs(model: TrellisSparseStructureFlowModel):
    generator = torch.Generator().manual_seed(10)
    hidden_states = torch.randn(
        2,
        model.config.in_channels,
        model.config.resolution,
        model.config.resolution,
        model.config.resolution,
        generator=generator,
    )
    timesteps = torch.tensor([250.0, 750.0])
    context = torch.randn(2, 7, model.config.cond_channels, generator=generator)
    return hidden_states, timesteps, context


def _sparse_inputs(model: TrellisSLatFlowModel):
    generator = torch.Generator().manual_seed(11)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 1, 0], [1, 7, 7, 7]],
        dtype=torch.int64,
    )
    hidden_states = TrellisSparseTensor(
        coordinates=coordinates,
        features=torch.randn(4, model.config.in_channels, generator=generator),
    )
    timesteps = torch.tensor([200.0, 800.0])
    context = torch.randn(2, 7, model.config.cond_channels, generator=generator)
    return hidden_states, timesteps, context


def test_sparse_structure_flow_forward_checkpointing_and_save_load(tmp_path):
    model = TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config())
    with torch.no_grad():
        model.out_layer.weight.normal_(std=0.02)
    model.enable_gradient_checkpointing()
    hidden_states, timesteps, context = _dense_inputs(model)
    output = model(hidden_states, timesteps, context).sample
    tuple_output = model(hidden_states, timesteps, context, return_dict=False)[0]
    assert output.shape == hidden_states.shape
    torch.testing.assert_close(output, tuple_output)
    output.square().mean().backward()
    assert model.gradient_checkpointing
    assert model.input_layer.weight.grad is not None
    assert model.blocks[0].self_attn.to_qkv.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TrellisSparseStructureFlowModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(hidden_states, timesteps, context).sample, model(hidden_states, timesteps, context).sample
        )


def test_slat_flow_portable_core_forward_checkpointing_and_save_load(tmp_path):
    model = TrellisSLatFlowModel(**TrellisSLatFlowModel.tiny_config())
    with torch.no_grad():
        model.out_layer.weight.normal_(std=0.02)
    model.enable_gradient_checkpointing()
    hidden_states, timesteps, context = _sparse_inputs(model)
    output = model(hidden_states, timesteps, context).sample
    assert torch.equal(output.coordinates, hidden_states.coordinates)
    assert output.features.shape == hidden_states.features.shape
    output.features.square().mean().backward()
    assert model.input_layer.weight.grad is not None
    assert model.blocks[0].cross_attn.to_kv.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TrellisSLatFlowModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        reloaded = loaded(hidden_states, timesteps, context).sample
    torch.testing.assert_close(output.features.detach(), reloaded.features)


def test_state_names_match_pinned_trellis_layouts():
    dense_keys = set(TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config()).state_dict())
    assert {
        "t_embedder.mlp.0.weight",
        "pos_emb",
        "input_layer.weight",
        "blocks.0.self_attn.to_qkv.weight",
        "blocks.0.cross_attn.to_kv.weight",
        "blocks.0.adaLN_modulation.1.weight",
        "out_layer.weight",
    }.issubset(dense_keys)

    sparse_keys = set(TrellisSLatFlowModel(**TrellisSLatFlowModel.tiny_config()).state_dict())
    assert {
        "t_embedder.mlp.0.weight",
        "input_layer.weight",
        "blocks.0.self_attn.to_qkv.weight",
        "blocks.0.cross_attn.to_kv.weight",
        "out_layer.weight",
    }.issubset(sparse_keys)
    assert not any(key.startswith("input_blocks.") or key.startswith("out_blocks.") for key in sparse_keys)


def test_slat_flow_io_stages_forward_and_released_key_layout(tmp_path):
    model = TrellisSLatFlowModel(**TrellisSLatFlowModel.small_config())
    with torch.no_grad():
        model.out_layer.weight.normal_(std=0.02)
        for block in (*model.input_blocks, *model.out_blocks):
            block.conv2.conv.weight.normal_(std=0.02)
    assert [block.downsample for block in model.input_blocks] == [False, True]
    assert [block.upsample for block in model.out_blocks] == [True, False]
    keys = model.state_dict()
    # spconv layout: (out, k, k, k, in) under ``conv.weight``; skip connections double the input width.
    assert keys["input_blocks.1.conv1.conv.weight"].shape == (16, 3, 3, 3, 8)
    assert keys["out_blocks.0.conv1.conv.weight"].shape == (8, 3, 3, 3, 32)
    assert keys["out_blocks.0.skip_connection.weight"].shape == (8, 32)
    assert keys["out_layer.weight"].shape == (4, 8)

    generator = torch.Generator().manual_seed(12)
    coordinates = torch.nonzero(torch.rand(2, 8, 8, 8, generator=generator) < 0.3)
    hidden_states = TrellisSparseTensor(coordinates, torch.randn(coordinates.shape[0], 4, generator=generator))
    timesteps = torch.tensor([200.0, 800.0])
    context = torch.randn(2, 7, model.config.cond_channels, generator=generator)
    output = model(hidden_states, timesteps, context).sample
    assert torch.equal(output.coordinates, coordinates)
    assert output.features.shape == hidden_states.features.shape
    output.features.square().mean().backward()
    assert model.input_blocks[1].conv1.conv.weight.grad is not None
    assert model.out_blocks[0].skip_connection.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = TrellisSLatFlowModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(hidden_states, timesteps, context).sample.features, output.features)


def test_slat_flow_rejects_mismatched_io_stages():
    with pytest.raises(ValueError, match="io_block_channels"):
        TrellisSLatFlowModel(**{**TrellisSLatFlowModel.small_config(), "patch_size": 4})
