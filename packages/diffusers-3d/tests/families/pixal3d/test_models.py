from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    Pixal3DNAFUpsampler,
    Pixal3DSLatFlowModel,
    Pixal3DSparseStructureFlowModel,
    Trellis2SLatFlowModel,
    TrellisSparseTensor,
)

# Read from the safetensors headers of TencentARC/Pixal3D@b0cb2e1b794cab9aa0ac38a95d794a4d9337437f (single-view and
# ``_mv`` flows share the layout): 700 tensors per flow, plus ``rope_phases`` in the dense flow, which the converter
# checks and drops. The NAF release holds 37 tensors, ``image_encoder.rope.periods`` among them.
RELEASED_FLOW_TENSORS = 700
RELEASED_PROJECTION_SHAPES = {"sparse_structure": (1536, 1024), "slat": (1536, 2048)}
RELEASED_NAF_TENSORS = 36


def test_sparse_structure_flow_adds_projected_features_and_round_trips(tmp_path):
    model = Pixal3DSparseStructureFlowModel(**Pixal3DSparseStructureFlowModel.tiny_config()).eval()
    with torch.no_grad():
        model.out_layer.weight.normal_(std=0.02)
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(2, 2, 2, 2, 2, generator=generator)
    timesteps = torch.tensor([250.0, 750.0])
    global_tokens = torch.randn(2, 3, 12, generator=generator)
    projected = torch.randn(2, 8, 12, generator=generator)

    with torch.no_grad():
        output = model(latents, timesteps, (global_tokens, projected)).sample
        without_projection = model(latents, timesteps, (global_tokens, torch.zeros_like(projected))).sample
    assert output.shape == latents.shape
    assert not torch.allclose(output, without_projection)
    assert {
        "blocks.0.cross_attn.cross_attn_block.to_q.weight",
        "blocks.0.cross_attn.cross_attn_block.k_rms_norm.gamma",
        "blocks.0.cross_attn.proj_linear.weight",
    }.issubset(model.state_dict())
    with pytest.raises(ValueError, match="projected_hidden_states"):
        model(latents, timesteps, (global_tokens, projected[:, :4]))

    model.save_pretrained(tmp_path)
    loaded = Pixal3DSparseStructureFlowModel.from_pretrained(tmp_path, local_files_only=True).eval()
    assert loaded.config.proj_in_channels == 12
    with torch.no_grad():
        torch.testing.assert_close(loaded(latents, timesteps, (global_tokens, projected)).sample, output)


def test_slat_flow_requires_coordinate_aligned_projected_features(tmp_path):
    model = Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.tiny_config(texture=True))
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 1, 0], [1, 7, 7, 7]])
    generator = torch.Generator().manual_seed(1)
    latents = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    shape = TrellisSparseTensor(coordinates, torch.randn(4, 4, generator=generator))
    projected = TrellisSparseTensor(coordinates, torch.randn(4, 24, generator=generator))
    context = (torch.randn(2, 3, 12, generator=generator), projected)

    model.enable_gradient_checkpointing()
    output = model(latents, torch.tensor([200.0, 800.0]), context, concat_cond=shape).sample
    assert torch.equal(output.coordinates, coordinates)
    output.features.square().mean().backward()
    assert model.blocks[0].cross_attn.proj_linear.weight.grad is not None

    misaligned = TrellisSparseTensor(coordinates.flip(0), projected.features)
    with pytest.raises(ValueError, match="coordinate-aligned"):
        model(latents, torch.tensor([1.0, 2.0]), (context[0], misaligned), concat_cond=shape)

    model.eval().save_pretrained(tmp_path)
    loaded = Pixal3DSLatFlowModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        restored = loaded(latents, torch.tensor([200.0, 800.0]), context, concat_cond=shape).sample
        expected = model(latents, torch.tensor([200.0, 800.0]), context, concat_cond=shape).sample
    torch.testing.assert_close(restored.features, expected.features)


def test_production_layouts_match_the_released_checkpoints():
    with torch.device("meta"):
        dense = Pixal3DSparseStructureFlowModel(**Pixal3DSparseStructureFlowModel.production_config())
        shape = Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.production_config())
        texture = Pixal3DSLatFlowModel(**Pixal3DSLatFlowModel.production_config(texture=True))
        trellis2 = Trellis2SLatFlowModel(**Trellis2SLatFlowModel.production_config())
    for model, projection in ((dense, "sparse_structure"), (shape, "slat"), (texture, "slat")):
        assert len(model.state_dict()) == RELEASED_FLOW_TENSORS
        assert model.blocks[0].cross_attn.proj_linear.weight.shape == RELEASED_PROJECTION_SHAPES[projection]
        assert model.blocks[0].cross_attn.proj_linear.weight.dtype is torch.bfloat16
    # Everything but the per-block projection is the TRELLIS.2 layout, one wrapper level deeper.
    remapped = {key.replace("cross_attn.cross_attn_block.", "cross_attn.") for key in shape.state_dict()}
    assert remapped - set(trellis2.state_dict()) == {
        f"blocks.{i}.cross_attn.proj_linear.{p}" for i in range(30) for p in ("weight", "bias")
    }
    assert texture.input_layer.weight.shape == (1536, 64)

    upsampler = Pixal3DNAFUpsampler(**Pixal3DNAFUpsampler.production_config())
    assert len(upsampler.state_dict()) == RELEASED_NAF_TENSORS


def test_naf_upsampler_output_and_validation():
    model = Pixal3DNAFUpsampler(**Pixal3DNAFUpsampler.tiny_config()).eval()
    image = torch.rand(2, 3, 24, 24)
    features = torch.randn(2, 6, 4, 4)
    with torch.no_grad():
        output = model(image, features, (12, 12)).sample
    assert output.shape == (2, 6, 12, 12)
    with pytest.raises(ValueError, match="integer multiple"):
        model(image, features, (10, 10))
    with pytest.raises(ValueError, match="at least 3 cells"):
        model(image, features[..., :2, :2], (8, 8))
