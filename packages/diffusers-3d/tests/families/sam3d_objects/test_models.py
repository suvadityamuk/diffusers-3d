from __future__ import annotations

import torch
from transformers import Dinov2WithRegistersModel

from diffusers_3d import Sam3DConditionEmbedder, Sam3DObjectsPipeline, Sam3DSparseStructureFlowModel

from .conftest import tiny_dinov2_config


def _randomize(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _flow_inputs(model: Sam3DSparseStructureFlowModel, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    latents = {name: torch.randn(shape, generator=generator) for name, shape in model.latent_shapes(2).items()}
    return latents, torch.tensor([150.0, 700.0]), torch.randn(2, 5, model.config.cond_channels, generator=generator)


def test_flow_layout_save_load_and_gradient_checkpointing(tmp_path):
    model = Sam3DSparseStructureFlowModel(**Sam3DSparseStructureFlowModel.tiny_config())
    _randomize(model, 0)
    latents, timesteps, context = _flow_inputs(model)
    assert list(latents) == ["shape", "rotation", "translation", "scale", "translation_scale"]
    assert {
        "d_embedder.mlp.0.weight",
        "latent_mapping.shape.pos_emb",
        "latent_mapping.rotation.pos_emb",
        "blocks.0.self_attn.streams.pose.to_qkv.weight",
        "blocks.0.self_attn.streams.shape.k_rms_norm.gamma",
        "blocks.0.cross_attn.pose.to_kv.weight",
        "blocks.0.mlp.shape.mlp.2.weight",
        "blocks.0.norm2.pose.weight",
    }.issubset(model.state_dict())

    with torch.no_grad():
        output = model(latents, timesteps, context).sample
        (tuple_output,) = model(latents, timesteps, context, return_dict=False)
    assert {name: value.shape for name, value in output.items()} == {
        name: value.shape for name, value in latents.items()
    }
    for name in output:
        torch.testing.assert_close(output[name], tuple_output[name])

    model.enable_gradient_checkpointing()
    model(latents, timesteps, context).sample["rotation"].square().sum().backward()
    assert model.blocks[0].self_attn.streams["pose"].to_qkv.weight.grad is not None

    model.eval().save_pretrained(tmp_path)
    loaded = Sam3DSparseStructureFlowModel.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        reloaded = loaded(latents, timesteps, context).sample
    for name in output:
        torch.testing.assert_close(reloaded[name], output[name])


def test_shape_stream_ignores_pose_tokens_but_not_the_shortcut_step():
    model = Sam3DSparseStructureFlowModel(**Sam3DSparseStructureFlowModel.tiny_config()).eval()
    _randomize(model, 1)
    latents, timesteps, context = _flow_inputs(model, seed=2)
    perturbed = {**latents, "translation": latents["translation"] + 3.0}
    with torch.no_grad():
        reference = model(latents, timesteps, context).sample
        moved = model(perturbed, timesteps, context).sample
        shortcut = model(latents, timesteps, context, shortcut_step=torch.tensor(250.0)).sample
    # Shape tokens only attend to themselves; pose tokens attend to the shape stream as well.
    torch.testing.assert_close(moved["shape"], reference["shape"])
    assert not torch.allclose(moved["rotation"], reference["rotation"])
    assert not torch.allclose(shortcut["shape"], reference["shape"])


def test_condition_embedder_token_layout_and_pointmap_dropout(tmp_path):
    embedder = Sam3DConditionEmbedder(**Sam3DConditionEmbedder.tiny_config()).eval()
    _randomize(embedder, 3)
    encoder = Dinov2WithRegistersModel(tiny_dinov2_config()).eval()
    generator = torch.Generator().manual_seed(4)
    images = torch.rand(2, 2, 3, 8, 8, generator=generator)
    masks = (torch.rand(2, 2, 1, 8, 8, generator=generator) > 0.5).float()
    pointmaps = torch.randn(2, 2, 3, 8, 8, generator=generator)
    pointmaps[:, :, :, 0, 0] = float("nan")
    with torch.no_grad():
        image_tokens = Sam3DObjectsPipeline.encode_views(encoder, images)
        mask_tokens = Sam3DObjectsPipeline.encode_views(encoder, masks)
        tokens = embedder(image_tokens["normed"], mask_tokens["normed"], pointmaps).embeddings
        dropped = embedder(image_tokens["normed"], mask_tokens["normed"]).embeddings
    # Normed DINOv2 tokens are 1 class + 4 patch tokens per view; pre-norm tokens keep the 2 registers.
    assert image_tokens["normed"].shape == (2, 2, 5, 12) and image_tokens["prenorm"].shape == (2, 2, 7, 12)
    # The crop and full views of the image and the mask, then 4 + 4 pointmap tokens.
    assert tokens.shape == (2, 28, embedder.embed_dim) == (2, 28, 12)
    torch.testing.assert_close(dropped[:, :20], tokens[:, :20])
    assert not bool(dropped[:, 20:].any())
    assert bool(tokens.isfinite().all())

    embedder.save_pretrained(tmp_path)
    loaded = Sam3DConditionEmbedder.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(image_tokens["normed"], mask_tokens["normed"], pointmaps).embeddings, tokens)
