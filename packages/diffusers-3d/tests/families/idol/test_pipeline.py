from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from diffusers_3d import AutoPipelineForImageTo3D, CoordinateSystem, GaussianSplatAsset, ImageCondition
from diffusers_3d.families.idol import IdolImageTo3DPipeline, decode_idol_camera, prepare_idol_image
from diffusers_3d.families.triposplat import TripoSplatGaussianDecoder

from .configs import tiny_decoder_config
from .conftest import SMALL


def test_prepare_image_crop_margin_follows_the_mode(rgba_condition):
    tight = np.asarray(prepare_idol_image(rgba_condition, image_size=60, crop_margin=1.2))
    loose = np.asarray(prepare_idol_image(rgba_condition, image_size=60, crop_margin=1.7))
    assert tight.shape == loose.shape == (60, 60, 3)
    # The foreground fills 1 / margin of the canvas: a looser crop leaves a wider black border.
    assert (loose[:, :9] == 0).all() and tight[:, 9:12].any()


def test_prepare_image_uses_the_mask_and_rejects_inputs_without_a_background(rgba_condition):
    rgb, mask = rgba_condition.image[:3], rgba_condition.image[3:4]
    np.testing.assert_array_equal(
        np.asarray(prepare_idol_image(ImageCondition(image=rgb, mask=mask), image_size=32)),
        np.asarray(prepare_idol_image(rgba_condition, image_size=32)),
    )
    with pytest.raises(ValueError, match="background removal"):
        prepare_idol_image(ImageCondition(image=rgb))
    with pytest.raises(ValueError, match="must mark a background"):
        prepare_idol_image(ImageCondition(image=rgb, mask=torch.ones_like(mask)))
    with pytest.raises(ValueError, match="no foreground"):
        prepare_idol_image(ImageCondition(image=rgb, mask=torch.zeros_like(mask)))


def test_decode_camera_matches_the_perspective_and_orthographic_formulas():
    camera = decode_idol_camera(torch.tensor([0.0, 2.0, 2.0, 0.5, 0.25]))
    assert camera["yaw"] == 0.0 and math.isclose(camera["pitch"], math.pi / 4, rel_tol=1e-6)
    assert camera["radius"] == 2.0 and math.isclose(camera["fov"], math.degrees(2 * math.atan(0.125)))
    assert decode_idol_camera(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))["radius"] == 1000.0


def test_pipeline_generates_z_up_splats_and_round_trips(
    tiny_idol_pipeline, rgba_condition, tmp_path, local_external_components
):
    pipeline = tiny_idol_pipeline
    calls = local_external_components(IdolImageTo3DPipeline, image_encoder=pipeline.image_encoder, vae=pipeline.vae)
    output = pipeline([rgba_condition, rgba_condition], generator=torch.Generator().manual_seed(0), **SMALL)
    assert len(output.objects) == 2 and output.latents.latents.shape == (2, 10, 4)
    for asset in output.objects:
        assert type(asset) is GaussianSplatAsset
        assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
        assert asset.means.shape == (24, 3)
        assert asset.metadata["family"] == "idol" and set(asset.metadata["camera"]) == {
            "yaw",
            "pitch",
            "radius",
            "fov",
        }
    # 27 rounds down to a multiple of 4 Gaussians per token, as upstream.
    rounded = pipeline(rgba_condition, generator=torch.Generator().manual_seed(0), **{**SMALL, "num_gaussians": 27})
    assert rounded.objects[0].means.shape == (24, 3)
    single = pipeline(rgba_condition, generator=torch.Generator().manual_seed(0), **SMALL).objects[0]
    unguided = pipeline(rgba_condition, guidance_scale=1.0, generator=torch.Generator().manual_seed(0), **SMALL)
    real = pipeline(rgba_condition, mode="real", generator=torch.Generator().manual_seed(0), **SMALL)
    assert not torch.equal(unguided.objects[0].means, single.means)
    assert not torch.equal(real.objects[0].means, single.means)

    pipeline.save_pretrained(tmp_path)
    index = json.loads((tmp_path / "model_index.json").read_text())
    assert index["_class_name"] == "IdolImageTo3DPipeline"
    assert index["image_encoder"] == ["transformers", "DINOv3ViTModel"]
    # DINOv3 and the Flux.2 VAE are gated: never saved, and loaded from their pinned repositories.
    assert not (tmp_path / "image_encoder").exists() and not (tmp_path / "vae").exists()
    # Without a dtype DINOv3 and the VAE load in bfloat16 and the rest in float32, as upstream loads them.
    default = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert [(call[1], call[3]) for call in calls] == [
        ("facebook/dinov3-vith16plus-pretrain-lvd1689m", torch.bfloat16),
        ("black-forest-labs/FLUX.2-dev", torch.bfloat16),
    ]
    assert type(default) is IdolImageTo3DPipeline
    dtypes = [getattr(default, name).dtype for name in ("image_encoder", "vae", "transformer", "decoder")]
    assert dtypes == [torch.bfloat16, torch.bfloat16, torch.float32, torch.float32]
    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True, dtype=torch.float32)
    loaded.set_progress_bar_config(disable=True)
    reloaded = loaded(rgba_condition.image, generator=torch.Generator().manual_seed(0), **SMALL).objects[0]
    assert torch.equal(reloaded.means, single.means)
    assert torch.equal(reloaded.opacity_logits, single.opacity_logits)
    direct = IdolImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    blocks = direct.transformer.blocks[0]
    assert blocks.attn.to_qkv.weight.dtype is torch.float16 and blocks.norm1.weight is None
    assert direct.transformer.blocks[0].shift_table.dtype is torch.float32
    assert direct.transformer.proj_in.weight.dtype is torch.float32
    assert direct.decoder.gaussian_blocks[0].ff.net[2].weight.dtype is torch.float16
    assert all(
        torch.equal(direct.transformer.state_dict()[key], value.to(direct.transformer.state_dict()[key].dtype))
        for key, value in pipeline.transformer.state_dict().items()
    )
    direct.set_progress_bar_config(disable=True)
    mixed = direct(rgba_condition, generator=torch.Generator().manual_seed(0), **SMALL).objects[0]
    torch.testing.assert_close(mixed.means, single.means, atol=5e-2, rtol=0)


def test_given_latents_skip_the_noise_draw_and_bad_arguments_raise(tiny_idol_pipeline, rgba_condition):
    latents = torch.randn(1, 10, 4, generator=torch.Generator().manual_seed(1))
    output = tiny_idol_pipeline(rgba_condition, latents=latents, num_inference_steps=1, num_gaussians=8, image_size=32)
    assert output.latents.latents.shape == (1, 10, 4)
    with pytest.raises(ValueError, match="latents must have shape"):
        tiny_idol_pipeline(rgba_condition, latents=latents[:, :5], image_size=32)
    with pytest.raises(ValueError, match="mode must be one of"):
        tiny_idol_pipeline(rgba_condition, mode="photo", image_size=32)
    # Upstream places at least one anchor: fewer than 4 Gaussians per token still yields one token's worth.
    fewest = tiny_idol_pipeline(rgba_condition, num_inference_steps=1, num_gaussians=3, image_size=32)
    assert fewest.objects[0].means.shape == (4, 3)


def test_mismatched_components_are_rejected(tiny_idol_components):
    components = {**tiny_idol_components}
    components["decoder"] = TripoSplatGaussianDecoder(**{**tiny_decoder_config(), "latent_channels": 8})
    with pytest.raises(ValueError, match="decoder latent channels"):
        IdolImageTo3DPipeline(**components)


def test_hub_ids_download_model_index_and_component_folders_first(
    tiny_idol_pipeline, tmp_path, monkeypatch, local_external_components
):
    import diffusers_3d.execution.pipelines as pipelines

    local_external_components(IdolImageTo3DPipeline, vae=tiny_idol_pipeline.vae)
    tiny_idol_pipeline.save_pretrained(tmp_path)
    calls = {}

    def snapshot_download(repo_id, **kwargs):
        calls.update(kwargs, repo_id=repo_id)
        return str(tmp_path)

    monkeypatch.setattr(pipelines.huggingface_hub, "snapshot_download", snapshot_download)
    load_config = DiffusionPipeline.load_config.__func__

    def offline_load_config(cls, name, **kwargs):
        kwargs.pop("revision", None)
        return load_config(cls, tmp_path if name == "org/tiny-idol" else name, **kwargs)

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(offline_load_config))
    image_encoder = tiny_idol_pipeline.image_encoder
    loaded = AutoPipelineForImageTo3D.from_pretrained("org/tiny-idol", image_encoder=image_encoder, revision="v1")
    assert type(loaded) is IdolImageTo3DPipeline and loaded.image_encoder is image_encoder
    assert calls["repo_id"] == "org/tiny-idol" and calls["revision"] == "v1"
    assert "transformer/*" in calls["allow_patterns"] and "image_encoder/*" not in calls["allow_patterns"]


def test_enable_background_removal_lets_rgb_photos_through(tiny_idol_pipeline, rgba_condition):
    rgb = rgba_condition.image[:3]
    with pytest.raises(ValueError, match="background removal"):
        tiny_idol_pipeline.encode_image(rgb, image_size=32)
    alpha = Image.fromarray((rgba_condition.image[3].numpy() * 255).astype("uint8"), "L")
    sizes = []

    def remover(image):
        sizes.append(image.size)
        image = image.convert("RGBA")
        image.putalpha(alpha.resize(image.size, Image.Resampling.NEAREST))
        return image

    tiny_idol_pipeline.enable_background_removal(remover)
    try:
        expected = tiny_idol_pipeline.encode_image(rgba_condition, image_size=32)[0]
        assert tiny_idol_pipeline.encode_image(rgb, image_size=32)[0].shape == expected.shape
        assert min(sizes[0]) == 1024  # upstream cuts out after resizing the shorter side to 1024
    finally:
        tiny_idol_pipeline.disable_background_removal()
