from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from _configs import tiny_config
from diffusers import DiffusionPipeline
from PIL import Image

from diffusers_3d import AutoPipelineForImageTo3D, CoordinateSystem, GaussianSplatAsset, ImageCondition
from diffusers_3d.families.triposplat import TripoSplatImageTo3DPipeline, prepare_triposplat_image

from .conftest import SMALL


def test_prepare_image_crops_a_square_around_the_eroded_foreground_on_black(rgba_condition):
    image = np.asarray(prepare_triposplat_image(rgba_condition, image_size=60, erode_radius=0))
    assert image.shape == (60, 60, 3)
    # The 30-pixel-wide input is upscaled 2x, so the foreground is 40x30 inside a 1.2x square crop of 47 pixels.
    assert (image[:5] == 0).all() and (image[:, :8] == 0).all() and image[30, 30].any()
    eroded = np.asarray(prepare_triposplat_image(rgba_condition, image_size=60))
    assert not np.array_equal(eroded, image)


def test_prepare_image_uses_the_mask_and_rejects_inputs_without_a_background(rgba_condition):
    rgb, mask = rgba_condition.image[:3], rgba_condition.image[3:4]
    np.testing.assert_array_equal(
        np.asarray(prepare_triposplat_image(ImageCondition(image=rgb, mask=mask), image_size=32)),
        np.asarray(prepare_triposplat_image(rgba_condition, image_size=32)),
    )
    with pytest.raises(ValueError, match="background removal"):
        prepare_triposplat_image(ImageCondition(image=rgb))
    with pytest.raises(ValueError, match="must mark a background"):
        prepare_triposplat_image(ImageCondition(image=rgb, mask=torch.ones_like(mask)))
    with pytest.raises(ValueError, match="no foreground"):
        prepare_triposplat_image(ImageCondition(image=rgb, mask=torch.zeros_like(mask)))


def test_pipeline_generates_z_up_splats_and_round_trips(
    tiny_triposplat_pipeline, rgba_condition, tmp_path, local_external_components
):
    pipeline = tiny_triposplat_pipeline
    calls = local_external_components(
        TripoSplatImageTo3DPipeline, image_encoder=pipeline.image_encoder, vae=pipeline.vae
    )
    output = pipeline([rgba_condition, rgba_condition], generator=torch.Generator().manual_seed(0), **SMALL)
    assert len(output.objects) == 2 and output.latents.latents.shape == (2, 10, 4)
    for asset in output.objects:
        assert type(asset) is GaussianSplatAsset
        assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
        assert asset.means.shape == (32768, 3)
    # 32770 rounds to the nearest multiple of 4 Gaussians per token, as upstream.
    rounded = pipeline(rgba_condition, generator=torch.Generator().manual_seed(0), **{**SMALL, "num_gaussians": 32770})
    assert rounded.objects[0].means.shape == (32768, 3)
    # Upstream asserts the range before rounding.
    for count in (32767, 262145):
        with pytest.raises(ValueError, match="num_gaussians must be in"):
            pipeline(rgba_condition, **{**SMALL, "num_gaussians": count})
    unguided = pipeline(rgba_condition, guidance_scale=1.0, generator=torch.Generator().manual_seed(0), **SMALL)
    single_output = pipeline(rgba_condition, generator=torch.Generator().manual_seed(0), **SMALL)
    single = single_output.objects[0]
    assert not torch.equal(unguided.objects[0].means, single.means)
    # Guidance passes run one after the other by default (as upstream); batching them is an opt-in.
    batched = pipeline(rgba_condition, batch_guidance=True, generator=torch.Generator().manual_seed(0), **SMALL)
    torch.testing.assert_close(batched.latents.latents, single_output.latents.latents, atol=1e-4, rtol=1e-4)

    pipeline.save_pretrained(tmp_path)
    # DINOv3 and the Flux.2 VAE are gated: never saved, and loaded from their pinned repositories.
    index = json.loads((tmp_path / "model_index.json").read_text())
    assert index["image_encoder"] == ["transformers", "DINOv3ViTModel"] and index["vae"][1] == "TripoSplatImageVAE"
    assert not (tmp_path / "image_encoder").exists() and not (tmp_path / "vae").exists()
    # Without a dtype each component loads in the precision upstream runs it in.
    default = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert calls == [
        (
            "image_encoder",
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "c807c9eeea853df70aec4069e6f56b28ddc82acc",
            torch.bfloat16,
        ),
        ("vae", "black-forest-labs/FLUX.2-dev", "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c", torch.bfloat16),
    ]
    assert type(default) is TripoSplatImageTo3DPipeline
    dtypes = {name: getattr(default, name).dtype for name in ("image_encoder", "vae", "transformer", "decoder")}
    assert dtypes == {
        "image_encoder": torch.bfloat16,
        "vae": torch.bfloat16,
        "transformer": torch.float16,
        "decoder": torch.float16,
    }
    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True, dtype=torch.float32)
    loaded.set_progress_bar_config(disable=True)
    reloaded = loaded(rgba_condition.image, generator=torch.Generator().manual_seed(0), **SMALL).objects[0]
    assert torch.equal(reloaded.means, single.means)
    assert torch.equal(reloaded.opacity_logits, single.opacity_logits)


def test_given_latents_skip_the_noise_draw(tiny_triposplat_pipeline, rgba_condition):
    latents = torch.randn(1, 10, 4, generator=torch.Generator().manual_seed(1))
    output = tiny_triposplat_pipeline(rgba_condition, latents=latents, num_inference_steps=1, image_size=32)
    assert output.latents.latents.shape == (1, 10, 4)
    with pytest.raises(ValueError, match="latents must have shape"):
        tiny_triposplat_pipeline(rgba_condition, latents=latents[:, :5], image_size=32)


def test_mismatched_components_are_rejected(tiny_triposplat_components):
    from diffusers_3d.families.triposplat import TripoSplatGaussianDecoder

    components = {**tiny_triposplat_components}
    components["decoder"] = TripoSplatGaussianDecoder(
        **{**tiny_config(TripoSplatGaussianDecoder), "latent_channels": 8}
    )
    with pytest.raises(ValueError, match="decoder latent channels"):
        TripoSplatImageTo3DPipeline(**components)


def test_hub_ids_download_model_index_and_component_folders_first(
    tiny_triposplat_pipeline, tmp_path, monkeypatch, local_external_components
):
    import diffusers_3d.execution.pipelines as pipelines

    pipeline = tiny_triposplat_pipeline
    local_external_components(TripoSplatImageTo3DPipeline, image_encoder=pipeline.image_encoder, vae=pipeline.vae)
    tiny_triposplat_pipeline.save_pretrained(tmp_path)
    calls = {}

    def snapshot_download(repo_id, **kwargs):
        calls.update(kwargs, repo_id=repo_id)
        return str(tmp_path)

    monkeypatch.setattr(pipelines.huggingface_hub, "snapshot_download", snapshot_download)
    load_config = DiffusionPipeline.load_config.__func__

    def offline_load_config(cls, name, **kwargs):
        kwargs.pop("revision", None)
        return load_config(cls, tmp_path if name == "org/tiny-triposplat" else name, **kwargs)

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(offline_load_config))
    decoder = tiny_triposplat_pipeline.decoder
    loaded = AutoPipelineForImageTo3D.from_pretrained("org/tiny-triposplat", decoder=decoder, revision="v1")
    assert type(loaded) is TripoSplatImageTo3DPipeline and loaded.decoder is decoder
    assert calls["repo_id"] == "org/tiny-triposplat" and calls["revision"] == "v1"
    assert "transformer/*" in calls["allow_patterns"] and "decoder/*" not in calls["allow_patterns"]
    assert "image_encoder/*" not in calls["allow_patterns"] and "vae/*" not in calls["allow_patterns"]


def test_enable_background_removal_lets_rgb_photos_through(tiny_triposplat_pipeline, rgba_condition):
    rgb = rgba_condition.image[:3]
    with pytest.raises(ValueError, match="background removal"):
        tiny_triposplat_pipeline.encode_image(rgb, image_size=32)
    alpha = Image.fromarray((rgba_condition.image[3].numpy() * 255).astype("uint8"), "L")
    sizes = []

    def remover(image):
        sizes.append(image.size)
        image = image.convert("RGBA")
        image.putalpha(alpha.resize(image.size, Image.Resampling.NEAREST))
        return image

    tiny_triposplat_pipeline.enable_background_removal(remover)
    try:
        expected = tiny_triposplat_pipeline.encode_image(rgba_condition, image_size=32)[0]
        assert tiny_triposplat_pipeline.encode_image(rgb, image_size=32)[0].shape == expected.shape
        assert min(sizes[0]) == 1024  # upstream cuts out after resizing the shorter side to 1024
    finally:
        tiny_triposplat_pipeline.disable_background_removal()
