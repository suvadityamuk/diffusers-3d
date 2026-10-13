from __future__ import annotations

import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    CoordinateSystem,
    ImageCondition,
    MeshAsset,
    Object3DPipelineOutput,
    Step1X3DImageTo3DPipeline,
    preprocess_step1x3d_image,
)

pytestmark = pytest.mark.integration


def test_preprocessing_fits_foreground_on_white_square():
    red = torch.zeros(3, 20, 20)
    red[0] = 1
    framed = preprocess_step1x3d_image(ImageCondition(red), image_size=20)
    assert torch.equal(framed[:, :19, :19], red[:, :19, :19])
    assert torch.equal(framed[:, 19], torch.ones(3, 20)) and torch.equal(framed[:, :, 19], torch.ones(3, 20))

    transparent = torch.zeros(4, 6, 6)
    with pytest.raises(ValueError, match="non-transparent"):
        preprocess_step1x3d_image(ImageCondition(transparent))
    with pytest.raises(ValueError, match="foreground_ratio"):
        preprocess_step1x3d_image(ImageCondition(red), foreground_ratio=0.0)


def test_batch_conditioning_matches_single_image_preprocessing(tiny_step1x3d_pipeline, rgba_condition):
    pipeline = tiny_step1x3d_pipeline
    masked = ImageCondition(rgba_condition.image[:3], mask=rgba_condition.image[3:4])
    images = pipeline.preprocess((rgba_condition, masked))
    expected = preprocess_step1x3d_image(rgba_condition, image_size=pipeline.image_encoder.config.image_size)

    assert images.shape == (2, 3, 8, 8)
    torch.testing.assert_close(images[0], expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(images[1], expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(pipeline.preprocess(rgba_condition.image), expected[None], atol=0.0, rtol=0.0)
    conditional, negative = pipeline.encode_conditioning(images)
    assert conditional.shape == negative.shape == (2, 5, 12)
    assert not negative.any()


def test_sampling_zeroes_the_first_velocity_and_guides_negative_first(tiny_step1x3d_pipeline):
    pipeline = tiny_step1x3d_pipeline
    transformer, scheduler = pipeline.transformer, pipeline.scheduler
    latents = pipeline.prepare_latents(1, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(5)
    conditional, negative = torch.randn(2, 1, 5, 12).unbind(0)

    with torch.no_grad():
        assert torch.equal(pipeline.sample_latents(latents, conditional, negative, num_inference_steps=1), latents)
        actual = pipeline.sample_latents(latents, conditional, negative, num_inference_steps=3, guidance_scale=3.0)
        unguided = pipeline.sample_latents(latents, conditional, negative, num_inference_steps=3, guidance_scale=1.0)

        scheduler.set_timesteps(3)
        expected = latents
        for index, timestep in enumerate(scheduler.timesteps):
            uncond = transformer(expected, timestep, negative).sample
            cond = transformer(expected, timestep, conditional).sample
            velocity = (uncond + 3.0 * (cond - uncond)) * (0.0 if index == 0 else 1.0)
            expected = scheduler.step(velocity, timestep, expected).prev_sample

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, unguided)
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([1000.0, 500.5, 1.0]))


def test_latent_preparation_validates_shapes_and_generators(tiny_step1x3d_pipeline):
    pipeline = tiny_step1x3d_pipeline
    first = pipeline.prepare_latents(2, generator=torch.Generator().manual_seed(0))
    second = pipeline.prepare_latents(2, generator=torch.Generator().manual_seed(0))
    assert first.shape == (2, 8, 4)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="latents must have shape"):
        pipeline.prepare_latents(1, latents=torch.zeros(1, 7, 4))
    with pytest.raises(ValueError, match="one generator per batch item"):
        pipeline.prepare_latents(2, generator=[torch.Generator()])
    with pytest.raises(TypeError, match="ImageCondition"):
        pipeline.preprocess(["image.png"])


@pytest.mark.portable
def test_mesh_extraction_follows_released_marching_cubes_mapping(tiny_step1x3d_pipeline, sphere_field, monkeypatch):
    measure = pytest.importorskip("skimage.measure")
    pipeline = tiny_step1x3d_pipeline
    field = sphere_field(16)
    monkeypatch.setattr(
        pipeline.vae, "decode_grid", lambda latents, **kwargs: field.expand(latents.shape[0], -1, -1, -1)
    )

    mesh = pipeline.extract_meshes(torch.zeros(1, 8, 4), octree_resolution=16)[0]
    vertices, faces, _, _ = measure.marching_cubes(field[0].numpy(), 0.0, method="lewiner")
    expected_vertices = torch.from_numpy(vertices / [17, 17, 17] * 2.1 - 1.05).float()
    expected_faces = torch.from_numpy(faces[:, [2, 1, 0]].copy()).long()

    assert type(mesh) is MeshAsset and mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Y_UP
    assert torch.equal(mesh.vertices, expected_vertices)
    assert torch.equal(mesh.faces, expected_faces)
    triangles = mesh.vertices[mesh.faces]
    signed_volume = torch.linalg.det(triangles.double()).sum() / 6
    assert signed_volume > 0, "faces must wind outward"

    output = pipeline(
        ImageCondition(torch.rand(3, 8, 8)), num_inference_steps=2, octree_resolution=16, generator=torch.Generator()
    )
    assert type(output) is Object3DPipelineOutput
    assert torch.equal(output.objects[0].faces, expected_faces)
    assert output.latents.latents.shape == (1, 8, 4)


def test_pipeline_save_load_auto_and_metadata(tmp_path, tiny_step1x3d_pipeline):
    tiny_step1x3d_pipeline.save_pretrained(tmp_path)
    loaded = Step1X3DImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    automatic = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is type(automatic) is Step1X3DImageTo3DPipeline
    assert type(automatic.scheduler) is type(tiny_step1x3d_pipeline.scheduler)
    for name in ("image_encoder", "transformer", "vae"):
        for key, value in getattr(tiny_step1x3d_pipeline, name).state_dict().items():
            torch.testing.assert_close(getattr(automatic, name).state_dict()[key], value, atol=0.0, rtol=0.0)

    metadata = Step1X3DImageTo3DPipeline.object3d_model_index()
    assert metadata.task_ids == ("image-to-3d",)
    assert metadata.output_representations == ("mesh",)
    assert metadata.required_backends == ("scikit-image",)
    assert {component.name for component in metadata.components} == {
        "image_encoder",
        "transformer",
        "vae",
        "scheduler",
    }


def test_mismatched_components_are_rejected(tiny_step1x3d_components):
    from diffusers_3d import Step1X3DShapeDecoder

    components = dict(tiny_step1x3d_components)
    components["vae"] = Step1X3DShapeDecoder(**{**Step1X3DShapeDecoder.tiny_config(), "latent_channels": 3})
    with pytest.raises(ValueError, match="latent_channels"):
        Step1X3DImageTo3DPipeline(**components)
