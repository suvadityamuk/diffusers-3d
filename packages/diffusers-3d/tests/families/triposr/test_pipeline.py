from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from diffusers import DiffusionPipeline

from diffusers_3d import AutoPipelineForImageTo3D, CoordinateSystem, ImageCondition, MeshAsset
from diffusers_3d.families.triposr.pipeline import TripoSRImageTo3DPipeline, prepare_triposr_image


def test_prepare_image_crops_pads_and_composites_on_grey_like_upstream(rgba_condition):
    image = prepare_triposr_image(rgba_condition)
    # The crop keeps rows 10-28 and columns 5-18 (upstream drops the last foreground row and column): 19x14, padded
    # to a 19x19 square (2 + 3 columns), then to int(19 / 0.85) = 22 (1 + 2 on each axis).
    assert image.shape == (3, 22, 22)
    torch.testing.assert_close(image[:, :1], torch.full_like(image[:, :1], 127 / 255))
    torch.testing.assert_close(image[:, :, :3], torch.full_like(image[:, :, :3], 127 / 255))
    # The composite is re-quantized by truncation, as upstream saves it with ``astype(np.uint8)``.
    expected = rgba_condition.image[:3, 10:29, 5:19].mul(255).round() / 255
    torch.testing.assert_close(image[:, 1:20, 3:17], expected, atol=1.01 / 255, rtol=0)


def test_prepare_image_uses_the_mask_and_passes_rgb_through(rgba_condition):
    rgb = rgba_condition.image[:3]
    torch.testing.assert_close(
        prepare_triposr_image(ImageCondition(image=rgb, mask=rgba_condition.image[3:])),
        prepare_triposr_image(rgba_condition),
    )
    torch.testing.assert_close(prepare_triposr_image(ImageCondition(image=rgb)), rgb)
    with pytest.raises(ValueError, match="no foreground"):
        prepare_triposr_image(ImageCondition(image=rgb, mask=torch.zeros_like(rgb[:1])))
    with pytest.raises(ValueError, match="foreground_ratio"):
        prepare_triposr_image(rgba_condition, foreground_ratio=0)


def test_extract_mesh_rejects_batches_and_empty_surfaces(tiny_triposr_pipeline):
    with pytest.raises(ValueError, match="one shape at a time"):
        tiny_triposr_pipeline.extract_mesh(torch.zeros(2, 3, 6, 8, 8))
    with pytest.raises(ValueError, match="never crosses threshold"):
        tiny_triposr_pipeline.extract_mesh(torch.zeros(1, 3, 6, 8, 8), resolution=6, threshold=1e9)


def test_extract_mesh_places_and_winds_a_dense_sphere_outward(tmp_path):
    def decoder(triplanes, points, chunk_size):
        density = 50 * torch.exp(-points.norm(dim=-1, keepdim=True))
        return SimpleNamespace(density=density, color=(points + 1) / 2)

    decoder.config = SimpleNamespace(radius=0.87)
    # Only ``self.decoder`` is used; the density crosses 25 on the sphere of radius log(2).
    mesh = TripoSRImageTo3DPipeline.extract_mesh(
        SimpleNamespace(decoder=decoder), torch.zeros(1, 3, 1, 2, 2), resolution=48
    )
    torch.testing.assert_close(
        mesh.vertices.norm(dim=-1), torch.full((len(mesh.vertices),), math.log(2)), atol=2e-3, rtol=0
    )
    torch.testing.assert_close(mesh.colors, (mesh.vertices + 1) / 2)
    assert torch.linalg.det(mesh.vertices[mesh.faces]).sum() > 0
    # Saved raw in the Z-up frame, as upstream ``run.py`` exports its trimesh meshes.
    trimesh = pytest.importorskip("trimesh")
    mesh.save(tmp_path / "mesh.ply")
    saved = trimesh.load(tmp_path / "mesh.ply", process=False)
    assert torch.equal(torch.from_numpy(saved.vertices).float(), mesh.vertices)
    assert torch.equal(torch.from_numpy(saved.faces).long(), mesh.faces)


def test_pipeline_generates_coloured_z_up_meshes_and_round_trips(tiny_triposr_pipeline, rgba_condition, tmp_path):
    pipeline = tiny_triposr_pipeline
    output = pipeline([rgba_condition, rgba_condition.image[:3]], mc_resolution=12, chunk_size=500)
    assert len(output.objects) == 2
    assert output.latents.latents.shape == (2, 3, 6, 8, 8)
    for mesh in output.objects:
        assert type(mesh) is MeshAsset
        assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
        assert mesh.colors.shape == mesh.vertices.shape
        assert mesh.vertices.abs().max() <= 0.87 + 1e-6
    assert not torch.equal(output.objects[0].vertices, output.objects[1].vertices)

    single = pipeline(rgba_condition, mc_resolution=12).objects[0]
    pipeline.save_pretrained(tmp_path)
    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is TripoSRImageTo3DPipeline
    reloaded = loaded(rgba_condition.image, mc_resolution=12).objects[0]
    assert torch.equal(reloaded.vertices, single.vertices)
    assert torch.equal(reloaded.faces, single.faces)
    assert torch.equal(reloaded.colors, single.colors)


def test_render_views_composites_over_white_and_tolerates_rays_missing_the_box(tiny_triposr_pipeline):
    triplanes = torch.randn(2, 3, 6, 8, 8, generator=torch.Generator().manual_seed(0))
    frames = tiny_triposr_pipeline.render_views(triplanes, num_views=3, height=6, width=5, num_samples_per_ray=4)
    assert frames.shape == (2, 3, 6, 5, 3)
    assert not torch.equal(frames[0], frames[1])
    # From far away with a wide field of view the corner rays miss the box (upstream raises there) and stay white.
    far = tiny_triposr_pipeline.render_views(
        triplanes[:1], num_views=1, height=6, width=6, num_samples_per_ray=4, camera_distance=5.0, fovy_deg=60.0
    )
    assert torch.equal(far[0, 0, 0, 0], torch.ones(3)) and not torch.equal(far[0, 0, 3, 3], torch.ones(3))


def test_hub_ids_download_model_index_and_component_folders_first(tiny_triposr_pipeline, tmp_path, monkeypatch):
    import diffusers_3d.execution.pipelines as pipelines

    tiny_triposr_pipeline.save_pretrained(tmp_path)
    calls = {}

    def snapshot_download(repo_id, **kwargs):
        calls.update(kwargs, repo_id=repo_id)
        return str(tmp_path)

    monkeypatch.setattr(pipelines.huggingface_hub, "snapshot_download", snapshot_download)
    load_config = DiffusionPipeline.load_config.__func__

    def offline_load_config(cls, name, **kwargs):
        kwargs.pop("revision", None)
        return load_config(cls, tmp_path if name == "org/tiny-triposr" else name, **kwargs)

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(offline_load_config))
    decoder = tiny_triposr_pipeline.decoder
    loaded = AutoPipelineForImageTo3D.from_pretrained("org/tiny-triposr", decoder=decoder, revision="v1")
    assert type(loaded) is TripoSRImageTo3DPipeline and loaded.decoder is decoder
    assert calls["repo_id"] == "org/tiny-triposr" and calls["revision"] == "v1"
    assert "transformer/*" in calls["allow_patterns"] and "image_encoder/*" in calls["allow_patterns"]
    assert "decoder/*" not in calls["allow_patterns"]


def test_enable_background_removal_lets_rgb_photos_through(tiny_triposr_pipeline, rgba_condition):
    from PIL import Image

    rgb = rgba_condition.image[:3]
    alpha = Image.fromarray((rgba_condition.image[3].numpy() * 255).astype("uint8"), "L")

    def remover(image):
        image = image.convert("RGBA")
        image.putalpha(alpha)
        return image

    expected = tiny_triposr_pipeline.preprocess(rgba_condition)
    assert tiny_triposr_pipeline.preprocess(rgb).shape == expected.shape
    assert not torch.equal(tiny_triposr_pipeline.preprocess(rgb), expected)
    tiny_triposr_pipeline.enable_background_removal(remover)
    try:
        # The remover sees 8-bit RGB, so pixels match the RGBA path to within quantization.
        torch.testing.assert_close(tiny_triposr_pipeline.preprocess(rgb), expected, atol=0.05, rtol=0)
    finally:
        tiny_triposr_pipeline.disable_background_removal()
