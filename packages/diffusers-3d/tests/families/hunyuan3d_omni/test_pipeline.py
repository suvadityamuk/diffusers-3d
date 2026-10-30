from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline

from diffusers_3d import AutoPipelineForImageTo3D, MeshAsset, Object3DPipelineOutput
from diffusers_3d.families.hunyuan3d_omni.pipeline import Hunyuan3DOmniImageTo3DPipeline, recenter_image

from .conftest import make_tiny_omni_components, point_cloud, rgba_condition, sample_latents

pytestmark = pytest.mark.integration

_SMALL = {"num_inference_steps": 2, "octree_resolution": 12, "num_chunks": 500, "postprocess": False}


def test_construction_warns_about_the_territorial_license():
    with pytest.warns(UserWarning, match="European Union, the United Kingdom, or South Korea"):
        Hunyuan3DOmniImageTo3DPipeline(**make_tiny_omni_components())


def test_recenter_resizes_the_alpha_box_with_cubic_interpolation():
    rgba = np.zeros((10, 6, 4), dtype=np.uint8)
    rgba[2:7, 1:4] = (200, 100, 50, 255)
    square = recenter_image(rgba, border_ratio=0.0)
    assert square.shape == (10, 10, 3)
    # Upstream's crop drops the last foreground row and column: a 4x2 box scaled by 10/4.
    assert (square[:, :2] == 255).all() and (square[:, 7:] == 255).all()
    assert (square[:, 3:7] == (200, 100, 50)).all()
    with pytest.raises(ValueError, match="no foreground"):
        recenter_image(np.zeros((4, 4, 4), dtype=np.uint8), border_ratio=0.15)


def test_unconditional_context_zeroes_image_tokens_and_keeps_control_tokens(tiny_omni_pipeline):
    images = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    with torch.no_grad():
        context = tiny_omni_pipeline.encode_conditions(images, point_cloud(), "point_cloud", True)
    # 5 DINOv2 tokens, then 7 point tokens and 10 type tokens; the control is shared by both images.
    assert context.shape == (4, 22, 24)
    assert context[:2, :5].any() and not context[2:, :5].any()
    assert torch.equal(context[2:, 5:], context[:2, 5:]) and torch.equal(context[0, 5:], context[1, 5:])


def test_controls_are_validated(tiny_omni_pipeline):
    with pytest.raises(ValueError, match="exactly one"):
        tiny_omni_pipeline(rgba_condition(), **_SMALL)
    with pytest.raises(ValueError, match="exactly one"):
        tiny_omni_pipeline(rgba_condition(), bbox=torch.ones(3), point_cloud=point_cloud(), **_SMALL)
    with pytest.raises(ValueError, match="2 entries for 1 images"):
        tiny_omni_pipeline(rgba_condition(), point_cloud=torch.zeros(2, 4, 3), **_SMALL)


@pytest.mark.portable
def test_extract_mesh_winds_outward_and_divides_by_the_cell_count():
    axis = torch.linspace(-1, 1, 17)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    mesh = Hunyuan3DOmniImageTo3DPipeline.extract_mesh(
        0.5 - points.norm(dim=-1), (-1.0,) * 3 + (1.0,) * 3, 0.0, postprocess_passes=0
    )
    triangles = mesh.vertices[mesh.faces]
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert ((normals * triangles.mean(dim=1)).sum(dim=-1) > 0).all()
    torch.testing.assert_close(mesh.vertices.norm(dim=-1).mean(), torch.tensor(0.5), atol=1e-2, rtol=0)


@pytest.mark.portable
@pytest.mark.parametrize(
    "control",
    [
        {"point_cloud": point_cloud()},
        {"voxels": point_cloud()},
        {"bbox": torch.tensor([0.8, 0.6, 1.0])},
        {"skeleton": torch.linspace(-0.5, 0.5, 4 * 6).reshape(4, 6)},
    ],
    ids=lambda control: next(iter(control)),
)
def test_tiny_pipeline_returns_meshes_and_is_deterministic(tiny_omni_pipeline, control):
    generator = [torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)]
    first = tiny_omni_pipeline([rgba_condition(), rgba_condition()], generator=generator, **control, **_SMALL)
    second = tiny_omni_pipeline(rgba_condition(), generator=torch.Generator().manual_seed(0), **control, **_SMALL)
    assert type(first) is Object3DPipelineOutput
    assert len(first.objects) == 2 and all(type(mesh) is MeshAsset for mesh in first.objects)
    assert first.objects[0].coordinate_system.value == "right_handed_y_up"
    assert first.objects[0].vertices.abs().max() <= 1.01
    assert first.latents.metadata["control"] == next(iter(control))
    # Batched and single-image matmuls round differently, which moves interpolated vertices slightly.
    torch.testing.assert_close(first.objects[0].vertices, second.objects[0].vertices, atol=1e-4, rtol=0)
    torch.testing.assert_close(first.latents.latents[:1], second.latents.latents)


@pytest.mark.portable
def test_clean_up_needs_pymeshlab_and_runs_twice_for_bounding_boxes(tiny_omni_pipeline, monkeypatch):
    from diffusers_3d.families.hunyuan3d_omni import pipeline as module

    monkeypatch.setitem(sys.modules, "pymeshlab", None)
    arguments = {**_SMALL, "postprocess": True}
    with pytest.raises(ImportError, match=r"diffusers-3d\[meshlab\]"):
        tiny_omni_pipeline(rgba_condition(), point_cloud=point_cloud(), **arguments)
    calls = []
    monkeypatch.setattr(module, "remove_floaters_and_degenerate_faces", lambda mesh: calls.append(mesh) or mesh)
    tiny_omni_pipeline(rgba_condition(), point_cloud=point_cloud(), **arguments)
    tiny_omni_pipeline(rgba_condition(), bbox=torch.tensor([0.8, 0.6, 1.0]), **arguments)
    assert len(calls) == 3


def test_the_control_changes_the_sample(tiny_omni_pipeline):
    assert not torch.equal(
        sample_latents(tiny_omni_pipeline), sample_latents(tiny_omni_pipeline, bbox=torch.tensor([0.8, 0.6, 1.0]))
    )


def test_pipeline_save_load_and_auto_pipeline(tmp_path, tiny_omni_pipeline, local_external_components):
    tiny_omni_pipeline.save_pretrained(tmp_path)
    # DINOv2 is Meta's facebook/dinov2-large: never written, loaded from there.
    assert not (tmp_path / "image_encoder").exists()
    calls = local_external_components(Hunyuan3DOmniImageTo3DPipeline, image_encoder=tiny_omni_pipeline.image_encoder)
    with pytest.warns(UserWarning, match="Tencent Hunyuan 3D Omni"):
        loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is Hunyuan3DOmniImageTo3DPipeline
    assert loaded.config.image_size == 16
    torch.testing.assert_close(sample_latents(loaded), sample_latents(tiny_omni_pipeline))
    direct = Hunyuan3DOmniImageTo3DPipeline.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(sample_latents(direct), sample_latents(tiny_omni_pipeline))
    revision = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
    assert calls == [("image_encoder", "facebook/dinov2-large", revision, None)] * 2


def test_hub_ids_download_model_index_and_component_folders_first(
    tiny_omni_pipeline, tmp_path, monkeypatch, local_external_components
):
    import diffusers_3d.execution.pipelines as pipelines

    tiny_omni_pipeline.save_pretrained(tmp_path)
    external = local_external_components(
        Hunyuan3DOmniImageTo3DPipeline, image_encoder=tiny_omni_pipeline.image_encoder
    )
    calls = {}

    def snapshot_download(repo_id, **kwargs):
        calls.update(kwargs, repo_id=repo_id)
        return str(tmp_path)

    monkeypatch.setattr(pipelines.huggingface_hub, "snapshot_download", snapshot_download)
    load_config = DiffusionPipeline.load_config.__func__

    def offline_load_config(cls, name, **kwargs):
        kwargs.pop("revision", None)
        return load_config(cls, tmp_path if name == "org/tiny-omni" else name, **kwargs)

    monkeypatch.setattr(DiffusionPipeline, "load_config", classmethod(offline_load_config))
    with pytest.warns(UserWarning, match="Tencent Hunyuan 3D Omni"):
        loaded = AutoPipelineForImageTo3D.from_pretrained("org/tiny-omni", vae=tiny_omni_pipeline.vae, revision="v1")
    assert type(loaded) is Hunyuan3DOmniImageTo3DPipeline and loaded.vae is tiny_omni_pipeline.vae
    assert calls["repo_id"] == "org/tiny-omni" and calls["revision"] == "v1"
    assert {"control_encoder/*", "transformer/*"} <= set(calls["allow_patterns"])
    assert "vae/*" not in calls["allow_patterns"] and "image_encoder/*" not in calls["allow_patterns"]
    assert [call[:2] for call in external] == [("image_encoder", "facebook/dinov2-large")]
