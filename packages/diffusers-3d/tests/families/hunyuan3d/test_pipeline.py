from __future__ import annotations

import numpy as np
import pytest
import torch

from diffusers_3d import (
    AutoPipelineForImageTo3D,
    Hunyuan3DImageTo3DPipeline,
    ImageCondition,
    MeshAsset,
    Object3DPipelineOutput,
)
from diffusers_3d.families.hunyuan3d.pipeline import recenter_image

from .conftest import make_tiny_hunyuan3d_pipeline, sample_latents

pytestmark = pytest.mark.integration


def _condition() -> ImageCondition:
    rgba = torch.zeros(4, 20, 24)
    rgba[:3] = torch.linspace(0, 1, 3 * 20 * 24).reshape(3, 20, 24)
    rgba[3, 4:15, 6:20] = 1.0
    return ImageCondition(image=rgba)


def test_encode_image_returns_cls_and_patch_tokens_then_zero_unconditional_context(tiny_hunyuan3d_pipeline):
    images = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    with torch.no_grad():
        context = tiny_hunyuan3d_pipeline.encode_image(images, do_classifier_free_guidance=True)
    assert context.shape == (4, 5, 24)
    assert context[:2].any() and not context[2:].any()


def test_construction_warns_about_the_territorial_license():
    with pytest.warns(UserWarning, match="European Union, the United Kingdom, or South Korea"):
        Hunyuan3DImageTo3DPipeline(**make_tiny_hunyuan3d_pipeline().components)


def test_recenter_crops_the_alpha_box_and_composites_on_white():
    rgba = np.zeros((10, 6, 4), dtype=np.uint8)
    rgba[2:7, 1:4] = (200, 100, 50, 255)
    square = recenter_image(rgba, border_ratio=0.0)
    assert square.shape == (10, 10, 3)
    # Upstream's crop drops the last foreground row and column: a 4x2 box scaled by 10/4.
    assert (square[:, :2] == 255).all() and (square[:, 7:] == 255).all()
    assert (square[:, 3:7] == (200, 100, 50)).all()
    with pytest.raises(ValueError, match="no foreground"):
        recenter_image(np.zeros((4, 4, 4), dtype=np.uint8), border_ratio=0.15)


@pytest.mark.portable
def test_extract_mesh_winds_outward_and_keeps_upstream_vertex_scaling():
    axis = torch.linspace(-1, 1, 17)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    occupancy = 0.5 - points.norm(dim=-1)
    mesh = Hunyuan3DImageTo3DPipeline.extract_mesh(occupancy, (-1.0,) * 3 + (1.0,) * 3, 0.0)
    triangles = mesh.vertices[mesh.faces]
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert ((normals * triangles.mean(dim=1)).sum(dim=-1) > 0).all()
    # Vertices are divided by the 17 samples per axis rather than the 16 cells, as upstream does.
    center = torch.full((3,), -1 + 2 * 8 / 17)
    radius = (mesh.vertices - center).norm(dim=-1)
    torch.testing.assert_close(radius.mean(), torch.tensor(0.5 * 16 / 17), atol=1e-2, rtol=0)


@pytest.mark.portable
def test_tiny_pipeline_returns_meshes_and_is_deterministic(tiny_hunyuan3d_pipeline):
    arguments = {"num_inference_steps": 3, "octree_resolution": 12, "num_chunks": 500}
    first = tiny_hunyuan3d_pipeline(
        [_condition(), _condition()], generator=torch.Generator().manual_seed(0), **arguments
    )
    second = tiny_hunyuan3d_pipeline(_condition(), generator=torch.Generator().manual_seed(0), **arguments)
    assert type(first) is Object3DPipelineOutput
    assert len(first.objects) == 2 and all(type(mesh) is MeshAsset for mesh in first.objects)
    assert first.objects[0].coordinate_system.value == "right_handed_y_up"
    assert first.objects[0].vertices.abs().max() <= 1.01
    torch.testing.assert_close(first.objects[0].vertices, second.objects[0].vertices)
    torch.testing.assert_close(first.latents.latents[:1], second.latents.latents)


def test_pipeline_save_load_and_auto_pipeline(tmp_path, tiny_hunyuan3d_pipeline, local_external_components):
    tiny_hunyuan3d_pipeline.save_pretrained(tmp_path)
    # The conditioner is Meta's facebook/dinov2-large cast to float16: never written, loaded from there.
    assert not (tmp_path / "image_encoder").exists()
    calls = local_external_components(Hunyuan3DImageTo3DPipeline, image_encoder=tiny_hunyuan3d_pipeline.image_encoder)
    with pytest.warns(UserWarning, match="Tencent Hunyuan 3D 2.1"):
        loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is Hunyuan3DImageTo3DPipeline
    assert loaded.config.image_size == 16
    torch.testing.assert_close(sample_latents(loaded), sample_latents(tiny_hunyuan3d_pipeline))
    assert calls == [("image_encoder", "facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", None)]
