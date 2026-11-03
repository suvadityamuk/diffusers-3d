from __future__ import annotations

import warnings

import pytest
import torch

from diffusers_3d import (
    AutoPipelineFor3D,
    Hunyuan3DPaintPipeline,
    ImageCondition,
    MeshAsset,
    Object3DPipelineOutput,
)
from diffusers_3d.backends import TrimeshBackend
from diffusers_3d.families.hunyuan3d.paint_pipeline import (
    CANDIDATE_VIEWS,
    render_conditions,
    renderer_vertices,
    select_views,
)

from .conftest import TINY_PAINT_SIZES, make_tiny_paint_pipeline

pytestmark = [pytest.mark.integration, pytest.mark.portable]


def _mesh() -> MeshAsset:
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(subdivisions=2)
    return MeshAsset(vertices=torch.tensor(sphere.vertices, dtype=torch.float32), faces=torch.tensor(sphere.faces))


def _image() -> ImageCondition:
    rgba = torch.zeros(4, 24, 20)
    rgba[:3] = torch.linspace(0, 1, 3 * 24 * 20).reshape(3, 24, 20)
    rgba[3, 3:20, 4:16] = 1.0
    return ImageCondition(image=rgba)


def _paint(pipeline: Hunyuan3DPaintPipeline, **kwargs):
    options = {"num_inference_steps": 2, "max_faces": None, "generator": torch.Generator().manual_seed(0)}
    return pipeline(_mesh(), _image(), **TINY_PAINT_SIZES, **(options | kwargs))


def test_paints_a_uv_textured_mesh_with_a_pbr_material():
    output = _paint(make_tiny_paint_pipeline())
    assert isinstance(output, Object3DPipelineOutput)
    (mesh,) = output.objects
    material = mesh.materials[0]
    assert mesh.uvs is not None and mesh.uvs.shape == (mesh.vertices.shape[0], 2)
    assert material.base_color.shape == (16, 16, 3)
    assert material.metallic.shape == material.roughness.shape == (16, 16, 1)
    assert mesh.metadata["views"] == [list(view) for view in CANDIDATE_VIEWS[:6]]
    assert output.latents.latents.shape == (1, 2, 6, 4, 8, 8)
    assert [preview.shape for preview in output.previews] == [(6, 3, 16, 16)] * 2


def _external(pipeline: Hunyuan3DPaintPipeline) -> dict:
    """The third-party components, which save_pretrained leaves out and from_pretrained would download."""

    return {name: getattr(pipeline, name) for name in Hunyuan3DPaintPipeline.external_components}


def test_round_trips_through_save_pretrained_and_the_auto_pipeline(tmp_path):
    pipeline = make_tiny_paint_pipeline()
    expected = _paint(pipeline, return_dict=False)
    pipeline.save_pretrained(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model_index.json", "scheduler", "unet"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        loaded = AutoPipelineFor3D.from_pretrained(tmp_path, **_external(pipeline))
    assert type(loaded) is Hunyuan3DPaintPipeline
    actual = _paint(loaded, return_dict=False)
    torch.testing.assert_close(actual[0][0].materials[0].base_color, expected[0][0].materials[0].base_color)
    torch.testing.assert_close(actual[1].latents, expected[1].latents)


def test_the_upscaler_can_be_left_out_when_loading(tmp_path):
    pipeline = make_tiny_paint_pipeline()
    pipeline.save_pretrained(tmp_path)
    components = _external(pipeline) | {"upscaler": None}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Hunyuan3D-2.1 weights", category=UserWarning)
        assert AutoPipelineFor3D.from_pretrained(tmp_path, **components).upscaler is None
        assert Hunyuan3DPaintPipeline.from_pretrained(tmp_path, **components).upscaler is None


def test_paints_without_the_upscaler_or_guidance_and_exports_a_glb(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    output = _paint(make_tiny_paint_pipeline(upscaler=False), guidance_scale=1.0)
    path = tmp_path / "textured.glb"
    TrimeshBackend().export_mesh(output.objects[0], path)
    material = trimesh.load(path, force="mesh").visual.material
    assert material.baseColorTexture.size == material.metallicRoughnessTexture.size == (16, 16)


def test_view_selection_starts_from_the_six_fixed_views_and_renders_white_backgrounds():
    mesh = _mesh()
    vertices = renderer_vertices(mesh.vertices)
    assert torch.isclose((vertices - (vertices.amax(0) + vertices.amin(0)) / 2).norm(dim=1).max(), torch.tensor(0.575))
    views = select_views(vertices, mesh.faces, 9, 64)
    assert views[:6] == list(CANDIDATE_VIEWS[:6])
    normals, positions = render_conditions(vertices, mesh.faces, views[:2], 32)
    assert normals.shape == positions.shape == (2, 32, 32, 3) and normals.dtype is torch.uint8
    assert (normals[:, 0, 0] == 255).all() and (positions[:, 0, 0] == 255).all()
    assert (positions[:, 16, 16] < 255).all()


def test_construction_warns_about_the_territorial_license():
    with pytest.warns(UserWarning, match="European Union, the United Kingdom, or South Korea"):
        Hunyuan3DPaintPipeline(**make_tiny_paint_pipeline().components)


def test_prepare_mesh_without_remeshing_reads_the_glb_frame_and_unwraps():
    pytest.importorskip("xatlas")
    mesh = _mesh()
    rotated = MeshAsset(
        vertices=mesh.vertices, faces=mesh.faces, metadata={"export_rotation": [[1, 0, 0], [0, 0, 1], [0, -1, 0]]}
    )
    prepared = make_tiny_paint_pipeline().prepare_mesh(rotated, max_faces=None)
    assert prepared.uvs is not None and prepared.faces.shape == mesh.faces.shape
    assert prepared.vertices.dtype is torch.float32
    # Every unwrapped vertex is an input vertex as the GLB export writes it (y, z) -> (z, -y).
    expected = torch.stack([mesh.vertices[:, 0], mesh.vertices[:, 2], -mesh.vertices[:, 1]], dim=1)
    assert torch.equal(
        prepared.vertices[:, None].eq(expected[None]).all(-1).any(-1),
        torch.ones(len(prepared.vertices), dtype=torch.bool),
    )
