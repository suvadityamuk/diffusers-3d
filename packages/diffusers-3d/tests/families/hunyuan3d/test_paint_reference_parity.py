from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)

from diffusers_3d import HUNYUAN3D_REFERENCE_REVISION, Hunyuan3DPaintUNet
from diffusers_3d.backends import (
    look_at_view,
    merge_views,
    orthographic_projection,
    project_view,
    rasterize,
    rasterize_texels,
    splat_view,
)
from diffusers_3d.families.hunyuan3d.paint_conversion import convert_unet_state_dict, unet_config
from diffusers_3d.families.hunyuan3d.paint_models import position_voxel_indices, rotary_embedding
from diffusers_3d.families.hunyuan3d.paint_pipeline import CANDIDATE_VIEWS, renderer_vertices, select_views

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT", "/tmp/Hunyuan3D-2.1"))
REFERENCE_REPOSITORY = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
REFERENCE_PATHS = (
    "hy3dpaint/DifferentiableRenderer/MeshRender.py",
    "hy3dpaint/DifferentiableRenderer/camera_utils.py",
    "hy3dpaint/custom_rasterizer/custom_rasterizer/render.py",
    "hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py",
    "hy3dpaint/hunyuanpaintpbr/unet/modules.py",
    "hy3dpaint/utils/pipeline_utils.py",
)
REFERENCE_PACKAGE = "_diffusers_3d_hunyuan3d_paint_reference"
# The released UNet config with tiny sizes; cross-attention and DINOv2 widths are fixed by the reference model.
TINY_UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "act_fn": "silu",
    "attention_head_dim": [1, 2],
    "block_out_channels": [16, 32],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 8,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "sample_size": 8,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
    "use_linear_projection": True,
}


def _package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _rasterize_image(vertices, faces, depth_prior, width, height, occlusion_truncation, use_depth_prior):
    """``custom_rasterizer_kernel.rasterize_image`` backed by the package rasterizer (1-based face ids)."""

    raster = rasterize(vertices, faces.long(), (height, width), align_corners=True)
    return (raster.face_index[0] + 1).to(torch.int32), raster.barycentrics[0]


def _reference(module: str) -> types.ModuleType:
    """Import a pinned ``hy3dpaint`` module with the CUDA rasterizer kernel replaced by the package rasterizer."""

    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=HUNYUAN3D_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        for dependency in ("cv2", "einops", "trimesh"):
            import_reference_dependency(dependency)
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
    if REFERENCE_PACKAGE not in sys.modules:
        source = REFERENCE_ROOT / "hy3dpaint"
        _package(REFERENCE_PACKAGE, source)
        for package in ("DifferentiableRenderer", "hunyuanpaintpbr", "hunyuanpaintpbr/unet", "utils"):
            _package(f"{REFERENCE_PACKAGE}.{package.replace('/', '.')}", source / package)
        kernel = types.ModuleType("custom_rasterizer_kernel")
        kernel.rasterize_image = _rasterize_image
        sys.modules["custom_rasterizer_kernel"] = kernel
        _package("custom_rasterizer", source / "custom_rasterizer" / "custom_rasterizer")
        sys.modules["custom_rasterizer"].__dict__.update(vars(importlib.import_module("custom_rasterizer.render")))
        # The released material attention moves every material to "cuda:0"; drop that to run on the CPU.
        name = f"{REFERENCE_PACKAGE}.hunyuanpaintpbr.unet.attn_processor"
        path = source / "hunyuanpaintpbr" / "unet" / "attn_processor.py"
        attn_processor = types.ModuleType(name)
        attn_processor.__file__ = str(path)
        code = path.read_text(encoding="utf-8").replace('.to("cuda:0")', "")
        exec(compile(code, str(path), "exec"), attn_processor.__dict__)
        sys.modules[name] = attn_processor
    sys.dont_write_bytecode = True
    return importlib.import_module(f"{REFERENCE_PACKAGE}.{module}")


def _bored_block() -> tuple[np.ndarray, np.ndarray]:
    """A Y-up block with an oblique tunnel, wound outwards: the six fixed views miss most of the tunnel."""

    measure = pytest.importorskip("skimage.measure")
    elevation, azimuth = math.radians(20), math.radians(120)
    direction = np.array(
        [-math.cos(elevation) * math.cos(azimuth), math.sin(elevation), math.cos(elevation) * math.sin(azimuth)]
    )
    axis = np.linspace(-1, 1, 24)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    radial = np.linalg.norm(points - (points @ direction)[..., None] * direction, axis=-1)
    field = np.maximum(np.abs(points).max(axis=-1) - 0.8, 0.25 - radial)
    vertices, faces, _, _ = measure.marching_cubes(field, 0.0, spacing=(2 / 23,) * 3)
    return (vertices - 1).astype(np.float32), faces.astype(np.int64)


def _face_atlas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every face onto its own vertices and give it a triangle in its own cell of a square UV grid."""

    cells = math.ceil(math.sqrt(len(faces)))
    index = np.arange(len(faces))
    origin = np.stack([index % cells, index // cells], axis=1) / cells
    corners = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85]]) / cells
    uvs = (origin[:, None] + corners).reshape(-1, 2).astype(np.float32)
    return vertices[faces].reshape(-1, 3), np.arange(3 * len(faces)).reshape(-1, 3), uvs


def _mesh_render(resolution: int, texture_size: int, **options):
    vertices, faces, uvs = _face_atlas(*_bored_block())
    render = _reference("DifferentiableRenderer.MeshRender").MeshRender(
        default_resolution=resolution, texture_size=texture_size, device="cpu", **options
    )
    # set_mesh modifies the arrays it is given in place.
    render.set_mesh(vertices.copy(), faces.astype(np.int32), vtx_uv=uvs.copy(), uv_idx=faces.astype(np.int32))
    return render, vertices, torch.from_numpy(faces), torch.from_numpy(uvs)


def test_paint_unet_matches_reference():
    modules = _reference("hunyuanpaintpbr.unet.modules")
    from diffusers import UNet2DConditionModel

    reference = modules.UNet2p5DConditionModel(UNet2DConditionModel(**TINY_UNET_CONFIG))
    conv_in = reference.unet.conv_in
    reference.unet.conv_in = torch.nn.Conv2d(12, conv_in.out_channels, 3, padding=1)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for value in reference.state_dict().values():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.2)
    reference.eval()
    model = Hunyuan3DPaintUNet(**unet_config(TINY_UNET_CONFIG)).eval()
    model.load_state_dict(convert_unet_state_dict(reference.state_dict()), strict=True)

    batch, views = 3, 3
    sample = torch.randn(1, 2, views, 4, 8, 8, generator=generator).expand(batch, -1, -1, -1, -1, -1)
    normals, positions = (
        torch.randn(1, views, 4, 8, 8, generator=generator).expand(batch, -1, -1, -1, -1) for _ in "np"
    )
    reference_latents = torch.randn(1, 1, 4, 8, 8, generator=generator).expand(batch, -1, -1, -1, -1)
    position_maps = torch.rand(views, 3, 64, 64, generator=generator)
    position_maps[:, :, :20] = 1.0
    dino = torch.randn(1, 5, 1536, generator=generator)
    dino = torch.cat([torch.zeros_like(dino), torch.zeros_like(dino), dino])
    scale = torch.tensor([0.0, 1.0, 1.0])
    text = torch.stack([reference.unet.learned_text_clip_albedo, reference.unet.learned_text_clip_mr])
    with torch.no_grad():
        expected = reference(
            sample,
            torch.tensor(500),
            text.expand(batch, -1, -1, -1),
            embeds_normal=normals,
            embeds_position=positions,
            ref_latents=reference_latents,
            dino_hidden_states=dino,
            position_maps=position_maps.half().expand(batch, -1, -1, -1, -1),
            ref_scale=scale,
        )[0]
        features = model.encode_reference(reference_latents[:, 0])
        actual = model(
            sample,
            torch.tensor(500),
            torch.cat([normals[0], positions[0]], dim=1),
            position_maps,
            dino,
            features,
            scale,
        ).sample
    torch.testing.assert_close(actual.flatten(0, 2), expected, rtol=0, atol=1e-5)


def test_paint_voxel_indices_and_rotary_embedding_match_reference():
    modules = _reference("hunyuanpaintpbr.unet.modules")
    attn_processor = _reference("hunyuanpaintpbr.unet.attn_processor")
    generator = torch.Generator().manual_seed(1)
    position_maps = (torch.rand(2, 3, 64, 64, generator=generator) * 255).round() / 255
    position_maps[:, :, :24, :40] = 1.0
    # The released pipeline passes float16 maps, which the reference modifies in place.
    expected = modules.calc_multires_voxel_idxs(
        position_maps.half()[None], grid_resolutions=[16, 8, 4, 2], voxel_resolutions=[128, 64, 32, 16]
    )
    actual = position_voxel_indices(position_maps, 16)
    assert actual.keys() == expected.keys()
    rotary = attn_processor.RotaryEmbedding
    for length, (voxels, resolution) in actual.items():
        assert resolution == expected[length]["voxel_resolution"]
        assert torch.equal(voxels, expected[length]["voxel_indices"][0])
        cos, sin = rotary.get_3d_rotary_pos_embed(expected[length]["voxel_indices"], 64, resolution)
        embedding = rotary_embedding(voxels, resolution, 64)
        assert torch.equal(embedding[0], cos[0]) and torch.equal(embedding[1], sin[0])
        query = torch.randn(1, 2, length, 64, generator=generator)
        from diffusers.models.embeddings import apply_rotary_emb

        assert torch.equal(apply_rotary_emb(query, embedding), rotary.apply_rotary_emb(query, (cos, sin)))


def test_paint_view_selection_and_cameras_match_reference():
    render, vertices, faces, _ = _mesh_render(1024, 16)
    camera_utils = _reference("DifferentiableRenderer.camera_utils")
    processor = _reference("utils.pipeline_utils").ViewProcessor(types.SimpleNamespace(), render)
    elevations, azimuths, weights = (list(values) for values in zip(*CANDIDATE_VIEWS))
    expected = processor.bake_view_selection(elevations, azimuths, weights, 9)

    renderer = renderer_vertices(torch.from_numpy(vertices))
    torch.testing.assert_close(renderer, render.vtx_pos, rtol=0, atol=1e-7)
    for elevation, azimuth, _ in CANDIDATE_VIEWS:
        expected_camera = torch.from_numpy(camera_utils.get_mv_matrix(elevation, azimuth, 1.45))
        torch.testing.assert_close(look_at_view(elevation, azimuth, 1.45), expected_camera, rtol=0, atol=1e-7)
    assert torch.equal(orthographic_projection(1.2), torch.from_numpy(render.camera_proj_mat))
    actual = select_views(render.vtx_pos, faces, 9, 1024)
    assert len(actual) > 6
    assert [tuple(view) for view in zip(*expected)] == actual


@pytest.mark.parametrize("mode", ["back_sample", "linear"])
def test_paint_bake_and_merge_match_reference(mode):
    smooth = mode == "linear"
    render, _, faces, uvs = _mesh_render(256, 512, bake_mode=mode, shader_type="vertex" if smooth else "face")
    generator = torch.Generator().manual_seed(2)
    projection = torch.from_numpy(render.camera_proj_mat)
    texels = rasterize_texels(render.vtx_pos, faces, uvs, 512)
    expected_textures, expected_weights, textures, weights = [], [], [], []
    for elevation, azimuth, view_weight in CANDIDATE_VIEWS[:6]:
        image = torch.rand(256, 256, 3, generator=generator)
        texture, cosine, _ = render.back_project(image, elevation, azimuth)
        expected_textures.append(texture)
        expected_weights.append(view_weight * cosine**4)
        world_to_camera = look_at_view(elevation, azimuth, 1.45)
        options = {"unreliable_radius": render.bake_unreliable_kernel_size, "smooth_normals": smooth}
        if smooth:
            texture, cosine = splat_view(
                image, render.vtx_pos, faces, uvs, world_to_camera, projection, 512, **options
            )
        else:
            texture, cosine = project_view(
                image, render.vtx_pos, faces, world_to_camera, projection, texels, 512, **options
            )
        textures.append(texture)
        weights.append(view_weight * cosine**4)
        torch.testing.assert_close(texture, expected_textures[-1], rtol=0, atol=1e-5)
        torch.testing.assert_close(weights[-1], expected_weights[-1][..., 0], rtol=0, atol=1e-5)
    expected_texture, expected_mask = render.fast_bake_texture(expected_textures, expected_weights)
    texture, mask = merge_views(textures, weights)
    assert torch.equal(mask, expected_mask[..., 0])
    torch.testing.assert_close(texture, expected_texture, rtol=0, atol=1e-5)
    assert mask.sum() > 10000
    assert math.isclose(float(texture[mask].std()), float(expected_texture[expected_mask[..., 0]].std()), rel_tol=1e-4)
