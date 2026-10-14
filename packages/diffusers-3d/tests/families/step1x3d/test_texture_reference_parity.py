from __future__ import annotations

import ast
import contextlib
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler, EulerDiscreteScheduler
from PIL import Image

from diffusers_3d import ImageCondition, Step1X3DMultiViewAdapter, preprocess_step1x3d_reference_image
from diffusers_3d._reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from diffusers_3d.backends import look_at_view, merge_views, orthographic_projection
from diffusers_3d.backends.mesh_rasterizer import Rasterization, interpolate, rasterize
from diffusers_3d.data.image_processing import _quantized_rgba
from diffusers_3d.families.step1x3d import texture_pipeline
from diffusers_3d.families.step1x3d.conversion import STEP1X3D_REFERENCE_REVISION
from diffusers_3d.families.step1x3d.texture_conversion import shifted_snr_betas
from diffusers_3d.families.step1x3d.texture_pipeline import (
    STEP1X3D_BAKE_EXPONENT,
    STEP1X3D_BAKE_VIEWS,
    render_step1x3d_control_images,
    step1x3d_orthographic_projection,
    step1x3d_view_matrices,
)
from diffusers_3d.families.step1x3d.texture_sync import Step1X3DLatentTextureSync, nearest_valid_index

from .conftest import make_tiny_step1x3d_texture_components

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_STEP1X3D_REFERENCE_ROOT", "/tmp/Step1X-3D"))
REFERENCE_REPOSITORY = "https://github.com/stepfun-ai/Step1X-3D.git"
REFERENCE_PATHS = (
    "step1x3d_texture/custom_rasterizer/custom_rasterizer/render.py",
    "step1x3d_texture/differentiable_renderer/mesh_render.py",
    "step1x3d_texture/models/attention_processor.py",
    "step1x3d_texture/schedulers/scheduler_utils.py",
    "step1x3d_texture/schedulers/scheduling_shift_snr.py",
    "step1x3d_texture/texture_sync/step_sync.py",
    "step1x3d_texture/utils/camera.py",
    "step1x3d_texture/utils/logging.py",
    "step1x3d_texture/utils/render.py",
    "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py",
)
# stabilityai/stable-diffusion-xl-base-1.0 scheduler/scheduler_config.json
SDXL_SCHEDULER_CONFIG = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False,
}
_REFERENCE: types.SimpleNamespace | None = None


def _module(name: str, path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    return module


def _rasterize_image(vertices, faces, depth_prior, width, height, occlusion_truncation, use_depth_prior):
    """``custom_rasterizer_kernel.rasterize_image`` backed by the package rasterizer (1-based face ids)."""

    raster = rasterize(vertices, faces.long(), (height, width), align_corners=True)
    return (raster.face_index[0] + 1).to(torch.int32), raster.barycentrics[0]


@contextlib.contextmanager
def _upstream_import_environment():
    """Expose the pinned ``step1x3d_texture`` sources without their package ``__init__`` files.

    ``utils/__init__`` imports the training utilities, and ``utils/render`` imports nvdiffrast (CUDA-only), whose
    contexts the tests replace with the package rasterizer. ``custom_rasterizer``'s CUDA kernel is replaced the same
    way.
    """

    root = REFERENCE_ROOT / "step1x3d_texture"
    stubs = {"nvdiffrast": _module("nvdiffrast"), "nvdiffrast.torch": _module("nvdiffrast.torch")}
    stubs["step1x3d_texture"] = _module("step1x3d_texture", root)
    for package in ("models", "schedulers", "texture_sync", "utils", "differentiable_renderer"):
        stubs[f"step1x3d_texture.{package}"] = _module(f"step1x3d_texture.{package}", root / package)
    stubs["custom_rasterizer_kernel"] = _module("custom_rasterizer_kernel")
    stubs["custom_rasterizer_kernel"].rasterize_image = _rasterize_image
    stubs["custom_rasterizer"] = _module("custom_rasterizer", root / "custom_rasterizer" / "custom_rasterizer")
    saved = {name: sys.modules.get(name) for name in (*stubs, "custom_rasterizer.render")}
    sys.modules.update(stubs)
    sys.modules.pop("custom_rasterizer.render", None)
    try:
        stubs["custom_rasterizer"].__dict__.update(vars(importlib.import_module("custom_rasterizer.render")))
        yield
    finally:
        for name in [name for name in sys.modules if name.startswith("step1x3d_texture")]:
            del sys.modules[name]
        for name, value in saved.items():
            if value is not None:
                sys.modules[name] = value
            else:
                sys.modules.pop(name, None)


def _pipeline_definition(name: str):
    """A function or class of the texture pipeline module, which imports pytorch3d and BiRefNet, so it is parsed."""

    path = REFERENCE_ROOT / "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    namespace = {"np": np, "Image": Image, "torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _load_reference(*dependencies: str) -> types.SimpleNamespace:
    global _REFERENCE
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=STEP1X3D_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        for dependency in ("einops", "trimesh", "xatlas", *dependencies):
            import_reference_dependency(dependency)
    except ReferenceCheckoutError as error:
        try:
            reason = reference_unavailable(error)
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
        pytest.skip(reason)
    if _REFERENCE is None:
        with _upstream_import_environment():
            _REFERENCE = types.SimpleNamespace(
                attention=importlib.import_module("step1x3d_texture.models.attention_processor"),
                ShiftSNRScheduler=importlib.import_module(
                    "step1x3d_texture.schedulers.scheduling_shift_snr"
                ).ShiftSNRScheduler,
                step_tex_sync=importlib.import_module("step1x3d_texture.texture_sync.step_sync").step_tex_sync,
                camera=importlib.import_module("step1x3d_texture.utils.camera"),
                render=importlib.import_module("step1x3d_texture.utils.render"),
                preprocess_image=_pipeline_definition("preprocess_image"),
            )
    return _REFERENCE


def test_shifted_snr_scheduler_matches_reference():
    reference = _load_reference()
    upstream = reference.ShiftSNRScheduler.from_scheduler(
        EulerDiscreteScheduler.from_config(SDXL_SCHEDULER_CONFIG),
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    betas = shifted_snr_betas(DDPMScheduler.from_config(SDXL_SCHEDULER_CONFIG).alphas_cumprod)
    ours = DDPMScheduler.from_config(SDXL_SCHEDULER_CONFIG, trained_betas=betas.tolist())

    assert torch.equal(ours.betas, upstream.betas)
    assert torch.equal(ours.alphas_cumprod, upstream.alphas_cumprod)
    upstream.set_timesteps(50)
    ours.set_timesteps(50)
    assert torch.equal(ours.timesteps, upstream.timesteps)


def test_multiview_attention_processors_match_reference(tiny_clip_tokenizer):
    reference = _load_reference()
    components = make_tiny_step1x3d_texture_components(tiny_clip_tokenizer)
    unet = components["unet"].eval()
    stock = unet.attn_processors

    def processor(use_branches):
        def make(name, hidden_size, cross_attention_dim, original):
            return reference.attention.DecoupledMVRowColSelfAttnProcessor2_0(
                query_dim=hidden_size,
                inner_dim=hidden_size,
                num_views=6,
                name=name,
                use_mv=use_branches,
                use_ref=use_branches,
            )

        return make

    reference.attention.set_unet_2d_condition_attn_processor(
        unet, set_self_attn_proc_func=processor(True), set_cross_attn_proc_func=processor(False)
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in unet.named_parameters():
            if ".processor." in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    adapter = Step1X3DMultiViewAdapter(hidden_sizes=Step1X3DMultiViewAdapter.hidden_sizes_for(unet))
    adapter.load_state_dict({key: value for key, value in unet.state_dict().items() if ".processor." in key})

    latents, reference_latents = (
        torch.randn(12, 4, 8, 8, generator=generator),
        torch.randn(1, 4, 8, 8, generator=generator),
    )
    context = torch.randn(12, 5, 16, generator=generator)
    added = {
        "text_embeds": torch.randn(12, 8, generator=generator),
        "time_ids": torch.rand(12, 6, generator=generator),
    }
    reference_added = {name: value[-1:] for name, value in added.items()}
    timestep = torch.tensor(500)

    with torch.no_grad():
        cache = {}
        unet(
            reference_latents,
            0,
            encoder_hidden_states=context[-1:],
            added_cond_kwargs=reference_added,
            cross_attention_kwargs={"cache_hidden_states": cache, "use_mv": False, "use_ref": False},
        )
        states = {key: value.repeat_interleave(6, dim=0) for key, value in cache.items()}
        states = {key: torch.cat([torch.zeros_like(value), value]) for key, value in states.items()}
        expected = unet(
            latents,
            timestep,
            encoder_hidden_states=context,
            added_cond_kwargs=added,
            cross_attention_kwargs={"mv_scale": 1.0, "ref_hidden_states": states, "ref_scale": 0.7, "num_views": 6},
        ).sample

        unet.set_attn_processor(stock)
        unet.set_attn_processor(adapter.attn_processors_for(unet))
        cache = {}
        unet(
            reference_latents,
            0,
            encoder_hidden_states=context[-1:],
            added_cond_kwargs=reference_added,
            cross_attention_kwargs={"reference_cache": cache},
        )
        states = {key: value.repeat_interleave(6, dim=0) for key, value in cache.items()}
        states = {key: torch.cat([torch.zeros_like(value), value]) for key, value in states.items()}
        actual = unet(
            latents,
            timestep,
            encoder_hidden_states=context,
            added_cond_kwargs=added,
            cross_attention_kwargs={"reference_hidden_states": states, "ref_scale": 0.7},
        ).sample

    error = (actual - expected).abs().max().item()
    print(f"multi-view attention max abs error {error:.3e}")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _test_mesh():
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    box = trimesh.creation.box(extents=(0.3, 0.2, 0.25))
    box.apply_translation((0.45, 0.2, 0.3))
    return trimesh.util.concatenate([sphere, box])


class _NvdiffrastStandIn:
    """nvdiffrast's ``rasterize`` and ``interpolate`` outputs, computed by the package rasterizer."""

    def rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        raster = rasterize(pos, tri.long(), resolution)
        weights = raster.barycentrics
        return torch.cat(
            [weights[..., :2], raster.depth[..., None], (raster.face_index + 1)[..., None].float()], -1
        ), None

    def interpolate(self, attr, rast, tri, rast_db=None, diff_attrs=None):
        face_index = rast[..., 3].long() - 1
        weights = torch.cat([rast[..., :2], 1 - rast[..., :2].sum(dim=-1, keepdim=True)], dim=-1)
        weights = torch.where(face_index[..., None] >= 0, weights, torch.zeros_like(weights))
        raster = Rasterization(face_index, weights, rast[..., 2])
        return interpolate(attr[0] if attr.ndim == 3 else attr, raster, tri.long()), None


def test_control_images_match_reference_render_maths(tiny_clip_tokenizer):
    reference = _load_reference()
    from diffusers_3d import MeshAsset, Step1X3DMeshTexturePipeline

    native = _test_mesh()
    upstream_mesh, _ = reference.render.load_mesh(native.copy(), rescale=True)
    cameras = reference.camera.get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )
    upstream = reference.render.render(
        _NvdiffrastStandIn(), upstream_mesh, cameras, 64, 64, render_attr=False, normal_background=0.0
    )
    expected = torch.cat([(upstream.pos + 0.5).clamp(0, 1), (upstream.normal / 2 + 0.5).clamp(0, 1)], dim=-1)

    pipeline = Step1X3DMeshTexturePipeline(**make_tiny_step1x3d_texture_components(tiny_clip_tokenizer))
    mesh = MeshAsset(vertices=torch.tensor(native.vertices, dtype=torch.float32), faces=torch.tensor(native.faces))
    prepared, positions = pipeline.prepare_mesh(mesh, clean=False)
    world_to_clip = step1x3d_orthographic_projection() @ step1x3d_view_matrices()
    actual = render_step1x3d_control_images(positions, prepared.faces, world_to_clip, 64).permute(0, 2, 3, 1)

    assert torch.equal(prepared.faces, upstream_mesh.t_pos_idx)
    print(f"vertex max abs error {(positions - upstream_mesh.v_pos).abs().max().item():.3e}")
    torch.testing.assert_close(positions, upstream_mesh.v_pos, atol=1e-6, rtol=0)
    torch.testing.assert_close(world_to_clip[:, :2], cameras.mvp_mtx[:, :2], atol=1e-6, rtol=0)
    error = (actual - expected).abs()
    print(f"control image max abs error {error.max().item():.3e}, mean {error.mean().item():.3e}")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


class _UVProjectionStandIn:
    """Upstream ``UVProjection`` maths over shared fragments: renders and samples through ``pixel_texel`` (the
    package rasterizer with nearest texel sampling), bakes by differentiating renders of zero textures, and fills
    with the package nearest-texel search in place of the CuPy Voronoi solver."""

    def __init__(self, sync, positions, faces, uvs, world_to_clip, camera_positions, visibility_size):
        self.sync, self.size = sync, sync.texture_size
        self.gradient_maps = []
        for view in range(6):
            zero = torch.zeros(self.size, self.size, 4, requires_grad=True)
            ((1 - self._render(zero, view)) ** 2).sum().backward()
            self.gradient_maps.append(-zero.grad)

        flipped = faces[:, [0, 2, 1]]
        clip = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1) @ world_to_clip.transpose(1, 2)
        pix2face = rasterize(clip, flipped, (visibility_size,) * 2, cull_backfaces=True).face_index
        uv_clip = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=1)
        uv_clip[:, 1] *= -1
        uv_pix2face = rasterize(uv_clip, flipped, (self.size,) * 2).face_index[0][..., None]
        self.visible_triangles = []
        for view in range(6):
            valid = torch.unique(pix2face[view])
            valid = valid[1:] if valid[0] == -1 else valid
            mask = torch.isin(uv_pix2face, valid)
            triangle_mask = torch.ones(self.size, self.size, 1)
            triangle_mask[~mask] = 0
            triangle_mask[:, 1:][triangle_mask[:, :-1] > 0] = 1
            triangle_mask[:, :-1][triangle_mask[:, 1:] > 0] = 1
            triangle_mask[1:, :][triangle_mask[:-1, :] > 0] = 1
            triangle_mask[:-1, :][triangle_mask[1:, :] > 0] = 1
            self.visible_triangles.append(triangle_mask)

        # pytorch3d vertex normals and HardGeometryShader cosines.
        corners = positions[faces]
        normals = torch.zeros_like(positions)
        for a, b, c in ((1, 2, 0), (2, 0, 1), (0, 1, 2)):
            normals.index_add_(
                0, faces[:, a], torch.cross(corners[:, b] - corners[:, a], corners[:, c] - corners[:, a], dim=1)
            )
        normals = torch.nn.functional.normalize(normals, eps=1e-6, dim=1)
        raster = rasterize(clip, flipped, (sync.image_size,) * 2, cull_backfaces=True)
        points = interpolate(positions, raster, flipped)
        pixel_normals = torch.nn.functional.normalize(interpolate(normals, raster, flipped), p=2, dim=-1, eps=1e-6)
        to_camera = torch.nn.functional.normalize(camera_positions[:, None, None] - points, p=2, dim=-1, eps=1e-6)
        cos_angles = (to_camera * pixel_normals).sum(dim=-1, keepdim=True).clamp(0, 1)
        self.cos_maps = []
        for view in range(6):
            zero = torch.zeros(self.size, self.size, 4, requires_grad=True)
            ((cos_angles[view] - self._render(zero, view)) ** 2).sum().backward()
            self.cos_maps.append(
                self._voronoi(-zero.grad / (self.gradient_maps[view] + 1e-8), self.gradient_maps[view][..., 0])
            )

    def _render(self, texture_map, view):
        pixel_texel = self.sync.pixel_texel[view]
        colors = texture_map.reshape(-1, 4)[pixel_texel.clamp(min=0)]
        colors = torch.where(pixel_texel[:, None] >= 0, colors, torch.ones_like(colors))
        alpha = (pixel_texel >= 0).float()[:, None]
        return torch.cat([colors, alpha], dim=-1).reshape(self.sync.image_size, self.sync.image_size, 5)

    def _voronoi(self, texture, mask):
        return texture.reshape(-1, texture.shape[-1])[nearest_valid_index(mask > 0)].reshape(texture.shape)

    def set_texture_map(self, texture):
        self.texture = texture.permute(1, 2, 0)

    def render_textured_views(self):
        return [self._render(self.texture, view).permute(2, 0, 1) for view in range(6)]

    @torch.enable_grad()
    def bake_texture(self, views=None, main_views=(), cos_weighted=True, exp=None):
        exp = exp or 1
        views = [view.permute(1, 2, 0) for view in views]
        bake_maps = [torch.zeros(self.size, self.size, 4, requires_grad=True) for _ in views]
        loss = sum(((self._render(bake_maps[i], i)[..., :-1] - views[i]) ** 2).sum() for i in range(6))
        loss.backward()
        total_weights, baked = 0, 0
        for i in range(6):
            bake_map = self._voronoi(
                -bake_maps[i].grad / (self.gradient_maps[i] + 1e-8), self.gradient_maps[i][..., 0]
            )
            weight = self.visible_triangles[i] * self.cos_maps[i] ** exp
            total_weights = total_weights + weight
            baked = baked + bake_map * weight
        baked = baked / (total_weights + 1e-8)
        baked = self._voronoi(baked, total_weights[..., 0])
        self.set_texture_map(baked.permute(2, 0, 1))
        return self.render_textured_views(), baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)


def test_texture_sync_step_matches_reference_maths(tiny_clip_tokenizer):
    reference = _load_reference()
    from diffusers_3d import MeshAsset, Step1X3DMeshTexturePipeline

    native = _test_mesh()
    pipeline = Step1X3DMeshTexturePipeline(**make_tiny_step1x3d_texture_components(tiny_clip_tokenizer))
    mesh = MeshAsset(vertices=torch.tensor(native.vertices, dtype=torch.float32), faces=torch.tensor(native.faces))
    prepared, positions = pipeline.prepare_mesh(mesh, clean=False)
    world_to_camera = step1x3d_view_matrices()
    world_to_clip = step1x3d_orthographic_projection() @ world_to_camera
    camera_positions = torch.linalg.inv(world_to_camera)[:, :3, 3]
    sync = Step1X3DLatentTextureSync(
        positions,
        prepared.faces,
        prepared.uvs,
        world_to_clip,
        camera_positions,
        image_size=24,
        texture_size=64,
        visibility_size=96,
    )
    stand_in = _UVProjectionStandIn(sync, positions, prepared.faces, prepared.uvs, world_to_clip, camera_positions, 96)

    betas = shifted_snr_betas(DDPMScheduler.from_config(SDXL_SCHEDULER_CONFIG).alphas_cumprod)
    scheduler = DDPMScheduler.from_config(SDXL_SCHEDULER_CONFIG, trained_betas=betas.tolist())
    scheduler.set_timesteps(50)
    generator = torch.Generator().manual_seed(0)
    sample, model_output = torch.randn(2, 6, 4, 24, 24, generator=generator).unbind(0)
    texture = torch.randn(4, 64, 64, generator=generator)
    for timestep, exp in ((scheduler.timesteps[3], 0.5), (scheduler.timesteps[20], 2.5)):
        expected = reference.step_tex_sync(
            scheduler=scheduler,
            uvp=stand_in,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            texture=texture,
            generator=torch.Generator().manual_seed(1),
            main_views=[],
            exp=exp,
        )
        views, actual_texture = sync.step(
            scheduler, model_output, timestep, sample, texture, exp, generator=torch.Generator().manual_seed(1)
        )
        texture_error = (actual_texture - expected["prev_tex"]).abs().max().item()
        view_error = (views - expected["prev_sample"]).abs().max().item()
        print(f"texture sync t={int(timestep)} exp={exp}: texture {texture_error:.3e}, views {view_error:.3e}")
        torch.testing.assert_close(actual_texture, expected["prev_tex"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(views, expected["prev_sample"], atol=1e-5, rtol=1e-5)


def test_reference_image_preprocessing_matches_reference():
    reference = _load_reference()
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(50, 37, 4), dtype=np.uint8)
    pixels[..., 3] = 0
    pixels[6:41, 4:30, 3] = 255
    pixels[6:12, 4:30, 3] = 90
    condition = ImageCondition(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255)
    _, rgba = _quantized_rgba(condition)
    expected = reference.preprocess_image(None, Image.fromarray(rgba, "RGBA"), 96, 96)
    expected = torch.from_numpy(np.asarray(expected).copy()).permute(2, 0, 1).float() / 255

    assert torch.equal(preprocess_step1x3d_reference_image(condition, image_size=96), expected)


def test_final_bake_matches_reference_mesh_render(tiny_clip_tokenizer, monkeypatch):
    _load_reference("cv2")
    from diffusers_3d import MeshAsset, Step1X3DMeshTexturePipeline

    config = _pipeline_definition("Step1X3DTextureConfig")()
    bake_from_multiview = _pipeline_definition("bake_from_multiview")
    native = _test_mesh()
    pipeline = Step1X3DMeshTexturePipeline(**make_tiny_step1x3d_texture_components(tiny_clip_tokenizer))
    mesh = MeshAsset(vertices=torch.tensor(native.vertices, dtype=torch.float32), faces=torch.tensor(native.faces))
    prepared, positions = pipeline.prepare_mesh(mesh, clean=False)
    with _upstream_import_environment():
        mesh_render = importlib.import_module("step1x3d_texture.differentiable_renderer.mesh_render")
        camera_utils = importlib.import_module("step1x3d_texture.differentiable_renderer.camera_utils")
        render = mesh_render.MeshRender(default_resolution=256, texture_size=256, device="cpu")
    # The released pipeline's mesh_bp: the Y-up mesh scaled to max |x| = 0.5, loaded without centring.
    vertices = prepared.vertices.numpy()
    faces = prepared.faces.numpy().astype(np.int32)
    render.set_mesh(
        vertices / np.abs(vertices).max() * 0.5,
        faces,
        prepared.uvs.numpy().copy(),
        faces,
        scale_factor=1.0,
        auto_center=False,
    )

    generator = torch.Generator().manual_seed(0)
    views = torch.randint(0, 256, (6, 3, 64, 64), generator=generator, dtype=torch.uint8)
    images = [Image.fromarray(view.permute(1, 2, 0).numpy()).resize((256, 256), Image.LANCZOS) for view in views]
    elevations, azimuths, weights = (
        config.selected_camera_elevs,
        config.selected_camera_azims,
        config.selected_view_weights,
    )
    expected, expected_mask = bake_from_multiview(
        None, render, images, elevations, azimuths, weights, "fast", config.bake_exp
    )
    merged = []
    monkeypatch.setattr(texture_pipeline, "merge_views", lambda *args: merged.append(merge_views(*args)) or merged[-1])
    actual = pipeline.bake_views(views.float() / 255, positions, prepared.faces, prepared.uvs, texture_size=256)

    assert STEP1X3D_BAKE_VIEWS == tuple(zip(elevations, azimuths, weights))
    assert STEP1X3D_BAKE_EXPONENT == config.bake_exp
    torch.testing.assert_close(-positions, render.vtx_pos, rtol=0, atol=1e-7)
    for elevation, azimuth, _ in STEP1X3D_BAKE_VIEWS:
        expected_camera = torch.from_numpy(camera_utils.get_mv_matrix(elevation, azimuth, config.camera_distance))
        torch.testing.assert_close(look_at_view(elevation, azimuth, 1.8), expected_camera, rtol=0, atol=1e-7)
    assert torch.equal(orthographic_projection(1.1), torch.from_numpy(render.camera_proj_mat))
    (texture, mask), expected_mask = merged[0], expected_mask[..., 0]
    assert torch.equal(mask, expected_mask) and mask.sum() > 20000
    error = (texture - expected).abs()[mask]
    print(f"final bake max abs error {error.max().item():.3e}, mean {error.mean().item():.3e}")
    torch.testing.assert_close(texture, expected, rtol=0, atol=1e-5)
    assert torch.equal(actual[mask], texture[mask])
