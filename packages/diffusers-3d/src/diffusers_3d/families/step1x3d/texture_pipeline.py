# Portions of this file reproduce StepFun Step1X-3D texture pipeline semantics:
# https://github.com/stepfun-ai/Step1X-3D (step1x3d_texture/pipelines/)
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
# The image-guided multi-view loop derives from MV-Adapter (https://github.com/huanngzh/MV-Adapter) and the
# Diffusers StableDiffusionXLPipeline (Copyright The HuggingFace Team).
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified for typed mesh and image inputs, plain-PyTorch geometry renders, and a separate baking step.

from __future__ import annotations

import math
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ...backends import inpaint_texture, look_at_view, merge_views, orthographic_projection, splat_view
from ...backends._optional import require
from ...backends.mesh_rasterizer import interpolate, rasterize
from ...backends.multiview_baking import file_geometry
from ...data import ImageCondition, ImageInput
from ...data.background import BackgroundRemoval, lacks_alpha_channel
from ...data.birefnet import BiRefNetBackgroundRemover
from ...data.image_processing import _quantized_rgba, validate_image_condition_pixels
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    Latent3DOutput,
    MeshAsset,
    Object3DPipelineOutput,
    PBRMaterial,
)
from .texture_attention import STEP1X3D_NUM_VIEWS, Step1X3DMultiViewAdapter
from .texture_components import STEP1X3D_TEXTURE_EXTERNAL_COMPONENTS
from .texture_sync import Step1X3DLatentTextureSync, vertex_normals

# Released view layout: front, right, back, left, top, bottom (Z-up, front facing -Y).
STEP1X3D_VIEW_ELEVATIONS = (0.0, 0.0, 0.0, 0.0, 89.99, -89.99)
STEP1X3D_VIEW_AZIMUTHS = (-90.0, 0.0, 90.0, 180.0, 90.0, 90.0)
STEP1X3D_CAMERA_DISTANCE = 1.8
STEP1X3D_ORTHO_HALF_EXTENT = 0.55
# Final bake views as ``MeshRender`` (elevation, azimuth, weight), in the same order.
STEP1X3D_BAKE_VIEWS = ((0, 0, 1.0), (0, 90, 0.1), (0, 180, 0.5), (0, 270, 0.1), (90, 180, 0.05), (-90, 180, 0.05))
STEP1X3D_BAKE_EXPONENT = 4
# Solid colours (in [-1, 1]) composited behind the synchronised views; the first two are black and white.
_BACKGROUND_COLORS = (
    (-1, -1, -1),
    (1, 1, 1),
    (0, -1, -1),
    (1, -1, -1),
    (0, 0, -1),
    (1, 1, -1),
    (-1, 0, -1),
    (-1, 1, -1),
    (-1, 0, 0),
    (-1, 1, 1),
    (-1, -1, 0),
    (-1, -1, 1),
    (0, -1, 0),
    (1, -1, 1),
)


NVDIFFRAST_INSTALL = 'pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git" (needs CUDA)'
STEP1X3D_MAX_FACES = 50000  # ``reduce_face``'s max_facenum, which inference.py and app.py run before texturing
_COLOR_NAMES = (
    "black",
    "white",
    "maroon",
    "red",
    "olive",
    "yellow",
    "green",
    "lime",
    "teal",
    "aqua",
    "navy",
    "blue",
    "purple",
    "fuchsia",
)


@dataclass(frozen=True)
class Step1X3DTextureMeshes:
    """The three UV layouts of one released texturing run, all over ``faces`` of ``vertices`` (float64, the GLB's
    right-handed Y-up frame, scaled to ``max |x| = 0.5``)."""

    vertices: np.ndarray  # mesh_bp: (V, 3) float64
    faces: np.ndarray  # (F, 3) int64, split by the first unwrap
    sync_uvs: np.ndarray  # second unwrap (texture synchronisation): (U, 2) float32
    sync_uv_faces: np.ndarray  # (F, 3) int64 into sync_uvs
    final_mapping: np.ndarray  # third unwrap (bake and output): (W,) int64 into vertices
    final_faces: np.ndarray  # (F, 3) int64 into final_mapping
    final_uvs: np.ndarray  # (W, 2) float32


def _pymeshlab_round_trip(mesh, directory: str, operation=None):
    """``trimesh2pymeshlab``, an optional filter, then ``pymeshlab2trimesh`` (both through PLY files)."""

    pymeshlab = require("pymeshlab", "meshlab")
    trimesh = require("trimesh", "portable")
    path = os.path.join(directory, "mesh.ply")
    mesh.export(path)
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(path)
    if operation is not None:
        operation(meshes)
    meshes.save_current_mesh(path)
    return trimesh.load(path)


def step1x3d_texture_meshes(mesh: MeshAsset, max_faces: int | None = STEP1X3D_MAX_FACES) -> Step1X3DTextureMeshes:
    """The meshes the released texturing run builds from the GLB of ``mesh``.

    ``remove_degenerate_face`` and ``reduce_face(mesh, max_faces)`` (pymeshlab quadric decimation above
    ``max_faces``; ``None`` skips both), then ``load_mesh(rescale=True)`` (xatlas unwrap, scaled to
    ``max |x| = 0.5``), then the texture synchronisation's own xatlas unwrap (``max_iterations=4`` on float32
    vertices) and the final ``mesh_uv_wrap``, both of the first unwrap's split mesh.
    """

    trimesh = require("trimesh", "portable")
    xatlas = require("xatlas", "portable")
    vertices, faces = file_geometry(mesh)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "mesh.glb")
        MeshAsset(vertices=vertices, faces=faces).save(path)
        loaded = trimesh.load(path)
        loaded = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        if max_faces is not None:

            def reload(meshes) -> None:  # remove_degenerate_face saves and reloads the pymeshlab mesh
                meshes.save_current_mesh(os.path.join(directory, "reloaded.ply"))
                meshes.clear()
                meshes.load_new_mesh(os.path.join(directory, "reloaded.ply"))

            loaded = _pymeshlab_round_trip(loaded, directory, reload)

            def decimate(meshes) -> None:
                if max_faces <= meshes.current_mesh().face_number():
                    meshes.apply_filter(
                        "meshing_decimation_quadric_edge_collapse",
                        targetfacenum=max_faces,
                        qualitythr=1.0,
                        preserveboundary=True,
                        boundaryweight=3,
                        preservenormal=True,
                        preservetopology=True,
                        autoclean=True,
                    )

            loaded = _pymeshlab_round_trip(loaded, directory, decimate)
    mapping, faces, _ = xatlas.parametrize(loaded.vertices, loaded.faces)
    vertices = np.asarray(loaded.vertices)[mapping]
    vertices = vertices / np.abs(vertices).max() * 0.5
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.int64).astype(np.int32))
    options = xatlas.ChartOptions()
    options.max_iterations = 4
    atlas.generate(chart_options=options)
    _, sync_uv_faces, sync_uvs = atlas[0]
    final_mapping, final_faces, final_uvs = xatlas.parametrize(vertices, faces)
    return Step1X3DTextureMeshes(
        vertices=vertices,
        faces=faces.astype(np.int64),
        sync_uvs=sync_uvs.astype(np.float32),
        sync_uv_faces=sync_uv_faces.astype(np.int64),
        final_mapping=final_mapping.astype(np.int64),
        final_faces=final_faces.astype(np.int64),
        final_uvs=final_uvs.astype(np.float32),
    )


def view_frame(vertices: np.ndarray) -> torch.Tensor:
    """``load_mesh``'s ``mesh2std`` (Y-up, +x front) as float32: ``(x, y, z) -> (x, -z, y)``."""

    return torch.from_numpy(np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1).astype(np.float32))


def export_texture(texture: torch.Tensor) -> torch.Tensor:
    """``MeshRender.set_texture`` then ``save_mesh`` of an inpainted ``(T, T, 3)`` texture: two 8-bit truncations."""

    first = Image.fromarray((texture.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
    second = torch.from_numpy(np.array(first) / 255.0).float()
    return torch.from_numpy((second.numpy() * 255).astype(np.uint8)).float() / 255


def _upcast_vae(vae: AutoencoderKL) -> None:
    """``StableDiffusionXLPipeline.upcast_vae``: float32, except the decoder's input layers keep their dtype.

    Only a half-precision VAE is changed (upstream runs in float16).
    """

    dtype = vae.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return
    vae.to(dtype=torch.float32)
    vae.post_quant_conv.to(dtype)
    vae.decoder.conv_in.to(dtype)
    vae.decoder.mid_block.to(dtype)


def step1x3d_view_matrices(
    elevations: Sequence[float] = STEP1X3D_VIEW_ELEVATIONS,
    azimuths: Sequence[float] = STEP1X3D_VIEW_AZIMUTHS,
    distance: float = STEP1X3D_CAMERA_DISTANCE,
) -> torch.Tensor:
    """``(N, 4, 4)`` OpenGL world-to-camera matrices of cameras looking at the origin of a Z-up frame."""

    elevation = torch.tensor(elevations, dtype=torch.float32) * math.pi / 180
    azimuth = torch.tensor(azimuths, dtype=torch.float32) * math.pi / 180
    position = distance * torch.stack(
        [elevation.cos() * azimuth.cos(), elevation.cos() * azimuth.sin(), elevation.sin()], dim=-1
    )
    forward = F.normalize(-position, dim=-1)
    right = F.normalize(torch.cross(forward, torch.tensor([0.0, 0.0, 1.0]).expand_as(forward), dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1)
    camera_to_world = torch.eye(4).repeat(len(elevations), 1, 1)
    camera_to_world[:, :3, :3] = torch.stack([right, up, -forward], dim=-1)
    camera_to_world[:, :3, 3] = position
    return torch.linalg.inv(camera_to_world)


def step1x3d_orthographic_projection(
    half_extent: float = STEP1X3D_ORTHO_HALF_EXTENT, distance: float = STEP1X3D_CAMERA_DISTANCE
) -> torch.Tensor:
    """Orthographic projection with image row 0 at the top; depth spans ``distance +- 1``."""

    scale = 1 / half_extent
    return torch.tensor(
        [[scale, 0.0, 0.0, 0.0], [0.0, -scale, 0.0, 0.0], [0.0, 0.0, -1.0, -distance], [0.0, 0.0, 0.0, 1.0]]
    )


def render_step1x3d_control_images(
    positions: torch.Tensor, faces: torch.Tensor, world_to_clip: torch.Tensor, image_size: int
) -> torch.Tensor:
    """``(N, 6, H, W)`` geometry conditions in ``[0, 1]``: position ``+ 0.5`` and normal ``/ 2 + 0.5``.

    Normals are area-weighted over vertices merged by position; the background is 0.5. On CUDA the views are
    rasterized and interpolated with nvdiffrast, as upstream renders them (an optional dependency,
    :data:`NVDIFFRAST_INSTALL`); elsewhere with the package rasterizer, which differs on a fraction of a percent of
    the pixels.
    """

    clip = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1) @ world_to_clip.transpose(1, 2)
    welded, index = positions.unique(dim=0, return_inverse=True)
    normals = vertex_normals(welded, index[faces], eps=1e-12)[index]
    if positions.is_cuda:
        dr = require("nvdiffrast.torch", NVDIFFRAST_INSTALL)
        triangles = faces.to(torch.int32).contiguous()
        raster, _ = dr.rasterize(
            dr.RasterizeCudaContext(device=positions.device), clip.float().contiguous(), triangles, (image_size,) * 2
        )
        position = dr.interpolate(positions[None].float().contiguous(), raster, triangles)[0] + 0.5
        normal = F.normalize(dr.interpolate(normals[None].float().contiguous(), raster, triangles)[0], dim=-1)
        normal = torch.where(raster[..., 3:] > 0, normal, 0.0) / 2 + 0.5
    else:
        raster = rasterize(clip, faces, (image_size, image_size))
        position = interpolate(positions, raster, faces) + 0.5
        normal = F.normalize(interpolate(normals, raster, faces), dim=-1) / 2 + 0.5
    return torch.cat([position, normal], dim=-1).clamp(0, 1).permute(0, 3, 1, 2).to(positions.dtype)


def preprocess_step1x3d_reference_image(condition: ImageCondition, *, image_size: int = 768) -> torch.Tensor:
    """Released reference framing: crop to the non-transparent box (one extra pixel above and left), fit the long
    side to 90% of ``image_size``, centre, and composite on grey. Returns ``(3, image_size, image_size)`` in [0, 1].

    Alpha (times the optional mask) selects the foreground; no background-removal model runs.
    """

    validate_image_condition_pixels(condition)
    pixels, rgba = _quantized_rgba(condition)
    rows, columns = np.nonzero(rgba[..., 3])
    if rows.size == 0:
        raise ValueError("alpha and mask must contain at least one non-transparent pixel")
    height, width = rgba.shape[:2]
    crop = rgba[
        max(rows.min() - 1, 0) : min(rows.max() + 1, height), max(columns.min() - 1, 0) : min(columns.max() + 1, width)
    ]
    crop_height, crop_width = crop.shape[:2]
    if crop_height > crop_width:
        crop_width, crop_height = int(crop_width * (image_size * 0.9) / crop_height), int(image_size * 0.9)
    else:
        crop_height, crop_width = int(crop_height * (image_size * 0.9) / crop_width), int(image_size * 0.9)
    canvas = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    top, left = (image_size - crop_height) // 2, (image_size - crop_width) // 2
    canvas[top : top + crop_height, left : left + crop_width] = Image.fromarray(crop).resize((crop_width, crop_height))
    canvas = canvas.astype(np.float32) / 255
    rgb = canvas[..., :3] * canvas[..., 3:] + (1 - canvas[..., 3:]) * 0.5
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device=pixels.device, dtype=pixels.dtype) / 255


class Step1X3DMeshTexturePipeline(Object3DPipeline):
    """Step1X-3D mesh texturing: six orthographic views generated by SDXL with MV-Adapter, baked onto the mesh UVs.

    The SDXL UNet runs with the MV-Adapter attention processors installed for the call, a T2I-Adapter over
    position and normal renders, and the reference image's cached self-attention inputs. For the first
    ``texture_sync_ratio`` of the schedule the six latent views step together through a shared latent texture.
    """

    # Upstream cuts out three-channel reference images with BiRefNet (ZhengPeng7/BiRefNet, MIT) at full size.
    background_removal = BackgroundRemoval(BiRefNetBackgroundRemover, trigger=lacks_alpha_channel)
    model_cpu_offload_seq = "text_encoder->text_encoder_2->adapter->unet->vae"
    # The SDXL base (unchanged by the release) and the fp16-fix VAE load from their original repositories.
    external_components = STEP1X3D_TEXTURE_EXTERNAL_COMPONENTS
    _exclude_from_cpu_offload = ["mv_adapter"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        adapter: T2IAdapter,
        mv_adapter: Step1X3DMultiViewAdapter,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        if set(mv_adapter.config.hidden_sizes) != set(Step1X3DMultiViewAdapter.hidden_sizes_for(unet)):
            raise ValueError("mv_adapter must cover every self-attention layer of the UNet")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            adapter=adapter,
            mv_adapter=mv_adapter,
            scheduler=scheduler,
        )

    def prepare_mesh(
        self, mesh: MeshAsset, *, max_faces: int | None = STEP1X3D_MAX_FACES
    ) -> tuple[Step1X3DTextureMeshes, torch.Tensor]:
        """The released meshes (:func:`step1x3d_texture_meshes`) and the float32 view-frame positions of their vertices
        (Z-up, scaled to ``max |x| = 0.5``) on the execution device."""

        meshes = step1x3d_texture_meshes(mesh, max_faces)
        return meshes, view_frame(meshes.vertices).to(self._execution_device)

    def encode_prompt(self, prompt: str, negative_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """SDXL prompt embeddings and pooled embeddings for ``[negative_prompt, prompt]``."""

        hidden_states = []
        for tokenizer, text_encoder in ((self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)):
            input_ids = tokenizer(
                [negative_prompt, prompt],
                padding="max_length",
                max_length=text_encoder.config.max_position_embeddings,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output = text_encoder(input_ids.to(text_encoder.device), output_hidden_states=True)
            hidden_states.append(output.hidden_states[-2])
        return torch.cat(hidden_states, dim=-1), output.text_embeds

    def generate_views(
        self,
        control_images: torch.Tensor,
        reference_image: torch.Tensor,
        texture_sync: Step1X3DLatentTextureSync,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        reference_conditioning_scale: float,
        control_conditioning_scale: float,
        texture_sync_ratio: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise the six views. Returns them as ``(6, 3, H, W)`` images in ``[0, 1]`` and as latents."""

        device, dtype = self._execution_device, self.unet.dtype
        views, size = STEP1X3D_NUM_VIEWS, control_images.shape[-1]
        guided = guidance_scale > 1
        prompt_embeds, pooled_prompt_embeds = (
            x.to(device, dtype) if guided else x[1:].to(device, dtype) for x in (prompt_embeds, pooled_prompt_embeds)
        )
        time_ids = torch.tensor([[size, size, 0, 0, size, size]], device=device, dtype=dtype)
        conditions = {
            "encoder_hidden_states": prompt_embeds.repeat_interleave(views, dim=0),
            "added_cond_kwargs": {
                "text_embeds": pooled_prompt_embeds.repeat_interleave(views, dim=0),
                "time_ids": time_ids.expand(len(prompt_embeds) * views, -1),
            },
        }
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        train_steps = self.scheduler.config.num_train_timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        latent_size = size // 2 ** (len(self.vae.config.block_out_channels) - 1)
        scaling = self.vae.config.scaling_factor

        def composite(background: torch.Tensor, foreground: torch.Tensor, timestep: int) -> torch.Tensor:
            # composite_rendered_view: one global-RNG normal draw per view.
            if timestep > 0:
                noise = torch.stack(
                    [torch.normal(0, 1, view.shape, generator=global_generator, device=device) for view in background]
                )
                background = (1 - alphas_cumprod[timestep]) * noise + alphas_cumprod[timestep] * background
            return (foreground * texture_sync.masks + background * (1 - texture_sync.masks)).to(dtype)

        def pick_backgrounds(names: Sequence[str]) -> torch.Tensor:
            return torch.tensor([_COLOR_NAMES.index(colors_random.choice(names)) for _ in range(views)], device=device)

        # Upstream draws the latents, the reference posterior and the scheduler noise from ``generator``; the noise
        # texture, every background noise and the colour posteriors from the global torch RNG; the background
        # colours from Python's ``random``. It never seeds the global RNGs. The package draws the torch ones in the
        # same order from a generator seeded like a fresh global RNG (``torch.initial_seed()``), and the colours from
        # a ``random.Random`` seeded from ``generator``, so a run is reproducible and leaves the global RNGs alone.
        global_generator = torch.Generator(device).manual_seed(torch.initial_seed())
        colors_random = random.Random(generator.initial_seed() if generator is not None else None)
        latents = randn_tensor((views, 4, latent_size, latent_size), generator=generator, device=device, dtype=dtype)
        backgrounds = pick_backgrounds(_COLOR_NAMES)
        texture = torch.normal(0, 1, (4, *(texture_sync.texture_size,) * 2), generator=global_generator, device=device)
        if texture_sync_ratio > 0:
            latents = composite(latents, texture_sync.render(texture), int(timesteps[0]) + 1)

        vae = self.vae
        reference = reference_image[None].to(device, vae.dtype) * 2 - 1
        reference_latents = vae.encode(reference).latent_dist.sample(generator) * scaling
        # The released pipeline then upcasts the VAE (keeping the decoder's input layers) for the colour latents and
        # the final decode; ``__call__`` restores its dtype.
        _upcast_vae(vae)
        colors = torch.tensor(_BACKGROUND_COLORS, device=device, dtype=vae.dtype)[:, :, None, None]
        colors = vae.encode(colors.expand(-1, -1, size, size)).latent_dist.sample(global_generator) * scaling
        colors = colors.to(dtype)

        cache = {}
        self.unet(
            reference_latents.to(dtype),
            torch.zeros_like(timesteps[0]),
            encoder_hidden_states=prompt_embeds[-1:],
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds[-1:], "time_ids": time_ids},
            cross_attention_kwargs={"reference_cache": cache},
        )
        reference_states = {
            key: torch.cat([torch.zeros_like(value), value]) if guided else value
            for key, value in ((key, value.repeat_interleave(views, dim=0)) for key, value in cache.items())
        }
        control = control_images.to(device, dtype).repeat(2 if guided else 1, 1, 1, 1)
        adapter_states = [state * control_conditioning_scale for state in self.adapter(control)]

        for index, timestep in enumerate(self.progress_bar(timesteps)):
            model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2) if guided else latents, timestep)
            noise = self.unet(
                model_input,
                timestep,
                cross_attention_kwargs={
                    "reference_hidden_states": reference_states,
                    "ref_scale": reference_conditioning_scale,
                },
                down_intrablock_additional_residuals=[state.clone() for state in adapter_states],
                **conditions,
            ).sample
            if guided:
                negative, positive = noise.chunk(2)
                noise = negative + guidance_scale * (positive - negative)
            if timestep > (1 - texture_sync_ratio) * train_steps:
                # The cosine exponent ramps from 0 to 6; the released bake treats 0 as 1.
                exp = 6.0 * index / num_inference_steps or 1.0
                latents, texture = texture_sync.step(
                    self.scheduler, noise, timestep, latents, texture, exp, generator=generator
                )
                latents = composite(colors[backgrounds], latents, timestep)
            else:
                latents = self.scheduler.step(noise, timestep, latents, generator=generator).prev_sample
            # Random colours early on, then only black or white backgrounds.
            progress = 1 - timestep / train_steps
            if progress < 0.4:
                backgrounds = pick_backgrounds(_COLOR_NAMES)
            elif progress < 0.99:
                backgrounds = pick_backgrounds(("black", "white"))

        images = vae.decode(latents / scaling).sample
        return (images.float() / 2 + 0.5).clamp(0, 1), latents

    def bake_views(
        self,
        views: torch.Tensor,
        positions: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        *,
        texture_size: int,
    ) -> torch.Tensor:
        """Bake ``(6, 3, H, W)`` views in ``[0, 1]`` onto ``uvs``; returns a ``(T, T, 3)`` texture, row 0 at ``v = 1``.

        ``positions`` are in the view frame. The views are quantized to 8 bits and upscaled with Lanczos to
        ``texture_size`` before baking, as in the released pipeline.
        """

        vertices = -positions  # MeshRender's frame is the view frame mirrored through the origin
        projection = orthographic_projection(2 * STEP1X3D_ORTHO_HALF_EXTENT).to(vertices)
        images = (views * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        textures, weights = [], []
        for image, (elevation, azimuth, view_weight) in zip(images, STEP1X3D_BAKE_VIEWS):
            image = Image.fromarray(image).resize((texture_size, texture_size), Image.Resampling.LANCZOS)
            texture, cosine = splat_view(
                torch.from_numpy(np.array(image) / 255.0).to(vertices),
                vertices,
                faces,
                uvs,
                look_at_view(elevation, azimuth, STEP1X3D_CAMERA_DISTANCE).to(vertices),
                projection,
                texture_size,
                unreliable_radius=texture_size // 256,
            )
            textures.append(texture)
            weights.append(view_weight * cosine**STEP1X3D_BAKE_EXPONENT)
        texture, mask = merge_views(textures, weights)
        return inpaint_texture(texture, mask, vertices, faces, uvs, arithmetic="numpy")

    @torch.no_grad()
    def __call__(
        self,
        mesh: MeshAsset | Sequence[MeshAsset],
        image: ImageInput | list[ImageInput],
        *,
        prompt: str = "high quality",
        negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        reference_conditioning_scale: float = 1.0,
        control_conditioning_scale: float = 1.0,
        texture_sync_ratio: float = 0.5,
        resolution: int = 768,
        latent_texture_size: int = 1536,
        texture_size: int = 2048,
        max_faces: int | None = STEP1X3D_MAX_FACES,
        generator: torch.Generator | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Texture each mesh from its reference image (defaults follow the released ``inference.py``).

        Each mesh is read as its GLB export holds it, cleaned and decimated to ``max_faces`` with pymeshlab
        (``None`` skips both), and unwrapped three times as upstream does (:func:`step1x3d_texture_meshes`). Noise
        comes from ``generator`` where upstream uses its generator; where upstream draws from its (unseeded) global
        torch RNG the package uses a generator seeded with ``torch.initial_seed()``, which is what a fresh upstream
        process holds, and Python ``random`` draws come from ``random.Random(generator.initial_seed())``. Returns the meshes upstream exports:
        right-handed Y-up, scaled to ``max |x| = 0.5``, with the final xatlas UVs and the 8-bit texture;
        ``previews`` holds each mesh's six ``(6, 3, H, W)`` views.
        """

        meshes = (mesh,) if isinstance(mesh, MeshAsset) else tuple(mesh)
        images = tuple(self.image_conditions(image))
        if len(meshes) != len(images) or not meshes:
            raise ValueError("pass one reference image per mesh")
        device = self._execution_device
        world_to_camera = step1x3d_view_matrices().to(device)
        world_to_clip = step1x3d_orthographic_projection().to(device) @ world_to_camera
        camera_positions = torch.linalg.inv(world_to_camera)[:, :3, 3]
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        original_processors, vae_dtype = self.unet.attn_processors, self.vae.dtype
        self.unet.set_attn_processor(self.mv_adapter.attn_processors_for(self.unet))
        textured, all_latents, previews = [], [], []
        try:
            for mesh, image in zip(meshes, images):
                prepared, positions = self.prepare_mesh(mesh, max_faces=max_faces)
                faces = torch.from_numpy(prepared.faces).to(device)
                control_images = render_step1x3d_control_images(positions, faces, world_to_clip, resolution)
                latent_size = resolution // 2 ** (len(self.vae.config.block_out_channels) - 1)
                sync = Step1X3DLatentTextureSync(
                    positions,
                    faces,
                    torch.from_numpy(prepared.sync_uvs).to(device),
                    world_to_clip,
                    camera_positions,
                    image_size=latent_size,
                    texture_size=latent_texture_size,
                    uv_faces=torch.from_numpy(prepared.sync_uv_faces).to(device),
                )
                views, latents = self.generate_views(
                    control_images,
                    preprocess_step1x3d_reference_image(image, image_size=resolution),
                    sync,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    reference_conditioning_scale=reference_conditioning_scale,
                    control_conditioning_scale=control_conditioning_scale,
                    texture_sync_ratio=texture_sync_ratio,
                    generator=generator,
                )
                self.vae.to(vae_dtype)
                vertices = prepared.vertices[prepared.final_mapping]
                final_faces = torch.from_numpy(prepared.final_faces).to(device)
                uvs = torch.from_numpy(prepared.final_uvs).to(device)
                texture = self.bake_views(
                    views, view_frame(vertices).to(device), final_faces, uvs, texture_size=texture_size
                )
                textured.append(
                    MeshAsset(
                        vertices=torch.from_numpy(vertices.astype(np.float32)).to(device),
                        faces=final_faces,
                        uvs=uvs,
                        materials=(PBRMaterial(base_color=export_texture(texture).to(device)),),
                        metadata={"family": "step1x3d", "representation": "mesh"},
                    )
                )
                all_latents.append(latents)
                previews.append(views)
        finally:
            self.unet.set_attn_processor(original_processors)
            self.vae.to(vae_dtype)

        latent_output = (
            Latent3DOutput(latents=torch.stack(all_latents), metadata={"family": "step1x3d"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(textured), latent_output
        return Object3DPipelineOutput(objects=tuple(textured), latents=latent_output, previews=tuple(previews))


__all__ = [
    "STEP1X3D_BAKE_EXPONENT",
    "STEP1X3D_BAKE_VIEWS",
    "STEP1X3D_CAMERA_DISTANCE",
    "STEP1X3D_MAX_FACES",
    "STEP1X3D_ORTHO_HALF_EXTENT",
    "STEP1X3D_VIEW_AZIMUTHS",
    "STEP1X3D_VIEW_ELEVATIONS",
    "Step1X3DMeshTexturePipeline",
    "Step1X3DTextureMeshes",
    "export_texture",
    "preprocess_step1x3d_reference_image",
    "render_step1x3d_control_images",
    "step1x3d_orthographic_projection",
    "step1x3d_texture_meshes",
    "step1x3d_view_matrices",
    "view_frame",
]
