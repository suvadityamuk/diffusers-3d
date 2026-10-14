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
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ...backends import (
    TrimeshBackend,
    XAtlasBackend,
    inpaint_texture,
    look_at_view,
    merge_views,
    orthographic_projection,
    splat_view,
)
from ...backends.mesh_rasterizer import interpolate, rasterize
from ...data import ImageCondition
from ...data.image_processing import _quantized_rgba, validate_image_condition_pixels
from ...execution.metadata import ContributionStatus, Object3DComponentSpec, ReviewStatus, fully_qualified_class_name
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    CoordinateSystem,
    Latent3DOutput,
    MeshAsset,
    Object3DKind,
    Object3DPipelineOutput,
    PBRMaterial,
)
from .texture_attention import STEP1X3D_NUM_VIEWS, Step1X3DMultiViewAdapter
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

    scale = 2 / (2 * half_extent)
    return torch.tensor(
        [[scale, 0.0, 0.0, 0.0], [0.0, -scale, 0.0, 0.0], [0.0, 0.0, -1.0, -distance], [0.0, 0.0, 0.0, 1.0]]
    )


def render_step1x3d_control_images(
    positions: torch.Tensor, faces: torch.Tensor, world_to_clip: torch.Tensor, image_size: int
) -> torch.Tensor:
    """``(N, 6, H, W)`` geometry conditions in ``[0, 1]``: position ``+ 0.5`` and normal ``/ 2 + 0.5``.

    Normals are area-weighted over vertices merged by position; the background is 0.5.
    """

    clip = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1) @ world_to_clip.transpose(1, 2)
    raster = rasterize(clip, faces, (image_size, image_size))
    welded, index = positions.unique(dim=0, return_inverse=True)
    normals = vertex_normals(welded, index[faces], eps=1e-12)[index]
    position = interpolate(positions, raster, faces) + 0.5
    normal = F.normalize(interpolate(normals, raster, faces), dim=-1) / 2 + 0.5
    return torch.cat([position, normal], dim=-1).clamp(0, 1).permute(0, 3, 1, 2)


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


def _component_spec(name: str, component_type: type) -> Object3DComponentSpec:
    return Object3DComponentSpec(
        name=name,
        expected_class=fully_qualified_class_name(component_type),
        subfolder=name,
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )


class Step1X3DMeshTexturePipeline(Object3DPipeline):
    """Step1X-3D mesh texturing: six orthographic views generated by SDXL with MV-Adapter, baked onto the mesh UVs.

    The SDXL UNet runs with the MV-Adapter attention processors installed for the call, a T2I-Adapter over
    position and normal renders, and the reference image's cached self-attention inputs. For the first
    ``texture_sync_ratio`` of the schedule the six latent views step together through a shared latent texture.
    """

    family_id = "step1x3d"
    task_ids = ("mesh-texturing",)
    output_object_types = (MeshAsset,)
    output_representations = ("mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("trimesh", "xatlas")
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    component_specs = tuple(
        _component_spec(name, component_type)
        for name, component_type in (
            ("vae", AutoencoderKL),
            ("text_encoder", CLIPTextModel),
            ("text_encoder_2", CLIPTextModelWithProjection),
            ("tokenizer", CLIPTokenizer),
            ("tokenizer_2", CLIPTokenizer),
            ("unet", UNet2DConditionModel),
            ("adapter", T2IAdapter),
            ("mv_adapter", Step1X3DMultiViewAdapter),
            ("scheduler", DDPMScheduler),
        )
    )
    model_cpu_offload_seq = "text_encoder->text_encoder_2->adapter->unet->vae"
    _exclude_from_cpu_offload = ["mv_adapter"]
    _optional_components = []

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

    def prepare_mesh(self, mesh: MeshAsset, *, clean: bool = True) -> tuple[MeshAsset, torch.Tensor]:
        """UV-unwrap ``mesh`` with xatlas (after trimesh's degenerate-face cleanup when ``clean``).

        Returns the unwrapped mesh and its world positions in the Z-up view frame, scaled to ``max |x| = 0.5``.
        """

        if clean:
            mesh = TrimeshBackend().process_geometry(mesh, operation="cleanup")
        mesh = XAtlasBackend().process_geometry(mesh, operation="unwrap_uv")
        z_up = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Z_UP)
        positions = z_up.vertices @ z_up.transform[:3, :3].T + z_up.transform[:3, 3]
        return mesh, (positions / positions.abs().max() * 0.5).to(self._execution_device, torch.float32)

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
            if timestep > 0:
                noise = randn_tensor(background.shape, generator=generator, device=device, dtype=background.dtype)
                background = (1 - alphas_cumprod[timestep]) * noise + alphas_cumprod[timestep] * background
            return (foreground * texture_sync.masks + background * (1 - texture_sync.masks)).to(dtype)

        def pick_backgrounds(count: int) -> torch.Tensor:
            index = torch.randint(count, (views,), generator=generator, device=getattr(generator, "device", None))
            return index.to(device)

        latents = randn_tensor((views, 4, latent_size, latent_size), generator=generator, device=device, dtype=dtype)
        texture = randn_tensor((4, *(texture_sync.texture_size,) * 2), generator=generator, device=device)
        if texture_sync_ratio > 0:
            latents = composite(latents, texture_sync.render(texture), int(timesteps[0]) + 1)

        vae = self.vae
        reference = reference_image[None].to(device, vae.dtype) * 2 - 1
        reference_latents = vae.encode(reference).latent_dist.sample(generator) * scaling
        colors = torch.tensor(_BACKGROUND_COLORS, device=device, dtype=vae.dtype)[:, :, None, None]
        colors = vae.encode(colors.expand(-1, -1, size, size)).latent_dist.sample(generator) * scaling
        backgrounds = pick_backgrounds(len(_BACKGROUND_COLORS))

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
                backgrounds = pick_backgrounds(len(_BACKGROUND_COLORS))
            elif progress < 0.99:
                backgrounds = pick_backgrounds(2)

        images = vae.decode(latents.to(vae.dtype) / scaling).sample
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
                torch.from_numpy(np.array(image)).to(vertices) / 255,
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
        return inpaint_texture(texture, mask, vertices, faces, uvs)

    @torch.no_grad()
    def __call__(
        self,
        mesh: MeshAsset | Sequence[MeshAsset],
        image: ImageCondition | Sequence[ImageCondition],
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
        clean_mesh: bool = True,
        generator: torch.Generator | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Texture each mesh from its reference image (defaults follow the released pipeline).

        Returns UV-textured meshes; ``previews`` holds each mesh's six ``(6, 3, H, W)`` views.
        """

        meshes = (mesh,) if isinstance(mesh, MeshAsset) else tuple(mesh)
        images = (image,) if isinstance(image, ImageCondition) else tuple(image)
        if len(meshes) != len(images) or not meshes:
            raise ValueError("pass one reference image per mesh")
        device = self._execution_device
        world_to_camera = step1x3d_view_matrices().to(device)
        world_to_clip = step1x3d_orthographic_projection().to(device) @ world_to_camera
        camera_positions = torch.linalg.inv(world_to_camera)[:, :3, 3]
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        original_processors = self.unet.attn_processors
        self.unet.set_attn_processor(self.mv_adapter.attn_processors_for(self.unet))
        textured, all_latents, previews = [], [], []
        try:
            for mesh, image in zip(meshes, images):
                mesh, positions = self.prepare_mesh(mesh, clean=clean_mesh)
                faces, uvs = mesh.faces.to(device), mesh.uvs.to(device)
                control_images = render_step1x3d_control_images(positions, faces, world_to_clip, resolution)
                latent_size = resolution // 2 ** (len(self.vae.config.block_out_channels) - 1)
                sync = Step1X3DLatentTextureSync(
                    positions,
                    faces,
                    uvs,
                    world_to_clip,
                    camera_positions,
                    image_size=latent_size,
                    texture_size=latent_texture_size,
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
                texture = self.bake_views(views, positions, faces, uvs, texture_size=texture_size)
                textured.append(
                    MeshAsset(
                        vertices=mesh.vertices.to(device),
                        faces=faces,
                        transform=mesh.transform.to(device),
                        coordinate_system=mesh.coordinate_system,
                        uvs=uvs,
                        materials=(PBRMaterial(base_color=texture.to(device)),),
                        metadata={"family": "step1x3d", "representation": "mesh"},
                    )
                )
                all_latents.append(latents)
                previews.append(views)
        finally:
            self.unet.set_attn_processor(original_processors)

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
    "STEP1X3D_ORTHO_HALF_EXTENT",
    "STEP1X3D_VIEW_AZIMUTHS",
    "STEP1X3D_VIEW_ELEVATIONS",
    "Step1X3DMeshTexturePipeline",
    "preprocess_step1x3d_reference_image",
    "render_step1x3d_control_images",
    "step1x3d_orthographic_projection",
    "step1x3d_view_matrices",
]
