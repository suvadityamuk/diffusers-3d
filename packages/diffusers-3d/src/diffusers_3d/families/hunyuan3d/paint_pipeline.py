# Portions of this file reproduce pipeline semantics from Tencent Hunyuan3D-2.1 Paint:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: reimplemented on stock Diffusers and Transformers components with typed mesh and image inputs, the
# plain-PyTorch rasterizer and baker of diffusers-3d, and MeshAsset outputs.

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import BitImageProcessorPil, Dinov2Model

from ...backends import TrimeshBackend, XAtlasBackend
from ...backends.mesh_rasterizer import interpolate, rasterize
from ...backends.multiview_baking import (
    inpaint_texture,
    look_at_view,
    merge_views,
    orthographic_projection,
    project_view,
    rasterize_texels,
)
from ...data import ImageCondition, validate_image_condition_pixels
from ...data.image_processing import _quantized_rgba
from ...execution.metadata import ContributionStatus, Object3DComponentSpec, ReviewStatus, fully_qualified_class_name
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DKind, Object3DPipelineOutput, PBRMaterial
from .paint_models import Hunyuan3DPaintRRDBNet, Hunyuan3DPaintUNet

# (elevation, azimuth, bake weight) in degrees: six fixed views, then 12 azimuths at +-20 degrees elevation.
CANDIDATE_VIEWS = (
    (0, 0, 1.0),
    (0, 90, 0.1),
    (0, 180, 0.5),
    (0, 270, 0.1),
    (90, 0, 0.05),
    (-90, 180, 0.05),
    *((elevation, azimuth, 0.01) for azimuth in range(0, 360, 30) for elevation in (20, -20)),
)
CAMERA_DISTANCE = 1.45
ORTHOGRAPHIC_SCALE = 1.2
MESH_SCALE = 1.15  # diameter of the normalized mesh
BAKE_EXPONENT = 4


def renderer_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Y-up vertices in ``MeshRender``'s frame ``(-x, z, -y)``, centred on their bounding box with diameter 1.15."""

    vertices = torch.stack([-vertices[:, 0], vertices[:, 2], -vertices[:, 1]], dim=1)
    center = (vertices.amax(dim=0) + vertices.amin(dim=0)) / 2
    scale = (vertices - center).norm(dim=1).max() * 2.0
    return (vertices - center) * (MESH_SCALE / float(scale))


def project_vertices(vertices: torch.Tensor, elevation: float, azimuth: float) -> torch.Tensor:
    """Clip-space ``(V, 4)`` positions of renderer-frame vertices for a released Paint camera."""

    camera = (
        torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
        @ look_at_view(elevation, azimuth, CAMERA_DISTANCE).to(vertices).T
    )
    return camera @ orthographic_projection(ORTHOGRAPHIC_SCALE).to(vertices).T


def select_views(
    vertices: torch.Tensor, faces: torch.Tensor, max_num_views: int, image_size: int
) -> list[tuple[float, float, float]]:
    """The released view selection: the six fixed views, then greedily the candidate adding the most new face area.

    Candidates are added while one uncovers more than 1% of the surface area, up to ``max_num_views``. Faces count as
    seen when they cover a pixel of an ``image_size`` render.
    """

    clip = torch.stack([project_vertices(vertices, elevation, azimuth) for elevation, azimuth, _ in CANDIDATE_VIEWS])
    face_index = rasterize(clip, faces, (image_size, image_size), align_corners=True).face_index.flatten(1)
    seen = torch.zeros(len(CANDIDATE_VIEWS), faces.shape[0] + 1, dtype=torch.bool, device=faces.device)
    seen.scatter_(1, face_index + 1, True)
    seen = seen[:, 1:]
    corners = vertices[faces]
    areas = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1).norm(dim=1) * 0.5
    ratios = areas / areas.sum()

    selected = list(range(6))
    covered = seen[:6].any(dim=0)
    while len(selected) < max_num_views:
        gains = (seen & ~covered).to(ratios.dtype) @ ratios
        gains[selected] = 0
        best = int(gains.argmax())
        if gains[best] <= 0.01:
            break
        selected.append(best)
        covered |= seen[best]
    return [CANDIDATE_VIEWS[index] for index in selected]


def render_conditions(
    vertices: torch.Tensor, faces: torch.Tensor, views: list[tuple[float, float, float]], render_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(views, render_size, render_size, 3)`` uint8 normal and position maps on white, as ``MeshRender`` renders them.

    Normals are flat face normals in the renderer frame mapped by ``(n + 1) / 2``; positions are ``0.5 - p / 1.15``.
    """

    clip = torch.stack([project_vertices(vertices, elevation, azimuth) for elevation, azimuth, _ in views])
    raster = rasterize(clip, faces, (render_size, render_size), align_corners=True)
    visible = (raster.face_index >= 0)[..., None]
    corners = vertices[faces]
    face_normals = F.normalize(torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1), dim=1)
    normals = torch.where(visible, face_normals[raster.face_index.clamp(min=0)], 1.0)
    positions = torch.where(visible, interpolate(0.5 - vertices / MESH_SCALE, raster, faces), 1.0)
    return ((normals + 1) * 0.5 * 255).to(torch.uint8), (positions * 255).to(torch.uint8)


def _resize(images: torch.Tensor, size: int) -> torch.Tensor:
    """Pillow bicubic resize of ``(N, H, W, 3)`` uint8 images to ``(N, 3, size, size)`` floats in ``[0, 1]``."""

    resized = [
        np.array(Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC))
        for image in images.cpu().numpy()
    ]
    return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255


class Hunyuan3DPaintPipeline(Object3DPipeline):
    """Hunyuan3D-Paint 2.1: PBR textures (base colour and metallic-roughness) for a mesh from one reference image.

    The mesh is optionally simplified, UV-unwrapped with xatlas, and rendered from the released view selection as
    normal and position maps. A multiview UNet generates albedo and metallic-roughness images for every view,
    conditioned on those maps, the reference image (through a reference UNet and DINOv2), and three-way
    classifier-free guidance. The views are optionally upscaled 4x with Real-ESRGAN, baked into a UV texture,
    inpainted, and returned as a ``MeshAsset`` with one ``PBRMaterial``; export it as GLB with ``TrimeshBackend``.
    """

    family_id = "hunyuan3d-2.1"
    task_ids = ("mesh-texturing",)
    output_object_types = (MeshAsset,)
    output_representations = ("mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("trimesh", "xatlas")
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    component_specs = tuple(
        Object3DComponentSpec(
            name=name,
            expected_class=fully_qualified_class_name(component_type),
            subfolder=name,
            optional=name == "upscaler",
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        )
        for name, component_type in (
            ("feature_extractor", BitImageProcessorPil),
            ("image_encoder", Dinov2Model),
            ("unet", Hunyuan3DPaintUNet),
            ("vae", AutoencoderKL),
            ("scheduler", UniPCMultistepScheduler),
            ("upscaler", Hunyuan3DPaintRRDBNet),
        )
    )
    _optional_components = ["upscaler"]

    def __init__(
        self,
        feature_extractor: BitImageProcessorPil,
        image_encoder: Dinov2Model,
        unet: Hunyuan3DPaintUNet,
        vae: AutoencoderKL,
        scheduler: UniPCMultistepScheduler,
        upscaler: Hunyuan3DPaintRRDBNet | None = None,
    ) -> None:
        super().__init__()
        warnings.warn(
            "Hunyuan3D-2.1 weights are licensed under the Tencent Hunyuan 3D 2.1 Community License, which does not "
            "apply in the European Union, the United Kingdom, or South Korea. You are responsible for making sure "
            "your use of it complies with that license.",
            UserWarning,
            stacklevel=2,
        )
        self.register_modules(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            upscaler=upscaler,
        )

    def prepare_mesh(self, mesh: MeshAsset, max_faces: int | None) -> MeshAsset:
        """World-space Y-up geometry, merged and simplified to ``max_faces`` (unless ``None``), then UV-unwrapped."""

        mesh = mesh.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP)
        vertices = mesh.vertices @ mesh.transform[:3, :3].T + mesh.transform[:3, 3]
        mesh = MeshAsset(vertices=vertices.detach().cpu().float(), faces=mesh.faces.detach().cpu())
        if max_faces is not None:
            backend = TrimeshBackend()
            mesh = backend.process_geometry(mesh, operation="merge_vertices")
            mesh = MeshAsset(vertices=mesh.vertices, faces=mesh.faces)
            if mesh.faces.shape[0] > max_faces:
                mesh = backend.process_geometry(
                    mesh, operation="simplify_quadric_decimation", parameters={"face_count": max_faces}
                )
                mesh = MeshAsset(vertices=mesh.vertices, faces=mesh.faces)
        return XAtlasBackend().process_geometry(mesh, operation="unwrap_uv")

    def prepare_image(self, image: ImageCondition, size: int) -> Image.Image:
        """The reference image resized to ``size`` with Pillow's bicubic filter and composited on white."""

        validate_image_condition_pixels(image)
        rgba = Image.fromarray(_quantized_rgba(image)[1], "RGBA").resize((size, size), Image.Resampling.BICUBIC)
        reference = Image.new("RGB", rgba.size, (255, 255, 255))
        reference.paste(rgba, mask=rgba.getchannel("A"))
        return reference

    def encode_latents(self, images: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """VAE latents of ``(N, 3, H, W)`` images in ``[0, 1]``, sampled from the posterior."""

        images = images.to(device=self._execution_device, dtype=self.vae.dtype)
        posterior = self.vae.encode((images - 0.5) * 2).latent_dist
        return posterior.sample(generator) * self.vae.config.scaling_factor

    def generate_views(
        self,
        reference: Image.Image,
        normal_maps: torch.Tensor,
        position_maps: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Denoise albedo and metallic-roughness latents for every view; returns them as ``(2, views, C, h, w)``.

        With ``guidance_scale > 1`` the batch holds an unconditional branch (no reference or DINOv2 features), a
        reference-only branch, and the full branch, combined as ``u + g (r - u) + g (f - r)``.
        """

        device, dtype = self._execution_device, self.unet.dtype
        views = normal_maps.shape[0]
        # The initial noise is drawn first, so a fresh generator gives the noise upstream draws from its seeded one.
        size = reference.width // 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (2 * views, self.unet.config.latent_channels, size, size)
        latents = (
            randn_tensor(shape, generator=generator, device=device, dtype=dtype) * self.scheduler.init_noise_sigma
        )
        reference_image = torch.from_numpy(np.array(reference)).permute(2, 0, 1)[None].float() / 255
        reference_latents = self.encode_latents(reference_image, generator)
        condition_latents = torch.cat(
            [self.encode_latents(normal_maps, generator), self.encode_latents(position_maps, generator)], dim=1
        )
        pixel_values = self.feature_extractor(reference, return_tensors="pt").pixel_values
        image_embeds = self.image_encoder(pixel_values.to(device, self.image_encoder.dtype)).last_hidden_state
        reference_scale = torch.ones(1, device=device, dtype=dtype)
        if guidance_scale > 1:
            image_embeds = torch.cat([torch.zeros_like(image_embeds), torch.zeros_like(image_embeds), image_embeds])
            reference_scale = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
        reference_features = self.unet.encode_reference(reference_latents.to(dtype))
        position_maps = position_maps.to(device, dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            sample = self.scheduler.scale_model_input(latents, timestep).unflatten(0, (2, views))
            noise = self.unet(
                sample.expand(reference_scale.shape[0], -1, -1, -1, -1, -1),
                timestep,
                condition_latents.to(dtype),
                position_maps,
                image_embeds.to(dtype),
                reference_features,
                reference_scale,
            ).sample.flatten(1, 2)
            if guidance_scale > 1:
                unconditional, reference_only, full = noise
                noise = unconditional + guidance_scale * (reference_only - unconditional)
                noise = noise + guidance_scale * (full - reference_only)
            else:
                noise = noise[0]
            latents = self.scheduler.step(noise, timestep, latents).prev_sample
        return latents.unflatten(0, (2, views))

    def decode_views(self, latents: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` images in ``[0, 1]``, quantized to 8 bits as the released pipeline saves them."""

        images = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        return ((images.float() / 2 + 0.5).clamp(0, 1) * 255).round() / 255

    def upscale(self, images: torch.Tensor) -> torch.Tensor:
        """Real-ESRGAN 4x of 8-bit ``(N, 3, H, W)`` images, one view at a time, quantized to 8 bits."""

        outputs = []
        for image in images:
            # The released pipeline hands Real-ESRGAN's BGR interface RGB images, so the network sees swapped channels.
            output = self.upscaler(image[None].flip(1).to(self.upscaler.dtype)).float().clamp(0, 1).flip(1)
            outputs.append((output * 255).round() / 255)
        return torch.cat(outputs)

    def bake(
        self,
        images: torch.Tensor,
        views: list[tuple[float, float, float]],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        texture_size: int,
        unreliable_radius: int,
    ) -> torch.Tensor:
        """Back-project ``(views, H, W, C)`` images onto a ``texture_size`` atlas, merge them, and inpaint the rest."""

        texels = rasterize_texels(vertices, faces, uvs, texture_size)
        projection = orthographic_projection(ORTHOGRAPHIC_SCALE).to(vertices)
        textures, weights = [], []
        for image, (elevation, azimuth, view_weight) in zip(images, views):
            world_to_camera = look_at_view(elevation, azimuth, CAMERA_DISTANCE).to(vertices)
            texture, cosine = project_view(
                image,
                vertices,
                faces,
                world_to_camera,
                projection,
                texels,
                texture_size,
                unreliable_radius=unreliable_radius,
            )
            textures.append(texture)
            weights.append(view_weight * cosine**BAKE_EXPONENT)
        texture, mask = merge_views(textures, weights)
        return inpaint_texture(texture, mask, vertices, faces, uvs)

    @torch.no_grad()
    def __call__(
        self,
        mesh: MeshAsset,
        image: ImageCondition,
        *,
        num_inference_steps: int = 15,
        guidance_scale: float = 3.0,
        max_num_views: int = 6,
        view_size: int = 512,
        render_size: int = 2048,
        texture_size: int = 2048,
        max_faces: int | None = 40000,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset], Latent3DOutput]:
        """Texture ``mesh`` from ``image`` (whose alpha or mask isolates the object) with the released defaults.

        The mesh is read in its own coordinate system and world transform and returned in right-handed Y-up world
        coordinates with xatlas UVs. ``max_num_views`` (6 to 9 upstream) caps the view selection; views are
        generated at ``view_size``, rendered and baked at ``render_size`` (``render_size // 512`` pixels of each
        view's silhouette and depth edges are not trusted), baked into a ``2 * texture_size`` atlas, and
        box-filtered to ``texture_size``. The latents are the denoised ``(1, 2, views, C, h, w)`` albedo and
        metallic-roughness latents; ``previews`` holds the decoded ``(views, 3, view_size, view_size)`` albedo and
        metallic-roughness images.
        """

        device = self._execution_device
        prepared = self.prepare_mesh(mesh, max_faces)
        vertices = renderer_vertices(prepared.vertices).to(device)
        faces, uvs = prepared.faces.to(device), prepared.uvs.to(device)
        views = select_views(vertices, faces, max_num_views, render_size // 2)
        normal_maps, position_maps = render_conditions(vertices, faces, views, render_size)
        normal_maps, position_maps = _resize(normal_maps, view_size), _resize(position_maps, view_size)

        latents = self.generate_views(
            self.prepare_image(image, view_size),
            normal_maps,
            position_maps,
            num_inference_steps,
            guidance_scale,
            generator,
        )
        albedo, metallic_roughness = self.decode_views(latents.flatten(0, 1)).chunk(2)
        images = torch.cat([albedo, metallic_roughness], dim=1)
        if self.upscaler is not None:
            images = torch.cat([self.upscale(part) for part in images.split(3, dim=1)], dim=1)
        if images.shape[-1] != render_size:
            images = F.interpolate(images, size=(render_size, render_size), mode="bicubic", antialias=True)
            images = (images.clamp(0, 1) * 255).round() / 255
        texture = self.bake(
            images.permute(0, 2, 3, 1).float(),
            views,
            vertices,
            faces,
            uvs,
            2 * texture_size,
            unreliable_radius=render_size // 512,
        )
        texture = F.avg_pool2d(texture.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0).clamp(0, 1).cpu()
        material = PBRMaterial(base_color=texture[..., :3], metallic=texture[..., 3:4], roughness=texture[..., 4:5])
        textured = MeshAsset(
            vertices=prepared.vertices,
            faces=prepared.faces,
            uvs=prepared.uvs,
            materials=(material,),
            metadata={"family": "hunyuan3d-2.1", "views": [list(view) for view in views]},
        )
        latent_output = Latent3DOutput(latents=latents[None], metadata={"family": "hunyuan3d-2.1"})
        self.maybe_free_model_hooks()
        if not return_dict:
            return (textured,), latent_output
        return Object3DPipelineOutput(
            objects=(textured,), latents=latent_output, previews=(albedo.cpu(), metallic_roughness.cpu())
        )


__all__ = [
    "CANDIDATE_VIEWS",
    "Hunyuan3DPaintPipeline",
    "project_vertices",
    "render_conditions",
    "renderer_vertices",
    "select_views",
]
