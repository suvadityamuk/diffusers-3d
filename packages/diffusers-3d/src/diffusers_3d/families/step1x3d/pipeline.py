# Portions of this file reproduce StepFun Step1X-3D geometry pipeline semantics:
# https://github.com/stepfun-ai/Step1X-3D
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
#
# Apache License 2.0. Copyright (c) StepFun.
# Modified for typed image conditions and MeshAsset outputs.

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import Dinov2WithRegistersModel

from ...backends._optional import require
from ...backends.occupancy import hierarchical_occupancy, octree_resolutions
from ...data import ImageCondition, ImageInput
from ...data.background import BackgroundRemoval, RembgBackgroundRemover, lacks_alpha
from ...data.image_processing import _quantized_rgba, normalize_imagenet, validate_image_condition_pixels
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DPipelineOutput
from .models import Step1X3DTransformer1DModel
from .scheduler import Step1X3DFlowMatchEulerScheduler
from .vae import Step1X3DShapeDecoder


def preprocess_step1x3d_image(
    condition: ImageCondition, *, image_size: int = 518, foreground_ratio: float = 0.95
) -> torch.Tensor:
    """Released Step1X-3D framing: crop to the non-transparent bounding box, composite on white, scale the crop to
    ``foreground_ratio`` of the original frame, centre it on a white square, then resize bicubically.

    Alpha (times the optional mask) selects the foreground; a fully opaque image is used as a full-frame crop, so no
    background-removal model ever runs. Returns a ``(3, image_size, image_size)`` RGB tensor in ``[0, 1]``.
    """

    validate_image_condition_pixels(condition)
    if not 0 < foreground_ratio <= 1:
        raise ValueError("foreground_ratio must lie in (0, 1]")
    pixels, rgba = _quantized_rgba(condition)
    image = Image.fromarray(rgba)
    width, height = image.size
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("alpha and mask must contain at least one non-transparent pixel")
    left, upper, right, lower = box
    scale = min(height * foreground_ratio / (lower - upper), width * foreground_ratio / (right - left))
    foreground = Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image).crop(box)
    foreground = foreground.resize((int((right - left) * scale), int((lower - upper) * scale)))
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(
        foreground,
        (
            (side - width) // 2 + (width - foreground.width) // 2,
            (side - height) // 2 + (height - foreground.height) // 2,
        ),
    )
    canvas = np.asarray(canvas.resize((image_size, image_size), Image.Resampling.BICUBIC)).copy()
    return torch.from_numpy(canvas).permute(2, 0, 1).to(device=pixels.device, dtype=pixels.dtype) / 255


def _to_meshset(mesh, directory: str, pymeshlab):
    """Upstream's ``trimesh2pymeshlab``: a PLY written by trimesh, read back by pymeshlab."""

    path = os.path.join(directory, "trimesh.ply")
    mesh.export(path)
    meshset = pymeshlab.MeshSet()
    meshset.load_new_mesh(path)
    return meshset


def _to_trimesh(meshset, directory: str, trimesh):
    """Upstream's ``pymeshlab2trimesh``: a PLY written by pymeshlab, loaded (and processed) by trimesh."""

    path = os.path.join(directory, "pymeshlab.ply")
    meshset.save_current_mesh(path)
    return trimesh.load(path)


def postprocess_step1x3d_mesh(
    mesh,
    *,
    do_remove_floater: bool = True,
    do_reduce_face: bool = True,
    max_facenum: int = 200000,
    do_shade_smooth: bool = True,
):
    """The released clean-up of a ``trimesh.Trimesh``: pymeshlab floater removal and quadric decimation to
    ``max_facenum`` faces, then trimesh smooth shading.

    Each pymeshlab step round-trips through PLY files as upstream does, so trimesh re-merges vertices after it.
    Floater removal drops connected components with fewer than 0.1% of the faces; decimation keeps boundaries,
    normals and topology. pymeshlab (GPL) is needed for either: ``pip install "diffusers-3d[meshlab]"``.
    """

    if do_remove_floater or (do_reduce_face and max_facenum > 0):
        pymeshlab = require("pymeshlab", "meshlab")
        trimesh = require("trimesh", "portable")
        with tempfile.TemporaryDirectory() as directory:
            if do_remove_floater:
                meshset = _to_meshset(mesh, directory, pymeshlab)
                meshset.apply_filter("compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.001)
                meshset.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
                meshset.apply_filter("meshing_remove_selected_vertices_and_faces")
                mesh = _to_trimesh(meshset, directory, trimesh)
            if do_reduce_face and max_facenum > 0:
                meshset = _to_meshset(mesh, directory, pymeshlab)
                if max_facenum <= meshset.current_mesh().face_number():
                    meshset.apply_filter(
                        "meshing_decimation_quadric_edge_collapse",
                        targetfacenum=max_facenum,
                        qualitythr=1.0,
                        preserveboundary=True,
                        boundaryweight=3,
                        preservenormal=True,
                        preservetopology=True,
                        autoclean=True,
                    )
                mesh = _to_trimesh(meshset, directory, trimesh)
    if do_shade_smooth:
        mesh = mesh.smooth_shaded
    return mesh


class Step1X3DImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """Step1X-3D image-to-geometry: DINOv2 tokens condition a rectified-flow transformer over the shape VAE's latent
    set, the VAE decoder evaluates occupancy on the released hierarchical grid, and marching cubes extracts a mesh.

    Meshes go through the released clean-up (see :func:`postprocess_step1x3d_mesh`): trimesh merges duplicate
    vertices and fixes the winding, pymeshlab removes floaters and decimates to 200k faces, and trimesh smooth-shades
    the result. The pymeshlab steps need ``pip install "diffusers-3d[meshlab]"`` (GPL) unless they are turned off.
    """

    # Upstream cuts out inputs without usable alpha with rembg (u2net) at full size.
    background_removal = BackgroundRemoval(RembgBackgroundRemover, trigger=lacks_alpha)
    model_cpu_offload_seq = "image_encoder->transformer->vae"
    _lora_loadable_modules = ["transformer"]

    def __init__(
        self,
        image_encoder: Dinov2WithRegistersModel,
        transformer: Step1X3DTransformer1DModel,
        vae: Step1X3DShapeDecoder,
        scheduler: Step1X3DFlowMatchEulerScheduler,
    ) -> None:
        super().__init__()
        if image_encoder.config.hidden_size != transformer.config.cross_attention_dim:
            raise ValueError("image encoder hidden size must match the transformer cross_attention_dim")
        if transformer.config.in_channels != vae.config.latent_channels:
            raise ValueError("transformer in_channels must match the VAE latent_channels")
        self.register_modules(image_encoder=image_encoder, transformer=transformer, vae=vae, scheduler=scheduler)

    def preprocess(self, image: ImageInput | list[ImageInput], *, foreground_ratio: float = 0.95) -> torch.Tensor:
        conditions = self.image_conditions(image)
        images = [
            preprocess_step1x3d_image(
                condition, image_size=self.image_encoder.config.image_size, foreground_ratio=foreground_ratio
            )
            for condition in conditions
        ]
        return torch.stack(images).to(device=self._execution_device)

    def encode_conditioning(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``(batch, 3, height, width)`` RGB images in ``[0, 1]``; other sizes are resized bicubically.

        Returns the final layer-normed class and patch tokens with the registers dropped, and the all-zero null
        condition the released checkpoints were trained with.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        size = self.image_encoder.config.image_size
        images = images.to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)
        if images.shape[-2:] != (size, size):
            images = F.interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)
        hidden_states = self.image_encoder(normalize_imagenet(images)).last_hidden_state
        registers = self.image_encoder.config.num_register_tokens
        conditional = torch.cat([hidden_states[:, :1], hidden_states[:, 1 + registers :]], dim=1)
        return conditional, torch.zeros_like(conditional)

    def prepare_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.vae.config.num_latents, self.transformer.config.in_channels)
        dtype = self.transformer.dtype
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}")
            return latents.to(device=self._execution_device, dtype=dtype)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=dtype)

    def sample_latents(
        self,
        latents: torch.Tensor,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        zero_init_steps: int = 1,
    ) -> torch.Tensor:
        """Euler flow sampling with ``[negative, conditional]`` classifier-free guidance when ``guidance_scale > 1``.

        The velocity of the first ``zero_init_steps`` steps is zeroed, as the released pipeline does.
        """

        do_guidance = guidance_scale > 1
        context = torch.cat([negative, conditional]) if do_guidance else conditional
        context = context.to(dtype=latents.dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_input = torch.cat([latents] * 2) if do_guidance else latents
            velocity = self.transformer(model_input, timestep.expand(model_input.shape[0]), context).sample
            if do_guidance:
                negative_velocity, conditional_velocity = velocity.chunk(2)
                velocity = negative_velocity + guidance_scale * (conditional_velocity - negative_velocity)
            if index < zero_init_steps:
                velocity = velocity * 0.0
            latents = self.scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    def decode_occupancy(
        self,
        latents: torch.Tensor,
        *,
        bounds: float = 1.05,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        chunk_size: int = 65536,
    ) -> list[torch.Tensor]:
        """One occupancy-logit grid per latent set, on the released hierarchical grid.

        The batch is decoded at once and the grid is held, and refined points computed, in the VAE dtype, as
        upstream (which first moves a bfloat16 VAE to float16). Samples away from the surface are never evaluated and
        hold NaN.
        """

        if self.vae.dtype == torch.bfloat16:
            # Upstream moves a bfloat16 VAE to float16 before decoding.
            self.vae.to(torch.float16)
        dtype = self.vae.dtype
        resolutions = octree_resolutions(octree_resolution)
        grids = []
        for tokens in self.vae.decode(latents.to(dtype=dtype)).split(1):
            grids.append(
                hierarchical_occupancy(
                    lambda points, tokens=tokens: self.vae.query(points, tokens)[..., 0],
                    (-bounds,) * 3 + (bounds,) * 3,
                    resolutions,
                    dtype=dtype,
                    device=latents.device,
                    level=mc_level,
                    num_chunks=chunk_size,
                )
            )
        return grids

    @staticmethod
    def extract_mesh(
        grid: torch.Tensor,
        *,
        bounds: float = 1.05,
        mc_level: float = 0.0,
        do_remove_floater: bool = True,
        do_reduce_face: bool = True,
        max_facenum: int = 200000,
        do_shade_smooth: bool = True,
    ) -> MeshAsset:
        """Lewiner marching cubes at ``mc_level``, then the released trimesh and pymeshlab steps.

        Vertices use the released mapping ``index / (octree_resolution + 1) * 2 * bounds - bounds``. Trimesh merges
        duplicate vertices (dropping any interpolated from unevaluated samples) and makes the winding consistent
        and outward (``fix_normals``); :func:`postprocess_step1x3d_mesh` follows.
        """

        marching_cubes = require("skimage.measure", "portable").marching_cubes
        trimesh = require("trimesh", "portable")
        vertices, faces, _, _ = marching_cubes(grid.float().cpu().numpy(), mc_level, method="lewiner")
        vertices = vertices.astype(np.float64) / grid.shape[0] * (2 * bounds) - bounds
        mesh = trimesh.Trimesh(vertices.astype(np.float32), faces[:, ::-1])
        mesh.fix_normals()
        # Upstream caches the normals (so its PLY round trips carry them) and attaches a white PBR material.
        mesh.face_normals
        mesh.vertex_normals
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=(255, 255, 255), main_color=(255, 255, 255), metallicFactor=0.05, roughnessFactor=1.0
            )
        )
        mesh = postprocess_step1x3d_mesh(
            mesh,
            do_remove_floater=do_remove_floater,
            do_reduce_face=do_reduce_face,
            max_facenum=max_facenum,
            do_shade_smooth=do_shade_smooth,
        )
        return MeshAsset(
            vertices=torch.from_numpy(mesh.vertices).to(device=grid.device, dtype=torch.float32),
            faces=torch.from_numpy(mesh.faces).to(device=grid.device, dtype=torch.int64),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "step1x3d", "representation": "mesh", "octree_resolution": grid.shape[0] - 1},
        )

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        foreground_ratio: float = 0.95,
        bounds: float = 1.05,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        chunk_size: int = 65536,
        do_remove_floater: bool = True,
        do_reduce_face: bool = True,
        max_facenum: int = 200000,
        do_shade_smooth: bool = True,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per input image (defaults follow the released ``inference.py``).

        ``do_remove_floater``, ``do_reduce_face`` (to ``max_facenum`` faces) and ``do_shade_smooth`` are the released
        mesh clean-up steps, all on by default; the first two need pymeshlab.
        """

        images = self.preprocess(image, foreground_ratio=foreground_ratio)
        conditional, negative = self.encode_conditioning(images)
        latents = self.prepare_latents(images.shape[0], generator=generator, latents=latents)
        latents = self.sample_latents(
            latents, conditional, negative, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
        )
        grids = self.decode_occupancy(
            latents, bounds=bounds, octree_resolution=octree_resolution, mc_level=mc_level, chunk_size=chunk_size
        )
        meshes = tuple(
            self.extract_mesh(
                grid,
                bounds=bounds,
                mc_level=mc_level,
                do_remove_floater=do_remove_floater,
                do_reduce_face=do_reduce_face,
                max_facenum=max_facenum,
                do_shade_smooth=do_shade_smooth,
            )
            for grid in grids
        )
        latent_output = Latent3DOutput(latents=latents, metadata={"family": "step1x3d"}) if return_latents else None
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["Step1X3DImageTo3DPipeline", "postprocess_step1x3d_mesh", "preprocess_step1x3d_image"]
