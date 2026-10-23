# Portions of this file reproduce image preprocessing, sampling, and hierarchical surface extraction from TripoSG:
# https://github.com/VAST-AI-Research/TripoSG
# Revision: fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
#
# MIT License. Copyright (c) 2025 VAST-AI-Research and contributors.
# Modified for typed image conditions, device-agnostic PyTorch, and the package marching-cubes backend. The FlashVDM
# decoder and diso dual marching cubes are not ported; see this family's README.

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import BitImageProcessorPil, Dinov2Model

from ...backends import ScikitImageBackend
from ...data import ImageCondition
from ...data.image_processing import validate_image_condition_pixels
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.metadata import (
    ContributionStatus,
    Object3DComponentSpec,
    ReviewStatus,
    fully_qualified_class_name,
)
from ...execution.pipelines import Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DKind, Object3DPipelineOutput
from .models import TripoSGTransformerModel
from .vae import TripoSGVAEDecoder

# The eight children of a coarse cell, in the order upstream concatenates them.
_CHILD_OFFSETS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))


def _component(name: str, component_type: type) -> Object3DComponentSpec:
    return Object3DComponentSpec(
        name=name,
        expected_class=fully_qualified_class_name(component_type),
        subfolder=name,
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )


def prepare_triposg_image(condition: ImageCondition, *, padding_ratio: float = 0.1) -> Image.Image:
    """Crop the alpha foreground, composite it on white, and pad it square, as upstream ``load_image`` does.

    Alpha comes from an RGBA image and/or ``mask``. Upstream replaces a missing or ambiguous alpha with the
    ``briaai/RMBG-1.4`` background remover (non-commercial license); this package never runs one, so such inputs
    raise instead. Upstream's 2000-pixel downscale of very large files is not applied.
    """

    validate_image_condition_pixels(condition)
    pixels = condition.image
    if pixels.shape[0] != 4 and condition.mask is None:
        raise ValueError(
            "TripoSG needs the foreground as RGBA alpha or a mask; upstream obtains it with briaai/RMBG-1.4, "
            "which this package does not run"
        )
    rgb = pixels[:3].expand(3, -1, -1).float()
    alpha = pixels[3:4].float() if pixels.shape[0] == 4 else torch.ones_like(rgb[:1])
    if condition.mask is not None:
        alpha = alpha * condition.mask.float()
    alpha = alpha.mul(255).round().to(torch.uint8)
    # Upstream's ``is_valid_alpha``: both extreme bins of a 20-bin histogram must hold 1% of the pixels.
    minimum_count = alpha.numel() * 0.01
    if (alpha <= 12).sum() < minimum_count or (alpha >= 244).sum() < minimum_count:
        raise ValueError("alpha/mask must contain clear foreground and background (at least 1% of pixels each)")

    rows, columns = torch.nonzero(alpha[0], as_tuple=True)
    top, left = int(rows.min()), int(columns.min())
    height, width = int(rows.max()) - top + 1, int(columns.max()) - left + 1
    alpha = alpha.float() / 255
    image = rgb * alpha + (1 - alpha)
    if width > height:
        pad_x = int(width * padding_ratio)
        pad_y = int(pad_x + (width - height) / 2)
    else:
        pad_y = int(height * padding_ratio)
        pad_x = int(pad_y + (height - width) / 2)
    image = F.pad(image[:, top : top + height, left : left + width], (pad_x, pad_x, pad_y, pad_y), value=1.0)
    return Image.fromarray(image.mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


class TripoSGImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """TripoSG image-to-shape: rectified flow over 2048 shape tokens, then a hierarchical SDF grid and marching cubes.

    The input must carry its foreground as RGBA alpha or a mask (see :func:`prepare_triposg_image`). Surfaces are
    extracted with upstream's MIT hierarchical path (a dense ``2**dense_octree_depth`` grid refined to
    ``2**hierarchical_octree_depth`` near the surface) and scikit-image marching cubes; upstream's default FlashVDM
    decoder is not ported. Meshes are returned on the CPU in TripoSG's right-handed Y-up frame.
    """

    family_id = "triposg"
    task_ids = ("image-to-3d",)
    output_object_types = (MeshAsset,)
    output_representations = ("mesh",)
    object_kinds = (Object3DKind.MESH,)
    required_backends = ("scikit-image",)
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    component_specs = (
        _component("feature_extractor", BitImageProcessorPil),
        _component("image_encoder", Dinov2Model),
        _component("scheduler", FlowMatchEulerDiscreteScheduler),
        _component("transformer", TripoSGTransformerModel),
        _component("vae", TripoSGVAEDecoder),
    )
    model_cpu_offload_seq = "image_encoder->transformer->vae"
    _lora_loadable_modules = ["transformer"]

    def __init__(
        self,
        feature_extractor: BitImageProcessorPil,
        image_encoder: Dinov2Model,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: TripoSGTransformerModel,
        vae: TripoSGVAEDecoder,
    ) -> None:
        super().__init__()
        warnings.warn(
            "TripoSG's transformer is derived from Tencent HunyuanDiT, whose Tencent Hunyuan Community License does not "
            "apply in the European Union, the United Kingdom, or South Korea. You are responsible for making sure your "
            "use of it complies with that license.",
            UserWarning,
            stacklevel=2,
        )
        if transformer.config.cross_attention_dim != image_encoder.config.hidden_size:
            raise ValueError("transformer cross_attention_dim must match the image encoder hidden size")
        if transformer.config.in_channels != vae.config.latent_channels:
            raise ValueError("transformer in_channels must match the VAE latent channels")
        self.register_modules(
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            scheduler=scheduler,
            transformer=transformer,
            vae=vae,
        )

    def preprocess(self, image: ImageCondition | Sequence[ImageCondition] | torch.Tensor) -> torch.Tensor:
        """Turn RGBA tensors ``(4, H, W)``/``(batch, 4, H, W)`` or image conditions into DINOv2 pixel values."""

        if isinstance(image, torch.Tensor):
            conditions = [ImageCondition(image=value) for value in (image[None] if image.ndim == 3 else image)]
        elif type(image) is ImageCondition:
            conditions = [image]
        elif isinstance(image, Sequence) and image and all(type(value) is ImageCondition for value in image):
            conditions = list(image)
        else:
            raise TypeError("image must be an ImageCondition, a sequence of them, or an RGBA tensor")
        images = [prepare_triposg_image(condition) for condition in conditions]
        pixel_values = self.feature_extractor(images, return_tensors="pt").pixel_values
        return pixel_values.to(device=self._execution_device, dtype=self.image_encoder.dtype)

    def prepare_latents(
        self,
        batch_size: int,
        num_tokens: int,
        dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, num_tokens, self.transformer.config.in_channels)
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}")
            return latents.to(device=self._execution_device, dtype=dtype)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per image")
        return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=dtype)

    @torch.no_grad()
    def extract_mesh(
        self,
        latents: torch.Tensor,
        *,
        bounds: float = 1.005,
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        num_chunks: int = 50000,
    ) -> MeshAsset:
        """Decode one shape ``(1, tokens, channels)`` on upstream's hierarchical grid and mesh its zero level.

        The cube ``[-bounds, bounds]^3`` is sampled densely at ``2**dense_octree_depth`` points per axis. Each
        refinement doubles the grid by nearest-neighbour upsampling and re-decodes only the children of cells
        whose ``sigmoid`` band value is not saturated, dilated by a max-pool. Query coordinates and the grid are
        float16, as upstream. Marching cubes uses upstream's scikit-image Lewiner settings.
        """

        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError("extract_mesh decodes one shape at a time: latents must have shape (1, tokens, channels)")
        if not 0 < dense_octree_depth <= hierarchical_octree_depth:
            raise ValueError("octree depths must satisfy 0 < dense_octree_depth <= hierarchical_octree_depth")
        device = latents.device

        def decode(points: torch.Tensor) -> torch.Tensor:
            return self.vae(latents, points[None], num_chunks=num_chunks).sample[0, :, 0].to(torch.float16)

        size = 2**dense_octree_depth
        axis = torch.linspace(
            torch.tensor(-bounds, device=device),
            torch.tensor(bounds, device=device),
            size,
            dtype=torch.float16,
            device=device,
        )
        points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        grid = decode(points).view(size, size, size)
        offsets = torch.tensor(_CHILD_OFFSETS, device=device)
        for _ in range(hierarchical_octree_depth - dense_octree_depth):
            refined = F.interpolate(grid[None, None], scale_factor=2)[0, 0]
            band = torch.nonzero((torch.sigmoid(grid[1:-1, 1:-1, 1:-1]) * 2 - 1).abs() < 1.0) + 1
            cells = torch.zeros(size, size, size, device=device)
            cells[band[:, 0], band[:, 1], band[:, 2]] = 1
            kernel = 5 if size < 512 else 3
            cells = F.max_pool3d(cells[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
            children = (torch.nonzero(cells)[None] * 2 + offsets[:, None]).reshape(-1, 3).to(torch.float16)
            half = size
            size *= 2
            # Upstream maps indices to coordinates with spacing bounds/half (in float16) and back by truncation.
            points = (children - half) * (bounds / half)
            indices = (points * (half / bounds) + half).to(torch.int32)
            refined[indices[:, 0], indices[:, 1], indices[:, 2]] = decode(points)
            grid = refined

        mesh = ScikitImageBackend().extract_surface(
            grid.float(), level=0.0, gradient_direction="descent", allow_degenerate=True
        )
        bbox_min = torch.tensor(-bounds)
        bbox_size = torch.tensor(bounds) - bbox_min
        return MeshAsset(
            vertices=mesh.vertices / size * bbox_size + bbox_min,
            faces=mesh.faces,
            normals=mesh.normals,
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "triposg", "representation": "mesh", "resolution": size, "bounds": bounds},
        )

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        num_tokens: int = 2048,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        bounds: float = 1.005,
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        num_chunks: int = 50000,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per image with upstream's defaults: 50 Euler steps, CFG 7.0, a 512^3 refined grid.

        Classifier-free guidance (``guidance_scale > 1``) uses all-zero image tokens as the unconditional input.
        ``bounds`` is the half-extent of the cube the field is sampled in.
        """

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        image_embeds = self.image_encoder(self.preprocess(image)).last_hidden_state
        batch_size = image_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1
        if do_classifier_free_guidance:
            image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds])
        latents = self.prepare_latents(batch_size, num_tokens, image_embeds.dtype, generator, latents)

        sigmas = [1.0 - index / num_inference_steps for index in range(num_inference_steps)]
        self.scheduler.set_timesteps(sigmas=sigmas, device=self._execution_device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            velocity = self.transformer(model_input, timestep.expand(model_input.shape[0]), image_embeds).sample
            if do_classifier_free_guidance:
                unconditional, conditional = velocity.chunk(2)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            # TripoSG integrates x + (sigma - sigma_next) * v; the stock Euler step adds (sigma_next - sigma) * v.
            latents = self.scheduler.step(-velocity, timestep, latents).prev_sample

        meshes = tuple(
            self.extract_mesh(
                shape[None],
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
                num_chunks=num_chunks,
            )
            for shape in latents
        )
        latent_output = (
            Latent3DOutput(latents=latents, metadata={"family": "triposg", "stage": "shape_tokens"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["TripoSGImageTo3DPipeline", "prepare_triposg_image"]
