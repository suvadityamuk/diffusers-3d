# Portions of this file reproduce image preprocessing, sampling, and hierarchical surface extraction from TripoSG:
# https://github.com/VAST-AI-Research/TripoSG
# Revision: fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
#
# MIT License. Copyright (c) 2025 VAST-AI-Research and contributors.
# Modified for typed image conditions, device-agnostic PyTorch, and the package marching-cubes backend. The FlashVDM
# grid reuses the package's port of the Hunyuan3D-2.1 refinement, from which upstream's own copy derives.

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import BitImageProcessorPil, Dinov2Model

from ...backends import ScikitImageBackend
from ...backends._optional import require
from ...backends.occupancy import dense_grid_points, octree_resolutions, refinement_indices
from ...data import ImageCondition, ImageInput, as_image_conditions
from ...data.background import BackgroundRemoval, has_ambiguous_alpha
from ...data.image_processing import validate_image_condition_pixels
from ...data.rmbg import RMBG14BackgroundRemover
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DPipelineOutput
from ..hunyuan3d.models import DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, load_dinov2
from .models import TripoSGTransformerModel
from .vae import TripoSGVAEDecoder

# The eight children of a coarse cell, in the order upstream concatenates them.
_CHILD_OFFSETS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
_DISO_HINT = "pip install diso (CUDA only; CC BY-NC 4.0), or pass use_flash_decoder=False"
# FlashVDM samples marked unevaluated before they become NaN, and upstream's batching of its queries.
_UNEVALUATED = -10000.0
_FLASH_NUM_CHUNKS = 10000
_FLASH_MINI_GRIDS = 4
_FLASH_QUERY_CELLS = 6


def _downscale_to_2000(image: Image.Image) -> Image.Image:
    """Upstream's ``INTER_AREA`` shrink of images longer than 2000 pixels, which runs before its background remover."""

    import cv2

    scale = 2000 / max(image.size)
    if scale >= 1:
        return image
    size = (int(image.width * scale), int(image.height * scale))
    return Image.fromarray(cv2.resize(np.asarray(image), size, interpolation=cv2.INTER_AREA))


def prepare_triposg_image(
    condition: ImageCondition, *, padding_ratio: float = 0.1, validate_alpha: bool = True
) -> Image.Image:
    """Crop the alpha foreground, composite it on white, and pad it square, as upstream ``load_image`` does.

    Alpha comes from an RGBA image and/or ``mask``. Images whose longer side exceeds 2000 pixels are first shrunk to
    2000 with OpenCV's ``INTER_AREA``, as upstream. With ``validate_alpha``, the alpha must pass upstream's
    ``is_valid_alpha``; upstream replaces a missing or failing alpha with its ``briaai/RMBG-1.4`` mask, which the
    pipeline's background removal reproduces, and never checks that mask (``validate_alpha=False``).
    """

    import cv2

    validate_image_condition_pixels(condition)
    pixels = condition.image
    if pixels.shape[0] != 4 and condition.mask is None:
        raise ValueError(
            "TripoSG needs the foreground as RGBA alpha or a mask; pass one, or keep background removal on "
            "(pipeline.enable_background_removal()) to cut it out with briaai/RMBG-1.4"
        )
    rgb = pixels[:3].expand(3, -1, -1).float()
    alpha = pixels[3:4].float() if pixels.shape[0] == 4 else torch.ones_like(rgb[:1])
    if condition.mask is not None:
        alpha = alpha * condition.mask.float()
    height, width = rgb.shape[-2:]
    scale = 2000 / max(height, width)
    if scale < 1:
        rgba = torch.cat([rgb, alpha]).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        rgba = cv2.resize(rgba, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        rgb, alpha = torch.from_numpy(rgba).to(rgb.device).permute(2, 0, 1).float().div(255).split([3, 1])
    alpha = alpha.mul(255).round().to(torch.uint8)
    # Upstream's ``is_valid_alpha``: both extreme bins of a 20-bin histogram must hold 1% of the pixels.
    histogram = cv2.calcHist([alpha[0].cpu().numpy()], [0], None, [20], [0, 256]).ravel()
    minimum_count = alpha.shape[1] * alpha.shape[2] * 0.01
    if validate_alpha and (histogram[0] < minimum_count or histogram[-1] < minimum_count):
        raise ValueError(
            "alpha/mask must contain clear foreground and background (at least 1% of pixels each); pass one, or keep "
            "background removal on (pipeline.enable_background_removal()) to cut the image out with briaai/RMBG-1.4"
        )

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


def load_dinov2_image_processor(repository: str, *, revision: str | None = None, dtype=None, **hub_kwargs):
    """The DINOv2 preprocessor of ``repository`` as the PIL-backed ``BitImageProcessorPil`` the release was made with."""

    return BitImageProcessorPil.from_pretrained(repository, revision=revision, **hub_kwargs)


class TripoSGImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """TripoSG image-to-shape: rectified flow over 2048 shape tokens, then an SDF grid and surface extraction.

    The input must carry its foreground as RGBA alpha or a mask (see :func:`prepare_triposg_image`). On CUDA,
    surfaces are extracted as upstream's default does: the FlashVDM grid and DISO's ``DiffDMC`` (``pip install
    diso``). Elsewhere, or with ``use_flash_decoder=False``, upstream's hierarchical path runs instead (a dense
    ``2**dense_octree_depth`` grid refined to ``2**hierarchical_octree_depth`` near the surface, then scikit-image
    marching cubes). Meshes are returned on the CPU in TripoSG's right-handed Y-up frame.
    """

    # Upstream cuts out missing or ambiguous alpha with briaai/RMBG-1.4 (non-commercial) after its 2000 px shrink.
    background_removal = BackgroundRemoval(
        RMBG14BackgroundRemover, trigger=has_ambiguous_alpha, resize=_downscale_to_2000
    )
    # The release's DINOv2 and its preprocessor equal Meta's DINOv2-L (float32, as released) tensor for tensor.
    external_components = {
        "feature_extractor": ExternalComponent(
            DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, load_dinov2_image_processor
        ),
        "image_encoder": ExternalComponent(DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, load_dinov2),
    }
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

    def preprocess(self, image: ImageInput | list[ImageInput]) -> torch.Tensor:
        """Turn RGBA tensors ``(4, H, W)``/``(batch, 4, H, W)`` or image conditions into DINOv2 pixel values."""

        conditions = self.image_conditions(image)
        removal = self.background_removal.automatic
        if self._background_removal_enabled is not None:
            removal = self._background_removal_enabled
        # Removal output is used as upstream uses its RMBG mask, unchecked; other inputs passed ``is_valid_alpha``.
        removed = [removal and has_ambiguous_alpha(original) for original in as_image_conditions(image)]
        images = [
            prepare_triposg_image(condition, validate_alpha=not skip) for condition, skip in zip(conditions, removed)
        ]
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
        flash_octree_depth: int = 9,
        use_flash_decoder: bool | None = None,
        num_chunks: int = 50000,
    ) -> MeshAsset:
        """Mesh the zero level of one shape ``(1, tokens, channels)``.

        ``use_flash_decoder`` (default: on CUDA devices) is upstream's default FlashVDM grid meshed with DISO's
        ``DiffDMC`` (:meth:`flash_extract_mesh`). Otherwise it is upstream's ``use_flash_decoder=False`` path
        (:meth:`hierarchical_extract_mesh`), which also runs on the CPU.
        """

        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError("extract_mesh decodes one shape at a time: latents must have shape (1, tokens, channels)")
        if use_flash_decoder is None:
            use_flash_decoder = latents.device.type == "cuda"
        if use_flash_decoder:
            return self.flash_extract_mesh(
                latents, bounds=bounds, octree_depth=flash_octree_depth, num_chunks=num_chunks
            )
        return self.hierarchical_extract_mesh(
            latents,
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            num_chunks=num_chunks,
        )

    def _mesh_asset(self, vertices: np.ndarray, faces: np.ndarray, size: int, bounds: float) -> MeshAsset:
        # Upstream wraps the result in a trimesh, whose default processing merges duplicate vertices.
        mesh = require("trimesh", "portable").Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
        return MeshAsset(
            vertices=torch.from_numpy(mesh.vertices).float(),
            faces=torch.from_numpy(mesh.faces).to(torch.int64),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "triposg", "representation": "mesh", "resolution": size, "bounds": bounds},
        )

    def flash_occupancy(
        self, latents: torch.Tensor, *, bounds: float = 1.005, octree_depth: int = 9, num_chunks: int = 50000
    ) -> torch.Tensor:
        """Upstream's FlashVDM grid of one shape, with unevaluated samples at ``-10000``.

        The coarse grid, ``round(2**octree_depth / 2**k / 4) * 4 - 1`` cells per axis (63 at depth 9), is decoded
        in ``4**3`` blocks, and each level doubles it near the surface (504 cells at depth 9). Candidate samples are
        dilated twice, refined points are computed in float32 and decoded in 6^3 spatial bins. Upstream's
        top-k key/value selection is switched off on this path, so attention is exact.
        """

        dtype, device = latents.dtype, latents.device
        bounds = (-bounds,) * 3 + (bounds,) * 3
        bbox_min = np.array(bounds[:3])
        bbox_size = np.array(bounds[3:]) - bbox_min
        resolutions = octree_resolutions(2**octree_depth)
        coarse = round(resolutions[0] / _FLASH_MINI_GRIDS) * _FLASH_MINI_GRIDS - 1
        resolutions = [coarse * 2**index for index in range(len(resolutions))]

        def decode(points: torch.Tensor) -> torch.Tensor:
            batch = latents.expand(points.shape[0], -1, -1)
            return self.vae(batch, points, num_chunks=num_chunks).sample

        # The coarse grid is decoded as 4^3 blocks of cubes, as many blocks per call as fit in 10000 points.
        blocks, side = _FLASH_MINI_GRIDS, (coarse + 1) // _FLASH_MINI_GRIDS
        points = dense_grid_points(bounds, coarse).to(device=device, dtype=dtype)
        points = points.view(blocks, side, blocks, side, blocks, side, 3).permute(0, 2, 4, 1, 3, 5, 6)
        points = points.reshape(-1, side**3, 3)
        per_call = max(_FLASH_NUM_CHUNKS // points.shape[1], 1)
        logits = torch.cat([decode(points[start : start + per_call]) for start in range(0, len(points), per_call)])
        grid = logits.reshape(blocks, blocks, blocks, side, side, side).permute(0, 3, 1, 4, 2, 5).contiguous()
        grid = grid.view((coarse + 1,) * 3)

        for resolution in resolutions[1:]:
            fine = refinement_indices(
                grid, resolution, finest=resolution == resolutions[-1], candidate_dilations=2, dtype=dtype
            )
            points = torch.stack(fine, dim=1) * torch.tensor(
                bbox_size / resolution, dtype=torch.float32, device=device
            )
            points = points + torch.tensor(bbox_min, dtype=torch.float32, device=device)
            # Upstream sorts the queries into 6^3 spatial bins and decodes whole bins, about 10000 points at a time.
            minimum, maximum = points.min(dim=0).values, points.max(dim=0).values
            bins = torch.floor((points - minimum) / (maximum - minimum) * (_FLASH_QUERY_CELLS - 0.001)).long()
            bins = bins[..., 0] * _FLASH_QUERY_CELLS**2 + bins[..., 1] * _FLASH_QUERY_CELLS + bins[..., 2]
            bins = bins.sort()
            points = points[bins.indices].unsqueeze(0).contiguous()
            chunks, total = [], 0
            for count in torch.unique(bins.values, return_counts=True)[1].tolist():
                if total + count < _FLASH_NUM_CHUNKS or total == 0:
                    total += count
                else:
                    chunks.append(total)
                    total = count
            chunks.append(total)
            logits = torch.cat(
                [self.vae(latents, chunk, num_chunks=num_chunks).sample for chunk in points.split(chunks, 1)], 1
            )
            values = torch.zeros(points.shape[1], dtype=dtype, device=device)
            values[bins.indices] = logits[0, :, 0]
            grid = torch.full((resolution + 1,) * 3, _UNEVALUATED, dtype=dtype, device=device)
            grid[fine] = values
        return grid

    @torch.no_grad()
    def flash_extract_mesh(
        self, latents: torch.Tensor, *, bounds: float = 1.005, octree_depth: int = 9, num_chunks: int = 50000
    ) -> MeshAsset:
        """Upstream's default extraction: the :meth:`flash_occupancy` grid meshed by DISO's CUDA ``DiffDMC``.

        As upstream, the signed distance handed to ``DiffDMC`` is the negated grid divided by 32 (at depths of 6 and
        more), unevaluated samples are NaN, and vertices are scaled by ``2**octree_depth`` although the grid has
        fewer cells.
        """

        diff_dmc = require("diso", _DISO_HINT).DiffDMC
        grid = self.flash_occupancy(latents, bounds=bounds, octree_depth=octree_depth, num_chunks=num_chunks)
        grid[grid == _UNEVALUATED] = float("nan")
        # Upstream divides by what its resolution loop leaves behind, not by the grid size.
        divisor = 2**octree_depth
        while divisor >= 63:
            divisor //= 2
        sdf = (-grid / divisor).to(torch.float32).contiguous()
        vertices, faces = diff_dmc(dtype=torch.float32).to(grid.device)(
            sdf, deform=None, return_quads=False, normalize=False
        )
        bbox_min = np.array((-bounds,) * 3)
        vertices = vertices.cpu().numpy() / 2**octree_depth * (np.array((bounds,) * 3) - bbox_min) + bbox_min
        return self._mesh_asset(vertices, faces.cpu().numpy()[:, ::-1], grid.shape[0] - 1, bounds)

    @torch.no_grad()
    def hierarchical_extract_mesh(
        self,
        latents: torch.Tensor,
        *,
        bounds: float = 1.005,
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        num_chunks: int = 50000,
    ) -> MeshAsset:
        """Decode one shape on upstream's ``use_flash_decoder=False`` hierarchical grid and mesh its zero level.

        The cube ``[-bounds, bounds]^3`` is sampled densely at ``2**dense_octree_depth`` points per axis. Each
        refinement doubles the grid by nearest-neighbour upsampling and re-decodes only the children of cells
        whose ``sigmoid`` band value is not saturated, dilated by a max-pool. Query coordinates and the grid are
        float16, as upstream. Marching cubes uses upstream's scikit-image Lewiner settings, and trimesh merges duplicate
        vertices as upstream's output does.
        """

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
        vertices = (mesh.vertices / size * bbox_size + bbox_min).numpy()
        return self._mesh_asset(vertices, mesh.faces.numpy(), size, bounds)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        num_tokens: int = 2048,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        bounds: float = 1.005,
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool | None = None,
        num_chunks: int = 50000,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per image with upstream's defaults: 50 Euler steps, CFG 7.0, a refined SDF grid.

        Classifier-free guidance (``guidance_scale > 1``) uses all-zero image tokens as the unconditional input.
        ``bounds`` is the half-extent of the cube the field is sampled in. ``use_flash_decoder`` defaults to on for
        CUDA devices (see :meth:`extract_mesh`); ``flash_octree_depth`` applies to that path and the two other
        depths to the hierarchical one.
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
                flash_octree_depth=flash_octree_depth,
                use_flash_decoder=use_flash_decoder,
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


__all__ = ["TripoSGImageTo3DPipeline", "load_dinov2_image_processor", "prepare_triposg_image"]
