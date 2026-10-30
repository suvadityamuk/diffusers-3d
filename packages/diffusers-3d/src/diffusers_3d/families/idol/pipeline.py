# Portions of this file reproduce image preprocessing, conditioning, camera decoding, and sampling from IDOL:
# https://github.com/speridlabs/idol-sdk
# Revision: 4702a7f00850e34b6e83a2170cc123c3e9570744
#
# Apache License 2.0. Copyright (c) 2026 Speridlabs.
# Modified for typed image conditions, batched classifier-free guidance, and the stock flow-matching scheduler.

from __future__ import annotations

import dataclasses
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageFilter
from transformers import DINOv3ViTModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...data import ImageCondition, ImageInput
from ...data.background import BackgroundRemoval, lacks_alpha, resize_shortest_side
from ...data.birefnet import BiRefNetBackgroundRemover
from ...data.image_processing import validate_image_condition_pixels
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import GaussianSplatAsset, Latent3DOutput, Object3DPipelineOutput
from ..trellis2.models import DINOV3_REVISIONS, DINOV3_VITH16PLUS_REPO, load_dinov3
from ..triposplat.decoder import TripoSplatGaussianDecoder
from ..triposplat.models import (
    FLUX2_REPO,
    FLUX2_REVISION,
    TripoSplatImageVAE,
    TripoSplatTransformerModel,
    load_flux2_vae_encoder,
)

# Square crop side as a multiple of the foreground's longer side: tight for renders, looser for photos.
CROP_MARGINS = {"synthetic": 1.2, "real": 1.7}
# The precision upstream ``IdolPipeline`` loads each component in (``dinov3_dtype="bf16"`` by default). Upstream also
# keeps the linear layers of the transformer and decoder blocks in float16 (``convert_module_to_f16``).
UPSTREAM_DTYPES = {"image_encoder": torch.bfloat16, "vae": torch.bfloat16, "default": torch.float32}
_BLOCKS = {
    TripoSplatTransformerModel: ("blocks", "noise_refiner", "context_refiner"),
    TripoSplatGaussianDecoder: ("octree_blocks", "gaussian_blocks"),
}


def float16_blocks(model: TripoSplatTransformerModel | TripoSplatGaussianDecoder):
    """Cast the linear layers of ``model``'s transformer blocks to float16 in place, as upstream IDOL keeps them."""

    for name in _BLOCKS[type(model)]:
        for module in getattr(model, name).modules():
            if isinstance(module, torch.nn.Linear):
                module.half()
    return model


def prepare_idol_image(condition: ImageCondition, *, image_size: int = 1024, crop_margin: float = 1.2) -> Image.Image:
    """Cut out, square-crop, and composite the foreground on black, as upstream ``preprocess_image`` does.

    The shorter side is resized to ``image_size``, the alpha is eroded by one pixel, and a square ``crop_margin``
    times the foreground's longer side is cropped around its centre and resized to ``image_size``. Alpha comes from an
    RGBA image and/or ``mask``; upstream cuts out inputs without transparency with BiRefNet, the pipeline's default
    background removal does the same before this runs.
    """

    validate_image_condition_pixels(condition)
    pixels = condition.image
    if pixels.shape[0] != 4 and condition.mask is None:
        raise ValueError(
            "IDOL needs the foreground as RGBA alpha or a mask; pass one, or keep background removal on "
            "(the default) for RGB inputs"
        )
    alpha = pixels[3:4] if pixels.shape[0] == 4 else torch.ones_like(pixels[:1])
    if condition.mask is not None:
        alpha = alpha * condition.mask
    rgba = torch.cat([pixels[:3].expand(3, -1, -1), alpha]).mul(255).round().to(torch.uint8)
    image = Image.fromarray(rgba.permute(1, 2, 0).cpu().numpy(), "RGBA")

    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    if np.array(image.getchannel(3)).min() == 255:
        raise ValueError("the alpha/mask must mark a background (some pixels must be transparent)")
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    rows, columns = np.nonzero(np.array(image.getchannel(3)))
    if rows.size == 0:
        raise ValueError("the alpha/mask has no foreground")
    centre_x, centre_y = (columns.min() + columns.max()) / 2, (rows.min() + rows.max()) / 2
    half = max(columns.max() - columns.min(), rows.max() - rows.min()) / 2 * crop_margin
    box = [int(centre_x - half), int(centre_y - half), int(centre_x + half), int(centre_y + half)]
    image = image.crop(box).resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite


def decode_idol_camera(camera: torch.Tensor) -> dict[str, float]:
    """Decode a predicted camera token ``[direction (3), 1 / distance, half-width]``, as upstream ``decode_camera``.

    Returns yaw and pitch in radians, the radius, and the vertical field of view in degrees; an inverse distance
    below ``1e-3`` is an orthographic camera at radius 1000.
    """

    direction, inverse_distance, half_width = camera[:3].float(), float(camera[3]), float(camera[4])
    norm = float(torch.linalg.vector_norm(direction))
    dx, dy, dz = (direction / norm).tolist() if norm > 1e-6 else (0.0, 0.0, 1.0)
    if inverse_distance < 1e-3:
        radius, fov = 1000.0, math.degrees(2 * math.atan(half_width / 1000.0))
    else:
        radius, fov = 1.0 / inverse_distance, math.degrees(2 * math.atan(half_width * inverse_distance))
    return {"yaw": math.atan2(dx, dy), "pitch": math.asin(min(max(dz, -1.0), 1.0)), "radius": radius, "fov": fov}


class IdolImageTo3DPipeline(Object3DPipeline):
    """IDOL image-to-Gaussians: rectified flow over 8192 latent tokens, then an octree Gaussian decoder.

    IDOL keeps TripoSplat's architecture, so its networks are the TripoSplat classes with IDOL's weights. The image is
    conditioned through DINOv3 ViT-H/16+ tokens plus token-aligned Flux.2 VAE latents. The input must carry its
    foreground as RGBA alpha or a mask (see :func:`prepare_idol_image`). One
    :class:`~diffusers_3d.GaussianSplatAsset` (SH degree 0) is returned per image in a right-handed Z-up frame, with
    the predicted input camera in ``metadata["camera"]``.
    """

    # Upstream cuts out inputs without usable alpha with its bundled BiRefNet (ZhengPeng7/BiRefNet weights, MIT) after
    # resizing the shorter side to 1024 px in the image's own mode.
    background_removal = BackgroundRemoval(
        partial(BiRefNetBackgroundRemover, resize="tensor", cuda_dtype=torch.float16),
        trigger=lacks_alpha,
        resize=resize_shortest_side,
        resize_in_source_mode=True,
    )
    # Gated third-party weights, loaded from their repositories and never saved with the pipeline. IDOL loads the VAE
    # through ``AutoencoderKLFlux2``: the config's batch-norm epsilon and a clamped posterior log-variance.
    external_components = {
        "image_encoder": ExternalComponent(
            DINOV3_VITH16PLUS_REPO, DINOV3_REVISIONS[DINOV3_VITH16PLUS_REPO], load_dinov3
        ),
        "vae": ExternalComponent(FLUX2_REPO, FLUX2_REVISION, partial(load_flux2_vae_encoder, clamp_logvar=True)),
    }
    model_cpu_offload_seq = "image_encoder->vae->transformer->decoder"

    def __init__(
        self,
        image_encoder: DINOv3ViTModel,
        vae: TripoSplatImageVAE,
        transformer: TripoSplatTransformerModel,
        decoder: TripoSplatGaussianDecoder,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ) -> None:
        super().__init__()
        if transformer.config.cross_attention_dim != image_encoder.config.hidden_size:
            raise ValueError("transformer cross_attention_dim must match the image encoder hidden size")
        if transformer.config.cross_attention_dim_2 != 4 * vae.config.latent_channels:
            raise ValueError("transformer cross_attention_dim_2 must be four times the VAE latent channels")
        if transformer.config.in_channels != decoder.config.latent_channels:
            raise ValueError("transformer in_channels must match the decoder latent channels")
        self.register_modules(
            image_encoder=image_encoder, vae=vae, transformer=transformer, decoder=decoder, scheduler=scheduler
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Without ``dtype``, load each component in upstream's precision (:data:`UPSTREAM_DTYPES`, with float16
        transformer and decoder blocks); a ``dtype`` loads every component in that precision instead."""

        if "torch_dtype" in kwargs:
            kwargs.setdefault("dtype", kwargs.pop("torch_dtype"))
        if kwargs.get("dtype") is not None:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **{**kwargs, "dtype": UPSTREAM_DTYPES})
        float16_blocks(pipeline.transformer), float16_blocks(pipeline.decoder)
        return pipeline

    @torch.no_grad()
    def encode_image(
        self,
        image: ImageInput | list[ImageInput],
        *,
        mode: str = "synthetic",
        crop_margin: float | None = None,
        image_size: int = 1024,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the DINOv3 tokens and the token-aligned Flux.2 VAE tokens of the prepared images.

        ``mode`` picks the crop (``"synthetic"``: 1.2, ``"real"``: 1.7) unless ``crop_margin`` is given. Upstream
        layer-normalizes the DINOv3 output once more and prepends zero tokens to the VAE tokens in place of the class
        and register tokens. The VAE posterior is sampled with ``generator``.
        """

        if mode not in CROP_MARGINS:
            raise ValueError(f"mode must be one of {sorted(CROP_MARGINS)}")
        margin = CROP_MARGINS[mode] if crop_margin is None else crop_margin
        conditions = self.image_conditions(image)
        images = [prepare_idol_image(c, image_size=image_size, crop_margin=margin) for c in conditions]
        pixels = torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images])
        pixels = pixels.to(self._execution_device).float().div(255)
        mean, std = (
            pixels.new_tensor(value).view(1, 3, 1, 1) for value in (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        )
        image_embeds = self.image_encoder(pixel_values=((pixels - mean) / std).to(self.image_encoder.dtype))
        image_embeds = image_embeds.last_hidden_state
        image_embeds = F.layer_norm(image_embeds.float(), image_embeds.shape[-1:])
        latent_embeds = self.vae(pixels.to(self.vae.dtype) * 2 - 1, generator=generator).float()
        prefix = image_embeds.shape[1] - latent_embeds.shape[1]
        if prefix != 1 + self.image_encoder.config.num_register_tokens:
            raise ValueError("the image encoder patch grid must match the VAE latent grid")
        return image_embeds, F.pad(latent_embeds, (0, 0, prefix, 0))

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        mode: str = "synthetic",
        crop_margin: float | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_gaussians: int = 262144,
        image_size: int = 1024,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        camera_latents: torch.Tensor | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[GaussianSplatAsset, ...], Latent3DOutput | None]:
        """Generate one Gaussian splat per image with upstream's defaults: 50 Euler steps, shift 3.0, CFG 3.0.

        ``guidance_scale`` uses the Diffusers convention ``s * cond - (s - 1) * uncond`` with all-zero image tokens as
        the unconditional input, so it is upstream's ``cfg_strength + 1``; 1.0 disables guidance. ``num_gaussians``
        is rounded down to a multiple of the decoder's ``gaussians_per_token`` (32), but to at least one token's worth,
        as upstream does (it documents 32768 to 1048576 without enforcing it). ``generator`` drives, in order, the VAE posterior, the initial latent and camera noise, and the
        octree resampling.
        """

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        per_token = self.decoder.config.gaussians_per_token
        num_gaussians = max(1, int(num_gaussians) // per_token) * per_token
        image_embeds, latent_embeds = self.encode_image(
            image, mode=mode, crop_margin=crop_margin, image_size=image_size, generator=generator
        )
        config = self.transformer.config
        batch_size, device = image_embeds.shape[0], self._execution_device
        # Upstream keeps the flow state in float32 whatever precision the transformer runs in.
        dtype = torch.promote_types(self.transformer.dtype, torch.float32)
        shapes = {
            "latents": (batch_size, config.num_latent_tokens, config.in_channels),
            "camera_latents": (batch_size, 1, config.camera_channels),
        }
        noise = {"latents": latents, "camera_latents": camera_latents}
        for name, shape in shapes.items():
            if noise[name] is None:
                noise[name] = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            elif tuple(noise[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        latents, camera_latents = (noise[name].to(device=device, dtype=dtype) for name in shapes)

        do_classifier_free_guidance = guidance_scale > 1
        if do_classifier_free_guidance:
            image_embeds = torch.cat([image_embeds, torch.zeros_like(image_embeds)])
            latent_embeds = torch.cat([latent_embeds, torch.zeros_like(latent_embeds)])
        # Latent and camera tokens step together as one flat state, so the stock scheduler steps once per timestep.
        sample = torch.cat([latents.flatten(1), camera_latents.flatten(1)], dim=1)
        split = latents[0].numel()
        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1], device=device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            latents = sample[:, :split].view(shapes["latents"])
            camera_latents = sample[:, split:].view(shapes["camera_latents"])
            if do_classifier_free_guidance:
                latents, camera_latents = torch.cat([latents] * 2), torch.cat([camera_latents] * 2)
            output = self.transformer(
                latents, camera_latents, timestep.expand(latents.shape[0]), image_embeds, latent_embeds
            )
            velocity = torch.cat([output.sample.flatten(1), output.camera.flatten(1)], dim=1)
            if do_classifier_free_guidance:
                conditional, unconditional = velocity.chunk(2)
                velocity = guidance_scale * conditional - (guidance_scale - 1) * unconditional
            sample = self.scheduler.step(velocity.to(sample.dtype), timestep, sample).prev_sample

        latents = sample[:, :split].view(shapes["latents"])
        cameras = sample[:, split:]
        assets = self.decoder(latents, num_gaussians, generator=generator).assets
        objects = tuple(
            dataclasses.replace(
                asset, metadata={**asset.metadata, "family": "idol", "camera": decode_idol_camera(camera)}
            )
            for asset, camera in zip(assets, cameras)
        )
        latent_output = (
            Latent3DOutput(latents=latents, metadata={"family": "idol", "stage": "gaussian_latents"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return objects, latent_output
        return Object3DPipelineOutput(objects=objects, latents=latent_output, previews=None)


__all__ = [
    "CROP_MARGINS",
    "UPSTREAM_DTYPES",
    "float16_blocks",
    "IdolImageTo3DPipeline",
    "decode_idol_camera",
    "prepare_idol_image",
]
