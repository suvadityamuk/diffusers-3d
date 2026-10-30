# Portions of this file reproduce image preprocessing, conditioning, and sampling from TripoSplat:
# https://github.com/VAST-AI-Research/TripoSplat
# Revision: d8db9e018b413dd9c4a9fe22463781bf98e8e68d
#
# MIT License. Copyright (c) 2026 VAST.
# Modified for typed image conditions and the stock flow-matching scheduler.

from __future__ import annotations

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
from .decoder import TripoSplatGaussianDecoder
from .models import FLUX2_REPO, FLUX2_REVISION, TripoSplatImageVAE, TripoSplatTransformerModel, load_flux2_vae_encoder

# Upstream ``TripoSplatPipeline`` asserts this range on ``num_gaussians`` before rounding it.
NUM_GAUSSIANS_RANGE = (32768, 262144)
# The precision upstream ``TripoSplatPipeline`` loads each component in.
UPSTREAM_DTYPES = {"image_encoder": torch.bfloat16, "vae": torch.bfloat16, "default": torch.float16}


def prepare_triposplat_image(
    condition: ImageCondition, *, image_size: int = 1024, erode_radius: int = 1
) -> Image.Image:
    """Cut out, square-crop, and composite the foreground on black, as upstream ``preprocess_image`` does.

    The shorter side is resized to ``image_size``, the alpha is eroded by ``erode_radius`` pixels, and a square 1.2
    times the foreground's longer side is cropped around its centre and resized to ``image_size``. Alpha comes from
    an RGBA image and/or ``mask``; upstream cuts out inputs without transparency with BiRefNet, the pipeline's
    default background removal does the same before this runs.
    """

    validate_image_condition_pixels(condition)
    pixels = condition.image
    if pixels.shape[0] != 4 and condition.mask is None:
        raise ValueError(
            "TripoSplat needs the foreground as RGBA alpha or a mask; pass one, or keep background removal on "
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
    if erode_radius > 0:
        image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))
    rows, columns = np.nonzero(np.array(image.getchannel(3)))
    if rows.size == 0:
        raise ValueError("the alpha/mask has no foreground")
    centre_x, centre_y = (columns.min() + columns.max()) / 2, (rows.min() + rows.max()) / 2
    half = max(columns.max() - columns.min(), rows.max() - rows.min()) / 2 * 1.2
    box = [int(centre_x - half), int(centre_y - half), int(centre_x + half), int(centre_y + half)]
    image = image.crop(box).resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite


class TripoSplatImageTo3DPipeline(Object3DPipeline):
    """TripoSplat image-to-Gaussians: rectified flow over 8192 latent tokens, then an octree Gaussian decoder.

    The image is conditioned through DINOv3 ViT-H/16+ tokens plus token-aligned Flux.2 VAE latents. The input must
    carry its foreground as RGBA alpha or a mask (see :func:`prepare_triposplat_image`). One
    :class:`~diffusers_3d.GaussianSplatAsset` (SH degree 0) is returned per image in a right-handed Z-up frame.

    DINOv3 and the Flux.2 VAE are gated third-party weights: they load from ``facebook/dinov3-vith16plus-pretrain-
    lvd1689m`` and ``black-forest-labs/FLUX.2-dev`` (accept their licenses on the Hub and log in first) and are never
    saved with the pipeline. TripoSplat's release bundles exactly these tensors.
    """

    # Upstream cuts out inputs without usable alpha with its bundled BiRefNet (ZhengPeng7/BiRefNet weights, MIT) after
    # resizing the shorter side to 1024 px in the image's own mode.
    background_removal = BackgroundRemoval(
        partial(BiRefNetBackgroundRemover, resize="tensor"),
        trigger=lacks_alpha,
        resize=resize_shortest_side,
        resize_in_source_mode=True,
    )
    external_components = {
        "image_encoder": ExternalComponent(
            DINOV3_VITH16PLUS_REPO, DINOV3_REVISIONS[DINOV3_VITH16PLUS_REPO], load_dinov3
        ),
        # TripoSplat's own Flux.2 VAE encoder uses PyTorch's default batch-norm epsilon and no log-variance clamp.
        "vae": ExternalComponent(FLUX2_REPO, FLUX2_REVISION, partial(load_flux2_vae_encoder, batch_norm_eps=1e-5)),
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
        """Without ``dtype``, load each component in upstream's precision (:data:`UPSTREAM_DTYPES`)."""

        if "torch_dtype" in kwargs:
            kwargs.setdefault("dtype", kwargs.pop("torch_dtype"))
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = UPSTREAM_DTYPES
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @torch.no_grad()
    def encode_image(
        self,
        image: ImageInput | list[ImageInput],
        *,
        image_size: int = 1024,
        erode_radius: int = 1,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the DINOv3 tokens and the token-aligned Flux.2 VAE tokens of the prepared images.

        Upstream layer-normalizes the DINOv3 output once more and prepends zero tokens to the VAE tokens in place of
        the class and register tokens. The VAE posterior is sampled with ``generator``.
        """

        conditions = self.image_conditions(image)
        images = [prepare_triposplat_image(c, image_size=image_size, erode_radius=erode_radius) for c in conditions]
        pixels = torch.stack([torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images])
        pixels = pixels.to(self._execution_device).float().div(255)
        mean, std = (
            pixels.new_tensor(value).view(1, 3, 1, 1) for value in (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        )
        image_embeds = self.image_encoder(pixel_values=((pixels - mean) / std).to(self.image_encoder.dtype))
        image_embeds = image_embeds.last_hidden_state
        image_embeds = F.layer_norm(image_embeds.float(), image_embeds.shape[-1:])
        latent_embeds = self.vae(pixels.to(self.vae.dtype) * 2 - 1, generator=generator)
        prefix = image_embeds.shape[1] - latent_embeds.shape[1]
        if prefix != 1 + self.image_encoder.config.num_register_tokens:
            raise ValueError("the image encoder patch grid must match the VAE latent grid")
        latent_embeds = F.pad(latent_embeds, (0, 0, prefix, 0))
        return image_embeds, latent_embeds

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        num_gaussians: int = 262144,
        image_size: int = 1024,
        erode_radius: int = 1,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        camera_latents: torch.Tensor | None = None,
        batch_guidance: bool = False,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[GaussianSplatAsset, ...], Latent3DOutput | None]:
        """Generate one Gaussian splat per image with upstream's defaults: 20 Euler steps, CFG 3.0, shift 3.0.

        ``num_gaussians`` must lie in :data:`NUM_GAUSSIANS_RANGE` (32768 to 262144), as upstream asserts, and is
        rounded to the nearest multiple of the decoder's ``gaussians_per_token`` (32). Classifier-free guidance
        (``guidance_scale > 1``) uses all-zero image tokens as the unconditional input and blends
        ``s * cond - (s - 1) * uncond``. Like upstream, the conditional and unconditional passes run one after the
        other; ``batch_guidance=True`` runs them as one batch, which is faster but not bit-identical. ``generator``
        drives, in order, the VAE posterior, the initial latent and camera noise, and the octree resampling.
        """

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if not NUM_GAUSSIANS_RANGE[0] <= num_gaussians <= NUM_GAUSSIANS_RANGE[1]:
            raise ValueError(f"num_gaussians must be in {list(NUM_GAUSSIANS_RANGE)}, got {num_gaussians}")
        per_token = self.decoder.config.gaussians_per_token
        num_gaussians = round(num_gaussians / per_token) * per_token
        image_embeds, latent_embeds = self.encode_image(
            image, image_size=image_size, erode_radius=erode_radius, generator=generator
        )
        config = self.transformer.config
        batch_size, device = image_embeds.shape[0], self._execution_device
        # Upstream keeps the flow state in float32 whatever precision the transformer runs in.
        dtype = torch.promote_types(self.transformer.dtype, torch.float32)
        shapes = {"latents": (batch_size, config.num_latent_tokens, config.in_channels)}
        shapes["camera_latents"] = (batch_size, 1, config.camera_channels)
        noise = {"latents": latents, "camera_latents": camera_latents}
        for name, shape in shapes.items():
            if noise[name] is None:
                noise[name] = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            elif tuple(noise[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        latents, camera_latents = (noise[name].to(device=device, dtype=dtype) for name in shapes)

        do_classifier_free_guidance = guidance_scale > 1
        conditions = [(image_embeds, latent_embeds)]
        if do_classifier_free_guidance:
            conditions.append((torch.zeros_like(image_embeds), torch.zeros_like(latent_embeds)))
            if batch_guidance:
                conditions = [tuple(torch.cat(pair) for pair in zip(*conditions))]

        def predict(latents: torch.Tensor, camera_latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            velocities = []
            for embeds, embeds_2 in conditions:
                repeat = embeds.shape[0] // latents.shape[0]
                states, cameras = latents.repeat(repeat, 1, 1), camera_latents.repeat(repeat, 1, 1)
                output = self.transformer(states, cameras, timestep.expand(states.shape[0]), embeds, embeds_2)
                velocities.extend(torch.cat([output.sample.flatten(1), output.camera.flatten(1)], 1).chunk(repeat))
            if not do_classifier_free_guidance:
                return velocities[0]
            return guidance_scale * velocities[0] - (guidance_scale - 1) * velocities[1]

        # Latent and camera tokens step together as one flat state, so the stock scheduler steps once per timestep.
        sample = torch.cat([latents.flatten(1), camera_latents.flatten(1)], dim=1)
        split = latents[0].numel()
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            velocity = predict(
                sample[:, :split].view(shapes["latents"]), sample[:, split:].view(shapes["camera_latents"]), timestep
            )
            sample = self.scheduler.step(velocity.to(sample.dtype), timestep, sample).prev_sample

        latents = sample[:, :split].view(shapes["latents"])
        objects = self.decoder(latents, num_gaussians, generator=generator).assets
        latent_output = (
            Latent3DOutput(latents=latents, metadata={"family": "triposplat", "stage": "gaussian_latents"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return objects, latent_output
        return Object3DPipelineOutput(objects=objects, latents=latent_output, previews=None)


__all__ = ["NUM_GAUSSIANS_RANGE", "UPSTREAM_DTYPES", "TripoSplatImageTo3DPipeline", "prepare_triposplat_image"]
