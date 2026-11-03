# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for typed, object-native Diffusers stages.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, Dinov2WithRegistersModel

from ...data import ImageInput, TextCondition, preprocess_image_condition
from ...data.background import BackgroundRemoval, RembgBackgroundRemover, downscale_longest_side, lacks_alpha
from ...data.image_processing import normalize_imagenet
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import (
    Latent3DOutput,
    Object3D,
    Object3DPipelineOutput,
    SparseVoxelAsset,
)
from .decoders import (
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
)
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .scheduler import TrellisFlowEulerScheduler
from .sparse import TrellisSparseTensor

# The encoders upstream conditions on, at pinned revisions. The DINOv2 weights equal torch.hub's ``dinov2_vitl14_reg``
# tensor for tensor; upstream loads the CLIP checkpoint from the Hub itself.
DINOV2_REPO, DINOV2_REVISION = "facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
CLIP_REPO, CLIP_REVISION = "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"


def _encoder(model_type: type[Any]):
    def load(repository: str, *, revision: str, dtype: torch.dtype | None = None, **hub_kwargs: Any) -> Any:
        model = model_type.from_pretrained(repository, revision=revision, dtype=dtype or torch.float32, **hub_kwargs)
        return model.eval()

    return load


def _tokenizer(repository: str, *, revision: str, dtype: torch.dtype | None = None, **hub_kwargs: Any) -> Any:
    return CLIPTokenizer.from_pretrained(repository, revision=revision, **hub_kwargs)


class _TrellisTwoStagePipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """Shared TRELLIS stages: sparse structure -> SLAT -> Gaussian splats, FlexiCubes meshes, and/or radiance fields.

    Subclasses supply the encoder (DINOv2 image tokens or CLIP text tokens) and ``__call__``. ``formats``
    covers ``"sparse_structure"``, ``"slat"``, ``"gaussian"``, ``"mesh"``, and ``"radiance_field"``; every network
    runs in plain PyTorch.
    """

    _model_cpu_offload_seq = (
        "sparse_structure_flow_model->sparse_structure_decoder->slat_flow_model"
        "->gaussian_decoder->mesh_decoder->radiance_field_decoder"
    )
    _optional_components = [
        "slat_flow_model",
        "slat_scheduler",
        "gaussian_decoder",
        "mesh_decoder",
        "radiance_field_decoder",
    ]
    _lora_loadable_modules = ["sparse_structure_flow_model", "slat_flow_model"]

    def __init__(
        self,
        cond_channels: int,
        sparse_structure_flow_model: TrellisSparseStructureFlowModel,
        sparse_structure_decoder: TrellisSparseStructureDecoder,
        sparse_structure_scheduler: TrellisFlowEulerScheduler,
        slat_flow_model: TrellisSLatFlowModel | None = None,
        slat_scheduler: TrellisFlowEulerScheduler | None = None,
        gaussian_decoder: TrellisSLatGaussianDecoder | None = None,
        mesh_decoder: TrellisSLatMeshDecoder | None = None,
        radiance_field_decoder: TrellisSLatRadianceFieldDecoder | None = None,
        slat_mean: Sequence[float] | None = None,
        slat_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if sparse_structure_flow_model.config.cond_channels != cond_channels:
            raise ValueError("sparse-structure encoder and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")
        if slat_flow_model is not None:
            if slat_scheduler is None:
                raise ValueError("slat_scheduler is required when slat_flow_model is provided")
            if slat_flow_model.config.cond_channels != cond_channels:
                raise ValueError("SLAT encoder and flow context dimensions must match")
            if slat_mean is None or slat_std is None:
                raise ValueError("SLAT normalization is required when slat_flow_model is provided")
            if (
                len(slat_mean) != slat_flow_model.config.out_channels
                or len(slat_std) != slat_flow_model.config.out_channels
            ):
                raise ValueError("SLAT normalization must contain one value per output feature channel")
            if any(float(value) <= 0 for value in slat_std):
                raise ValueError("SLAT standard deviations must be positive")
        elif any(
            component is not None
            for component in (slat_scheduler, gaussian_decoder, mesh_decoder, radiance_field_decoder)
        ):
            raise ValueError("SLAT decoders and scheduler require slat_flow_model")
        for name, decoder in (
            ("Gaussian", gaussian_decoder),
            ("mesh", mesh_decoder),
            ("radiance-field", radiance_field_decoder),
        ):
            if decoder is not None and decoder.config.latent_channels != slat_flow_model.config.out_channels:
                raise ValueError(f"SLAT flow output channels must match the {name} decoder")

        self.register_modules(
            sparse_structure_flow_model=sparse_structure_flow_model,
            sparse_structure_decoder=sparse_structure_decoder,
            sparse_structure_scheduler=sparse_structure_scheduler,
            slat_flow_model=slat_flow_model,
            slat_scheduler=slat_scheduler,
            gaussian_decoder=gaussian_decoder,
            mesh_decoder=mesh_decoder,
            radiance_field_decoder=radiance_field_decoder,
        )
        self.register_to_config(
            slat_mean=None if slat_mean is None else [float(value) for value in slat_mean],
            slat_std=None if slat_std is None else [float(value) for value in slat_std],
        )

    def _randn(
        self, shape: tuple[int, ...], generator: torch.Generator | list[torch.Generator] | None, dtype: torch.dtype
    ) -> torch.Tensor:
        # Upstream draws float32 noise from the CPU RNG and then moves it, so ``torch.manual_seed`` reproduces it.
        device = torch.device("cpu") if generator is None else self._execution_device
        noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        return noise.to(device=self._execution_device, dtype=dtype)

    def prepare_sparse_structure_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.sparse_structure_flow_model
        shape = (
            batch_size,
            model.config.in_channels,
            model.config.resolution,
            model.config.resolution,
            model.config.resolution,
        )
        parameter = next(model.parameters())
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        if latents is None:
            return self._randn(shape, generator, parameter.dtype)
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"sparse-structure latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

    @staticmethod
    def _sample_flow(
        model: TrellisSparseStructureFlowModel | TrellisSLatFlowModel,
        scheduler: TrellisFlowEulerScheduler,
        latents: torch.Tensor | TrellisSparseTensor,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_interval: tuple[float, float],
        rescale_t: float,
    ) -> torch.Tensor | TrellisSparseTensor:
        """Euler sampling with CFG applied only inside ``guidance_interval``, as released."""

        if len(guidance_interval) != 2 or not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        batch_size = latents.batch_size if isinstance(latents, TrellisSparseTensor) else latents.shape[0]
        scheduler.set_timesteps(num_inference_steps, device=latents.device, rescale_t=rescale_t)
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(timestep, batch_size, device=latents.device)
            velocity = model(latents, model_timestep, conditional).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unconditional_velocity = model(latents, model_timestep, unconditional).sample
                velocity = scheduler.apply_guidance(velocity, unconditional_velocity, guidance_scale)
            latents = scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    def sample_sparse_structure(
        self,
        latents: torch.Tensor,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
    ) -> torch.Tensor:
        """Sample dense sparse-structure latents with released TRELLIS CFG semantics."""

        return self._sample_flow(
            self.sparse_structure_flow_model,
            self.sparse_structure_scheduler,
            latents,
            conditional,
            unconditional,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
        )

    def extract_sparse_structure(self, latents: torch.Tensor) -> tuple[SparseVoxelAsset, ...]:
        return self.sparse_structure_decoder.decode_to_sparse_voxels(latents)

    def prepare_slat_latents(
        self,
        sparse_structures: tuple[SparseVoxelAsset, ...],
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        if self.slat_flow_model is None:
            raise RuntimeError("SLAT sampling requires slat_flow_model")
        coordinates = TrellisSparseTensor.from_sparse_voxel_assets(sparse_structures).coordinates
        expected_channels = self.slat_flow_model.config.in_channels
        if latents is not None:
            if not torch.equal(latents.coordinates, coordinates) or latents.channels != expected_channels:
                raise ValueError("SLAT latents must match extracted coordinates and model input channels")
            return latents.to(device=self._execution_device, dtype=self.slat_flow_model.dtype)
        if isinstance(generator, list):
            if len(generator) != len(sparse_structures):
                raise ValueError("a generator list must contain one generator per sparse structure")
            feature_parts = []
            for batch_index, item_generator in enumerate(generator):
                count = int((coordinates[:, 0] == batch_index).sum())
                feature_parts.append(
                    self._randn((count, expected_channels), item_generator, self.slat_flow_model.dtype)
                )
            features = torch.cat(feature_parts)
        else:
            features = self._randn((coordinates.shape[0], expected_channels), generator, self.slat_flow_model.dtype)
        return TrellisSparseTensor(coordinates.to(device=features.device), features)

    def sample_slat(
        self,
        latents: TrellisSparseTensor,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
    ) -> TrellisSparseTensor:
        """Sample and denormalize portable SLAT features over fixed sparse coordinates."""

        if self.slat_flow_model is None or self.slat_scheduler is None:
            raise RuntimeError("SLAT sampling requires slat_flow_model and slat_scheduler")
        latents = self._sample_flow(
            self.slat_flow_model,
            self.slat_scheduler,
            latents,
            conditional,
            unconditional,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
        )
        mean = latents.features.new_tensor(self.config.slat_mean)
        std = latents.features.new_tensor(self.config.slat_std)
        return latents.denormalize(mean, std)

    def decode_slat(self, slat: TrellisSparseTensor, *, formats: tuple[str, ...]) -> tuple[Object3D, ...]:
        objects: list[Object3D] = []
        if "slat" in formats:
            objects.extend(slat.to_sparse_voxel_assets(resolution=self.slat_flow_model.config.resolution))
        if "gaussian" in formats:
            if self.gaussian_decoder is None:
                raise RuntimeError("format 'gaussian' requires gaussian_decoder")
            objects.extend(self.gaussian_decoder(slat).assets)
        if "mesh" in formats:
            if self.mesh_decoder is None:
                raise RuntimeError("format 'mesh' requires mesh_decoder")
            objects.extend(self.mesh_decoder(slat).assets)
        if "radiance_field" in formats:
            if self.radiance_field_decoder is None:
                raise RuntimeError("format 'radiance_field' requires radiance_field_decoder")
            objects.extend(self.radiance_field_decoder(slat).assets)
        return tuple(objects)

    def _generate(
        self,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        *,
        formats: tuple[str, ...] | list[str] | None,
        sparse_structure_num_inference_steps: int,
        slat_num_inference_steps: int,
        guidance_scale: float,
        guidance_interval: tuple[float, float],
        rescale_t: float,
        generator: torch.Generator | list[torch.Generator] | None,
        sparse_structure_latents: torch.Tensor | None,
        slat_latents: TrellisSparseTensor | None,
        return_latents: bool,
        return_dict: bool,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Run both stages from encoded conditioning.

        ``formats`` defaults to every loaded SLAT decoder output (``"gaussian"``, ``"mesh"``, and/or
        ``"radiance_field"``), or ``"sparse_structure"`` when none is loaded.
        """

        decoders = {
            "gaussian": self.gaussian_decoder,
            "mesh": self.mesh_decoder,
            "radiance_field": self.radiance_field_decoder,
        }
        if formats is None:
            formats = tuple(name for name, decoder in decoders.items() if decoder is not None) or ("sparse_structure",)
        formats = tuple(formats)
        allowed_formats = {"sparse_structure", "slat", *decoders}
        if not formats or len(set(formats)) != len(formats) or set(formats).difference(allowed_formats):
            raise ValueError(f"formats must contain unique values from {sorted(allowed_formats)}")
        requires_slat = any(name in formats for name in ("slat", *decoders))
        if requires_slat and self.slat_flow_model is None:
            raise RuntimeError("formats requiring SLAT need slat_flow_model and slat_scheduler")
        for name, decoder in decoders.items():
            if name in formats and decoder is None:
                raise RuntimeError(f"format '{name}' requires {name}_decoder")
        if generator is None:
            # Upstream ``run`` seeds the global RNGs with its default seed, 42, right before sampling.
            torch.manual_seed(42)
        sparse_structure_latents = self.prepare_sparse_structure_latents(
            conditional.shape[0], generator=generator, latents=sparse_structure_latents
        )
        sparse_structure_latents = self.sample_sparse_structure(
            sparse_structure_latents,
            conditional,
            unconditional,
            num_inference_steps=sparse_structure_num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
        )
        sparse_structures = self.extract_sparse_structure(sparse_structure_latents)
        objects: list[Object3D] = []
        if "sparse_structure" in formats:
            objects.extend(sparse_structures)

        if requires_slat:
            slat_latents = self.prepare_slat_latents(sparse_structures, generator=generator, latents=slat_latents)
            slat = self.sample_slat(
                slat_latents,
                conditional,
                unconditional,
                num_inference_steps=slat_num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_interval=guidance_interval,
                rescale_t=rescale_t,
            )
            objects.extend(self.decode_slat(slat, formats=formats))
        latent_output = (
            Latent3DOutput(
                latents=sparse_structure_latents, metadata={"family": "trellis", "stage": "sparse_structure"}
            )
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(objects=tuple(objects), latents=latent_output)


class TrellisImageTo3DPipeline(_TrellisTwoStagePipeline):
    """TRELLIS image-to-3D: DINOv2 image tokens condition the sparse-structure and SLAT flows."""

    # Upstream cuts out inputs without usable alpha with rembg (u2net) after shrinking them to at most 1024 px.
    background_removal = BackgroundRemoval(RembgBackgroundRemover, trigger=lacks_alpha, resize=downscale_longest_side)
    external_components = {
        "image_encoder": ExternalComponent(DINOV2_REPO, DINOV2_REVISION, _encoder(Dinov2WithRegistersModel))
    }
    model_cpu_offload_seq = "image_encoder->" + _TrellisTwoStagePipeline._model_cpu_offload_seq

    def __init__(
        self,
        image_encoder: Dinov2WithRegistersModel,
        sparse_structure_flow_model: TrellisSparseStructureFlowModel,
        sparse_structure_decoder: TrellisSparseStructureDecoder,
        sparse_structure_scheduler: TrellisFlowEulerScheduler,
        slat_flow_model: TrellisSLatFlowModel | None = None,
        slat_scheduler: TrellisFlowEulerScheduler | None = None,
        gaussian_decoder: TrellisSLatGaussianDecoder | None = None,
        mesh_decoder: TrellisSLatMeshDecoder | None = None,
        radiance_field_decoder: TrellisSLatRadianceFieldDecoder | None = None,
        slat_mean: Sequence[float] | None = None,
        slat_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            image_encoder.config.hidden_size,
            sparse_structure_flow_model,
            sparse_structure_decoder,
            sparse_structure_scheduler,
            slat_flow_model,
            slat_scheduler,
            gaussian_decoder,
            mesh_decoder,
            radiance_field_decoder,
            slat_mean,
            slat_std,
        )
        self.register_modules(image_encoder=image_encoder)

    def preprocess(self, image: ImageInput | list[ImageInput]) -> torch.Tensor:
        """Apply pinned foreground recentering without implicit background removal."""

        conditions = self.image_conditions(image)

        processed = [
            preprocess_image_condition(
                condition, image_size=self.image_encoder.config.image_size, foreground_scale=1.2
            ).image
            for condition in conditions
        ]
        return torch.stack(processed).to(device=self._execution_device, dtype=self.image_encoder.dtype)

    def encode_conditioning(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[0, 1]`` images to layer-normed DINOv2 tokens and the released all-zero unconditional branch."""

        size = self.image_encoder.config.image_size
        images = images.to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)
        if images.shape[-2:] != (size, size):
            images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        # TRELLIS takes the class, register, and patch tokens before DINOv2's learned output norm.
        hidden_states = self.image_encoder(
            pixel_values=normalize_imagenet(images), output_hidden_states=True
        ).hidden_states
        conditional = F.layer_norm(hidden_states[-1], hidden_states[-1].shape[-1:])
        return conditional, torch.zeros_like(conditional)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        formats: tuple[str, ...] | list[str] | None = None,
        sparse_structure_num_inference_steps: int = 25,
        slat_num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        rescale_t: float = 3.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: torch.Tensor | None = None,
        slat_latents: TrellisSparseTensor | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Generate from one or more foreground-masked images with the released image guidance (5.0)."""

        images = self.preprocess(image)
        conditional, unconditional = self.encode_conditioning(images)
        return self._generate(
            conditional,
            unconditional,
            formats=formats,
            sparse_structure_num_inference_steps=sparse_structure_num_inference_steps,
            slat_num_inference_steps=slat_num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
            generator=generator,
            sparse_structure_latents=sparse_structure_latents,
            slat_latents=slat_latents,
            return_latents=return_latents,
            return_dict=return_dict,
        )


class TrellisTextTo3DPipeline(_TrellisTwoStagePipeline):
    """TRELLIS text-to-3D (``microsoft/TRELLIS-text-*``): CLIP text tokens condition both flows.

    The unconditional branch is the encoding of the empty prompt, as released; a ``negative_text`` on a
    :class:`TextCondition` replaces it for that batch item.
    """

    model_cpu_offload_seq = "text_encoder->" + _TrellisTwoStagePipeline._model_cpu_offload_seq
    external_components = {
        "text_encoder": ExternalComponent(CLIP_REPO, CLIP_REVISION, _encoder(CLIPTextModel)),
        "tokenizer": ExternalComponent(CLIP_REPO, CLIP_REVISION, _tokenizer),
    }

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        sparse_structure_flow_model: TrellisSparseStructureFlowModel,
        sparse_structure_decoder: TrellisSparseStructureDecoder,
        sparse_structure_scheduler: TrellisFlowEulerScheduler,
        slat_flow_model: TrellisSLatFlowModel | None = None,
        slat_scheduler: TrellisFlowEulerScheduler | None = None,
        gaussian_decoder: TrellisSLatGaussianDecoder | None = None,
        mesh_decoder: TrellisSLatMeshDecoder | None = None,
        radiance_field_decoder: TrellisSLatRadianceFieldDecoder | None = None,
        slat_mean: Sequence[float] | None = None,
        slat_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            text_encoder.config.hidden_size,
            sparse_structure_flow_model,
            sparse_structure_decoder,
            sparse_structure_scheduler,
            slat_flow_model,
            slat_scheduler,
            gaussian_decoder,
            mesh_decoder,
            radiance_field_decoder,
            slat_mean,
            slat_std,
        )
        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer)

    @staticmethod
    def preprocess(prompt: str | TextCondition | Sequence[str | TextCondition]) -> tuple[list[str], list[str]]:
        """Normalize prompts to parallel positive / negative lists; the released negative is the empty string."""

        if isinstance(prompt, (str, TextCondition)):
            items: Sequence[str | TextCondition] = (prompt,)
        elif isinstance(prompt, Sequence):
            items = tuple(prompt)
        else:
            raise TypeError("prompt must be a string, a TextCondition, or a sequence of them")
        if not items:
            raise ValueError("prompt must not be empty")
        positives, negatives = [], []
        for item in items:
            if isinstance(item, str):
                item = TextCondition(text=item)
            elif type(item) is not TextCondition:
                raise TypeError("prompt sequences must contain strings or exact TextCondition values")
            positives.append(item.text)
            negatives.append("" if item.negative_text is None else item.negative_text)
        return positives, negatives

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        """CLIP ``last_hidden_state`` of prompts padded and truncated to the encoder's context, as released."""

        input_ids = self.tokenizer(
            list(prompts),
            max_length=self.text_encoder.config.max_position_embeddings,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return self.text_encoder(input_ids=input_ids.to(self.text_encoder.device)).last_hidden_state

    def encode_conditioning(
        self, prompts: Sequence[str], negative_prompts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.encode_prompt(prompts)
        if all(negative == "" for negative in negative_prompts):
            unconditional = self.encode_prompt([""]).expand(len(prompts), -1, -1)
        else:
            unconditional = self.encode_prompt(negative_prompts)
        return conditional, unconditional

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | TextCondition | Sequence[str | TextCondition],
        *,
        formats: tuple[str, ...] | list[str] | None = None,
        sparse_structure_num_inference_steps: int = 25,
        slat_num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        guidance_interval: tuple[float, float] = (0.5, 0.95),
        rescale_t: float = 3.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: torch.Tensor | None = None,
        slat_latents: TrellisSparseTensor | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Generate from one or more prompts.

        ``guidance_scale`` and ``guidance_interval`` default to the released text sampler settings (7.5 and
        ``(0.5, 0.95)``) rather than the image pipeline's.
        """

        prompts, negative_prompts = self.preprocess(prompt)
        conditional, unconditional = self.encode_conditioning(prompts, negative_prompts)
        return self._generate(
            conditional,
            unconditional,
            formats=formats,
            sparse_structure_num_inference_steps=sparse_structure_num_inference_steps,
            slat_num_inference_steps=slat_num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=rescale_t,
            generator=generator,
            sparse_structure_latents=sparse_structure_latents,
            slat_latents=slat_latents,
            return_latents=return_latents,
            return_dict=return_dict,
        )


__all__ = ["TrellisImageTo3DPipeline", "TrellisTextTo3DPipeline"]
