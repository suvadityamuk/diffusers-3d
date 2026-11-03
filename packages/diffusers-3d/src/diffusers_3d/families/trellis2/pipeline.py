# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for typed object-native stages and explicit backend/license gates.
# Sparse convolution, attention, and mesh extraction run in plain PyTorch.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import DINOv3ViTModel

from ...backends import ovoxel_to_mesh, textured_mesh_from_ovoxel
from ...data import ImageCondition, ImageInput, preprocess_image_condition
from ...data.background import BackgroundRemoval, downscale_longest_side, lacks_alpha
from ...data.birefnet import RMBG2_REPO, RMBG2_REVISION, BiRefNetBackgroundRemover
from ...data.image_processing import _quantized_rgba, normalize_imagenet
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import (
    CameraRig,
    Latent3DOutput,
    MeshAsset,
    Object3D,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
)
from ..trellis.sparse import TrellisSparseTensor, trellis_grid_transform
from .decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
from .models import (
    DINOV3_REVISIONS,
    DINOV3_VITL16_REPO,
    Trellis2SLatFlowModel,
    Trellis2SparseStructureFlowModel,
    load_dinov3,
)
from .scheduler import Trellis2FlowEulerScheduler

_SPARSE_SAMPLER_DEFAULTS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.7,
    "guidance_interval": [0.6, 1.0],
    "rescale_t": 5.0,
}
_SHAPE_SAMPLER_DEFAULTS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "guidance_interval": [0.6, 1.0],
    "rescale_t": 3.0,
}
_TEXTURE_SAMPLER_DEFAULTS = {
    "steps": 12,
    "guidance_strength": 1.0,
    "guidance_rescale": 0.0,
    "guidance_interval": [0.6, 0.9],
    "rescale_t": 3.0,
}
_CAPABILITY_LIMITATIONS = {"official_full_checkpoint_parity": False, "production_gpu_quality_verified": False}


@dataclass(frozen=True)
class _PipelineType:
    """One released preset, relative to the components so tiny and production layouts both work.

    ``sparse_structure_pooling`` max-pools the decoded occupancy before the first SLAT stage; ``stages`` names the
    flow models to run; ``cascade_scale`` is the second stage's latent grid relative to the first (2 for 1024,
    3 for 1536), or ``None`` for a single stage.
    """

    sparse_structure_pooling: int
    stages: tuple[str, ...]
    cascade_scale: int | None = None


_PIPELINE_TYPES = {
    "512": _PipelineType(sparse_structure_pooling=2, stages=("512",)),
    "1024": _PipelineType(sparse_structure_pooling=1, stages=("1024",)),
    "1024_cascade": _PipelineType(sparse_structure_pooling=2, stages=("512", "1024"), cascade_scale=2),
    "1536_cascade": _PipelineType(sparse_structure_pooling=2, stages=("512", "1024"), cascade_scale=3),
}
# The 1024 stages condition on a 1024x1024 image, twice the 512 side (``get_cond(image, 1024)`` upstream).
_STAGE_IMAGE_SCALE = {"512": 1, "1024": 2}
_UPSTREAM_CASCADE_STEP = 128


# Released components TRELLIS.2 and Pixal3D use unchanged from Microsoft's TRELLIS releases (byte-identical files).
TRELLIS_IMAGE_LARGE_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_IMAGE_LARGE_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS2_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
RELEASED_STEMS = {
    "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
    "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
    "pbr_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
}
# Released ``pipeline.json`` model keys of those components.
RELEASED_KEYS = {
    "sparse_structure_decoder": "sparse_structure_decoder",
    "shape_slat_decoder": "shape_slat_decoder",
    "pbr_decoder": "tex_slat_decoder",
}


def load_released_component(
    repository: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    name: str,
    family: str = "trellis2",
    **hub_kwargs: Any,
):
    """Download a released TRELLIS-layout component (``ckpts/<stem>.json`` and ``.safetensors``) and convert it on
    load with the ``family`` converter (``"trellis2"`` or ``"pixal3d"``), so its tensors equal a converted folder's."""

    import tempfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    names = ("token", "cache_dir", "local_files_only", "force_download")
    download = {key: hub_kwargs[key] for key in names if hub_kwargs.get(key) is not None}
    stem = RELEASED_STEMS[name]
    paths = [
        hf_hub_download(repository, stem + suffix, revision=revision, **download)
        for suffix in (".json", ".safetensors")
    ]
    source = Path(paths[0]).with_suffix("")
    if family == "pixal3d":
        from ..pixal3d.conversion import _save_component as save
    else:
        from .conversion import _save_released_component as save
    with tempfile.TemporaryDirectory(prefix="trellis2-component-") as directory:
        model_type, _ = save(RELEASED_KEYS[name], source, Path(directory), safe_serialization=True)
        # The default loading path, exactly as a pipeline loads a converted folder.
        model = model_type.from_pretrained(directory, dtype=dtype)
    return model.eval()


def shrink_image_condition(condition: ImageCondition, max_size: int = 1024) -> ImageCondition:
    """Shrink an image whose longer side exceeds ``max_size``, as upstream ``preprocess_image`` does before cropping.

    The quantized RGBA image (alpha times the mask) is resized to ``int(side * max_size / longer side)`` with Pillow's
    LANCZOS, which resamples RGBA premultiplied; a camera's intrinsics are scaled to the new size.
    """

    height, width = condition.image.shape[-2:]
    scale = min(1, max_size / max(height, width))
    if scale >= 1:
        return condition
    size = (int(width * scale), int(height * scale))
    resized = np.asarray(Image.fromarray(_quantized_rgba(condition)[1], "RGBA").resize(size, Image.Resampling.LANCZOS))
    image = torch.from_numpy(resized.copy()).permute(2, 0, 1).to(condition.image).div(255)
    camera = condition.camera
    if camera is not None:
        resize = camera.intrinsics.new_tensor([size[0] / width, size[1] / height, 1.0]).diag()
        camera = CameraRig(
            world_to_camera=camera.world_to_camera,
            intrinsics=resize @ camera.intrinsics,
            image_sizes=camera.image_sizes.new_tensor([[size[1], size[0]]]),
            coordinate_system=camera.coordinate_system,
            metadata=camera.metadata,
        )
    return ImageCondition(image=image, camera=camera)


def _sampler_parameters(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = {**defaults, **({} if overrides is None else dict(overrides))}
    expected = {"steps", "guidance_strength", "guidance_rescale", "guidance_interval", "rescale_t"}
    unknown = set(values).difference(expected)
    if unknown:
        raise ValueError(f"unknown sampler parameters: {sorted(unknown)}")
    if not isinstance(values["steps"], int) or isinstance(values["steps"], bool) or values["steps"] <= 0:
        raise ValueError("sampler steps must be a positive integer")
    interval = tuple(float(value) for value in values["guidance_interval"])
    if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
        raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
    guidance_rescale = float(values["guidance_rescale"])
    if not 0 <= guidance_rescale <= 1:
        raise ValueError("guidance_rescale must lie in [0, 1]")
    rescale_t = float(values["rescale_t"])
    if rescale_t <= 0:
        raise ValueError("rescale_t must be positive")
    return {
        "steps": values["steps"],
        "guidance_strength": float(values["guidance_strength"]),
        "guidance_rescale": guidance_rescale,
        "guidance_interval": interval,
        "rescale_t": rescale_t,
    }


class Trellis2ImageTo3DPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """TRELLIS.2 image-to-3D: sparse structure -> shape SLAT -> texture SLAT -> O-Voxels, all in plain PyTorch.

    ``pipeline_type`` picks a released preset. ``"512"`` and ``"1024"`` run one shape and one texture flow model;
    the cascades sample the 512 shape model, grow its coordinates with the shape decoder, and sample the 1024
    models on the grown grid. ``shape_slat_flow_model`` / ``texture_slat_flow_model`` are the 512 models and the
    ``*_1024`` components the 1024 ones. Only the textured-mesh postprocess needs CuMesh.

    The DINOv3 ``image_encoder`` and the background remover's RMBG-2.0 weights are gated third-party weights: they
    load from their own repositories (accept their licenses on the Hub and log in first) and are never saved with
    the pipeline. The ``sparse_structure_decoder`` is TRELLIS-image-large's and likewise loads from that repository.
    """

    # As upstream: inputs without usable alpha are shrunk to at most 1024 px in their own mode and cut out with
    # BiRefNet using briaai/RMBG-2.0 weights (gated, non-commercial).
    background_removal = BackgroundRemoval(
        partial(BiRefNetBackgroundRemover, resize="pil", repository=RMBG2_REPO, revision=RMBG2_REVISION),
        trigger=lacks_alpha,
        resize=downscale_longest_side,
        resize_in_source_mode=True,
    )
    # TRELLIS.2 reuses TRELLIS-image-large's sparse-structure decoder, which loads (and converts) from there.
    external_components = {
        "image_encoder": ExternalComponent(DINOV3_VITL16_REPO, DINOV3_REVISIONS[DINOV3_VITL16_REPO], load_dinov3),
        "sparse_structure_decoder": ExternalComponent(
            TRELLIS_IMAGE_LARGE_REPO,
            TRELLIS_IMAGE_LARGE_REVISION,
            partial(load_released_component, name="sparse_structure_decoder"),
        ),
    }
    model_cpu_offload_seq = (
        "image_encoder->sparse_structure_flow_model->sparse_structure_decoder->"
        "shape_slat_flow_model->shape_slat_flow_model_1024->shape_slat_decoder->"
        "texture_slat_flow_model->texture_slat_flow_model_1024->pbr_decoder"
    )
    _optional_components = [
        "shape_slat_flow_model",
        "shape_slat_flow_model_1024",
        "shape_slat_scheduler",
        "shape_slat_decoder",
        "texture_slat_flow_model",
        "texture_slat_flow_model_1024",
        "texture_slat_scheduler",
        "pbr_decoder",
    ]
    _lora_loadable_modules = [
        "sparse_structure_flow_model",
        "shape_slat_flow_model",
        "shape_slat_flow_model_1024",
        "texture_slat_flow_model",
        "texture_slat_flow_model_1024",
    ]

    def __init__(
        self,
        image_encoder: DINOv3ViTModel,
        sparse_structure_flow_model: Trellis2SparseStructureFlowModel,
        sparse_structure_decoder: Trellis2SparseStructureDecoder,
        sparse_structure_scheduler: Trellis2FlowEulerScheduler,
        shape_slat_flow_model: Trellis2SLatFlowModel | None = None,
        shape_slat_flow_model_1024: Trellis2SLatFlowModel | None = None,
        shape_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        shape_slat_decoder: Trellis2ShapeDualGridDecoder | None = None,
        texture_slat_flow_model: Trellis2SLatFlowModel | None = None,
        texture_slat_flow_model_1024: Trellis2SLatFlowModel | None = None,
        texture_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        pbr_decoder: Trellis2PBRSparseDecoder | None = None,
        shape_slat_mean: Sequence[float] | None = None,
        shape_slat_std: Sequence[float] | None = None,
        texture_slat_mean: Sequence[float] | None = None,
        texture_slat_std: Sequence[float] | None = None,
        default_pipeline_type: str = "1024_cascade",
        image_size: int = 512,
        sparse_structure_sampler_defaults: Mapping[str, Any] | None = None,
        shape_slat_sampler_defaults: Mapping[str, Any] | None = None,
        texture_slat_sampler_defaults: Mapping[str, Any] | None = None,
        capability_limitations: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if image_size % image_encoder.config.patch_size:
            raise ValueError("image_size must be divisible by the DINOv3 patch size")
        cond_channels = image_encoder.config.hidden_size
        if sparse_structure_flow_model.config.cond_channels != cond_channels:
            raise ValueError("sparse-structure image encoder and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")

        shape_flows = {"512": shape_slat_flow_model, "1024": shape_slat_flow_model_1024}
        texture_flows = {"512": texture_slat_flow_model, "1024": texture_slat_flow_model_1024}
        has_shape = any(model is not None for model in shape_flows.values())
        has_texture = any(model is not None for model in texture_flows.values())
        if has_shape and (shape_slat_scheduler is None or shape_slat_decoder is None):
            raise ValueError("shape SLAT flow models require shape_slat_scheduler and shape_slat_decoder")
        if not has_shape and (shape_slat_scheduler is not None or shape_slat_decoder is not None or has_texture):
            raise ValueError("shape SLAT scheduler, decoder, and texture components require a shape SLAT flow model")
        if has_texture and (texture_slat_scheduler is None or pbr_decoder is None):
            raise ValueError("texture SLAT flow models require texture_slat_scheduler and pbr_decoder")
        if not has_texture and (texture_slat_scheduler is not None or pbr_decoder is not None):
            raise ValueError("texture SLAT scheduler and PBR decoder require a texture SLAT flow model")
        shape_channels = None
        for stage, model in shape_flows.items():
            if model is None:
                continue
            if model.config.cond_channels != cond_channels:
                raise ValueError(f"shape SLAT {stage} image encoder and flow context dimensions must match")
            if model.config.out_channels != shape_slat_decoder.config.latent_channels:
                raise ValueError(f"shape SLAT {stage} output channels must match the shape decoder")
            shape_channels = model.config.out_channels
        texture_channels = None
        for stage, model in texture_flows.items():
            if model is None:
                continue
            if shape_flows[stage] is None:
                raise ValueError(f"texture SLAT {stage} flow model requires the shape SLAT {stage} flow model")
            if model.config.cond_channels != cond_channels:
                raise ValueError(f"texture SLAT {stage} image encoder and flow context dimensions must match")
            if model.config.out_channels != pbr_decoder.config.latent_channels:
                raise ValueError(f"texture SLAT {stage} output channels must match the PBR decoder")
            if model.config.in_channels != model.config.out_channels + shape_flows[stage].config.out_channels:
                raise ValueError(f"texture SLAT {stage} input channels must concatenate texture noise and shape SLAT")
            texture_channels = model.config.out_channels
        default_pipeline_type = self._validate_pipeline_type(default_pipeline_type)

        def normalization(
            mean: Sequence[float] | None, std: Sequence[float] | None, channels: int | None, name: str
        ) -> tuple[list[float] | None, list[float] | None]:
            if mean is None and std is None:
                return None, None
            if mean is None or std is None or channels is None:
                raise ValueError(f"{name} mean and std must be supplied together with its flow model")
            normalized_mean = [float(value) for value in mean]
            normalized_std = [float(value) for value in std]
            if len(normalized_mean) != channels or len(normalized_std) != channels:
                raise ValueError(f"{name} normalization must have one value per output channel")
            if any(value <= 0 for value in normalized_std):
                raise ValueError(f"{name} standard deviations must be positive")
            return normalized_mean, normalized_std

        shape_mean, shape_std = normalization(shape_slat_mean, shape_slat_std, shape_channels, "shape SLAT")
        texture_mean, texture_std = normalization(
            texture_slat_mean, texture_slat_std, texture_channels, "texture SLAT"
        )
        sparse_defaults = _sampler_parameters(_SPARSE_SAMPLER_DEFAULTS, sparse_structure_sampler_defaults)
        shape_defaults = _sampler_parameters(_SHAPE_SAMPLER_DEFAULTS, shape_slat_sampler_defaults)
        texture_defaults = _sampler_parameters(_TEXTURE_SAMPLER_DEFAULTS, texture_slat_sampler_defaults)
        limitations = {**_CAPABILITY_LIMITATIONS, **({} if capability_limitations is None else capability_limitations)}
        self.register_modules(
            image_encoder=image_encoder,
            sparse_structure_flow_model=sparse_structure_flow_model,
            sparse_structure_decoder=sparse_structure_decoder,
            sparse_structure_scheduler=sparse_structure_scheduler,
            shape_slat_flow_model=shape_slat_flow_model,
            shape_slat_flow_model_1024=shape_slat_flow_model_1024,
            shape_slat_scheduler=shape_slat_scheduler,
            shape_slat_decoder=shape_slat_decoder,
            texture_slat_flow_model=texture_slat_flow_model,
            texture_slat_flow_model_1024=texture_slat_flow_model_1024,
            texture_slat_scheduler=texture_slat_scheduler,
            pbr_decoder=pbr_decoder,
        )
        self.register_to_config(
            shape_slat_mean=shape_mean,
            shape_slat_std=shape_std,
            texture_slat_mean=texture_mean,
            texture_slat_std=texture_std,
            default_pipeline_type=default_pipeline_type,
            image_size=image_size,
            sparse_structure_sampler_defaults=sparse_defaults,
            shape_slat_sampler_defaults=shape_defaults,
            texture_slat_sampler_defaults=texture_defaults,
            capability_limitations=limitations,
        )

    @staticmethod
    def _validate_pipeline_type(pipeline_type: str) -> str:
        if not isinstance(pipeline_type, str) or pipeline_type not in _PIPELINE_TYPES:
            raise ValueError(f"pipeline_type must be one of {sorted(_PIPELINE_TYPES)}")
        return pipeline_type

    @classmethod
    def _sparse_structure_pooling(cls, pipeline_type: str) -> int:
        """Max-pool ratio applied to the decoded occupancy before the first SLAT stage (2 for 512, 1 for 1024)."""

        return _PIPELINE_TYPES[cls._validate_pipeline_type(pipeline_type)].sparse_structure_pooling

    def _stage_models(self, stage: str) -> tuple[Trellis2SLatFlowModel | None, Trellis2SLatFlowModel | None]:
        suffix = "" if stage == "512" else "_1024"
        return getattr(self, f"shape_slat_flow_model{suffix}"), getattr(self, f"texture_slat_flow_model{suffix}")

    def _stage_image_size(self, stage: str) -> int:
        return self.config.image_size * _STAGE_IMAGE_SCALE[stage]

    def preprocess(self, image: ImageInput | list[ImageInput], *, image_size: int | None = None) -> torch.Tensor:
        conditions = [shrink_image_condition(condition) for condition in self.image_conditions(image)]
        image_size = self.config.image_size if image_size is None else image_size
        processed = [
            preprocess_image_condition(
                condition, image_size=image_size, foreground_scale=1.0, premultiply_before_resize=True
            ).image
            for condition in conditions
        ]
        return torch.stack(processed).to(device=self._execution_device, dtype=self.image_encoder.dtype)

    def encode_image(self, images: torch.Tensor, *, image_size: int | None = None) -> torch.Tensor:
        """DINOv3 tokens of ``[0, 1]`` images resized to ``image_size``, with TRELLIS.2's affine-free final norm."""

        image_size = self.config.image_size if image_size is None else image_size
        images = images.to(device=self._execution_device, dtype=self.image_encoder.dtype)
        if images.shape[-2:] != (image_size, image_size):
            images = F.interpolate(
                images, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True
            )
        output = self.image_encoder(pixel_values=normalize_imagenet(images), output_hidden_states=True)
        return F.layer_norm(output.hidden_states[-1], output.hidden_states[-1].shape[-1:])

    def encode_conditioning(
        self, images: torch.Tensor, *, image_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.encode_image(images, image_size=image_size)
        return conditional, torch.zeros_like(conditional)

    def prepare_sparse_structure_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.sparse_structure_flow_model
        shape = (batch_size, model.config.in_channels, *([model.config.resolution] * 3))
        parameter = next(model.parameters())
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("a generator list must contain one generator per batch item")
        if latents is None:
            return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=parameter.dtype)
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"sparse-structure latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

    def _sample_dense(
        self, latents: torch.Tensor, conditional: torch.Tensor, negative: torch.Tensor, parameters: Mapping[str, Any]
    ) -> torch.Tensor:
        return self._sample_sparse(
            self.sparse_structure_flow_model,
            self.sparse_structure_scheduler,
            latents,
            conditional,
            negative,
            parameters,
        )

    def prepare_slat_latents(
        self,
        structures: Sequence[SparseVoxelAsset],
        model: Trellis2SLatFlowModel,
        *,
        channels: int,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        source = TrellisSparseTensor.from_sparse_voxel_assets(tuple(structures))
        coordinates = source.coordinates
        parameter = next(model.parameters())
        if latents is not None:
            if not torch.equal(latents.coordinates, coordinates) or latents.channels != channels:
                raise ValueError("supplied SLAT latents must match extracted coordinates and channels")
            moved = latents.to(device=self._execution_device, dtype=parameter.dtype)
            source_assets = tuple(asset.to(device=moved.device) for asset in structures)
            return TrellisSparseTensor(moved.coordinates, moved.features, source_assets)
        if isinstance(generator, list):
            if len(generator) != len(structures):
                raise ValueError("a generator list must contain one generator per sparse structure")
            parts = [
                randn_tensor(
                    (structure.coordinates.shape[0], channels),
                    generator=generator[index],
                    device=self._execution_device,
                    dtype=parameter.dtype,
                )
                for index, structure in enumerate(structures)
            ]
            features = torch.cat(parts)
        else:
            features = randn_tensor(
                (coordinates.shape[0], channels),
                generator=generator,
                device=self._execution_device,
                dtype=parameter.dtype,
            )
        source_assets = tuple(asset.to(device=features.device) for asset in structures)
        return TrellisSparseTensor(coordinates.to(device=features.device), features, source_assets)

    @staticmethod
    def _sample_sparse(
        model: Trellis2SLatFlowModel,
        scheduler: Trellis2FlowEulerScheduler,
        latents: TrellisSparseTensor,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        parameters: Mapping[str, Any],
        *,
        concat_cond: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        """Euler sampling with upstream interval CFG; also drives the dense sparse-structure stage."""

        model_kwargs = {} if concat_cond is None else {"concat_cond": concat_cond}
        batch_size = latents.batch_size if isinstance(latents, TrellisSparseTensor) else latents.shape[0]
        scheduler.set_timesteps(parameters["steps"], device=latents.device, rescale_t=parameters["rescale_t"])
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(timestep, batch_size, device=latents.device)
            positive = model(latents, model_timestep, conditional, **model_kwargs).sample
            if scheduler.guidance_is_active(timestep, parameters["guidance_interval"]):
                negative_prediction = model(latents, model_timestep, negative, **model_kwargs).sample
                velocity = scheduler.apply_guidance(
                    positive,
                    negative_prediction,
                    parameters["guidance_strength"],
                    sample=latents,
                    timestep=timestep,
                    guidance_rescale=parameters["guidance_rescale"],
                )
            else:
                velocity = positive
            latents = scheduler.step(velocity, timestep, latents).prev_sample
        return latents

    @staticmethod
    def _denormalize(
        value: TrellisSparseTensor, mean: Sequence[float] | None, std: Sequence[float] | None
    ) -> TrellisSparseTensor:
        if mean is None or std is None:
            return value
        return value.denormalize(value.features.new_tensor(mean), value.features.new_tensor(std))

    @staticmethod
    def _slat_assets(value: TrellisSparseTensor, *, resolution: int, stage: str) -> tuple[SparseVoxelAsset, ...]:
        assets = []
        for batch_index in range(value.batch_size):
            mask = value.coordinates[:, 0] == batch_index
            source = value.source_assets[batch_index] if value.source_assets is not None else None
            assets.append(
                SparseVoxelAsset(
                    coordinates=value.coordinates[mask, 1:].to(dtype=torch.int64),
                    features=value.features[mask],
                    grid_transform=trellis_grid_transform(resolution, device=value.device, dtype=value.dtype),
                    transform=source.transform
                    if source is not None
                    else torch.eye(4, device=value.device, dtype=value.dtype),
                    coordinate_system=source.coordinate_system if source is not None else "right_handed_z_up",
                    metadata={
                        **({} if source is None else source.metadata),
                        "family": "trellis2",
                        "representation": "slat",
                        "stage": stage,
                        "resolution": resolution,
                    },
                )
            )
        return tuple(assets)

    def sample_shape_slat(
        self,
        structures: Sequence[SparseVoxelAsset],
        model: Trellis2SLatFlowModel,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        parameters: Mapping[str, Any],
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> tuple[TrellisSparseTensor, TrellisSparseTensor]:
        """Return the shape SLAT in normalized and denormalized form (the texture stage conditions on the former)."""

        assert self.shape_slat_scheduler is not None
        noise = self.prepare_slat_latents(
            structures, model, channels=model.config.in_channels, generator=generator, latents=latents
        )
        normalized = self._sample_sparse(model, self.shape_slat_scheduler, noise, conditional, negative, parameters)
        return normalized, self._denormalize(normalized, self.config.shape_slat_mean, self.config.shape_slat_std)

    def sample_texture_slat(
        self,
        structures: Sequence[SparseVoxelAsset],
        model: Trellis2SLatFlowModel,
        normalized_shape_slat: TrellisSparseTensor,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        parameters: Mapping[str, Any],
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        assert self.texture_slat_scheduler is not None
        channels = model.config.in_channels - normalized_shape_slat.channels
        noise = self.prepare_slat_latents(structures, model, channels=channels, generator=generator, latents=latents)
        normalized = self._sample_sparse(
            model,
            self.texture_slat_scheduler,
            noise,
            conditional,
            negative,
            parameters,
            concat_cond=normalized_shape_slat,
        )
        return self._denormalize(normalized, self.config.texture_slat_mean, self.config.texture_slat_std)

    @staticmethod
    def _quantize_cascade_coordinates(coordinates: torch.Tensor, resolution: int, target: int) -> torch.Tensor:
        """Map fine voxel coordinates on a ``resolution`` grid onto the cascade's ``target`` latent grid."""

        return ((coordinates + 0.5) / resolution * target).to(dtype=coordinates.dtype)

    def upsample_structures(
        self,
        shape_slat: TrellisSparseTensor,
        structures: Sequence[SparseVoxelAsset],
        *,
        cascade_scale: int,
        max_num_tokens: int,
    ) -> tuple[tuple[SparseVoxelAsset, ...], int]:
        """Released cascade step: grow the low-resolution shape SLAT into the next stage's active voxels.

        The shape decoder's upsampling stages produce the fine grid; those voxels are quantized onto a latent
        grid ``cascade_scale`` times the current one. If that exceeds ``max_num_tokens`` the target shrinks by
        the released 128-voxel step (in output units) until it fits or reaches the 1024 equivalent.
        """

        assert self.shape_slat_decoder is not None
        decoder = self.shape_slat_decoder
        factor = 2**decoder.num_upsamples
        low_resolution = int(structures[0].metadata["resolution"])
        fine_coordinates = decoder.upsample_coordinates(shape_slat, decoder.num_upsamples)
        fine_resolution = low_resolution * factor
        target = low_resolution * cascade_scale
        floor = low_resolution * 2
        step = max(_UPSTREAM_CASCADE_STEP // factor, 1)
        while True:
            quantized = torch.cat(
                [
                    fine_coordinates[:, :1],
                    self._quantize_cascade_coordinates(fine_coordinates[:, 1:], fine_resolution, target),
                ],
                dim=1,
            ).unique(dim=0)
            if quantized.shape[0] < max_num_tokens or target <= floor:
                break
            target = max(target - step, floor)
        grown = []
        for batch_index, structure in enumerate(structures):
            mask = quantized[:, 0] == batch_index
            coordinates = quantized[mask, 1:].to(dtype=torch.int64)
            if coordinates.shape[0] == 0:
                raise ValueError(f"cascade upsampling produced no voxels for batch item {batch_index}")
            grown.append(
                SparseVoxelAsset(
                    coordinates=coordinates,
                    features=torch.ones(coordinates.shape[0], 1, device=coordinates.device, dtype=shape_slat.dtype),
                    grid_transform=trellis_grid_transform(target, device=coordinates.device, dtype=shape_slat.dtype),
                    transform=structure.transform,
                    coordinate_system=structure.coordinate_system,
                    metadata={**structure.metadata, "resolution": target, "stage": "cascade_upsample"},
                )
            )
        return tuple(grown), target

    def decode_ovoxel(
        self, shape_slat: TrellisSparseTensor, texture_slat: TrellisSparseTensor | None = None, *, resolution: int
    ) -> tuple[OVoxelAsset, ...]:
        """Decode shape (and optionally texture) SLATs into O-Voxels on a ``resolution``-cubed grid."""

        if self.shape_slat_decoder is None:
            raise RuntimeError("O-Voxel shape decoding requires shape_slat_decoder")
        shape_output = self.shape_slat_decoder(shape_slat, resolution=resolution)
        if texture_slat is None:
            return shape_output.assets
        if self.pbr_decoder is None:
            raise RuntimeError("PBR O-Voxel decoding requires pbr_decoder")
        return self.pbr_decoder(texture_slat, shape_output.assets, shape_output.subdivisions).assets

    def postprocess_ovoxel(
        self,
        asset: OVoxelAsset,
        *,
        output_format: str,
        texturer: Callable[..., MeshAsset] = textured_mesh_from_ovoxel,
        **kwargs: Any,
    ) -> MeshAsset:
        """Mesh an O-Voxel (``"mesh"``) or bake it into a PBR-textured mesh (``"textured_mesh"``, needs CuMesh)."""

        if output_format == "mesh":
            return ovoxel_to_mesh(asset)
        if output_format == "textured_mesh":
            return texturer(asset, **kwargs)
        raise ValueError("output_format must be 'mesh' or 'textured_mesh'")

    def _require_stage_components(self, preset: _PipelineType, *, needs_texture: bool) -> None:
        if self.shape_slat_flow_model is None and self.shape_slat_flow_model_1024 is None:
            raise RuntimeError("SLAT, O-Voxel, and mesh formats require the shape SLAT flow, scheduler, and decoder")
        for stage in preset.stages:
            shape_model, texture_model = self._stage_models(stage)
            if shape_model is None:
                raise RuntimeError(f"this pipeline_type needs the {stage} shape SLAT flow model")
            if needs_texture and stage == preset.stages[-1] and texture_model is None:
                raise RuntimeError(f"texture, O-Voxel, and mesh formats need the {stage} texture SLAT flow model")

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        formats: tuple[str, ...] | list[str] | None = None,
        pipeline_type: str | None = None,
        max_num_tokens: int = 49152,
        sparse_structure_sampler_params: Mapping[str, Any] | None = None,
        shape_slat_sampler_params: Mapping[str, Any] | None = None,
        texture_slat_sampler_params: Mapping[str, Any] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: torch.Tensor | None = None,
        shape_slat_latents: TrellisSparseTensor | None = None,
        texture_slat_latents: TrellisSparseTensor | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Generate 3D objects from one or more images.

        ``formats`` picks what is returned, in this order: ``"sparse_structure"`` (occupancy voxels),
        ``"shape_slat"`` and ``"texture_slat"`` (structured latents), ``"o_voxel"`` (decoded PBR O-Voxels), and
        ``"mesh"`` (O-Voxel surfaces with vertex colours). It defaults to ``"o_voxel"`` when the texture
        components are loaded and to ``"sparse_structure"`` otherwise. ``pipeline_type`` selects the released
        preset; ``max_num_tokens`` caps the cascade's second-stage grid. ``shape_slat_latents`` and
        ``texture_slat_latents`` replace the sampled noise of the last stage.
        """

        if formats is None:
            formats = ("o_voxel",) if self.pbr_decoder is not None else ("sparse_structure",)
        formats = tuple(formats)
        allowed = {"sparse_structure", "shape_slat", "texture_slat", "o_voxel", "mesh"}
        if not formats or len(set(formats)) != len(formats) or set(formats).difference(allowed):
            raise ValueError(f"formats must contain unique values from {sorted(allowed)}")
        if not isinstance(max_num_tokens, int) or isinstance(max_num_tokens, bool) or max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be a positive integer")
        pipeline_type = self._validate_pipeline_type(
            self.config.default_pipeline_type if pipeline_type is None else pipeline_type
        )
        preset = _PIPELINE_TYPES[pipeline_type]
        needs_shape = bool(set(formats).difference({"sparse_structure"}))
        needs_texture = any(value in formats for value in ("texture_slat", "o_voxel", "mesh"))
        if needs_shape:
            self._require_stage_components(preset, needs_texture=needs_texture)

        # Conditioning is computed once per image size the run needs (512 stages share the sparse-structure one).
        image_sizes = {self._stage_image_size("512")}
        if needs_shape:
            image_sizes.update(self._stage_image_size(stage) for stage in preset.stages)
        conditioning: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for image_size in sorted(image_sizes):
            images = self.preprocess(image, image_size=image_size)
            conditioning[image_size] = self.encode_conditioning(images, image_size=image_size)
        batch_size = images.shape[0]
        conditional, negative = conditioning[self._stage_image_size("512")]

        dense_latents = self.prepare_sparse_structure_latents(
            batch_size, generator=generator, latents=sparse_structure_latents
        )
        sparse_parameters = _sampler_parameters(
            self.config.sparse_structure_sampler_defaults, sparse_structure_sampler_params
        )
        dense_latents = self._sample_dense(dense_latents, conditional, negative, sparse_parameters)
        structures = self.sparse_structure_decoder.decode_to_sparse_voxels(
            dense_latents, pooling=self._sparse_structure_pooling(pipeline_type)
        )
        objects: list[Object3D] = []
        if "sparse_structure" in formats:
            objects.extend(structures)

        if needs_shape:
            shape_parameters = _sampler_parameters(self.config.shape_slat_sampler_defaults, shape_slat_sampler_params)
            texture_parameters = _sampler_parameters(
                self.config.texture_slat_sampler_defaults, texture_slat_sampler_params
            )
            resolution = int(structures[0].metadata["resolution"])
            for index, stage in enumerate(preset.stages):
                last = index == len(preset.stages) - 1
                shape_model, texture_model = self._stage_models(stage)
                conditional, negative = conditioning[self._stage_image_size(stage)]
                normalized_shape, shape_slat = self.sample_shape_slat(
                    structures,
                    shape_model,
                    conditional,
                    negative,
                    shape_parameters,
                    generator=generator,
                    latents=shape_slat_latents if last else None,
                )
                if not last:
                    structures, resolution = self.upsample_structures(
                        shape_slat, structures, cascade_scale=preset.cascade_scale, max_num_tokens=max_num_tokens
                    )
            if "shape_slat" in formats:
                objects.extend(self._slat_assets(shape_slat, resolution=resolution, stage="shape"))
            texture_slat = None
            if needs_texture:
                texture_slat = self.sample_texture_slat(
                    structures,
                    texture_model,
                    normalized_shape,
                    conditional,
                    negative,
                    texture_parameters,
                    generator=generator,
                    latents=texture_slat_latents,
                )
                if "texture_slat" in formats:
                    objects.extend(self._slat_assets(texture_slat, resolution=resolution, stage="texture"))
            if "o_voxel" in formats or "mesh" in formats:
                output_resolution = resolution * 2**self.shape_slat_decoder.num_upsamples
                ovoxels = self.decode_ovoxel(shape_slat, texture_slat, resolution=output_resolution)
                if "o_voxel" in formats:
                    objects.extend(ovoxels)
                if "mesh" in formats:
                    objects.extend(ovoxel_to_mesh(asset) for asset in ovoxels)
        latent_output = (
            Latent3DOutput(latents=dense_latents, metadata={"family": "trellis2", "stage": "sparse_structure"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(objects=tuple(objects), latents=latent_output, previews=None)


__all__ = [
    "TRELLIS2_REPO",
    "TRELLIS2_REVISION",
    "TRELLIS_IMAGE_LARGE_REPO",
    "TRELLIS_IMAGE_LARGE_REVISION",
    "Trellis2ImageTo3DPipeline",
    "load_released_component",
    "shrink_image_condition",
]
