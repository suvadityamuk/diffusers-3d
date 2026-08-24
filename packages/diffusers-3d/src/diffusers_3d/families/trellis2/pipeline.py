# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for typed object-native stages and explicit backend/license gates.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from ...backends import OVoxelBackend, Trellis2PBRPostprocessFacade
from ...data import ImageCondition
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    Latent3DOutput,
    Object3D,
    Object3DKind,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
)
from ..trellis.sparse import TrellisSparseTensor
from .conditioner import Trellis2Dinov3Conditioner
from .decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
from .models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
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
_CAPABILITY_LIMITATIONS = {
    "reviewed_formats": ["sparse_structure"],
    "experimental_formats": ["shape_slat", "texture_slat", "o_voxel", "mesh"],
    "production_1024_cascade": "unsupported_until_flex_gemm_ovoxel_gpu_parity",
    "official_full_checkpoint_parity": False,
    "production_gpu_quality_verified": False,
}
_SPARSE_TARGET_RESOLUTIONS = {
    "512": 32,
    "1024": 64,
    "1024_cascade": 32,
    "1536_cascade": 32,
    "tiny": None,
}


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


class Trellis2ImageTo3DPipeline(Object3DPipeline):
    """TRELLIS.2 with a reviewed CPU sparse-structure path and gated O-Voxel stages."""

    family_id = "trellis2"
    task_ids = ("image-to-3d",)
    output_object_types = (SparseVoxelAsset,)
    output_representations = ("sparse-structure",)
    object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    model_cpu_offload_seq = (
        "conditioner->sparse_structure_flow_model->sparse_structure_decoder->"
        "shape_slat_flow_model->shape_slat_decoder->texture_slat_flow_model->pbr_decoder"
    )
    _optional_components = [
        "shape_slat_flow_model",
        "shape_slat_scheduler",
        "shape_slat_decoder",
        "texture_slat_flow_model",
        "texture_slat_scheduler",
        "pbr_decoder",
    ]

    def __init__(
        self,
        conditioner: Trellis2Dinov3Conditioner,
        sparse_structure_flow_model: Trellis2SparseStructureFlowModel,
        sparse_structure_decoder: Trellis2SparseStructureDecoder,
        sparse_structure_scheduler: Trellis2FlowEulerScheduler,
        shape_slat_flow_model: Trellis2SLatFlowModel | None = None,
        shape_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        shape_slat_decoder: Trellis2ShapeDualGridDecoder | None = None,
        texture_slat_flow_model: Trellis2SLatFlowModel | None = None,
        texture_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        pbr_decoder: Trellis2PBRSparseDecoder | None = None,
        shape_slat_mean: Sequence[float] | None = None,
        shape_slat_std: Sequence[float] | None = None,
        texture_slat_mean: Sequence[float] | None = None,
        texture_slat_std: Sequence[float] | None = None,
        default_pipeline_type: str = "1024_cascade",
        sparse_structure_sampler_defaults: Mapping[str, Any] | None = None,
        shape_slat_sampler_defaults: Mapping[str, Any] | None = None,
        texture_slat_sampler_defaults: Mapping[str, Any] | None = None,
        capability_limitations: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if sparse_structure_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
            raise ValueError("sparse-structure conditioner and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")
        experimental = (
            shape_slat_flow_model,
            shape_slat_scheduler,
            shape_slat_decoder,
            texture_slat_flow_model,
            texture_slat_scheduler,
            pbr_decoder,
        )
        if shape_slat_flow_model is not None:
            if shape_slat_scheduler is None or shape_slat_decoder is None:
                raise ValueError("shape SLAT flow requires its scheduler and shape decoder")
            if shape_slat_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
                raise ValueError("shape SLAT conditioner and flow context dimensions must match")
            if shape_slat_flow_model.config.out_channels != shape_slat_decoder.config.latent_channels:
                raise ValueError("shape SLAT output channels must match the shape decoder")
        elif any(value is not None for value in experimental[1:]):
            raise ValueError("experimental schedulers and decoders require shape_slat_flow_model")
        if texture_slat_flow_model is not None:
            if texture_slat_scheduler is None or pbr_decoder is None or shape_slat_flow_model is None:
                raise ValueError("texture SLAT flow requires shape SLAT, its scheduler, and the PBR decoder")
            if texture_slat_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
                raise ValueError("texture SLAT conditioner and flow context dimensions must match")
            if texture_slat_flow_model.config.out_channels != pbr_decoder.config.latent_channels:
                raise ValueError("texture SLAT output channels must match the PBR decoder")
            expected_concat = texture_slat_flow_model.config.out_channels + shape_slat_flow_model.config.out_channels
            if texture_slat_flow_model.config.in_channels != expected_concat:
                raise ValueError("texture SLAT input channels must concatenate texture noise and shape SLAT")
        default_pipeline_type = self._validate_pipeline_type(default_pipeline_type)

        def normalization(
            mean: Sequence[float] | None,
            std: Sequence[float] | None,
            channels: int | None,
            name: str,
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

        shape_mean, shape_std = normalization(
            shape_slat_mean,
            shape_slat_std,
            None if shape_slat_flow_model is None else shape_slat_flow_model.config.out_channels,
            "shape SLAT",
        )
        texture_mean, texture_std = normalization(
            texture_slat_mean,
            texture_slat_std,
            None if texture_slat_flow_model is None else texture_slat_flow_model.config.out_channels,
            "texture SLAT",
        )
        sparse_defaults = _sampler_parameters(
            _SPARSE_SAMPLER_DEFAULTS,
            sparse_structure_sampler_defaults,
        )
        shape_defaults = _sampler_parameters(_SHAPE_SAMPLER_DEFAULTS, shape_slat_sampler_defaults)
        texture_defaults = _sampler_parameters(_TEXTURE_SAMPLER_DEFAULTS, texture_slat_sampler_defaults)
        limitations = {**_CAPABILITY_LIMITATIONS, **({} if capability_limitations is None else capability_limitations)}
        self.register_modules(
            conditioner=conditioner,
            sparse_structure_flow_model=sparse_structure_flow_model,
            sparse_structure_decoder=sparse_structure_decoder,
            sparse_structure_scheduler=sparse_structure_scheduler,
            shape_slat_flow_model=shape_slat_flow_model,
            shape_slat_scheduler=shape_slat_scheduler,
            shape_slat_decoder=shape_slat_decoder,
            texture_slat_flow_model=texture_slat_flow_model,
            texture_slat_scheduler=texture_slat_scheduler,
            pbr_decoder=pbr_decoder,
        )
        self.register_to_config(
            shape_slat_mean=shape_mean,
            shape_slat_std=shape_std,
            texture_slat_mean=texture_mean,
            texture_slat_std=texture_std,
            default_pipeline_type=default_pipeline_type,
            sparse_structure_sampler_defaults=sparse_defaults,
            shape_slat_sampler_defaults=shape_defaults,
            texture_slat_sampler_defaults=texture_defaults,
            capability_limitations=limitations,
        )

    @staticmethod
    def _validate_pipeline_type(pipeline_type: str) -> str:
        if not isinstance(pipeline_type, str) or pipeline_type not in _SPARSE_TARGET_RESOLUTIONS:
            raise ValueError(f"pipeline_type must be one of {sorted(_SPARSE_TARGET_RESOLUTIONS)}")
        return pipeline_type

    @classmethod
    def _sparse_target_resolution(cls, pipeline_type: str) -> int | None:
        return _SPARSE_TARGET_RESOLUTIONS[cls._validate_pipeline_type(pipeline_type)]

    def preprocess(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                conditions = (ImageCondition(image=image),)
            elif image.ndim == 4:
                conditions = tuple(ImageCondition(image=value) for value in image)
            else:
                raise ValueError("image tensor must have shape (channels, height, width) or be batched")
        elif type(image) is ImageCondition:
            conditions = (image,)
        elif isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
            conditions = tuple(image)
            if not conditions or any(type(condition) is not ImageCondition for condition in conditions):
                raise TypeError("image sequences must contain exact ImageCondition values")
        else:
            raise TypeError("image must be an ImageCondition, sequence of ImageCondition values, or tensor")
        processed = []
        for condition in conditions:
            pixels = condition.image
            if pixels.shape[0] == 1:
                pixels = pixels.expand(3, -1, -1)
            elif pixels.shape[0] == 4:
                pixels = pixels[:3] * pixels[3:4]
            if condition.mask is not None:
                pixels = pixels * condition.mask
            if not bool(torch.isfinite(pixels).all()) or bool(((pixels < 0) | (pixels > 1)).any()):
                raise ValueError("TRELLIS.2 input images must contain finite values in [0, 1]")
            processed.append(
                F.interpolate(
                    pixels.unsqueeze(0),
                    size=(self.conditioner.image_size, self.conditioner.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            )
        parameter = next(self.conditioner.parameters())
        return torch.stack(processed).to(device=self._execution_device, dtype=parameter.dtype)

    def encode_conditioning(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.conditioner(images, value_range=(0.0, 1.0)).embeddings
        negative = self.conditioner.unconditional_embedding(
            images.shape[0],
            device=conditional.device,
            dtype=conditional.dtype,
        )
        return conditional, negative

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
            return randn_tensor(
                shape,
                generator=generator,
                device=self._execution_device,
                dtype=parameter.dtype,
            )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"sparse-structure latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

    def _sample_dense(
        self,
        latents: torch.Tensor,
        conditional: torch.Tensor,
        negative: torch.Tensor,
        parameters: Mapping[str, Any],
    ) -> torch.Tensor:
        scheduler = self.sparse_structure_scheduler
        scheduler.set_timesteps(parameters["steps"], device=latents.device, rescale_t=parameters["rescale_t"])
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(
                timestep,
                latents.shape[0],
                device=latents.device,
            )
            positive = self.sparse_structure_flow_model(latents, model_timestep, conditional).sample
            if scheduler.guidance_is_active(timestep, parameters["guidance_interval"]):
                negative_prediction = self.sparse_structure_flow_model(latents, model_timestep, negative).sample
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
    def _sparse_coordinates(structures: Sequence[SparseVoxelAsset]) -> torch.Tensor:
        return TrellisSparseTensor.from_sparse_voxel_assets(tuple(structures)).coordinates

    def prepare_slat_latents(
        self,
        structures: Sequence[SparseVoxelAsset],
        model: Trellis2SLatFlowModel,
        *,
        channels: int,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TrellisSparseTensor | None = None,
    ) -> TrellisSparseTensor:
        coordinates = self._sparse_coordinates(structures)
        parameter = next(model.parameters())
        if latents is not None:
            if not torch.equal(latents.coordinates, coordinates) or latents.channels != channels:
                raise ValueError("supplied SLAT latents must match extracted coordinates and channels")
            return latents.to(device=self._execution_device, dtype=parameter.dtype)
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
        return TrellisSparseTensor(coordinates.to(device=features.device), features)

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
        scheduler.set_timesteps(parameters["steps"], device=latents.device, rescale_t=parameters["rescale_t"])
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(timestep, latents.batch_size, device=latents.device)
            positive = model(
                latents,
                model_timestep,
                conditional,
                concat_cond=concat_cond,
            ).sample
            if scheduler.guidance_is_active(timestep, parameters["guidance_interval"]):
                negative_prediction = model(
                    latents,
                    model_timestep,
                    negative,
                    concat_cond=concat_cond,
                ).sample
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
        value: TrellisSparseTensor,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
    ) -> TrellisSparseTensor:
        if mean is None or std is None:
            return value
        return value.denormalize(value.features.new_tensor(mean), value.features.new_tensor(std))

    @staticmethod
    def _slat_assets(value: TrellisSparseTensor, *, resolution: int, stage: str) -> tuple[SparseVoxelAsset, ...]:
        assets = []
        for batch_index in range(value.batch_size):
            mask = value.coordinates[:, 0] == batch_index
            assets.append(
                SparseVoxelAsset(
                    coordinates=value.coordinates[mask, 1:].to(dtype=torch.int64),
                    features=value.features[mask],
                    voxel_size=1.0 / resolution,
                    coordinate_system=next(
                        iter(value.source_assets),
                        None,
                    ).coordinate_system
                    if value.source_assets
                    else "right_handed_z_up",
                    metadata={
                        "family": "trellis2",
                        "representation": "slat",
                        "stage": stage,
                        "resolution": resolution,
                        "official_checkpoint_parity": False,
                    },
                )
            )
        return tuple(assets)

    def decode_ovoxel(
        self,
        shape_slat: TrellisSparseTensor,
        texture_slat: TrellisSparseTensor | None = None,
    ) -> tuple[OVoxelAsset, ...]:
        if self.shape_slat_decoder is None:
            raise RuntimeError("O-Voxel shape decoding requires shape_slat_decoder")
        shape_assets = self.shape_slat_decoder(shape_slat).assets
        if texture_slat is None:
            return shape_assets
        if self.pbr_decoder is None:
            raise RuntimeError("PBR O-Voxel decoding requires pbr_decoder")
        return self.pbr_decoder(texture_slat, shape_assets).assets

    def postprocess_ovoxel(
        self,
        asset: OVoxelAsset,
        *,
        output_format: str,
        ovoxel_backend: OVoxelBackend | None = None,
        pbr_postprocess: Trellis2PBRPostprocessFacade | None = None,
        postprocess_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        kwargs = {} if postprocess_kwargs is None else dict(postprocess_kwargs)
        if output_format == "mesh":
            backend = OVoxelBackend() if ovoxel_backend is None else ovoxel_backend
            return backend.to_mesh(asset, **kwargs)
        if output_format == "glb":
            facade = Trellis2PBRPostprocessFacade() if pbr_postprocess is None else pbr_postprocess
            return facade.to_glb(asset, **kwargs)
        raise ValueError("output_format must be 'mesh' or 'glb'")

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        formats: tuple[str, ...] | list[str] = ("sparse_structure",),
        pipeline_type: str | None = None,
        sparse_structure_sampler_params: Mapping[str, Any] | None = None,
        shape_slat_sampler_params: Mapping[str, Any] | None = None,
        texture_slat_sampler_params: Mapping[str, Any] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: torch.Tensor | None = None,
        shape_slat_latents: TrellisSparseTensor | None = None,
        texture_slat_latents: TrellisSparseTensor | None = None,
        ovoxel_backend: OVoxelBackend | None = None,
        postprocess_kwargs: Mapping[str, Any] | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        formats = tuple(formats)
        allowed = {"sparse_structure", "shape_slat", "texture_slat", "o_voxel", "mesh"}
        if not formats or len(set(formats)) != len(formats) or set(formats).difference(allowed):
            raise ValueError(f"formats must contain unique values from {sorted(allowed)}")
        experimental_requested = bool(set(formats).difference({"sparse_structure"}))
        pipeline_type = self._validate_pipeline_type(
            self.config.default_pipeline_type if pipeline_type is None else pipeline_type
        )
        if experimental_requested and pipeline_type != "tiny":
            raise NotImplementedError(
                "the serialized 1024 cascade remains unsupported until production FlexGEMM/O-Voxel GPU parity; "
                "use pipeline_type='tiny' only with backend-free tiny experimental components"
            )
        if experimental_requested and (
            self.shape_slat_flow_model is None or self.shape_slat_scheduler is None or self.shape_slat_decoder is None
        ):
            raise RuntimeError("experimental formats require the shape SLAT flow, scheduler, and decoder")
        needs_texture = any(value in formats for value in ("texture_slat", "o_voxel", "mesh"))
        if needs_texture and (
            self.texture_slat_flow_model is None or self.texture_slat_scheduler is None or self.pbr_decoder is None
        ):
            raise RuntimeError("texture, O-Voxel, and mesh formats require texture SLAT and PBR components")
        images = self.preprocess(image)
        conditional, negative = self.encode_conditioning(images)
        dense_latents = self.prepare_sparse_structure_latents(
            images.shape[0],
            generator=generator,
            latents=sparse_structure_latents,
        )
        sparse_parameters = _sampler_parameters(
            self.config.sparse_structure_sampler_defaults,
            sparse_structure_sampler_params,
        )
        dense_latents = self._sample_dense(dense_latents, conditional, negative, sparse_parameters)
        structures = self.sparse_structure_decoder.decode_to_sparse_voxels(
            dense_latents,
            target_resolution=self._sparse_target_resolution(pipeline_type),
        )
        objects: list[Object3D] = []
        if "sparse_structure" in formats:
            objects.extend(structures)
        if experimental_requested:
            assert self.shape_slat_flow_model is not None
            assert self.shape_slat_scheduler is not None
            shape_noise = self.prepare_slat_latents(
                structures,
                self.shape_slat_flow_model,
                channels=self.shape_slat_flow_model.config.in_channels,
                generator=generator,
                latents=shape_slat_latents,
            )
            shape_parameters = _sampler_parameters(
                self.config.shape_slat_sampler_defaults,
                shape_slat_sampler_params,
            )
            normalized_shape = self._sample_sparse(
                self.shape_slat_flow_model,
                self.shape_slat_scheduler,
                shape_noise,
                conditional,
                negative,
                shape_parameters,
            )
            shape_slat = self._denormalize(
                normalized_shape,
                self.config.shape_slat_mean,
                self.config.shape_slat_std,
            )
            if "shape_slat" in formats:
                objects.extend(
                    self._slat_assets(
                        shape_slat,
                        resolution=self.shape_slat_flow_model.config.resolution,
                        stage="shape",
                    )
                )
            texture_slat = None
            if needs_texture:
                assert self.texture_slat_flow_model is not None
                assert self.texture_slat_scheduler is not None
                normalized_shape_condition = normalized_shape
                texture_channels = (
                    self.texture_slat_flow_model.config.in_channels - normalized_shape_condition.channels
                )
                texture_noise = self.prepare_slat_latents(
                    structures,
                    self.texture_slat_flow_model,
                    channels=texture_channels,
                    generator=generator,
                    latents=texture_slat_latents,
                )
                texture_parameters = _sampler_parameters(
                    self.config.texture_slat_sampler_defaults,
                    texture_slat_sampler_params,
                )
                normalized_texture = self._sample_sparse(
                    self.texture_slat_flow_model,
                    self.texture_slat_scheduler,
                    texture_noise,
                    conditional,
                    negative,
                    texture_parameters,
                    concat_cond=normalized_shape_condition,
                )
                texture_slat = self._denormalize(
                    normalized_texture,
                    self.config.texture_slat_mean,
                    self.config.texture_slat_std,
                )
                if "texture_slat" in formats:
                    objects.extend(
                        self._slat_assets(
                            texture_slat,
                            resolution=self.texture_slat_flow_model.config.resolution,
                            stage="texture",
                        )
                    )
            if any(value in formats for value in ("o_voxel", "mesh")):
                ovoxels = self.decode_ovoxel(shape_slat, texture_slat)
                if "o_voxel" in formats:
                    objects.extend(ovoxels)
                if "mesh" in formats:
                    objects.extend(
                        self.postprocess_ovoxel(
                            asset,
                            output_format="mesh",
                            ovoxel_backend=ovoxel_backend,
                            postprocess_kwargs=postprocess_kwargs,
                        )
                        for asset in ovoxels
                    )
        latent_output = (
            Latent3DOutput(
                latents=dense_latents,
                metadata={"family": "trellis2", "stage": "sparse_structure"},
            )
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(
            objects=tuple(objects),
            latents=latent_output,
            previews=None,
        )


__all__ = ["Trellis2ImageTo3DPipeline"]
