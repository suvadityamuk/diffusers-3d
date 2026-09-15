# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for typed object-native stages and explicit backend/license gates.
# Sparse convolution and attention run in plain PyTorch; only mesh extraction needs the O-Voxel runtime.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...backends import OVoxelBackend, Trellis2PBRPostprocessFacade
from ...data import ImageCondition, preprocess_image_condition
from ...execution.metadata import (
    ContributionStatus,
    Object3DComponentSpec,
    ReviewStatus,
    fully_qualified_class_name,
)
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    Latent3DOutput,
    MeshAsset,
    Object3D,
    Object3DKind,
    Object3DPipelineOutput,
    OVoxelAsset,
    SparseVoxelAsset,
)
from ..trellis.sparse import TrellisSparseTensor, trellis_grid_transform
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
    "official_full_checkpoint_parity": False,
    "production_gpu_quality_verified": False,
}


@dataclass(frozen=True)
class _PipelineType:
    """One released preset, expressed relative to the components so tiny and production layouts both work.

    ``sparse_structure_pooling`` max-pools the decoded occupancy grid before the first SLAT stage; ``stages``
    names the flow models to run (``"512"`` or ``"1024"``); ``cascade_scale`` is the second stage's latent grid
    relative to the first (2 for 1024, 3 for 1536), or ``None`` for a single stage.
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
_UPSTREAM_CASCADE_FLOOR = 1024


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
    """TRELLIS.2 image-to-3D: sparse structure -> shape SLAT -> texture SLAT -> O-Voxels, all in plain PyTorch.

    The four released presets are ``pipeline_type`` values. ``"512"`` and ``"1024"`` run one shape and one
    texture flow model; the cascades sample the 512 shape model first, grow its coordinates with the shape
    decoder, and sample the 1024 models on the grown grid. Component names follow the released
    ``pipeline.json``: ``shape_slat_flow_model`` / ``texture_slat_flow_model`` are the 512 models, the
    ``*_1024`` components are the 1024 models. Meshing O-Voxels needs ``OVoxelBackend``; everything else runs
    on CPU.
    """

    family_id = "trellis2"
    task_ids = ("image-to-3d",)
    output_object_types = (SparseVoxelAsset, OVoxelAsset, MeshAsset)
    output_representations = ("sparse-structure", "slat", "o-voxel", "mesh")
    object_kinds = (Object3DKind.SPARSE_VOXEL, Object3DKind.O_VOXEL, Object3DKind.MESH)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    component_specs = (
        Object3DComponentSpec(
            name="conditioner",
            expected_class=fully_qualified_class_name(Trellis2Dinov3Conditioner),
            subfolder="conditioner",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="sparse_structure_flow_model",
            expected_class=fully_qualified_class_name(Trellis2SparseStructureFlowModel),
            subfolder="sparse_structure_flow_model",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="sparse_structure_decoder",
            expected_class=fully_qualified_class_name(Trellis2SparseStructureDecoder),
            subfolder="sparse_structure_decoder",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="sparse_structure_scheduler",
            expected_class=fully_qualified_class_name(Trellis2FlowEulerScheduler),
            subfolder="sparse_structure_scheduler",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="shape_slat_flow_model",
            expected_class=fully_qualified_class_name(Trellis2SLatFlowModel),
            subfolder="shape_slat_flow_model",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="shape_slat_flow_model_1024",
            expected_class=fully_qualified_class_name(Trellis2SLatFlowModel),
            subfolder="shape_slat_flow_model_1024",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="shape_slat_scheduler",
            expected_class=fully_qualified_class_name(Trellis2FlowEulerScheduler),
            subfolder="shape_slat_scheduler",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="shape_slat_decoder",
            expected_class=fully_qualified_class_name(Trellis2ShapeDualGridDecoder),
            subfolder="shape_slat_decoder",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="texture_slat_flow_model",
            expected_class=fully_qualified_class_name(Trellis2SLatFlowModel),
            subfolder="texture_slat_flow_model",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="texture_slat_flow_model_1024",
            expected_class=fully_qualified_class_name(Trellis2SLatFlowModel),
            subfolder="texture_slat_flow_model_1024",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="texture_slat_scheduler",
            expected_class=fully_qualified_class_name(Trellis2FlowEulerScheduler),
            subfolder="texture_slat_scheduler",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="pbr_decoder",
            expected_class=fully_qualified_class_name(Trellis2PBRSparseDecoder),
            subfolder="pbr_decoder",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
    )
    model_cpu_offload_seq = (
        "conditioner->sparse_structure_flow_model->sparse_structure_decoder->"
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

    def __init__(
        self,
        conditioner: Trellis2Dinov3Conditioner,
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
        sparse_structure_sampler_defaults: Mapping[str, Any] | None = None,
        shape_slat_sampler_defaults: Mapping[str, Any] | None = None,
        texture_slat_sampler_defaults: Mapping[str, Any] | None = None,
        capability_limitations: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cond_channels = conditioner.model.config.hidden_size
        if sparse_structure_flow_model.config.cond_channels != cond_channels:
            raise ValueError("sparse-structure conditioner and flow context dimensions must match")
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
                raise ValueError(f"shape SLAT {stage} conditioner and flow context dimensions must match")
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
                raise ValueError(f"texture SLAT {stage} conditioner and flow context dimensions must match")
            if model.config.out_channels != pbr_decoder.config.latent_channels:
                raise ValueError(f"texture SLAT {stage} output channels must match the PBR decoder")
            if model.config.in_channels != model.config.out_channels + shape_flows[stage].config.out_channels:
                raise ValueError(f"texture SLAT {stage} input channels must concatenate texture noise and shape SLAT")
            texture_channels = model.config.out_channels
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

        shape_mean, shape_std = normalization(shape_slat_mean, shape_slat_std, shape_channels, "shape SLAT")
        texture_mean, texture_std = normalization(
            texture_slat_mean, texture_slat_std, texture_channels, "texture SLAT"
        )
        sparse_defaults = _sampler_parameters(_SPARSE_SAMPLER_DEFAULTS, sparse_structure_sampler_defaults)
        shape_defaults = _sampler_parameters(_SHAPE_SAMPLER_DEFAULTS, shape_slat_sampler_defaults)
        texture_defaults = _sampler_parameters(_TEXTURE_SAMPLER_DEFAULTS, texture_slat_sampler_defaults)
        limitations = {**_CAPABILITY_LIMITATIONS, **({} if capability_limitations is None else capability_limitations)}
        self.register_modules(
            conditioner=conditioner,
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
        return self.conditioner.image_size * _STAGE_IMAGE_SCALE[stage]

    def preprocess(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
        *,
        image_size: int | None = None,
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
        image_size = self.conditioner.image_size if image_size is None else image_size
        processed = [
            preprocess_image_condition(
                condition,
                image_size=image_size,
                foreground_scale=1.0,
                premultiply_before_resize=True,
            ).image
            for condition in conditions
        ]
        parameter = next(self.conditioner.parameters())
        return torch.stack(processed).to(device=self._execution_device, dtype=parameter.dtype)

    def encode_conditioning(
        self,
        images: torch.Tensor,
        *,
        image_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.conditioner(images, image_size=image_size, value_range=(0.0, 1.0)).embeddings
        negative = self.conditioner.unconditional_embedding(
            images.shape[0],
            image_size=image_size,
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
            return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=parameter.dtype)
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
            model_timestep = scheduler.model_timestep(timestep, latents.shape[0], device=latents.device)
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
        scheduler.set_timesteps(parameters["steps"], device=latents.device, rescale_t=parameters["rescale_t"])
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(timestep, latents.batch_size, device=latents.device)
            positive = model(latents, model_timestep, conditional, concat_cond=concat_cond).sample
            if scheduler.guidance_is_active(timestep, parameters["guidance_interval"]):
                negative_prediction = model(latents, model_timestep, negative, concat_cond=concat_cond).sample
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
                    ((fine_coordinates[:, 1:] + 0.5) / fine_resolution * target).to(dtype=fine_coordinates.dtype),
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
        self,
        shape_slat: TrellisSparseTensor,
        texture_slat: TrellisSparseTensor | None = None,
        *,
        resolution: int,
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
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
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
        ovoxel_backend: OVoxelBackend | None = None,
        postprocess_kwargs: Mapping[str, Any] | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Generate 3D objects from one or more images.

        ``formats`` picks what is returned, in this order: ``"sparse_structure"`` (occupancy voxels),
        ``"shape_slat"`` and ``"texture_slat"`` (structured latents), ``"o_voxel"`` (decoded PBR O-Voxels), and
        ``"mesh"`` (O-Voxels meshed through ``ovoxel_backend``). It defaults to ``"o_voxel"`` when the texture
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
            Latent3DOutput(latents=dense_latents, metadata={"family": "trellis2", "stage": "sparse_structure"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(objects=tuple(objects), latents=latent_output, previews=None)


__all__ = ["Trellis2ImageTo3DPipeline"]
