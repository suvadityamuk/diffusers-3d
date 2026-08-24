# The objectives in this file reproduce Microsoft TRELLIS flow matching:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for typed diffusers-3d training recipes.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ...data import ImageCondition
from ...objects import SparseVoxelAsset
from ...objects._validation import TensorShapeError, validate_shared_device, validate_tensor
from ...objects.base import TensorDataMixin
from ...training.exceptions import TrainingCheckpointError, TrainingTargetError
from ...training.recipe import TrainingRecipe3D
from ...training.types import ComponentPolicy, FineTuneKind, FineTuneStrategy3D, FullFineTune, TrainingStep3DOutput
from .conditioner import TrellisDinov2Conditioner
from .decoders import TrellisSparseStructureDecoder
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .pipeline import TrellisImageTo3DPipeline
from .scheduler import TrellisFlowEulerScheduler
from .sparse import TrellisSparseTensor


@dataclass(frozen=True, slots=True)
class TrellisSparseStructureExample(TensorDataMixin):
    condition: ImageCondition
    sparse_structure_latents: torch.Tensor
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.condition) is not ImageCondition or self.condition.image.shape[0] != 3:
            raise TrainingTargetError("condition must be an exact three-channel ImageCondition")
        validate_tensor("sparse_structure_latents", self.sparse_structure_latents, rank=4, floating=True)
        if min(self.sparse_structure_latents.shape) <= 0:
            raise TensorShapeError("sparse_structure_latents dimensions must be non-zero")
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise TrainingTargetError("example_id must be a non-empty string or None")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class TrellisSparseStructureBatch(TensorDataMixin):
    images: torch.Tensor
    sparse_structure_latents: torch.Tensor
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        validate_tensor("sparse_structure_latents", self.sparse_structure_latents, rank=5, floating=True)
        batch_size = self.images.shape[0]
        if batch_size == 0 or self.images.shape[1] != 3 or self.sparse_structure_latents.shape[0] != batch_size:
            raise TensorShapeError("images and sparse_structure_latents must share a non-zero batch")
        if self.noise is not None:
            validate_tensor("noise", self.noise, rank=5, floating=True)
            if self.noise.shape != self.sparse_structure_latents.shape:
                raise TensorShapeError("noise must match sparse_structure_latents")
        if self.timesteps is not None:
            validate_tensor("timesteps", self.timesteps, rank=1, floating=True)
            if self.timesteps.shape != (batch_size,) or bool(((self.timesteps < 0) | (self.timesteps > 1)).any()):
                raise TensorShapeError("timesteps must contain one [0, 1] value per batch item")
        if self.condition_dropout_mask is not None:
            if (
                not isinstance(self.condition_dropout_mask, torch.Tensor)
                or self.condition_dropout_mask.dtype is not torch.bool
                or self.condition_dropout_mask.shape != (batch_size,)
            ):
                raise TensorShapeError("condition_dropout_mask must be a bool tensor with one value per batch item")
        validate_shared_device(self.tensor_items())


TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY = ComponentPolicy(
    key="sparse_structure_flow_model",
    component_path="sparse_structure_flow_model",
    expected_types=(TrellisSparseStructureFlowModel,),
    supported_strategies=(FineTuneKind.FULL,),
)


class TrellisSparseStructureFlowRecipe(
    TrainingRecipe3D[
        TrellisImageTo3DPipeline,
        TrellisSparseStructureExample,
        TrellisSparseStructureBatch,
    ]
):
    """Released dense sparse-structure objective with frozen image conditioner and decoders."""

    recipe_id = "trellis-sparse-structure-flow"
    recipe_version = "1.0"
    family_id = "trellis"
    target_type = TrellisImageTo3DPipeline
    example_type = TrellisSparseStructureExample
    batch_type = TrellisSparseStructureBatch
    component_policies = (TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY,)

    def __init__(
        self,
        target: TrellisImageTo3DPipeline,
        *,
        sigma_min: float = 1e-5,
        timestep_mean: float = 1.0,
        timestep_std: float = 1.0,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__(target)
        if not math.isfinite(sigma_min) or not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must lie in [0, 1)")
        if not math.isfinite(timestep_mean):
            raise ValueError("timestep_mean must be finite")
        if not math.isfinite(timestep_std) or timestep_std <= 0:
            raise ValueError("timestep_std must be positive")
        if not math.isfinite(p_uncond) or not 0 <= p_uncond <= 1:
            raise ValueError("p_uncond must lie in [0, 1]")
        self.sigma_min = float(sigma_min)
        self.timestep_mean = float(timestep_mean)
        self.timestep_std = float(timestep_std)
        self.p_uncond = float(p_uncond)

    def collate(
        self,
        examples: Sequence[TrellisSparseStructureExample],
    ) -> TrellisSparseStructureBatch:
        if not examples or any(type(example) is not TrellisSparseStructureExample for example in examples):
            raise TrainingTargetError("examples must contain exact TrellisSparseStructureExample values")
        for example in examples:
            example.validate()
        try:
            images = torch.stack([example.condition.image for example in examples])
            latents = torch.stack([example.sparse_structure_latents for example in examples])
        except RuntimeError as error:
            raise TrainingTargetError("TRELLIS examples must have matching image and latent shapes") from error
        return TrellisSparseStructureBatch(images=images, sparse_structure_latents=latents)

    def validate_target(self) -> None:
        if type(self.target) is not TrellisImageTo3DPipeline:
            raise TrainingTargetError("target must be the exact TRELLIS image-to-3D pipeline")
        if type(self.target.sparse_structure_flow_model) is not TrellisSparseStructureFlowModel:
            raise TrainingTargetError("target sparse_structure_flow_model has the wrong exact type")
        if type(self.target.sparse_structure_decoder) is not TrellisSparseStructureDecoder:
            raise TrainingTargetError("target sparse_structure_decoder has the wrong exact type")
        if type(self.target.conditioner) is not TrellisDinov2Conditioner:
            raise TrainingTargetError("target conditioner has the wrong exact type")
        if type(self.target.sparse_structure_scheduler) is not TrellisFlowEulerScheduler:
            raise TrainingTargetError("target sparse_structure_scheduler has the wrong exact type")

    def compute_loss(self, batch: TrellisSparseStructureBatch) -> TrainingStep3DOutput:
        if type(batch) is not TrellisSparseStructureBatch:
            raise TrainingTargetError("batch must be an exact TrellisSparseStructureBatch")
        batch.validate()
        self.validate_target()
        clean = batch.sparse_structure_latents
        model = self.target.sparse_structure_flow_model
        expected_shape = (
            clean.shape[0],
            model.config.in_channels,
            model.config.resolution,
            model.config.resolution,
            model.config.resolution,
        )
        if tuple(clean.shape) != expected_shape:
            raise TensorShapeError(f"sparse_structure_latents must have shape {expected_shape}")
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images, value_range=(0.0, 1.0)).embeddings
        dropout_mask = (
            torch.rand(clean.shape[0], device=clean.device) < self.p_uncond
            if batch.condition_dropout_mask is None
            else batch.condition_dropout_mask
        )
        conditioning = torch.where(
            dropout_mask.reshape(-1, 1, 1),
            torch.zeros_like(conditioning),
            conditioning,
        )
        noise = torch.randn_like(clean) if batch.noise is None else batch.noise
        timesteps = (
            torch.sigmoid(
                torch.randn(clean.shape[0], device=clean.device) * self.timestep_std + self.timestep_mean
            ).to(dtype=clean.dtype)
            if batch.timesteps is None
            else batch.timesteps.to(dtype=clean.dtype)
        )
        interpolation = timesteps.reshape(-1, 1, 1, 1, 1)
        noisy = (1 - interpolation) * clean + (self.sigma_min + (1 - self.sigma_min) * interpolation) * noise
        target = (1 - self.sigma_min) * noise - clean
        prediction = model(noisy, timesteps * 1000, conditioning).sample
        loss = F.mse_loss(prediction, target)
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "flow_matching_mse": loss.detach(),
                "mean_timestep": timesteps.mean().detach(),
                "condition_dropout_fraction": dropout_mask.float().mean().detach(),
            },
        )

    def load_weights(
        self,
        save_directory: str | Path,
        strategy: FineTuneStrategy3D,
        components: Mapping[str, nn.Module],
    ) -> None:
        if type(strategy) is not FullFineTune or strategy.components != ("sparse_structure_flow_model",):
            raise TrainingCheckpointError("TRELLIS resume supports only full sparse-structure flow fine-tuning")
        model = components.get("sparse_structure_flow_model")
        if type(model) is not TrellisSparseStructureFlowModel:
            raise TrainingCheckpointError("TRELLIS checkpoint flow model has the wrong exact type")
        loaded = TrellisSparseStructureFlowModel.from_pretrained(
            Path(save_directory) / "sparse_structure_flow_model",
            local_files_only=True,
        )
        model.load_state_dict(loaded.state_dict(), strict=True)


@dataclass(frozen=True, slots=True)
class TrellisSLatExample(TensorDataMixin):
    condition: ImageCondition
    normalized_slat: SparseVoxelAsset
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.condition) is not ImageCondition or self.condition.image.shape[0] != 3:
            raise TrainingTargetError("condition must be an exact three-channel ImageCondition")
        if type(self.normalized_slat) is not SparseVoxelAsset:
            raise TrainingTargetError("normalized_slat must be an exact SparseVoxelAsset")
        self.normalized_slat.validate(expensive=True)
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise TrainingTargetError("example_id must be a non-empty string or None")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class TrellisSLatBatch(TensorDataMixin):
    images: torch.Tensor
    normalized_slat: TrellisSparseTensor
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        if not isinstance(self.normalized_slat, TrellisSparseTensor):
            raise TrainingTargetError("normalized_slat must be a TrellisSparseTensor")
        if self.images.shape[0] == 0 or self.images.shape[1] != 3:
            raise TensorShapeError("images must contain a non-zero batch of three-channel images")
        if self.images.shape[0] != self.normalized_slat.batch_size:
            raise TensorShapeError("images and normalized_slat must share a batch size")
        if self.noise is not None:
            validate_tensor("noise", self.noise, rank=2, floating=True)
            if self.noise.shape != self.normalized_slat.features.shape:
                raise TensorShapeError("noise must match normalized SLAT features")
        if self.timesteps is not None:
            validate_tensor("timesteps", self.timesteps, rank=1, floating=True)
            if self.timesteps.shape != (self.images.shape[0],) or bool(
                ((self.timesteps < 0) | (self.timesteps > 1)).any()
            ):
                raise TensorShapeError("timesteps must contain one [0, 1] value per batch item")
        if self.condition_dropout_mask is not None:
            if (
                not isinstance(self.condition_dropout_mask, torch.Tensor)
                or self.condition_dropout_mask.dtype is not torch.bool
                or self.condition_dropout_mask.shape != (self.images.shape[0],)
            ):
                raise TensorShapeError("condition_dropout_mask must be a bool tensor with one value per batch item")
        validate_shared_device(self.tensor_items())


TRELLIS_SLAT_FLOW_POLICY = ComponentPolicy(
    key="slat_flow_model",
    component_path="slat_flow_model",
    expected_types=(TrellisSLatFlowModel,),
    supported_strategies=(FineTuneKind.FULL,),
)


class TrellisSLatFlowRecipe(TrainingRecipe3D[TrellisImageTo3DPipeline, TrellisSLatExample, TrellisSLatBatch]):
    """Experimental FULL-only objective for the portable no-sparse-convolution SLAT core."""

    recipe_id = "trellis-slat-flow-experimental"
    recipe_version = "0.1"
    family_id = "trellis"
    target_type = TrellisImageTo3DPipeline
    example_type = TrellisSLatExample
    batch_type = TrellisSLatBatch
    component_policies = (TRELLIS_SLAT_FLOW_POLICY,)

    def __init__(
        self,
        target: TrellisImageTo3DPipeline,
        *,
        sigma_min: float = 1e-5,
        timestep_mean: float = 1.0,
        timestep_std: float = 1.0,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__(target)
        if not math.isfinite(sigma_min) or not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must lie in [0, 1)")
        if not math.isfinite(timestep_mean):
            raise ValueError("timestep_mean must be finite")
        if not math.isfinite(timestep_std) or timestep_std <= 0:
            raise ValueError("timestep_std must be positive")
        if not math.isfinite(p_uncond) or not 0 <= p_uncond <= 1:
            raise ValueError("p_uncond must lie in [0, 1]")
        self.sigma_min = float(sigma_min)
        self.timestep_mean = float(timestep_mean)
        self.timestep_std = float(timestep_std)
        self.p_uncond = float(p_uncond)

    def collate(self, examples: Sequence[TrellisSLatExample]) -> TrellisSLatBatch:
        if not examples or any(type(example) is not TrellisSLatExample for example in examples):
            raise TrainingTargetError("examples must contain exact TrellisSLatExample values")
        for example in examples:
            example.validate()
        try:
            images = torch.stack([example.condition.image for example in examples])
        except RuntimeError as error:
            raise TrainingTargetError("TRELLIS SLAT images must have matching shapes") from error
        sparse = TrellisSparseTensor.from_sparse_voxel_assets([example.normalized_slat for example in examples])
        return TrellisSLatBatch(images=images, normalized_slat=sparse)

    def validate_target(self) -> None:
        if type(self.target) is not TrellisImageTo3DPipeline:
            raise TrainingTargetError("target must be the exact TRELLIS image-to-3D pipeline")
        if type(self.target.slat_flow_model) is not TrellisSLatFlowModel:
            raise TrainingTargetError("target must contain the portable exact TrellisSLatFlowModel type")
        if self.target.slat_flow_model.config.io_block_channels is not None:
            raise TrainingTargetError("experimental SLAT training requires io_block_channels=None")

    def compute_loss(self, batch: TrellisSLatBatch) -> TrainingStep3DOutput:
        if type(batch) is not TrellisSLatBatch:
            raise TrainingTargetError("batch must be an exact TrellisSLatBatch")
        batch.validate()
        self.validate_target()
        sparse = batch.normalized_slat
        model = self.target.slat_flow_model
        if sparse.channels != model.config.in_channels:
            raise TensorShapeError(f"normalized SLAT must have {model.config.in_channels} channels")
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images, value_range=(0.0, 1.0)).embeddings
        dropout_mask = (
            torch.rand(sparse.batch_size, device=sparse.device) < self.p_uncond
            if batch.condition_dropout_mask is None
            else batch.condition_dropout_mask
        )
        conditioning = torch.where(
            dropout_mask.reshape(-1, 1, 1),
            torch.zeros_like(conditioning),
            conditioning,
        )
        noise = torch.randn_like(sparse.features) if batch.noise is None else batch.noise
        timesteps = (
            torch.sigmoid(
                torch.randn(sparse.batch_size, device=sparse.device) * self.timestep_std + self.timestep_mean
            ).to(dtype=sparse.dtype)
            if batch.timesteps is None
            else batch.timesteps.to(dtype=sparse.dtype)
        )
        per_voxel_timestep = timesteps[sparse.coordinates[:, 0].to(dtype=torch.int64)].unsqueeze(1)
        noisy_features = (1 - per_voxel_timestep) * sparse.features + (
            self.sigma_min + (1 - self.sigma_min) * per_voxel_timestep
        ) * noise
        target_features = (1 - self.sigma_min) * noise - sparse.features
        prediction = model(
            sparse.replace(noisy_features),
            timesteps * 1000,
            conditioning,
        ).sample
        loss = F.mse_loss(prediction.features, target_features)
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "flow_matching_mse": loss.detach(),
                "mean_timestep": timesteps.mean().detach(),
            },
        )


__all__ = [
    "TRELLIS_SLAT_FLOW_POLICY",
    "TRELLIS_SPARSE_STRUCTURE_FLOW_POLICY",
    "TrellisSLatBatch",
    "TrellisSLatExample",
    "TrellisSLatFlowRecipe",
    "TrellisSparseStructureBatch",
    "TrellisSparseStructureExample",
    "TrellisSparseStructureFlowRecipe",
]
