# The objectives in this file reproduce released Microsoft TRELLIS.2 flow matching:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for typed diffusers-3d training recipes.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...data import ImageCondition, preprocess_training_image_condition, validate_image_condition_pixels
from ...objects import SparseVoxelAsset
from ...objects._validation import TensorShapeError, validate_shared_device, validate_tensor
from ...objects.base import TensorDataMixin
from ...training.exceptions import TrainingTargetError
from ...training.recipe import TrainingRecipe3D
from ...training.types import (
    ComponentPolicy,
    FineTuneKind,
    FrozenComponentPolicy,
    TrainingStep3DOutput,
)
from ..trellis.sparse import TrellisSparseTensor
from .conditioner import Trellis2Dinov3Conditioner
from .decoders import Trellis2SparseStructureDecoder
from .models import Trellis2SLatFlowModel, Trellis2SparseStructureFlowModel
from .pipeline import Trellis2ImageTo3DPipeline
from .scheduler import Trellis2FlowEulerScheduler


def _validate_trellis2_condition(condition: ImageCondition) -> None:
    if type(condition) is not ImageCondition:
        raise TrainingTargetError("condition must be an exact ImageCondition")
    try:
        validate_image_condition_pixels(condition)
    except (TypeError, ValueError) as error:
        raise TrainingTargetError("condition must contain finite image values in [0, 1]") from error


def _preprocess_trellis2_conditions(
    conditions: Sequence[ImageCondition],
    *,
    image_size: int,
) -> torch.Tensor:
    try:
        images = [
            preprocess_training_image_condition(
                condition,
                image_size=image_size,
                foreground_scale=1.0,
            ).image
            for condition in conditions
        ]
        return torch.stack(images)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TrainingTargetError(
            "TRELLIS.2 conditions could not be preprocessed into a matching image batch"
        ) from error


def _validate_preprocessed_images(images: torch.Tensor) -> None:
    if bool(((images < 0) | (images > 1)).any()):
        raise TrainingTargetError("images must contain finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class Trellis2SparseStructureExample(TensorDataMixin):
    condition: ImageCondition
    sparse_structure_latents: torch.Tensor
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_trellis2_condition(self.condition)
        validate_tensor("sparse_structure_latents", self.sparse_structure_latents, rank=4, floating=True)
        if min(self.sparse_structure_latents.shape) <= 0:
            raise TensorShapeError("sparse_structure_latents dimensions must be non-zero")
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise TrainingTargetError("example_id must be a non-empty string or None")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class Trellis2SparseStructureBatch(TensorDataMixin):
    images: torch.Tensor
    sparse_structure_latents: torch.Tensor
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        _validate_preprocessed_images(self.images)
        validate_tensor("sparse_structure_latents", self.sparse_structure_latents, rank=5, floating=True)
        batch_size = self.images.shape[0]
        if batch_size == 0 or self.images.shape[1] != 3 or self.sparse_structure_latents.shape[0] != batch_size:
            raise TensorShapeError("images and sparse_structure_latents must share a non-zero batch")
        if self.noise is not None:
            validate_tensor("noise", self.noise, rank=5, floating=True)
            if self.noise.shape != self.sparse_structure_latents.shape:
                raise TensorShapeError("noise must match sparse_structure_latents")
        _validate_training_controls(
            batch_size,
            self.timesteps,
            self.condition_dropout_mask,
        )
        validate_shared_device(self.tensor_items())


TRELLIS2_SPARSE_STRUCTURE_FLOW_POLICY = ComponentPolicy(
    key="sparse_structure_flow_model",
    component_path="sparse_structure_flow_model",
    expected_types=(Trellis2SparseStructureFlowModel,),
    supported_strategies=(FineTuneKind.FULL,),
)
TRELLIS2_SPARSE_STRUCTURE_FROZEN_COMPONENT_POLICIES = (
    FrozenComponentPolicy(component_path="conditioner", expected_types=(Trellis2Dinov3Conditioner,)),
    FrozenComponentPolicy(
        component_path="sparse_structure_decoder",
        expected_types=(Trellis2SparseStructureDecoder,),
    ),
)


def _validate_hyperparameters(sigma_min: float, p_uncond: float) -> tuple[float, float]:
    if not math.isfinite(sigma_min) or not 0 <= sigma_min < 1:
        raise ValueError("sigma_min must lie in [0, 1)")
    if not math.isfinite(p_uncond) or not 0 <= p_uncond <= 1:
        raise ValueError("p_uncond must lie in [0, 1]")
    return float(sigma_min), float(p_uncond)


def _validate_training_controls(
    batch_size: int,
    timesteps: torch.Tensor | None,
    dropout_mask: torch.Tensor | None,
) -> None:
    if timesteps is not None:
        validate_tensor("timesteps", timesteps, rank=1, floating=True)
        if timesteps.shape != (batch_size,) or bool(((timesteps < 0) | (timesteps > 1)).any()):
            raise TensorShapeError("timesteps must contain one [0, 1] value per batch item")
    if dropout_mask is not None and (
        not isinstance(dropout_mask, torch.Tensor)
        or dropout_mask.dtype is not torch.bool
        or dropout_mask.shape != (batch_size,)
    ):
        raise TensorShapeError("condition_dropout_mask must be a bool tensor with one value per batch item")


def _drop_conditioning(
    conditioning: torch.Tensor,
    dropout_mask: torch.Tensor | None,
    p_uncond: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (
        torch.rand(conditioning.shape[0], device=conditioning.device) < p_uncond
        if dropout_mask is None
        else dropout_mask
    )
    return torch.where(mask.reshape(-1, 1, 1), torch.zeros_like(conditioning), conditioning), mask


class Trellis2SparseStructureFlowRecipe(
    TrainingRecipe3D[
        Trellis2ImageTo3DPipeline,
        Trellis2SparseStructureExample,
        Trellis2SparseStructureBatch,
    ]
):
    """Released FULL-only dense objective with frozen DINOv3 and decoder."""

    recipe_id = "trellis2-sparse-structure-flow"
    recipe_version = "1.0"
    family_id = "trellis2"
    target_type = Trellis2ImageTo3DPipeline
    example_type = Trellis2SparseStructureExample
    batch_type = Trellis2SparseStructureBatch
    component_policies = (TRELLIS2_SPARSE_STRUCTURE_FLOW_POLICY,)
    frozen_component_policies = TRELLIS2_SPARSE_STRUCTURE_FROZEN_COMPONENT_POLICIES

    def __init__(
        self,
        target: Trellis2ImageTo3DPipeline,
        *,
        sigma_min: float = 1e-5,
        timestep_mean: float = 1.0,
        timestep_std: float = 1.0,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__(target)
        self.sigma_min, self.p_uncond = _validate_hyperparameters(sigma_min, p_uncond)
        if not math.isfinite(timestep_mean):
            raise ValueError("timestep_mean must be finite")
        if not math.isfinite(timestep_std) or timestep_std <= 0:
            raise ValueError("timestep_std must be positive")
        self.timestep_mean = float(timestep_mean)
        self.timestep_std = float(timestep_std)

    def objective_config(self) -> Mapping[str, bool | float | int | str | None]:
        return {
            "p_uncond": self.p_uncond,
            "sigma_min": self.sigma_min,
            "stage": "sparse_structure",
            "timestep_distribution": "logit_normal",
            "timestep_mean": self.timestep_mean,
            "timestep_std": self.timestep_std,
        }

    def collate(
        self,
        examples: Sequence[Trellis2SparseStructureExample],
    ) -> Trellis2SparseStructureBatch:
        if not examples or any(type(example) is not Trellis2SparseStructureExample for example in examples):
            raise TrainingTargetError("examples must contain exact Trellis2SparseStructureExample values")
        for example in examples:
            example.validate()
        images = _preprocess_trellis2_conditions(
            [example.condition for example in examples],
            image_size=self.target.conditioner.image_size,
        )
        try:
            latents = torch.stack([example.sparse_structure_latents for example in examples])
        except RuntimeError as error:
            raise TrainingTargetError("TRELLIS.2 examples must have matching latent shapes") from error
        return Trellis2SparseStructureBatch(images=images, sparse_structure_latents=latents)

    def validate_target(self) -> None:
        if type(self.target) is not Trellis2ImageTo3DPipeline:
            raise TrainingTargetError("target must be the exact TRELLIS.2 image-to-3D pipeline")
        if type(self.target.sparse_structure_flow_model) is not Trellis2SparseStructureFlowModel:
            raise TrainingTargetError("target sparse_structure_flow_model has the wrong exact type")
        if type(self.target.sparse_structure_decoder) is not Trellis2SparseStructureDecoder:
            raise TrainingTargetError("target sparse_structure_decoder has the wrong exact type")
        if type(self.target.conditioner) is not Trellis2Dinov3Conditioner:
            raise TrainingTargetError("target conditioner has the wrong exact type")
        if type(self.target.sparse_structure_scheduler) is not Trellis2FlowEulerScheduler:
            raise TrainingTargetError("target sparse_structure_scheduler has the wrong exact type")

    def compute_loss(self, batch: Trellis2SparseStructureBatch) -> TrainingStep3DOutput:
        if type(batch) is not Trellis2SparseStructureBatch:
            raise TrainingTargetError("batch must be an exact Trellis2SparseStructureBatch")
        batch.validate()
        clean = batch.sparse_structure_latents
        model = self.target.sparse_structure_flow_model
        model_config = self.component_config(model)
        expected = (clean.shape[0], model_config.in_channels, *([model_config.resolution] * 3))
        if tuple(clean.shape) != expected:
            raise TensorShapeError(f"sparse_structure_latents must have shape {expected}")
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images, value_range=(0.0, 1.0)).embeddings
        conditioning, dropout_mask = _drop_conditioning(
            conditioning,
            batch.condition_dropout_mask,
            self.p_uncond,
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


@dataclass(frozen=True, slots=True)
class Trellis2SLatExample(TensorDataMixin):
    condition: ImageCondition
    normalized_slat: SparseVoxelAsset
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_trellis2_condition(self.condition)
        if type(self.normalized_slat) is not SparseVoxelAsset:
            raise TrainingTargetError("normalized_slat must be an exact SparseVoxelAsset")
        self.normalized_slat.validate(expensive=True)
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise TrainingTargetError("example_id must be a non-empty string or None")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class Trellis2SLatBatch(TensorDataMixin):
    images: torch.Tensor
    normalized_slat: TrellisSparseTensor
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        _validate_preprocessed_images(self.images)
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
        _validate_training_controls(
            self.images.shape[0],
            self.timesteps,
            self.condition_dropout_mask,
        )
        validate_shared_device(self.tensor_items())


TRELLIS2_SHAPE_SLAT_FLOW_POLICY = ComponentPolicy(
    key="shape_slat_flow_model",
    component_path="shape_slat_flow_model",
    expected_types=(Trellis2SLatFlowModel,),
    supported_strategies=(FineTuneKind.FULL,),
)
TRELLIS2_SHAPE_SLAT_FROZEN_COMPONENT_POLICIES = (
    FrozenComponentPolicy(component_path="conditioner", expected_types=(Trellis2Dinov3Conditioner,)),
)


class Trellis2ShapeSLatFlowRecipe(TrainingRecipe3D[Trellis2ImageTo3DPipeline, Trellis2SLatExample, Trellis2SLatBatch]):
    """Unregistered experimental uniform-t shape-SLAT objective."""

    recipe_id = "trellis2-shape-slat-flow-experimental"
    recipe_version = "0.1"
    family_id = "trellis2"
    target_type = Trellis2ImageTo3DPipeline
    example_type = Trellis2SLatExample
    batch_type = Trellis2SLatBatch
    component_policies = (TRELLIS2_SHAPE_SLAT_FLOW_POLICY,)
    frozen_component_policies = TRELLIS2_SHAPE_SLAT_FROZEN_COMPONENT_POLICIES

    def __init__(
        self,
        target: Trellis2ImageTo3DPipeline,
        *,
        sigma_min: float = 1e-5,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__(target)
        self.sigma_min, self.p_uncond = _validate_hyperparameters(sigma_min, p_uncond)

    def objective_config(self) -> Mapping[str, bool | float | int | str | None]:
        return {
            "p_uncond": self.p_uncond,
            "sigma_min": self.sigma_min,
            "stage": "shape_slat",
            "timestep_distribution": "uniform",
        }

    def collate(self, examples: Sequence[Trellis2SLatExample]) -> Trellis2SLatBatch:
        if not examples or any(type(example) is not Trellis2SLatExample for example in examples):
            raise TrainingTargetError("examples must contain exact Trellis2SLatExample values")
        for example in examples:
            example.validate()
        images = _preprocess_trellis2_conditions(
            [example.condition for example in examples],
            image_size=self.target.conditioner.image_size,
        )
        sparse = TrellisSparseTensor.from_sparse_voxel_assets([example.normalized_slat for example in examples])
        return Trellis2SLatBatch(images=images, normalized_slat=sparse)

    def validate_target(self) -> None:
        if type(self.target) is not Trellis2ImageTo3DPipeline:
            raise TrainingTargetError("target must be the exact TRELLIS.2 image-to-3D pipeline")
        if type(self.target.shape_slat_flow_model) is not Trellis2SLatFlowModel:
            raise TrainingTargetError("target must contain the exact tiny Trellis2SLatFlowModel shape component")
        if self.target.shape_slat_flow_model.config.require_flex_gemm:
            raise TrainingTargetError("experimental training requires the backend-free tiny shape-SLAT core")

    def compute_loss(self, batch: Trellis2SLatBatch) -> TrainingStep3DOutput:
        if type(batch) is not Trellis2SLatBatch:
            raise TrainingTargetError("batch must be an exact Trellis2SLatBatch")
        batch.validate()
        sparse = batch.normalized_slat
        model = self.target.shape_slat_flow_model
        model_config = self.component_config(model)
        if sparse.channels != model_config.in_channels:
            raise TensorShapeError(f"normalized shape SLAT must have {model_config.in_channels} channels")
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images, value_range=(0.0, 1.0)).embeddings
        conditioning, dropout_mask = _drop_conditioning(
            conditioning,
            batch.condition_dropout_mask,
            self.p_uncond,
        )
        noise = torch.randn_like(sparse.features) if batch.noise is None else batch.noise
        timesteps = (
            torch.rand(sparse.batch_size, device=sparse.device, dtype=sparse.dtype)
            if batch.timesteps is None
            else batch.timesteps.to(dtype=sparse.dtype)
        )
        per_voxel = timesteps[sparse.coordinates[:, 0].to(dtype=torch.int64)].unsqueeze(1)
        noisy = (1 - per_voxel) * sparse.features + (self.sigma_min + (1 - self.sigma_min) * per_voxel) * noise
        target = (1 - self.sigma_min) * noise - sparse.features
        prediction = model(sparse.replace(noisy), timesteps * 1000, conditioning).sample.features
        loss = F.mse_loss(prediction, target)
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "flow_matching_mse": loss.detach(),
                "mean_timestep": timesteps.mean().detach(),
                "condition_dropout_fraction": dropout_mask.float().mean().detach(),
            },
        )


@dataclass(frozen=True, slots=True)
class Trellis2TextureSLatExample(TensorDataMixin):
    condition: ImageCondition
    normalized_texture_slat: SparseVoxelAsset
    normalized_shape_slat: SparseVoxelAsset
    example_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_trellis2_condition(self.condition)
        for name in ("normalized_texture_slat", "normalized_shape_slat"):
            value = getattr(self, name)
            if type(value) is not SparseVoxelAsset:
                raise TrainingTargetError(f"{name} must be an exact SparseVoxelAsset")
            value.validate(expensive=True)
        if not torch.equal(self.normalized_texture_slat.coordinates, self.normalized_shape_slat.coordinates):
            raise TrainingTargetError("texture and shape SLAT coordinates must align exactly")
        if self.example_id is not None and (not isinstance(self.example_id, str) or not self.example_id):
            raise TrainingTargetError("example_id must be a non-empty string or None")
        validate_shared_device(self.tensor_items())


@dataclass(frozen=True, slots=True)
class Trellis2TextureSLatBatch(TensorDataMixin):
    images: torch.Tensor
    normalized_texture_slat: TrellisSparseTensor
    normalized_shape_slat: TrellisSparseTensor
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    condition_dropout_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        _validate_preprocessed_images(self.images)
        if not isinstance(self.normalized_texture_slat, TrellisSparseTensor) or not isinstance(
            self.normalized_shape_slat, TrellisSparseTensor
        ):
            raise TrainingTargetError("texture and shape values must be TrellisSparseTensor instances")
        if not torch.equal(self.normalized_texture_slat.coordinates, self.normalized_shape_slat.coordinates):
            raise TrainingTargetError("texture and shape batch coordinates must align exactly")
        batch_size = self.normalized_texture_slat.batch_size
        if self.images.shape[0] != batch_size or self.images.shape[1] != 3:
            raise TensorShapeError("images and sparse values must share a non-zero batch")
        if self.noise is not None:
            validate_tensor("noise", self.noise, rank=2, floating=True)
            if self.noise.shape != self.normalized_texture_slat.features.shape:
                raise TensorShapeError("noise must match normalized texture features")
        _validate_training_controls(batch_size, self.timesteps, self.condition_dropout_mask)
        validate_shared_device(self.tensor_items())


TRELLIS2_TEXTURE_SLAT_FLOW_POLICY = ComponentPolicy(
    key="texture_slat_flow_model",
    component_path="texture_slat_flow_model",
    expected_types=(Trellis2SLatFlowModel,),
    supported_strategies=(FineTuneKind.FULL,),
)
TRELLIS2_TEXTURE_SLAT_FROZEN_COMPONENT_POLICIES = (
    FrozenComponentPolicy(component_path="conditioner", expected_types=(Trellis2Dinov3Conditioner,)),
)


class Trellis2TextureSLatFlowRecipe(
    TrainingRecipe3D[
        Trellis2ImageTo3DPipeline,
        Trellis2TextureSLatExample,
        Trellis2TextureSLatBatch,
    ]
):
    """Unregistered experimental coordinate-aligned uniform-t texture objective."""

    recipe_id = "trellis2-texture-slat-flow-experimental"
    recipe_version = "0.1"
    family_id = "trellis2"
    target_type = Trellis2ImageTo3DPipeline
    example_type = Trellis2TextureSLatExample
    batch_type = Trellis2TextureSLatBatch
    component_policies = (TRELLIS2_TEXTURE_SLAT_FLOW_POLICY,)
    frozen_component_policies = TRELLIS2_TEXTURE_SLAT_FROZEN_COMPONENT_POLICIES

    def __init__(
        self,
        target: Trellis2ImageTo3DPipeline,
        *,
        sigma_min: float = 1e-5,
        p_uncond: float = 0.1,
    ) -> None:
        super().__init__(target)
        self.sigma_min, self.p_uncond = _validate_hyperparameters(sigma_min, p_uncond)

    def objective_config(self) -> Mapping[str, bool | float | int | str | None]:
        return {
            "p_uncond": self.p_uncond,
            "sigma_min": self.sigma_min,
            "stage": "texture_slat",
            "timestep_distribution": "uniform",
        }

    def collate(self, examples: Sequence[Trellis2TextureSLatExample]) -> Trellis2TextureSLatBatch:
        if not examples or any(type(example) is not Trellis2TextureSLatExample for example in examples):
            raise TrainingTargetError("examples must contain exact Trellis2TextureSLatExample values")
        for example in examples:
            example.validate()
        images = _preprocess_trellis2_conditions(
            [example.condition for example in examples],
            image_size=self.target.conditioner.image_size,
        )
        texture = TrellisSparseTensor.from_sparse_voxel_assets(
            [example.normalized_texture_slat for example in examples]
        )
        shape = TrellisSparseTensor.from_sparse_voxel_assets([example.normalized_shape_slat for example in examples])
        return Trellis2TextureSLatBatch(
            images=images,
            normalized_texture_slat=texture,
            normalized_shape_slat=shape,
        )

    def validate_target(self) -> None:
        if type(self.target) is not Trellis2ImageTo3DPipeline:
            raise TrainingTargetError("target must be the exact TRELLIS.2 image-to-3D pipeline")
        if type(self.target.texture_slat_flow_model) is not Trellis2SLatFlowModel:
            raise TrainingTargetError("target must contain the exact tiny Trellis2SLatFlowModel texture component")
        if self.target.texture_slat_flow_model.config.require_flex_gemm:
            raise TrainingTargetError("experimental training requires the backend-free tiny texture-SLAT core")

    def compute_loss(self, batch: Trellis2TextureSLatBatch) -> TrainingStep3DOutput:
        if type(batch) is not Trellis2TextureSLatBatch:
            raise TrainingTargetError("batch must be an exact Trellis2TextureSLatBatch")
        batch.validate()
        texture = batch.normalized_texture_slat
        shape = batch.normalized_shape_slat
        model = self.target.texture_slat_flow_model
        model_config = self.component_config(model)
        if (
            texture.channels != model_config.out_channels
            or texture.channels + shape.channels != model_config.in_channels
        ):
            raise TensorShapeError("texture and shape channels do not match the texture flow concat contract")
        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images, value_range=(0.0, 1.0)).embeddings
        conditioning, dropout_mask = _drop_conditioning(
            conditioning,
            batch.condition_dropout_mask,
            self.p_uncond,
        )
        noise = torch.randn_like(texture.features) if batch.noise is None else batch.noise
        timesteps = (
            torch.rand(texture.batch_size, device=texture.device, dtype=texture.dtype)
            if batch.timesteps is None
            else batch.timesteps.to(dtype=texture.dtype)
        )
        per_voxel = timesteps[texture.coordinates[:, 0].to(dtype=torch.int64)].unsqueeze(1)
        noisy = (1 - per_voxel) * texture.features + (self.sigma_min + (1 - self.sigma_min) * per_voxel) * noise
        target = (1 - self.sigma_min) * noise - texture.features
        prediction = model(
            texture.replace(noisy),
            timesteps * 1000,
            conditioning,
            concat_cond=shape,
        ).sample.features
        loss = F.mse_loss(prediction, target)
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "flow_matching_mse": loss.detach(),
                "mean_timestep": timesteps.mean().detach(),
                "condition_dropout_fraction": dropout_mask.float().mean().detach(),
            },
        )


__all__ = [
    "TRELLIS2_SHAPE_SLAT_FLOW_POLICY",
    "TRELLIS2_SHAPE_SLAT_FROZEN_COMPONENT_POLICIES",
    "TRELLIS2_SPARSE_STRUCTURE_FLOW_POLICY",
    "TRELLIS2_SPARSE_STRUCTURE_FROZEN_COMPONENT_POLICIES",
    "TRELLIS2_TEXTURE_SLAT_FLOW_POLICY",
    "TRELLIS2_TEXTURE_SLAT_FROZEN_COMPONENT_POLICIES",
    "Trellis2SLatBatch",
    "Trellis2SLatExample",
    "Trellis2ShapeSLatFlowRecipe",
    "Trellis2SparseStructureBatch",
    "Trellis2SparseStructureExample",
    "Trellis2SparseStructureFlowRecipe",
    "Trellis2TextureSLatBatch",
    "Trellis2TextureSLatExample",
    "Trellis2TextureSLatFlowRecipe",
]
