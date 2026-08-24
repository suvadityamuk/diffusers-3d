# The objective in this file follows the released Tencent Hunyuan3D-2.1
# flow-matching training code at revision
# 82920d643c0dc2f7bfd7255f45f62d386edfe60c.
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for typed diffusers-3d training integration.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...data import ImageCondition, Object3DExample
from ...objects import MeshAsset
from ...objects._validation import TensorShapeError, validate_shared_device, validate_tensor
from ...objects.base import TensorDataMixin
from ...training.exceptions import TrainingTargetError
from ...training.recipe import TrainingRecipe3D
from ...training.types import ComponentPolicy, FineTuneKind, TrainingStep3DOutput
from .models import Hunyuan3DShapeDiTModel
from .pipeline import Hunyuan3DImageToShapePipeline


@dataclass(frozen=True, slots=True)
class Hunyuan3DShapeBatch(TensorDataMixin):
    """Images plus exactly one shape source.

    ``shape_latents`` are the released-recipe ``x1`` values after any VAE
    scaling. ``surface_samples`` represent the upstream surface path, which is
    validated but rejected by the current decode-only VAE integration.
    ``noise`` and ``timesteps`` are optional deterministic-test inputs.
    """

    images: torch.Tensor
    shape_latents: torch.Tensor | None = None
    surface_samples: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        if self.images.shape[0] == 0 or self.images.shape[1] != 3:
            raise TensorShapeError("images must have shape (nonzero batch, 3, height, width)")
        if (self.shape_latents is None) == (self.surface_samples is None):
            raise TensorShapeError("exactly one of shape_latents or surface_samples must be provided")

        batch_size = self.images.shape[0]
        if self.shape_latents is not None:
            validate_tensor("shape_latents", self.shape_latents, rank=3, floating=True)
            if self.shape_latents.shape[0] != batch_size:
                raise TensorShapeError("shape_latents must match the image batch")
        if self.surface_samples is not None:
            validate_tensor("surface_samples", self.surface_samples, rank=3, floating=True)
            if self.surface_samples.shape[0] != batch_size or self.surface_samples.shape[-1] < 3:
                raise TensorShapeError("surface_samples must have shape (batch, points, at least 3)")
        if self.noise is not None:
            validate_tensor("noise", self.noise, rank=3, floating=True)
            if self.shape_latents is None or self.noise.shape != self.shape_latents.shape:
                raise TensorShapeError("noise requires shape_latents and must have the same shape")
        if self.timesteps is not None:
            validate_tensor("timesteps", self.timesteps, rank=1, floating=True)
            if self.timesteps.shape[0] != batch_size:
                raise TensorShapeError("timesteps must contain one value per batch item")
            if bool(((self.timesteps < 0) | (self.timesteps > 1)).any()):
                raise ValueError("timesteps must lie in [0, 1]")
        validate_shared_device(self.tensor_items())


HUNYUAN3D_DENOISER_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="denoiser",
    expected_types=(Hunyuan3DShapeDiTModel,),
    supported_strategies=(FineTuneKind.FULL,),
)


class Hunyuan3DShapeFlowMatchingRecipe(TrainingRecipe3D[Hunyuan3DImageToShapePipeline, Hunyuan3DShapeBatch]):
    """Released Hunyuan shape flow objective with frozen DINO and VAE."""

    recipe_id = "hunyuan3d-shape-flow-matching"
    recipe_version = "1.0"
    family_id = "hunyuan3d-2.1"
    target_type = Hunyuan3DImageToShapePipeline
    batch_type = Hunyuan3DShapeBatch
    component_policies = (HUNYUAN3D_DENOISER_POLICY,)

    def __init__(self, target: Hunyuan3DImageToShapePipeline) -> None:
        super().__init__(target)
        target.vae.requires_grad_(False)
        target.vae.eval()
        target.conditioner.requires_grad_(False)
        target.conditioner.eval()

    def collate(self, examples: Sequence[Object3DExample]) -> Hunyuan3DShapeBatch:
        images = []
        surface_samples = []
        for example in examples:
            if type(example) is not Object3DExample:
                raise TrainingTargetError("examples must contain exact Object3DExample values")
            if type(example.condition) is not ImageCondition or type(example.target) is not MeshAsset:
                raise TrainingTargetError("Hunyuan shape examples require ImageCondition and MeshAsset values")
            example.validate(expensive=True)
            images.append(example.condition.image[:3])
            mesh = example.target
            surface = mesh.vertices
            if mesh.normals is not None:
                surface = torch.cat([surface, mesh.normals], dim=-1)
            surface_samples.append(surface)
        if not images:
            raise TrainingTargetError("examples must not be empty")
        try:
            batched_surfaces = torch.stack(surface_samples)
        except RuntimeError as error:
            raise TrainingTargetError("surface samples must have the same point count") from error
        return Hunyuan3DShapeBatch(
            images=torch.stack(images),
            surface_samples=batched_surfaces,
        )

    def validate_target(self) -> None:
        if type(self.target) is not Hunyuan3DImageToShapePipeline:
            raise TrainingTargetError("target must be the exact Hunyuan3D image-to-shape pipeline")
        if type(self.target.denoiser) is not Hunyuan3DShapeDiTModel:
            raise TrainingTargetError("target denoiser must be Hunyuan3DShapeDiTModel")
        if any(parameter.requires_grad for parameter in self.target.vae.parameters()):
            raise TrainingTargetError("the Hunyuan shape VAE must remain frozen")
        if any(parameter.requires_grad for parameter in self.target.conditioner.parameters()):
            raise TrainingTargetError("the Hunyuan DINO conditioner must remain frozen")

    def compute_loss(self, batch: Hunyuan3DShapeBatch) -> TrainingStep3DOutput:
        if type(batch) is not Hunyuan3DShapeBatch:
            raise TrainingTargetError("batch must be an exact Hunyuan3DShapeBatch")
        batch.validate()
        self.validate_target()

        if batch.shape_latents is None:
            raise NotImplementedError(
                "surface-sample training is unavailable because Hunyuan3DShapeVAE.encode is intentionally unsupported; "
                "provide precomputed released-recipe shape_latents"
            )
        clean_latents = batch.shape_latents
        expected_shape = (
            clean_latents.shape[0],
            self.target.denoiser.config.input_size,
            self.target.denoiser.config.in_channels,
        )
        if clean_latents.shape != expected_shape:
            raise TensorShapeError(f"shape_latents must have shape {expected_shape}")

        with torch.no_grad():
            conditioning = self.target.conditioner(batch.images).embeddings
        noise = torch.randn_like(clean_latents) if batch.noise is None else batch.noise
        timesteps = (
            torch.rand(
                clean_latents.shape[0],
                device=clean_latents.device,
                dtype=clean_latents.dtype,
            )
            if batch.timesteps is None
            else batch.timesteps.to(dtype=clean_latents.dtype)
        )
        interpolation = timesteps.reshape(-1, 1, 1)
        noisy_latents = interpolation * clean_latents + (1.0 - interpolation) * noise
        velocity_target = clean_latents - noise
        velocity_prediction = self.target.denoiser(
            noisy_latents,
            timesteps,
            conditioning,
        ).sample
        loss = F.mse_loss(velocity_prediction, velocity_target)
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "flow_matching_mse": loss.detach(),
                "mean_timestep": timesteps.mean().detach(),
            },
        )


__all__ = [
    "HUNYUAN3D_DENOISER_POLICY",
    "Hunyuan3DShapeBatch",
    "Hunyuan3DShapeFlowMatchingRecipe",
]
