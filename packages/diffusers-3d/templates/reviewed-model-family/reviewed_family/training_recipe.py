from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffusers_3d import (
    ComponentPolicy,
    FineTuneKind,
    ImageCondition,
    MeshAsset,
    Object3DExample,
    TrainingRecipe3D,
    TrainingStep3DOutput,
)

from .model import ReviewedDenoiser
from .pipeline import ReviewedObject3DPipeline


@dataclass(frozen=True, slots=True)
class ReviewedBatch:
    """Exact typed batch owned by the reviewed recipe."""

    images: torch.Tensor
    vertices: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.images.ndim != 4 or self.vertices.ndim != 3 or self.vertices.shape[-1] != 3:
            raise ValueError("images and vertices must be batched channel-first images and XYZ vertices")
        if self.images.shape[0] == 0 or self.images.shape[0] != self.vertices.shape[0]:
            raise ValueError("images and vertices must have the same non-zero batch size")

    def to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ReviewedBatch:
        return type(self)(
            images=self.images.to(device=device, dtype=dtype, non_blocking=non_blocking),
            vertices=self.vertices.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )


REVIEWED_DENOISER_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="denoiser",
    expected_types=(ReviewedDenoiser,),
    supported_strategies=(FineTuneKind.FULL,),
    full_parameter_names=("projection.bias", "projection.weight"),
)


class ReviewedTrainingRecipe(TrainingRecipe3D[ReviewedObject3DPipeline, Object3DExample, ReviewedBatch]):
    """Exact objective starter; replace its math only with parity-tested reference math."""

    recipe_id = "reviewed-family"
    recipe_version = "1.0"
    family_id = "reviewed-family"
    target_type = ReviewedObject3DPipeline
    example_type = Object3DExample
    batch_type = ReviewedBatch
    component_policies = (REVIEWED_DENOISER_POLICY,)

    def collate(self, examples: Sequence[Object3DExample]) -> ReviewedBatch:
        images = []
        vertices = []
        for example in examples:
            if type(example) is not Object3DExample:
                raise TypeError("examples must contain exact Object3DExample values")
            if type(example.condition) is not ImageCondition or type(example.target) is not MeshAsset:
                raise TypeError("the reviewed family requires ImageCondition and MeshAsset values")
            example.validate(expensive=True)
            images.append(example.condition.image)
            vertices.append(example.target.vertices)
        return ReviewedBatch(images=torch.stack(images), vertices=torch.stack(vertices))

    def validate_target(self) -> None:
        if type(self.target) is not ReviewedObject3DPipeline:
            raise TypeError("target must be the exact reviewed pipeline")
        if type(self.target.denoiser) is not ReviewedDenoiser:
            raise TypeError("target denoiser must be the exact reviewed class")

    def compute_loss(self, batch: ReviewedBatch) -> TrainingStep3DOutput:
        batch.validate()
        values = batch.images.mean(dim=(1, 2, 3), keepdim=False)
        hidden_states = values.unsqueeze(-1).expand(-1, self.target.denoiser.config.latent_dim)
        predictions = self.target.denoiser(
            hidden_states,
            torch.zeros(hidden_states.shape[0], device=hidden_states.device),
        )
        loss = F.mse_loss(predictions[:, :9], batch.vertices.reshape(batch.vertices.shape[0], -1))
        return TrainingStep3DOutput(loss=loss, metrics={"mesh_mse": loss})
