# OBJECT3D_CONTRACT_VALIDATION_ONLY
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..data import ImageCondition, Object3DExample
from ..objects import MeshAsset
from ..objects._validation import TensorShapeError, validate_shared_device, validate_tensor
from ..objects.base import TensorDataMixin
from ..training.exceptions import TrainingCheckpointError, TrainingTargetError
from ..training.recipe import TRAINING_ADAPTER_NAME, TrainingRecipe3D
from ..training.types import (
    ComponentPolicy,
    FineTuneKind,
    FineTuneStrategy3D,
    FullFineTune,
    LoRAFineTune,
    TrainingStep3DOutput,
)
from .models import ContractReferenceDenoiser, ContractReferenceMeshDecoder
from .pipeline import ContractReferencePipeline


@dataclass(frozen=True, slots=True)
class ContractReferenceBatch(TensorDataMixin):
    """Exact image/mesh batch for the private validation objective."""

    images: torch.Tensor
    vertices: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor("images", self.images, rank=4, floating=True)
        validate_tensor("vertices", self.vertices, rank=3, trailing_shape=(3, 3), floating=True)
        if self.images.shape[0] == 0 or self.images.shape[0] != self.vertices.shape[0]:
            raise TensorShapeError("images and vertices must have the same non-zero batch size")
        if self.images.shape[1] != 3:
            raise TensorShapeError("images must contain exactly three channels")
        validate_shared_device(self.tensor_items())


class ContractReferenceDataset:
    """Small deterministic map-style dataset of exact object-3D examples."""

    def __init__(self, length: int = 2, image_size: int = 4) -> None:
        if length <= 0 or image_size <= 0:
            raise ValueError("length and image_size must be positive")
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Object3DExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        offset = float(index) / max(self.length, 1)
        image = torch.stack(
            [
                torch.full((self.image_size, self.image_size), offset),
                torch.full((self.image_size, self.image_size), 0.25 + offset),
                torch.full((self.image_size, self.image_size), 0.5 + offset),
            ]
        )
        vertices = torch.tensor(
            [
                [-0.5 + offset, -0.5, 0.0],
                [0.5 + offset, -0.5, 0.0],
                [offset, 0.5, 0.0],
            ]
        )
        mesh = MeshAsset(vertices=vertices, faces=torch.tensor([[0, 1, 2]], dtype=torch.int64))
        return Object3DExample(
            target=mesh,
            condition=ImageCondition(image=image),
            example_id=f"contract-reference-{index}",
        )


CONTRACT_REFERENCE_DENOISER_POLICY = ComponentPolicy(
    key="denoiser",
    component_path="denoiser",
    expected_types=(ContractReferenceDenoiser,),
    supported_strategies=(FineTuneKind.FULL, FineTuneKind.LORA),
    lora_target_modules=("projection",),
    full_parameter_names=(
        "conditioning_projection.bias",
        "conditioning_projection.weight",
        "output_projection.bias",
        "output_projection.weight",
        "projection.bias",
        "projection.weight",
    ),
)


class ContractReferenceRecipe(TrainingRecipe3D[ContractReferencePipeline, ContractReferenceBatch]):
    """Deterministic diffusion-like objective over exact pipeline components."""

    recipe_id = "contract-reference"
    recipe_version = "1.0"
    family_id = "contract-reference"
    target_type = ContractReferencePipeline
    batch_type = ContractReferenceBatch
    component_policies = (CONTRACT_REFERENCE_DENOISER_POLICY,)

    def collate(self, examples: Sequence[Object3DExample]) -> ContractReferenceBatch:
        images = []
        vertices = []
        for example in examples:
            if type(example) is not Object3DExample:
                raise TrainingTargetError("examples must contain exact Object3DExample values")
            if type(example.condition) is not ImageCondition or type(example.target) is not MeshAsset:
                raise TrainingTargetError(
                    "contract-reference examples require exact ImageCondition and MeshAsset values"
                )
            example.validate(expensive=True)
            images.append(example.condition.image)
            vertices.append(example.target.vertices)
        return ContractReferenceBatch(images=torch.stack(images), vertices=torch.stack(vertices))

    def validate_target(self) -> None:
        if type(self.target.denoiser) is not ContractReferenceDenoiser:
            raise TrainingTargetError("target denoiser must be the exact contract-reference class")
        if type(self.target.mesh_decoder) is not ContractReferenceMeshDecoder:
            raise TrainingTargetError("target mesh decoder must be the exact contract-reference class")
        if self.target.denoiser.config.latent_dim != 9 or self.target.mesh_decoder.config.num_vertices != 3:
            raise TrainingTargetError("contract-reference training requires nine latents and three mesh vertices")

    def compute_loss(self, batch: ContractReferenceBatch) -> TrainingStep3DOutput:
        batch.validate()
        clean_latents = batch.vertices.reshape(batch.vertices.shape[0], -1)
        conditioning = batch.images.mean(dim=(-2, -1))
        noise = torch.sin(clean_latents + conditioning.mean(dim=-1, keepdim=True))
        timesteps = torch.full(
            (clean_latents.shape[0],),
            self.target.scheduler.config.num_train_timesteps // 2,
            device=clean_latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.target.scheduler.add_noise(clean_latents, noise, timesteps)

        noise_prediction = self.target.denoiser(noisy_latents, timesteps, conditioning)
        alpha_product = self.target.scheduler.alphas_cumprod.to(
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )[timesteps].unsqueeze(-1)
        predicted_clean_latents = (
            noisy_latents - (1 - alpha_product).sqrt() * noise_prediction
        ) / alpha_product.sqrt()
        decoded_vertices = self.target.mesh_decoder(predicted_clean_latents)

        diffusion_loss = F.mse_loss(noise_prediction, noise)
        mesh_loss = F.mse_loss(decoded_vertices, batch.vertices)
        loss = diffusion_loss + 0.05 * mesh_loss
        return TrainingStep3DOutput(
            loss=loss,
            metrics={
                "diffusion_mse": diffusion_loss,
                "mesh_mse": mesh_loss,
            },
        )

    def load_weights(
        self,
        save_directory: str | Path,
        strategy: FineTuneStrategy3D,
        components: Mapping[str, nn.Module],
    ) -> None:
        directory = Path(save_directory)
        for key in strategy.components:
            component = components[key]
            component_directory = directory / key
            if type(strategy) is FullFineTune:
                loaded = type(component).from_pretrained(component_directory)
                component.load_state_dict(loaded.state_dict(), strict=True)
            elif type(strategy) is LoRAFineTune:
                if not isinstance(component, ContractReferenceDenoiser):
                    raise TrainingCheckpointError("LoRA checkpoint component has the wrong exact type")
                component.load_lora_adapter(
                    component_directory,
                    prefix=None,
                    adapter_name=TRAINING_ADAPTER_NAME,
                    hotswap=True,
                    use_safetensors=True,
                )
            else:
                raise TrainingCheckpointError(f"unsupported fine-tuning strategy {type(strategy).__name__}")


__all__ = [
    "CONTRACT_REFERENCE_DENOISER_POLICY",
    "ContractReferenceBatch",
    "ContractReferenceDataset",
    "ContractReferenceRecipe",
]
