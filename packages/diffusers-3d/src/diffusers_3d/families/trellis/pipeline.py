# Portions of this file reproduce pipeline semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for typed, object-native Diffusers stages.

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...data import ImageCondition, TextCondition, preprocess_image_condition
from ...execution.metadata import (
    ContributionStatus,
    Object3DComponentSpec,
    ReviewStatus,
    fully_qualified_class_name,
)
from ...execution.models import Object3DModel
from ...execution.pipelines import Object3DPipeline
from ...objects import (
    GaussianSplatAsset,
    Latent3DOutput,
    MeshAsset,
    Object3D,
    Object3DKind,
    Object3DPipelineOutput,
    RadianceFieldAsset,
    SparseVoxelAsset,
)
from .conditioner import TrellisDinov2Conditioner
from .decoders import (
    TrellisSLatGaussianDecoder,
    TrellisSLatMeshDecoder,
    TrellisSLatRadianceFieldDecoder,
    TrellisSparseStructureDecoder,
)
from .models import TrellisSLatFlowModel, TrellisSparseStructureFlowModel
from .scheduler import TrellisFlowEulerScheduler
from .sparse import TrellisSparseTensor
from .text_conditioner import TrellisClipTextConditioner


def _conditioner_spec(conditioner_type: type[Object3DModel]) -> Object3DComponentSpec:
    return Object3DComponentSpec(
        name="conditioner",
        expected_class=fully_qualified_class_name(conditioner_type),
        subfolder="conditioner",
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )


class _TrellisTwoStagePipeline(Object3DPipeline):
    """Shared TRELLIS stages: sparse structure -> SLAT -> Gaussian splats, FlexiCubes meshes, and/or radiance fields.

    Subclasses supply the conditioner (DINOv2 image tokens or CLIP text tokens) and ``__call__``. ``formats``
    covers ``"sparse_structure"``, ``"slat"``, ``"gaussian"``, ``"mesh"``, and ``"radiance_field"``; every network
    runs in plain PyTorch.
    """

    family_id = "trellis"
    output_object_types = (SparseVoxelAsset, GaussianSplatAsset, MeshAsset, RadianceFieldAsset)
    output_representations = ("sparse-structure", "slat", "gaussian-splat", "mesh", "radiance-field")
    object_kinds = (
        Object3DKind.SPARSE_VOXEL,
        Object3DKind.GAUSSIAN_SPLAT,
        Object3DKind.MESH,
        Object3DKind.RADIANCE_FIELD,
    )
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _shared_component_specs = (
        Object3DComponentSpec(
            name="sparse_structure_flow_model",
            expected_class=fully_qualified_class_name(TrellisSparseStructureFlowModel),
            subfolder="sparse_structure_flow_model",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="sparse_structure_decoder",
            expected_class=fully_qualified_class_name(TrellisSparseStructureDecoder),
            subfolder="sparse_structure_decoder",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="sparse_structure_scheduler",
            expected_class=fully_qualified_class_name(TrellisFlowEulerScheduler),
            subfolder="sparse_structure_scheduler",
            optional=False,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="slat_flow_model",
            expected_class=fully_qualified_class_name(TrellisSLatFlowModel),
            subfolder="slat_flow_model",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="slat_scheduler",
            expected_class=fully_qualified_class_name(TrellisFlowEulerScheduler),
            subfolder="slat_scheduler",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="gaussian_decoder",
            expected_class=fully_qualified_class_name(TrellisSLatGaussianDecoder),
            subfolder="gaussian_decoder",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="mesh_decoder",
            expected_class=fully_qualified_class_name(TrellisSLatMeshDecoder),
            subfolder="mesh_decoder",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
        Object3DComponentSpec(
            name="radiance_field_decoder",
            expected_class=fully_qualified_class_name(TrellisSLatRadianceFieldDecoder),
            subfolder="radiance_field_decoder",
            optional=True,
            review_status=ReviewStatus.REVIEWED,
            loading_eligible=True,
        ),
    )
    model_cpu_offload_seq = (
        "conditioner->sparse_structure_flow_model->sparse_structure_decoder->slat_flow_model"
        "->gaussian_decoder->mesh_decoder->radiance_field_decoder"
    )
    _optional_components = [
        "slat_flow_model",
        "slat_scheduler",
        "gaussian_decoder",
        "mesh_decoder",
        "radiance_field_decoder",
    ]

    def __init__(
        self,
        conditioner: Object3DModel,
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
        if sparse_structure_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
            raise ValueError("sparse-structure conditioner and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")
        if slat_flow_model is not None:
            if slat_scheduler is None:
                raise ValueError("slat_scheduler is required when slat_flow_model is provided")
            if slat_flow_model.config.cond_channels != conditioner.model.config.hidden_size:
                raise ValueError("SLAT conditioner and flow context dimensions must match")
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
            conditioner=conditioner,
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
            return randn_tensor(
                shape,
                generator=generator,
                device=self._execution_device,
                dtype=parameter.dtype,
            )
        if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != shape:
            raise ValueError(f"sparse-structure latents must have shape {shape}")
        return latents.to(device=self._execution_device, dtype=parameter.dtype)

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

        if len(guidance_interval) != 2 or not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        scheduler = self.sparse_structure_scheduler
        scheduler.set_timesteps(
            num_inference_steps,
            device=latents.device,
            rescale_t=rescale_t,
        )
        for timestep in scheduler.timesteps:
            model_timestep = scheduler.model_timestep(
                timestep,
                latents.shape[0],
                device=latents.device,
            )
            conditional_velocity = self.sparse_structure_flow_model(
                latents,
                model_timestep,
                conditional,
            ).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unconditional_velocity = self.sparse_structure_flow_model(
                    latents,
                    model_timestep,
                    unconditional,
                ).sample
                velocity = scheduler.apply_guidance(
                    conditional_velocity,
                    unconditional_velocity,
                    guidance_scale,
                )
            else:
                velocity = conditional_velocity
            latents = scheduler.step(velocity, timestep, latents).prev_sample
        return latents

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
                    randn_tensor(
                        (count, expected_channels),
                        generator=item_generator,
                        device=self._execution_device,
                        dtype=self.slat_flow_model.dtype,
                    )
                )
            features = torch.cat(feature_parts)
        else:
            features = randn_tensor(
                (coordinates.shape[0], expected_channels),
                generator=generator,
                device=self._execution_device,
                dtype=self.slat_flow_model.dtype,
            )
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
        if len(guidance_interval) != 2 or not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        self.slat_scheduler.set_timesteps(
            num_inference_steps,
            device=latents.device,
            rescale_t=rescale_t,
        )
        for timestep in self.slat_scheduler.timesteps:
            model_timestep = self.slat_scheduler.model_timestep(
                timestep,
                latents.batch_size,
                device=latents.device,
            )
            conditional_velocity = self.slat_flow_model(
                latents,
                model_timestep,
                conditional,
            ).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unconditional_velocity = self.slat_flow_model(
                    latents,
                    model_timestep,
                    unconditional,
                ).sample
                velocity = self.slat_scheduler.apply_guidance(
                    conditional_velocity,
                    unconditional_velocity,
                    guidance_scale,
                )
            else:
                velocity = conditional_velocity
            latents = self.slat_scheduler.step(velocity, timestep, latents).prev_sample
        mean = latents.features.new_tensor(self.config.slat_mean)
        std = latents.features.new_tensor(self.config.slat_std)
        return latents.denormalize(mean, std)

    def decode_slat(
        self,
        slat: TrellisSparseTensor,
        *,
        formats: tuple[str, ...],
    ) -> tuple[Object3D, ...]:
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
        if any(name in formats for name in ("slat", *decoders)) and self.slat_flow_model is None:
            raise RuntimeError("formats requiring SLAT need slat_flow_model and slat_scheduler")
        for name, decoder in decoders.items():
            if name in formats and decoder is None:
                raise RuntimeError(f"format '{name}' requires {name}_decoder")
        sparse_structure_latents = self.prepare_sparse_structure_latents(
            conditional.shape[0],
            generator=generator,
            latents=sparse_structure_latents,
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

        requires_slat = any(name in formats for name in ("slat", *decoders))
        if requires_slat:
            slat_latents = self.prepare_slat_latents(
                sparse_structures,
                generator=generator,
                latents=slat_latents,
            )
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
                latents=sparse_structure_latents,
                metadata={"family": self.family_id, "stage": "sparse_structure"},
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

    task_ids = ("image-to-3d",)
    component_specs = (_conditioner_spec(TrellisDinov2Conditioner), *_TrellisTwoStagePipeline._shared_component_specs)

    def __init__(
        self,
        conditioner: TrellisDinov2Conditioner,
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
        if type(conditioner) is not TrellisDinov2Conditioner:
            raise TypeError("TrellisImageTo3DPipeline requires a TrellisDinov2Conditioner")
        super().__init__(
            conditioner,
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

    def preprocess(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
    ) -> torch.Tensor:
        """Apply pinned foreground recentering without implicit background removal."""

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                conditions = (ImageCondition(image=image),)
            elif image.ndim == 4:
                conditions = tuple(ImageCondition(image=item) for item in image)
            else:
                raise ValueError("image tensor must have shape (channels, height, width) or be batched")
        elif type(image) is ImageCondition:
            conditions = (image,)
        elif isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
            conditions = tuple(image)
            if not conditions or any(type(condition) is not ImageCondition for condition in conditions):
                raise TypeError("image sequences must contain exact ImageCondition values")
        else:
            raise TypeError("image must be an ImageCondition, a sequence of ImageCondition values, or a tensor")

        processed = [
            preprocess_image_condition(
                condition,
                image_size=self.conditioner.image_size,
                foreground_scale=1.2,
            ).image
            for condition in conditions
        ]
        parameter = next(self.conditioner.parameters())
        return torch.stack(processed).to(device=self._execution_device, dtype=parameter.dtype)

    def encode_conditioning(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode released normalized image tokens and an all-zero unconditional branch."""

        conditional = self.conditioner(images, value_range=(0.0, 1.0)).embeddings
        unconditional = self.conditioner.unconditional_embedding(
            images.shape[0],
            device=conditional.device,
            dtype=conditional.dtype,
        )
        return conditional, unconditional

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | Sequence[ImageCondition] | torch.Tensor,
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

    task_ids = ("text-to-3d",)
    component_specs = (
        _conditioner_spec(TrellisClipTextConditioner),
        *_TrellisTwoStagePipeline._shared_component_specs,
    )

    def __init__(
        self,
        conditioner: TrellisClipTextConditioner,
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
        if type(conditioner) is not TrellisClipTextConditioner:
            raise TypeError("TrellisTextTo3DPipeline requires a TrellisClipTextConditioner")
        super().__init__(
            conditioner,
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

    def encode_conditioning(
        self, prompts: Sequence[str], negative_prompts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.conditioner(list(prompts)).embeddings
        if all(negative == "" for negative in negative_prompts):
            unconditional = self.conditioner.unconditional_embedding(
                len(prompts), device=conditional.device, dtype=conditional.dtype
            )
        else:
            unconditional = self.conditioner(list(negative_prompts)).embeddings
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
