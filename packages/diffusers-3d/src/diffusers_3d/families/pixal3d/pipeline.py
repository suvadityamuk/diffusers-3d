# Portions of this file reproduce pipeline semantics from Tencent ARC Pixal3D:
# https://github.com/TencentARC/Pixal3D
# Revision: f7cf38429b0bd264f1995f0f8743a88b1c728b94
#
# MIT License. Copyright (c) 2026 Tencent.
# Modified to reuse the package's TRELLIS.2 pipeline stages with typed, camera-explicit conditions.

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import DINOv3ViTModel

from ...data import ImageCondition, MultiViewCondition, preprocess_image_condition
from ...data.image_processing import _quantized_rgba
from ...execution.pipelines import ExternalComponent
from ...objects import CoordinateSystem, Latent3DOutput, Object3D, Object3DPipelineOutput, SparseVoxelAsset
from ..moge.conversion import MOGE2_WEIGHTS_REPOSITORY, MOGE2_WEIGHTS_REVISION, load_moge_estimator
from ..moge.models import MoGe2Model
from ..trellis.sparse import TrellisSparseTensor
from ..trellis2.decoders import Trellis2PBRSparseDecoder, Trellis2ShapeDualGridDecoder, Trellis2SparseStructureDecoder
from ..trellis2.pipeline import _PIPELINE_TYPES as _TRELLIS2_PIPELINE_TYPES
from ..trellis2.pipeline import (
    TRELLIS2_REPO,
    TRELLIS2_REVISION,
    TRELLIS_IMAGE_LARGE_REPO,
    TRELLIS_IMAGE_LARGE_REVISION,
    Trellis2ImageTo3DPipeline,
    _sampler_parameters,
    load_released_component,
    shrink_image_condition,
)
from ..trellis2.scheduler import Trellis2FlowEulerScheduler
from .conditioner import (
    NAF_CHECKPOINT_SHA256,
    NAF_CHECKPOINT_URL,
    Pixal3DNAFUpsampler,
    blender_camera_to_world,
    camera_distance_from_fov,
    front_view_camera,
    horizontal_fov,
    load_naf_upsampler,
    project_grid_features,
    relative_projection_cameras,
)
from .models import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel

_PIPELINE_TYPES = {name: _TRELLIS2_PIPELINE_TYPES[name] for name in ("1024_cascade", "1536_cascade")}
# ``IMAGE_COND_CONFIGS`` of the pinned inference scripts, relative to the pipeline ``image_size`` (512): the DINOv3
# input side and the NAF output side of each stage (``None``: no high-resolution branch). Every stage projects onto
# its own latent grid; the 1024 stages float with the cascade's token budget.
_STAGE_CONDITIONING = {
    "sparse_structure": (1, None),
    "512": (1, 1),
    "1024": (2, 1),
    "texture": (2, 2),
}
_FOREGROUND_SCALE = 1.1
_FLOW_COMPONENTS = {
    "sparse_structure_flow_model": Pixal3DSparseStructureFlowModel,
    "shape_slat_flow_model": Pixal3DSLatFlowModel,
    "shape_slat_flow_model_1024": Pixal3DSLatFlowModel,
    "texture_slat_flow_model": Pixal3DSLatFlowModel,
    "texture_slat_flow_model_1024": Pixal3DSLatFlowModel,
}


@dataclasses.dataclass(frozen=True)
class _Views:
    """Conditioning views per object: images ``{side: (batch, views, 3, side, side)}`` and projection cameras."""

    images: dict[int, torch.Tensor]
    camera_to_world: torch.Tensor
    camera_angle_x: torch.Tensor
    mesh_scale: float


def _multiview_images(condition: MultiViewCondition, image_size: int) -> torch.Tensor:
    """``inference_mv.to_cond_tensor``: views keep their framing, are LANCZOS-resized as RGBA, then premultiplied."""

    views = []
    for index in range(condition.images.shape[0]):
        mask = None if condition.masks is None else condition.masks[index]
        _, rgba = _quantized_rgba(ImageCondition(image=condition.images[index], mask=mask))
        resized = Image.fromarray(rgba).resize((image_size, image_size), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
        views.append(pixels[:3] * pixels[3:])
    return torch.stack(views)


def _cropped_image(condition: ImageCondition) -> torch.Tensor:
    """Upstream ``preprocess_image`` at native resolution: the ``alpha > 0.8`` box grown by 1.1 and premultiplied.

    This is the image upstream estimates the field of view on. Inputs without transparency keep their full frame.
    """

    rgba = _quantized_rgba(condition)[1]
    alpha = rgba[:, :, 3]
    image = Image.fromarray(rgba)
    if not np.all(alpha == 255):
        rows, columns = np.nonzero(alpha > 0.8 * 255)
        center_x, center_y = (columns.min() + columns.max()) / 2, (rows.min() + rows.max()) / 2
        half = int(max(columns.max() - columns.min(), rows.max() - rows.min()) * _FOREGROUND_SCALE) // 2
        image = image.crop((center_x - half, center_y - half, center_x + half, center_y + half))
    pixels = np.asarray(image).astype(np.float32) / 255
    pixels = (pixels[:, :, :3] * pixels[:, :, 3:] * 255).astype(np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255


class Pixal3DImageTo3DPipeline(Trellis2ImageTo3DPipeline):
    """Pixal3D: TRELLIS.2's cascade with pixel-aligned projection conditioning, in plain PyTorch.

    Each flow attends to the DINOv3 class and register tokens and, per latent cell, adds DINOv3 patch features
    sampled where the cell projects into the image (the SLAT stages concatenate a NAF-upsampled copy). Latents are
    view-aligned: the O-Voxel grid is right-handed Y-up with the conditioning camera on ``+z``, so outputs carry
    ``RIGHT_HANDED_Y_UP``. Projection needs the horizontal field of view of every view. Single-view checkpoints take
    ``ImageCondition`` inputs, multi-view ones posed ``MultiViewCondition`` inputs whose first view is the front view.
    """

    model_cpu_offload_seq = (
        "depth_estimator->image_encoder->feature_upsampler->"
        + Trellis2ImageTo3DPipeline.model_cpu_offload_seq.split("->", 1)[1]
    )
    _optional_components = [*Trellis2ImageTo3DPipeline._optional_components, "depth_estimator"]
    # Third-party networks Pixal3D uses unchanged load from their original sources and are never saved with it:
    # DINOv3 (gated), valeo.ai's NAF release, MoGe-2 (single-view checkpoints only), and the three decoders its
    # release copies byte for byte from TRELLIS-image-large and TRELLIS.2-4B (converted on load as Pixal3D's
    # converter did).
    external_components = {
        **Trellis2ImageTo3DPipeline.external_components,
        **{
            name: ExternalComponent(
                repository, revision, partial(load_released_component, name=name, family="pixal3d")
            )
            for name, repository, revision in (
                ("sparse_structure_decoder", TRELLIS_IMAGE_LARGE_REPO, TRELLIS_IMAGE_LARGE_REVISION),
                ("shape_slat_decoder", TRELLIS2_REPO, TRELLIS2_REVISION),
                ("pbr_decoder", TRELLIS2_REPO, TRELLIS2_REVISION),
            )
        },
        "feature_upsampler": ExternalComponent(NAF_CHECKPOINT_URL, NAF_CHECKPOINT_SHA256, load_naf_upsampler),
        "depth_estimator": ExternalComponent(MOGE2_WEIGHTS_REPOSITORY, MOGE2_WEIGHTS_REVISION, load_moge_estimator),
    }

    def __init__(
        self,
        image_encoder: DINOv3ViTModel,
        feature_upsampler: Pixal3DNAFUpsampler,
        sparse_structure_flow_model: Pixal3DSparseStructureFlowModel,
        sparse_structure_decoder: Trellis2SparseStructureDecoder,
        sparse_structure_scheduler: Trellis2FlowEulerScheduler,
        shape_slat_flow_model: Pixal3DSLatFlowModel | None = None,
        shape_slat_flow_model_1024: Pixal3DSLatFlowModel | None = None,
        shape_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        shape_slat_decoder: Trellis2ShapeDualGridDecoder | None = None,
        texture_slat_flow_model: Pixal3DSLatFlowModel | None = None,
        texture_slat_flow_model_1024: Pixal3DSLatFlowModel | None = None,
        texture_slat_scheduler: Trellis2FlowEulerScheduler | None = None,
        pbr_decoder: Trellis2PBRSparseDecoder | None = None,
        shape_slat_mean: Sequence[float] | None = None,
        shape_slat_std: Sequence[float] | None = None,
        texture_slat_mean: Sequence[float] | None = None,
        texture_slat_std: Sequence[float] | None = None,
        default_pipeline_type: str = "1536_cascade",
        image_size: int = 512,
        sparse_structure_sampler_defaults: Mapping[str, Any] | None = None,
        shape_slat_sampler_defaults: Mapping[str, Any] | None = None,
        texture_slat_sampler_defaults: Mapping[str, Any] | None = None,
        multiview: bool = False,
        capability_limitations: Mapping[str, Any] | None = None,
        depth_estimator: MoGe2Model | None = None,
    ) -> None:
        super().__init__(
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
            shape_slat_mean=shape_slat_mean,
            shape_slat_std=shape_slat_std,
            texture_slat_mean=texture_slat_mean,
            texture_slat_std=texture_slat_std,
            default_pipeline_type=default_pipeline_type,
            image_size=image_size,
            sparse_structure_sampler_defaults=sparse_structure_sampler_defaults,
            shape_slat_sampler_defaults=shape_slat_sampler_defaults,
            texture_slat_sampler_defaults=texture_slat_sampler_defaults,
            capability_limitations=capability_limitations,
        )
        hidden_size = image_encoder.config.hidden_size
        if hidden_size % feature_upsampler.config.num_heads:
            raise ValueError("DINOv3 features must split into the feature upsampler's attention heads")
        if sparse_structure_flow_model.config.proj_in_channels != hidden_size:
            raise ValueError("the sparse-structure flow projects raw DINOv3 patch features")
        for name in _FLOW_COMPONENTS:
            model = getattr(self, name)
            if name != "sparse_structure_flow_model" and model is not None:
                if model.config.proj_in_channels != 2 * hidden_size:
                    raise ValueError(f"{name} projects DINOv3 and NAF-upsampled features ({2 * hidden_size} channels)")
        if not isinstance(multiview, bool):
            raise TypeError("multiview must be a bool")
        self.register_modules(feature_upsampler=feature_upsampler, depth_estimator=depth_estimator)
        self.register_to_config(multiview=multiview)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load like any pipeline, but keep DINOv3 and the NAF upsampler in float32 whatever ``dtype`` the rest loads
        in, as upstream does (and MoGe-2, which runs under float16 autocast); a ``dtype`` dict that names any of them
        overrides this."""

        if "torch_dtype" in kwargs:
            kwargs.setdefault("dtype", kwargs.pop("torch_dtype"))
        dtype = kwargs.get("dtype")
        # Without a dtype every component keeps its own precision: float32 encoders and the flows' bfloat16 torso.
        if dtype is not None:
            dtypes = dict(dtype) if isinstance(dtype, dict) else {"default": dtype}
            kwargs["dtype"] = (
                dict.fromkeys(("image_encoder", "feature_upsampler", "depth_estimator"), torch.float32) | dtypes
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def _validate_pipeline_type(pipeline_type: str) -> str:
        if not isinstance(pipeline_type, str) or pipeline_type not in _PIPELINE_TYPES:
            raise ValueError(f"pipeline_type must be one of {sorted(_PIPELINE_TYPES)}")
        return pipeline_type

    @staticmethod
    def _quantize_cascade_coordinates(coordinates: torch.Tensor, resolution: int, target: int) -> torch.Tensor:
        # Unlike TRELLIS.2, Pixal3D rounds onto the ``target`` grid's ``linspace`` cell positions.
        return ((coordinates + 0.5) / resolution * (target - 1)).round().to(dtype=coordinates.dtype)

    def prepare_views(
        self,
        image: ImageCondition | MultiViewCondition | Sequence[ImageCondition | MultiViewCondition] | torch.Tensor,
        *,
        image_sizes: Sequence[int],
        camera_angle_x: float | Sequence[float] | None = None,
        mesh_scale: float = 1.0,
    ) -> _Views:
        """Preprocess conditions at every DINOv3 side in ``image_sizes`` and build their projection cameras.

        Single images larger than 1024 px are shrunk first, then get the released crop (``alpha > 0.8`` box grown by
        1.1, premultiplied on black). Their field of view is ``camera_angle_x`` (radians, one value or one per image),
        else the cropped condition camera's, else the ``depth_estimator``'s MoGe-2 estimate as upstream; the camera sits on the canonical front view at the distance where the unit cube fills the image width.
        Multi-view conditions are not cropped; their cameras are re-expressed relative to the first view.
        """

        if isinstance(image, MultiViewCondition) or (
            isinstance(image, (list, tuple))
            and image
            and all(isinstance(value, MultiViewCondition) for value in image)
        ):
            conditions = (image,) if isinstance(image, MultiViewCondition) else tuple(image)
        else:
            conditions = tuple(self.image_conditions(image))
        kinds = {type(condition) for condition in conditions}
        if not conditions or len(kinds) != 1 or not kinds <= {ImageCondition, MultiViewCondition}:
            raise TypeError("image must be ImageCondition or MultiViewCondition values of a single kind, or a tensor")
        multiview = kinds == {MultiViewCondition}
        if multiview != self.config.multiview:
            expected = "MultiViewCondition" if self.config.multiview else "ImageCondition"
            raise TypeError(f"this checkpoint was trained on {expected} inputs")
        if not isinstance(mesh_scale, (int, float)) or isinstance(mesh_scale, bool) or not mesh_scale > 0:
            raise ValueError("mesh_scale must be a positive number")
        device = self._execution_device
        dtype = self.image_encoder.dtype

        if multiview:
            if camera_angle_x is not None:
                raise ValueError("multi-view fields of view come from the condition cameras")
            if len({condition.images.shape[0] for condition in conditions}) != 1:
                raise ValueError("every multi-view condition in a batch must have the same number of views")
            if any(condition.images.shape[-1] != condition.images.shape[-2] for condition in conditions):
                raise ValueError("multi-view images must be square: they are projected without cropping")
            images = {size: torch.stack([_multiview_images(c, size) for c in conditions]) for size in image_sizes}
            camera_to_world = torch.stack(
                [relative_projection_cameras(blender_camera_to_world(c.cameras)) for c in conditions]
            )
            fov = torch.stack([horizontal_fov(c.cameras) for c in conditions])
        else:
            conditions = tuple(shrink_image_condition(condition) for condition in conditions)
            processed = {
                size: [
                    preprocess_image_condition(
                        condition,
                        image_size=size,
                        foreground_scale=_FOREGROUND_SCALE,
                        premultiply_before_resize=True,
                    )
                    for condition in conditions
                ]
                for size in image_sizes
            }
            images = {
                size: torch.stack([value.image for value in values])[:, None] for size, values in processed.items()
            }
            if camera_angle_x is None:
                cameras = [value.camera for value in next(iter(processed.values()))]
                if any(camera is None for camera in cameras) and self.depth_estimator is None:
                    raise ValueError(
                        "Pixal3D needs each image's horizontal field of view: pass camera_angle_x (radians), an "
                        "ImageCondition camera, or load the MoGe-2 depth_estimator"
                    )
                fov = torch.cat(
                    [
                        horizontal_fov(camera) if camera is not None else self.estimate_camera_angle_x(condition)
                        for camera, condition in zip(cameras, conditions)
                    ]
                )
            else:
                values = (
                    [camera_angle_x] * len(conditions) if isinstance(camera_angle_x, (int, float)) else camera_angle_x
                )
                if len(values) != len(conditions):
                    raise ValueError("camera_angle_x must be one value or one value per image")
                fov = torch.tensor([float(value) for value in values])
            if not bool(((fov > 0) & (fov < torch.pi)).all()):
                raise ValueError("fields of view must lie in (0, pi) radians")
            distance = torch.tensor([camera_distance_from_fov(float(value), mesh_scale) for value in fov])
            camera_to_world = front_view_camera(distance)[:, None]
            fov = fov[:, None]
        return _Views(
            images={size: value.to(device=device, dtype=dtype) for size, value in images.items()},
            camera_to_world=camera_to_world,
            camera_angle_x=fov.float(),
            mesh_scale=float(mesh_scale),
        )

    @torch.no_grad()
    def estimate_camera_angle_x(self, condition: ImageCondition) -> torch.Tensor:
        """Horizontal field of view ``(1,)`` in radians, from MoGe-2 on the cropped image as upstream estimates it."""

        image = _cropped_image(condition)
        device = self.depth_estimator.device
        # Upstream's default float16 autocast only runs on CUDA; elsewhere it raises, so run in float32 there.
        intrinsics = self.depth_estimator.infer(image.to(device), use_fp16=device.type == "cuda")["intrinsics"]
        # Upstream's ``get_camera_params_wild_moge``, in its NumPy float32 arithmetic.
        width = image.shape[-1]
        focal = intrinsics.squeeze().cpu().numpy()[0, 0] * width
        return torch.tensor([2 * math.atan(width / (2 * focal))])

    def encode_views(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """DINOv3 tokens of ``(batch, views, 3, side, side)`` images: view-averaged class + register tokens and the
        per-view patch-feature maps ``(batch, views, channels, side / patch, side / patch)``."""

        batch, views, _, side, _ = images.shape
        tokens = self.encode_image(images.flatten(0, 1), image_size=side)
        config = self.image_encoder.config
        global_tokens = tokens[:, : 1 + config.num_register_tokens].unflatten(0, (batch, views)).mean(dim=1)
        patches = tokens[:, 1 + config.num_register_tokens :].unflatten(1, (side // config.patch_size, -1))
        return global_tokens, patches.permute(0, 3, 1, 2).unflatten(0, (batch, views))

    def project_views(
        self,
        views: _Views,
        encoded: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        stage: str,
        indices: Sequence[torch.Tensor],
        *,
        grid_resolution: int,
    ) -> list[torch.Tensor]:
        """Per object, the view-averaged pixel-aligned features of its latent cells ``indices`` for ``stage``."""

        image_scale, upsample_scale = _STAGE_CONDITIONING[stage]
        side = self.config.image_size * image_scale
        _, feature_maps = encoded[side]
        outputs = []
        for batch_index, cells in enumerate(indices):
            total = None
            for view in range(feature_maps.shape[1]):
                maps = [feature_maps[batch_index, view]]
                if upsample_scale is not None:
                    target = self.config.image_size * upsample_scale
                    guide = views.images[side][batch_index, view : view + 1]
                    maps.append(self.feature_upsampler(guide, maps[0][None], (target, target)).sample[0])
                features = project_grid_features(
                    maps,
                    cells,
                    grid_resolution=grid_resolution,
                    camera_to_world=views.camera_to_world[batch_index, view],
                    camera_angle_x=views.camera_angle_x[batch_index, view],
                    mesh_scale=views.mesh_scale,
                    image_size=side,
                )
                total = features if total is None else total.add_(features)
            outputs.append(total / feature_maps.shape[1])
        return outputs

    def _slat_condition(
        self,
        views: _Views,
        encoded: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        stage: str,
        structures: Sequence[SparseVoxelAsset],
        resolution: int,
    ) -> tuple[tuple[torch.Tensor, TrellisSparseTensor], tuple[torch.Tensor, TrellisSparseTensor]]:
        coordinates = TrellisSparseTensor.from_sparse_voxel_assets(tuple(structures)).coordinates
        rows = [coordinates[:, 0] == index for index in range(len(structures))]
        projected = self.project_views(
            views, encoded, stage, [coordinates[mask, 1:] for mask in rows], grid_resolution=resolution
        )
        features = projected[0].new_empty(coordinates.shape[0], projected[0].shape[1])
        for mask, values in zip(rows, projected):
            features[mask] = values
        global_tokens = encoded[self.config.image_size * _STAGE_CONDITIONING[stage][0]][0]
        projected = TrellisSparseTensor(coordinates.to(features.device), features)
        negative = (torch.zeros_like(global_tokens), projected.replace(torch.zeros_like(features)))
        return (global_tokens, projected), negative

    @torch.no_grad()
    def __call__(
        self,
        image: ImageCondition | MultiViewCondition | Sequence[ImageCondition | MultiViewCondition] | torch.Tensor,
        *,
        camera_angle_x: float | Sequence[float] | None = None,
        mesh_scale: float = 1.0,
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
        """Generate view-aligned 3D objects; see :meth:`prepare_views` for ``camera_angle_x`` and ``mesh_scale``.

        ``formats``, the sampler parameters, latents, and ``max_num_tokens`` behave as in
        :class:`Trellis2ImageTo3DPipeline`; ``pipeline_type`` is ``"1024_cascade"`` or ``"1536_cascade"``.
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

        stages = ("sparse_structure", *(preset.stages if needs_shape else ()), *(("texture",) * needs_texture))
        sides = sorted({self.config.image_size * _STAGE_CONDITIONING[stage][0] for stage in stages})
        views = self.prepare_views(image, image_sizes=sides, camera_angle_x=camera_angle_x, mesh_scale=mesh_scale)
        encoded = {side: self.encode_views(views.images[side]) for side in sides}
        batch_size = views.camera_to_world.shape[0]

        resolution = self.sparse_structure_flow_model.config.resolution
        axis = torch.arange(resolution, device=self._execution_device)
        cells = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        projected = torch.stack(
            self.project_views(views, encoded, "sparse_structure", [cells] * batch_size, grid_resolution=resolution)
        )
        global_tokens = encoded[self.config.image_size][0]
        dense_latents = self.prepare_sparse_structure_latents(
            batch_size, generator=generator, latents=sparse_structure_latents
        )
        dense_latents = self._sample_dense(
            dense_latents,
            (global_tokens, projected),
            (torch.zeros_like(global_tokens), torch.zeros_like(projected)),
            _sampler_parameters(self.config.sparse_structure_sampler_defaults, sparse_structure_sampler_params),
        )
        # Every later asset inherits the view-aligned frame from these structures.
        structures = tuple(
            dataclasses.replace(structure, coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP)
            for structure in self.sparse_structure_decoder.decode_to_sparse_voxels(
                dense_latents, pooling=self._sparse_structure_pooling(pipeline_type)
            )
        )
        objects: list[Object3D] = list(structures) if "sparse_structure" in formats else []

        if needs_shape:
            shape_parameters = _sampler_parameters(self.config.shape_slat_sampler_defaults, shape_slat_sampler_params)
            resolution = int(structures[0].metadata["resolution"])
            for index, stage in enumerate(preset.stages):
                last = index == len(preset.stages) - 1
                conditional, negative = self._slat_condition(views, encoded, stage, structures, resolution)
                normalized_shape, shape_slat = self.sample_shape_slat(
                    structures,
                    self._stage_models(stage)[0],
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
                conditional, negative = self._slat_condition(views, encoded, "texture", structures, resolution)
                texture_slat = self.sample_texture_slat(
                    structures,
                    self._stage_models(preset.stages[-1])[1],
                    normalized_shape,
                    conditional,
                    negative,
                    _sampler_parameters(self.config.texture_slat_sampler_defaults, texture_slat_sampler_params),
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
                    objects.extend(self.postprocess_ovoxel(asset, output_format="mesh") for asset in ovoxels)
        latent_output = (
            Latent3DOutput(latents=dense_latents, metadata={"family": "pixal3d", "stage": "sparse_structure"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        # The reused TRELLIS.2 stages stamp their own family name.
        objects = [dataclasses.replace(value, metadata={**value.metadata, "family": "pixal3d"}) for value in objects]
        if not return_dict:
            return tuple(objects), latent_output
        return Object3DPipelineOutput(objects=tuple(objects), latents=latent_output, previews=None)


__all__ = ["Pixal3DImageTo3DPipeline"]
