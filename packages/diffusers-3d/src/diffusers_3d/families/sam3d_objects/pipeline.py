# Portions of this file reproduce inference semantics of SAM 3D Objects:
# https://github.com/facebookresearch/sam-3d-objects
# Revision: f91db411c50efee93d8db7aeb323885650f6f722
#
# SAM License. Copyright (c) Meta Platforms, Inc. and affiliates.
# This file has been modified for typed, object-native Diffusers stages built on the TRELLIS components.

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from transformers import Dinov2WithRegistersModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...data import ImageInput
from ...execution.lora import Object3DLoraLoaderMixin
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import (
    CoordinateSystem,
    Latent3DOutput,
    Object3D,
    Object3DPipelineOutput,
    SparseVoxelAsset,
    coordinate_change_matrix,
)
from ..moge.conversion import MOGE_WEIGHTS_REPOSITORY, MOGE_WEIGHTS_REVISION, load_moge_estimator
from ..moge.models import MoGeModel
from ..trellis.decoders import TrellisSLatGaussianDecoder, TrellisSLatMeshDecoder, TrellisSparseStructureDecoder
from ..trellis.models import TrellisSLatFlowModel
from ..trellis.scheduler import TrellisFlowEulerScheduler
from ..trellis.sparse import TrellisSparseTensor
from ..trellis.sparse_ops import submanifold_neighbors
from .conditioner import (
    DINOV2_REGISTERS_REPO,
    DINOV2_REGISTERS_REVISION,
    Sam3DConditionEmbedder,
    load_dinov2_with_registers,
)
from .models import Sam3DSparseStructureFlowModel

SAM_LICENSE_WARNING = (
    "SAM 3D Objects code and weights are released by Meta under the SAM License, which restricts use (including "
    "trade-control and military-use restrictions) and requires the license to accompany any redistribution. You are "
    "responsible for making sure your use of it complies with that license."
)
# Released normalization of the 6D rotation (two rotation-matrix columns) predicted by the layout stream.
ROTATION_6D_MEAN = (
    -0.06366084883674913,
    0.008438224692279752,
    0.00017084786438302483,
    0.0007126610473540038,
    -0.0030916726538816417,
    0.5166093753457688,
)
ROTATION_6D_STD = (
    0.6656971967514863,
    0.6787012271867754,
    0.30345010594844524,
    0.4394504420678794,
    0.39817973931717104,
    0.6176286868761914,
)
# Upstream halves structures with more active voxels than this so mesh decoding stays within int32 indexing.
MAX_STRUCTURE_VOXELS = 42000
_COMPONENTS = (
    ("image_encoder", Dinov2WithRegistersModel),
    ("sparse_structure_conditioner", Sam3DConditionEmbedder),
    ("sparse_structure_flow_model", Sam3DSparseStructureFlowModel),
    ("sparse_structure_decoder", TrellisSparseStructureDecoder),
    ("sparse_structure_scheduler", TrellisFlowEulerScheduler),
    ("slat_conditioner", Sam3DConditionEmbedder),
    ("slat_flow_model", TrellisSLatFlowModel),
    ("slat_scheduler", TrellisFlowEulerScheduler),
    ("gaussian_decoder", TrellisSLatGaussianDecoder),
    ("mesh_decoder", TrellisSLatMeshDecoder),
)
_OPTIONAL_COMPONENTS = (
    "slat_conditioner",
    "slat_flow_model",
    "slat_scheduler",
    "gaussian_decoder",
    "mesh_decoder",
    "depth_estimator",
)


def _pad_to_square(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    difference = abs(height - width)
    before, after = difference // 2, difference - difference // 2
    padding = (before, after, 0, 0) if height > width else (0, 0, before, after)
    return F.pad(tensor, padding)


def _resize(tensor: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    # ``torchvision.transforms.Resize`` on a square tensor: bilinear is antialiased, nearest is not.
    bilinear = mode == "bilinear"
    return F.interpolate(
        tensor[None], size=(size, size), mode=mode, antialias=bilinear, align_corners=False if bilinear else None
    )[0]


def _object_crop(mask: torch.Tensor, box_size_factor: float) -> tuple[int, int, int, int]:
    """Upstream ``compute_mask_bbox``: a square box around the mask, scaled about its center."""

    rows, columns = torch.nonzero(mask[0], as_tuple=True)
    if rows.numel() == 0:
        raise ValueError("mask must select at least one pixel")
    left, top, right, bottom = (int(value) for value in (columns.min(), rows.min(), columns.max(), rows.max()))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("the mask bounding box must be at least 2x2 pixels")
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = int(max(right - left, bottom - top, 2) * box_size_factor) // 2
    return int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)


def _crop_square(tensor: torch.Tensor, box: tuple[int, int, int, int], fill: float) -> torch.Tensor:
    # Out-of-image regions are zero (``torchvision`` crop), squaring pads use ``fill``.
    left, top, right, bottom = box
    height, width = tensor.shape[-2:]
    tensor = F.pad(tensor, (-left, right - width, -top, bottom - height))
    size = max(tensor.shape[-2:])
    vertical, horizontal = size - tensor.shape[-2], size - tensor.shape[-1]
    return F.pad(
        tensor, (horizontal // 2, horizontal - horizontal // 2, vertical // 2, vertical - vertical // 2), value=fill
    )


def _pointmap_scale_shift(pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream ``ObjectCentricSSI`` (scene scale): median object point as shift, median point extent as scale."""

    flat = pointmap.reshape(3, -1)
    points = flat[:, mask.reshape(-1) > 0.5]
    if not bool(points.isfinite().any()):
        return torch.ones_like(flat[:, 0]), torch.zeros_like(flat[:, 0])
    shift = points.nanmedian(dim=-1).values
    scale = (flat - shift[:, None]).abs().max(dim=0).values.nanmedian()
    return scale.expand_as(shift), shift


def preprocess_sam3d_condition(
    image: torch.Tensor,
    mask: torch.Tensor,
    pointmap: torch.Tensor | None = None,
    *,
    image_size: int = 518,
    box_size_factor: float = 1.2,
) -> dict[str, torch.Tensor]:
    """Build the released (object crop, full frame) views of one image, mask, and PyTorch3D-convention pointmap.

    Returns ``images`` ``(2, 3, S, S)``, ``masks`` ``(2, 1, S, S)``, and, with a pointmap, scale-shift normalized
    ``pointmaps`` ``(2, 3, S, S)`` plus the ``pointmap_scale`` and ``pointmap_shift`` that map the layout back to the
    pointmap's units. Like upstream, the RGB image is not masked.
    """

    rgb = image[:3].float()
    mask = mask.float()
    box = _object_crop(mask, box_size_factor)
    views = {"images": (rgb, "bilinear"), "masks": (mask, "nearest")}
    result = {}
    if pointmap is not None:
        scale, shift = _pointmap_scale_shift(pointmap, mask)
        # Upstream's normalized pointmap marks every invalid point, NaN or infinite (MoGe's masked points), as NaN.
        normalized = (pointmap - shift[:, None, None]) / scale[:, None, None]
        normalized = torch.where(pointmap.isfinite().all(dim=0), normalized, float("nan"))
        views["pointmaps"] = (normalized, "nearest")
        result |= {"pointmap_scale": scale, "pointmap_shift": shift}
    for name, (tensor, mode) in views.items():
        cropped = _crop_square(tensor, box, float("nan") if name == "pointmaps" else 0.0)
        result[name] = torch.stack([_resize(_pad_to_square(view), image_size, mode) for view in (cropped, tensor)])
    return result


def _rotation_from_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt the denormalized 6D prediction into a rotation matrix with columns ``b1, b2, b3``."""

    rotation = rotation * rotation.new_tensor(ROTATION_6D_STD) + rotation.new_tensor(ROTATION_6D_MEAN)
    first = F.normalize(rotation[..., :3], dim=-1)
    second = rotation[..., 3:]
    second = F.normalize(second - (first * second).sum(dim=-1, keepdim=True) * first, dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-1)


def _prune_interior(coordinates: torch.Tensor) -> torch.Tensor:
    """Keep voxels with at least one empty 26-neighbour (upstream ``prune_sparse_structure`` at distance 1)."""

    return coordinates[(submanifold_neighbors(coordinates) < 0).any(dim=1)]


def _downsample_structure(coordinates: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Upstream ``downsample_sparse_structure``: halve an oversized structure about its own bounds."""

    if coordinates.shape[0] <= MAX_STRUCTURE_VOXELS:
        return coordinates, 1
    points = coordinates.float()
    low, high = points.amin(dim=0), points.amax(dim=0)
    size = high - low + 1
    target = size / 2
    target_low = low + (size - target) / 2
    rescaled = torch.round((points - low) / (high - low) * (target - 1) + target_low).int()
    rescaled = torch.clamp(rescaled, target_low.int(), (target_low + target - 1).int())
    unique = torch.unique(rescaled, dim=0)
    if unique.shape[0] > MAX_STRUCTURE_VOXELS:
        unique = unique[torch.randperm(unique.shape[0], device=unique.device)[:MAX_STRUCTURE_VOXELS]]
    return unique.to(coordinates.dtype), 2


class Sam3DObjectsPipeline(Object3DPipeline, Object3DLoraLoaderMixin):
    """SAM 3D Objects: an image and an object mask (plus an optional pointmap) to Gaussian splats and/or a mesh.

    Stage one samples the 16^3 TRELLIS shape latent jointly with the object's layout; stage two samples TRELLIS
    structured latents on the decoded surface voxels, which the TRELLIS Gaussian and FlexiCubes decoders turn into
    assets. Every asset's ``transform`` places the object in the input camera's frame; see :meth:`__call__`.

    Both stages condition on tokens of the same frozen DINOv2 (``image_encoder``, run on the image and on its mask),
    which upstream keeps four identical copies of. That DINOv2 and the MoGe ``depth_estimator`` are third-party
    networks used unchanged: they load from ``facebook/dinov2-with-registers-large`` and ``Ruicheng/moge-vitl`` at
    pinned revisions and are never saved with the pipeline.
    """

    external_components = {
        "image_encoder": ExternalComponent(
            DINOV2_REGISTERS_REPO, DINOV2_REGISTERS_REVISION, load_dinov2_with_registers
        ),
        "depth_estimator": ExternalComponent(MOGE_WEIGHTS_REPOSITORY, MOGE_WEIGHTS_REVISION, load_moge_estimator),
    }
    _optional_components = list(_OPTIONAL_COMPONENTS)
    model_cpu_offload_seq = (
        "depth_estimator->image_encoder->sparse_structure_conditioner->sparse_structure_flow_model->sparse_structure_decoder"
        "->slat_conditioner->slat_flow_model->gaussian_decoder->mesh_decoder"
    )
    _lora_loadable_modules = ["sparse_structure_flow_model", "slat_flow_model"]

    def __init__(
        self,
        image_encoder: Dinov2WithRegistersModel,
        sparse_structure_conditioner: Sam3DConditionEmbedder,
        sparse_structure_flow_model: Sam3DSparseStructureFlowModel,
        sparse_structure_decoder: TrellisSparseStructureDecoder,
        sparse_structure_scheduler: TrellisFlowEulerScheduler,
        slat_conditioner: Sam3DConditionEmbedder | None = None,
        slat_flow_model: TrellisSLatFlowModel | None = None,
        slat_scheduler: TrellisFlowEulerScheduler | None = None,
        gaussian_decoder: TrellisSLatGaussianDecoder | None = None,
        mesh_decoder: TrellisSLatMeshDecoder | None = None,
        slat_mean: Sequence[float] | None = None,
        slat_std: Sequence[float] | None = None,
        depth_estimator: MoGeModel | None = None,
    ) -> None:
        super().__init__()
        warnings.warn(SAM_LICENSE_WARNING, UserWarning, stacklevel=2)
        if any(
            conditioner is not None and conditioner.config.image_embed_dim != image_encoder.config.hidden_size
            for conditioner in (sparse_structure_conditioner, slat_conditioner)
        ):
            raise ValueError("conditioner image_embed_dim must match the image encoder hidden size")
        if sparse_structure_flow_model.config.cond_channels != sparse_structure_conditioner.embed_dim:
            raise ValueError("sparse-structure conditioner and flow context dimensions must match")
        if sparse_structure_flow_model.config.in_channels != sparse_structure_decoder.config.latent_channels:
            raise ValueError("sparse-structure flow and decoder latent channels must match")
        slat_parts = (slat_conditioner, slat_flow_model, slat_scheduler, slat_mean, slat_std)
        if any(part is None for part in slat_parts):
            if any(part is not None for part in (*slat_parts, gaussian_decoder, mesh_decoder)):
                raise ValueError(
                    "the structured-latent stage needs its conditioner, flow model, scheduler, mean and std"
                )
        elif slat_flow_model.config.cond_channels != slat_conditioner.embed_dim:
            raise ValueError("structured-latent conditioner and flow context dimensions must match")
        elif not len(slat_mean) == len(slat_std) == slat_flow_model.config.out_channels:
            raise ValueError("SLAT normalization must contain one value per output feature channel")
        self.register_modules(
            image_encoder=image_encoder,
            sparse_structure_conditioner=sparse_structure_conditioner,
            sparse_structure_flow_model=sparse_structure_flow_model,
            sparse_structure_decoder=sparse_structure_decoder,
            sparse_structure_scheduler=sparse_structure_scheduler,
            slat_conditioner=slat_conditioner,
            slat_flow_model=slat_flow_model,
            slat_scheduler=slat_scheduler,
            gaussian_decoder=gaussian_decoder,
            mesh_decoder=mesh_decoder,
            depth_estimator=depth_estimator,
        )
        self.register_to_config(
            slat_mean=None if slat_mean is None else [float(value) for value in slat_mean],
            slat_std=None if slat_std is None else [float(value) for value in slat_std],
        )

    def preprocess(
        self,
        image: ImageInput | list[ImageInput],
        pointmap: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack the per-image views; ``pointmap`` is ``(3, H, W)`` OpenCV camera-space points (x right, y down).

        Without ``pointmap``, a loaded ``depth_estimator`` (MoGe) predicts it from the unmasked RGB image, as upstream
        ``compute_pointmap`` does.
        """

        conditions = self.image_conditions(image)
        pointmaps = (pointmap,) if pointmap is None or isinstance(pointmap, torch.Tensor) else tuple(pointmap)
        if pointmap is None:
            pointmaps *= len(conditions)
        elif len(pointmaps) != len(conditions) or any(item is None for item in pointmaps):
            raise ValueError("pass one pointmap per image, or none at all")
        items = []
        for condition, points in zip(conditions, pointmaps):
            if condition.mask is None:
                raise ValueError("SAM 3D Objects needs an object mask on every ImageCondition")
            if points is None and self.depth_estimator is not None:
                rgb = condition.image[:3].expand(3, -1, -1).to(self.depth_estimator.device)
                with torch.no_grad():
                    points = self.depth_estimator.infer(rgb, force_projection=False)["points"].permute(2, 0, 1)
            if points is not None:
                height, width = condition.image.shape[-2:]
                if points.ndim != 3 or points.shape[0] != 3:
                    raise ValueError("pointmaps must have shape (3, height, width)")
                # OpenCV camera axes to the PyTorch3D camera axes the model was trained in (x left, y up).
                points = points.float() * points.new_tensor([-1.0, -1.0, 1.0]).reshape(3, 1, 1)
                points = F.interpolate(points[None], size=(height, width), mode="nearest")[0]
            items.append(
                preprocess_sam3d_condition(
                    condition.image,
                    condition.mask,
                    points,
                    image_size=self.image_encoder.config.image_size,
                )
            )
        return {name: torch.stack([item[name] for item in items]).to(self._execution_device) for name in items[0]}

    @staticmethod
    def encode_views(encoder: Dinov2WithRegistersModel, views: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream ``Dino`` tokens of ``(batch, 2, channels, height, width)`` views in ``[0, 1]``, one pass per view.

        ``"normed"`` holds the normed class and patch tokens, ``"prenorm"`` the pre-norm tokens (registers included)
        under an unparameterized layer norm.
        """

        size = encoder.config.image_size
        views = views.to(encoder.dtype)
        mean, std = (
            views.new_tensor(value).reshape(1, 3, 1, 1) for value in (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        )
        normed, prenorm = [], []
        for pixels in views.unbind(1):
            if pixels.shape[-2:] != (size, size):
                pixels = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False)
            output = encoder((pixels.expand(-1, 3, -1, -1) - mean) / std, output_hidden_states=True)
            tokens, hidden_states = output.last_hidden_state, output.hidden_states[-1]
            normed.append(torch.cat([tokens[:, :1], tokens[:, 1 + encoder.config.num_register_tokens :]], dim=1))
            prenorm.append(F.layer_norm(hidden_states, hidden_states.shape[-1:]))
        return {"normed": torch.stack(normed, dim=1), "prenorm": torch.stack(prenorm, dim=1)}

    def prepare_sparse_structure_latents(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw the shape and layout noise in upstream's order (sorted modality names) or validate given latents."""

        model = self.sparse_structure_flow_model
        shapes = model.latent_shapes(batch_size)
        if latents is None:
            latents = {
                name: randn_tensor(shapes[name], generator=generator, device=self._execution_device, dtype=model.dtype)
                for name in sorted(shapes)
            }
        elif {name: tuple(value.shape) for name, value in latents.items()} != shapes:
            raise ValueError(f"sparse_structure_latents must have shapes {shapes}")
        return {name: latents[name].to(self._execution_device, model.dtype) for name in shapes}

    def _flow_timestep(
        self, scheduler: TrellisFlowEulerScheduler, timestep: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        # SAM 3D integrates from noise at t=0 to data at t=1; the TRELLIS scheduler runs t'=1-t.
        return scheduler.model_timestep(1 - timestep, batch_size, device=self._execution_device)

    def sample_sparse_structure(
        self,
        latents: dict[str, torch.Tensor],
        conditional: torch.Tensor,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_interval: tuple[float, float],
        rescale_t: float,
    ) -> dict[str, torch.Tensor]:
        """Euler-integrate shape and layout together; guidance applies to every modality inside the interval."""

        model, scheduler = self.sparse_structure_flow_model, self.sparse_structure_scheduler
        unconditional = torch.zeros_like(conditional)
        names = tuple(latents)
        sizes = [latents[name][0].numel() for name in names]
        sample = torch.cat([latents[name].flatten(1) for name in names], dim=1)
        scheduler.set_timesteps(num_inference_steps, device=sample.device, rescale_t=rescale_t)
        for timestep in scheduler.timesteps:
            model_timestep = self._flow_timestep(scheduler, timestep, sample.shape[0])
            current = {name: value.reshape(latents[name].shape) for name, value in zip(names, sample.split(sizes, 1))}
            velocity = model(current, model_timestep, conditional).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unguided = model(current, model_timestep, unconditional).sample
                velocity = {
                    name: scheduler.apply_guidance(velocity[name], unguided[name], guidance_scale) for name in names
                }
            flat_velocity = torch.cat([velocity[name].flatten(1) for name in names], dim=1)
            sample = scheduler.step(-flat_velocity, timestep, sample).prev_sample
        return {name: value.reshape(latents[name].shape) for name, value in zip(names, sample.split(sizes, 1))}

    def extract_sparse_structure(self, shape_latents: torch.Tensor) -> tuple[tuple[SparseVoxelAsset, ...], list[int]]:
        """Decode occupancy, keep its surface voxels, and halve structures too large for the mesh decoder."""

        resolution = self.sparse_structure_flow_model.config.resolution
        dense = shape_latents.permute(0, 2, 1).reshape(shape_latents.shape[0], -1, *(resolution,) * 3)
        structures, factors = [], []
        for asset in self.sparse_structure_decoder.decode_to_sparse_voxels(dense):
            coordinates = _prune_interior(F.pad(asset.coordinates, (1, 0)))[:, 1:]
            coordinates, factor = _downsample_structure(coordinates)
            structures.append(
                dataclasses.replace(
                    asset, coordinates=coordinates, features=asset.features.new_ones(coordinates.shape[0], 1)
                )
            )
            factors.append(factor)
        return tuple(structures), factors

    def decode_layout(
        self,
        pose: dict[str, torch.Tensor],
        *,
        pointmap_scale: torch.Tensor | None = None,
        pointmap_shift: torch.Tensor | None = None,
        downsample_factors: Sequence[int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode the ``ScaleShiftInvariant`` layout into ``rotation`` (3x3), ``translation`` and uniform ``scale``.

        Points map into the PyTorch3D camera frame as ``scale * local @ rotation + translation`` (upstream's row
        convention, with ``rotation`` the matrix of its ``rotation`` quaternion), in the pointmap's units.
        """

        batch_size = pose["rotation"].shape[0]
        rotation = _rotation_from_6d(pose["rotation"][:, 0].float())
        scene_scale = torch.ones(batch_size, 3) if pointmap_scale is None else pointmap_scale.float()
        scene_shift = torch.zeros(batch_size, 3) if pointmap_shift is None else pointmap_shift.float()
        scene_scale, scene_shift = scene_scale.to(rotation.device), scene_shift.to(rotation.device)
        factors = rotation.new_tensor(downsample_factors or [1] * batch_size)
        scale = (pose["scale"][:, 0].float().exp() * scene_scale).mean(dim=-1) * factors
        translation = pose["translation"][:, 0].float() * scene_scale + scene_shift
        return [
            {"rotation": rotation[index], "translation": translation[index], "scale": scale[index]}
            for index in range(batch_size)
        ]

    @staticmethod
    def _place(asset: Object3D, layout: dict[str, torch.Tensor], metric: bool) -> Object3D:
        """Attach the layout as the asset transform: local Z-up object to the PyTorch3D camera frame.

        The transform is expressed in the asset's Z-up convention, so after
        ``to_coordinate_system(RIGHT_HANDED_Y_UP)`` world coordinates are exactly PyTorch3D camera coordinates.
        """

        object_to_camera = torch.eye(4, dtype=torch.float64)
        object_to_camera[:3, :3] = layout["scale"].double() * layout["rotation"].double().T
        object_to_camera[:3, 3] = layout["translation"].double()
        z_up_to_y_up = torch.eye(4, dtype=torch.float64)
        z_up_to_y_up[:3, :3] = coordinate_change_matrix(
            CoordinateSystem.RIGHT_HANDED_Z_UP, CoordinateSystem.RIGHT_HANDED_Y_UP, dtype=torch.float64
        )
        transform = (z_up_to_y_up.T @ object_to_camera).to(device=asset.transform.device, dtype=asset.transform.dtype)
        metadata = {
            **asset.metadata,
            "family": "sam3d_objects",
            "layout": {
                "rotation": layout["rotation"].tolist(),
                "translation": layout["translation"].tolist(),
                "scale": float(layout["scale"]),
                "frame": "pytorch3d_camera",
                "metric": metric,
            },
        }
        return dataclasses.replace(asset, transform=transform @ asset.transform, metadata=metadata)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        pointmap: torch.Tensor | Sequence[torch.Tensor] | None = None,
        formats: tuple[str, ...] | list[str] | None = None,
        sparse_structure_num_inference_steps: int = 25,
        slat_num_inference_steps: int = 25,
        sparse_structure_guidance_scale: float = 7.0,
        slat_guidance_scale: float = 1.0,
        guidance_interval: tuple[float, float] = (0.5, 1.0),
        sparse_structure_rescale_t: float = 3.0,
        slat_rescale_t: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        sparse_structure_latents: dict[str, torch.Tensor] | None = None,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[Object3D, ...], Latent3DOutput | None]:
        """Reconstruct the masked object of each image with the released sampler settings.

        ``pointmap`` holds camera-space points per image (``(3, H, W)``, OpenCV axes, any units, non-finite where
        invalid), for example from MoGe as upstream uses. Without one the pointmap tokens are dropped, as in training,
        and the layout is in the model's normalized, scale-free units (``metadata["layout"]["metric"]`` is ``False``).

        ``formats`` picks ``"sparse_structure"``, ``"gaussian"`` and/or ``"mesh"`` (default: every loaded decoder).
        ``guidance_interval`` is in the scheduler's ``1 - t`` time, so the default guides the noisier half of each
        trajectory like upstream's ``[0, 500]``.
        """

        decoders = {"gaussian": self.gaussian_decoder, "mesh": self.mesh_decoder}
        formats = tuple(
            formats
            or tuple(name for name, decoder in decoders.items() if decoder is not None)
            or ("sparse_structure",)
        )
        if len(set(formats)) != len(formats) or set(formats) - {"sparse_structure", *decoders}:
            raise ValueError("formats must contain unique values from 'sparse_structure', 'gaussian' and 'mesh'")
        for name in set(formats) & set(decoders):
            if decoders[name] is None or self.slat_flow_model is None:
                raise RuntimeError(f"format {name!r} needs the structured-latent stage and {name}_decoder")

        inputs = self.preprocess(image, pointmap)
        batch_size = inputs["images"].shape[0]
        image_tokens = self.encode_views(self.image_encoder, inputs["images"])
        mask_tokens = self.encode_views(self.image_encoder, inputs["masks"])

        def condition(conditioner: Sam3DConditionEmbedder, pointmaps: torch.Tensor | None = None) -> torch.Tensor:
            features = "prenorm" if conditioner.config.prenorm_features else "normed"
            return conditioner(image_tokens[features], mask_tokens[features], pointmaps).embeddings

        conditional = condition(self.sparse_structure_conditioner, inputs.get("pointmaps"))
        latents = self.sample_sparse_structure(
            self.prepare_sparse_structure_latents(batch_size, generator=generator, latents=sparse_structure_latents),
            conditional,
            num_inference_steps=sparse_structure_num_inference_steps,
            guidance_scale=sparse_structure_guidance_scale,
            guidance_interval=guidance_interval,
            rescale_t=sparse_structure_rescale_t,
        )
        structures, factors = self.extract_sparse_structure(latents["shape"])
        layouts = self.decode_layout(
            latents,
            pointmap_scale=inputs.get("pointmap_scale"),
            pointmap_shift=inputs.get("pointmap_shift"),
            downsample_factors=factors,
        )
        objects: list[tuple[int, Object3D]] = []
        if "sparse_structure" in formats:
            objects.extend(enumerate(structures))
        if set(formats) & set(decoders):
            slat = self.sample_slat(
                structures,
                condition(self.slat_conditioner),
                generator=generator,
                num_inference_steps=slat_num_inference_steps,
                guidance_scale=slat_guidance_scale,
                guidance_interval=guidance_interval,
                rescale_t=slat_rescale_t,
            )
            for name in formats:
                if name in decoders:
                    objects.extend(enumerate(decoders[name](slat).assets))
        metric = "pointmaps" in inputs
        placed = tuple(self._place(asset, layouts[index], metric) for index, asset in objects)
        latent_output = (
            Latent3DOutput(latents=latents["shape"], metadata={"family": "sam3d_objects", "stage": "sparse_structure"})
            if return_latents
            else None
        )
        self.maybe_free_model_hooks()
        if not return_dict:
            return placed, latent_output
        return Object3DPipelineOutput(objects=placed, latents=latent_output)

    def sample_slat(
        self,
        structures: tuple[SparseVoxelAsset, ...],
        conditional: torch.Tensor,
        *,
        generator: torch.Generator | list[torch.Generator] | None,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_interval: tuple[float, float],
        rescale_t: float,
    ) -> TrellisSparseTensor:
        """Sample and denormalize structured latents on the extracted voxels."""

        model, scheduler = self.slat_flow_model, self.slat_scheduler
        coordinates = TrellisSparseTensor.from_sparse_voxel_assets(structures).coordinates.to(self._execution_device)
        generators = generator if isinstance(generator, list) else [generator] * len(structures)
        noise = [
            randn_tensor(
                (int((coordinates[:, 0] == index).sum()), model.config.in_channels),
                generator=item_generator,
                device=self._execution_device,
                dtype=model.dtype,
            )
            for index, item_generator in enumerate(generators)
        ]
        latents = TrellisSparseTensor(coordinates, torch.cat(noise))
        unconditional = torch.zeros_like(conditional)
        scheduler.set_timesteps(num_inference_steps, device=coordinates.device, rescale_t=rescale_t)
        for timestep in scheduler.timesteps:
            model_timestep = self._flow_timestep(scheduler, timestep, latents.batch_size)
            velocity = model(latents, model_timestep, conditional).sample
            if guidance_interval[0] <= float(timestep) <= guidance_interval[1]:
                unguided = model(latents, model_timestep, unconditional).sample
                velocity = scheduler.apply_guidance(velocity, unguided, guidance_scale)
            latents = scheduler.step(velocity.replace(-velocity.features), timestep, latents).prev_sample
        return latents.denormalize(
            latents.features.new_tensor(self.config.slat_mean), latents.features.new_tensor(self.config.slat_std)
        )


__all__ = ["SAM_LICENSE_WARNING", "Sam3DObjectsPipeline", "preprocess_sam3d_condition"]
