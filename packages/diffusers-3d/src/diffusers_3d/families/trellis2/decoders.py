# Portions of this file reproduce decoder contracts from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Production sparse convolution and O-Voxel extension source is not vendored.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import BACKEND_REGISTRY, BackendCapability, ovoxel_grid_transform
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import CoordinateSystem, Object3DKind, OVoxelAsset, SparseVoxelAsset
from ..trellis.decoders import TrellisSparseStructureDecoder
from ..trellis.sparse import TrellisSparseTensor, trellis_grid_transform


class Trellis2SparseStructureDecoder(TrellisSparseStructureDecoder):
    """Exact TRELLIS/TRELLIS.2 dense sparse-structure decoder architecture.

    TRELLIS.2 references the released TRELLIS image-large decoder unchanged.
    The inherited state layout is therefore intentional and exact; only family
    metadata and native output metadata differ.
    """

    family_id = "trellis2"
    component_role = "sparse_structure_decoder"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    def decode_to_sparse_voxels(
        self,
        hidden_states: torch.Tensor,
        *,
        target_resolution: int | None = None,
    ) -> tuple[SparseVoxelAsset, ...]:
        logits = self(hidden_states).sample
        if logits.shape[1] != 1:
            raise ValueError("SparseVoxelAsset conversion requires a one-channel occupancy decoder")
        decoded_resolution = logits.shape[2]
        if logits.shape[2:] != (decoded_resolution, decoded_resolution, decoded_resolution):
            raise ValueError("decoded occupancy grid must be cubic")
        resolution = decoded_resolution if target_resolution is None else target_resolution
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ValueError("target_resolution must be a positive integer or None")
        if resolution > decoded_resolution or decoded_resolution % resolution:
            raise ValueError("target_resolution must evenly divide the decoded occupancy resolution")
        occupancy = logits > 0
        if resolution != decoded_resolution:
            pooling_ratio = decoded_resolution // resolution
            occupancy = F.max_pool3d(occupancy.float(), pooling_ratio, pooling_ratio, 0) > 0.5
            feature_grid = occupancy.float()
        else:
            feature_grid = logits
        assets = []
        for batch_index in range(logits.shape[0]):
            coordinates = torch.argwhere(occupancy[batch_index, 0]).to(dtype=torch.int64)
            if coordinates.shape[0] == 0:
                raise ValueError(f"decoded sparse structure is empty for batch item {batch_index}")
            features = feature_grid[batch_index, 0][tuple(coordinates.unbind(dim=1))].unsqueeze(1)
            assets.append(
                SparseVoxelAsset(
                    coordinates=coordinates,
                    features=features,
                    grid_transform=trellis_grid_transform(
                        resolution,
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                    coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                    metadata={
                        "family": "trellis2",
                        "representation": "sparse_structure",
                        "resolution": resolution,
                        "decoded_resolution": decoded_resolution,
                        "occupancy_threshold": 0.0,
                        "decoder_checkpoint_semantics": "trellis-image-large-exact-reuse",
                    },
                )
            )
        return tuple(assets)


@dataclass
class Trellis2ShapeDecoderOutput(BaseOutput):
    assets: tuple[OVoxelAsset, ...]


@dataclass
class Trellis2PBRDecoderOutput(BaseOutput):
    assets: tuple[OVoxelAsset, ...]


def _production_sparse_gate(component: str, *, dtype: str) -> None:
    BACKEND_REGISTRY.select(
        BackendCapability.SPARSE_COMPUTE,
        name="flex_gemm",
        device="cuda",
        dtype=dtype,
        differentiable=True,
    )
    BACKEND_REGISTRY.select(
        BackendCapability.NATIVE_REPRESENTATION,
        name="o_voxel",
        device="cuda",
        dtype="float16" if dtype == "float16" else "float32",
        differentiable=False,
    )
    raise NotImplementedError(
        f"production {component} requires the released sparse UNet, pinned FlexGEMM, and compiled O-Voxel runtime; "
        "only portable_tiny=True has a backend-free implementation and no official checkpoint parity is claimed"
    )


class Trellis2ShapeDualGridDecoder(Object3DModel):
    """Experimental tiny dual-grid decoder with the released seven-channel contract."""

    family_id = "trellis2"
    component_role = "shape_slat_decoder"
    supported_object_kinds = (Object3DKind.O_VOXEL,)
    required_backends = ("flex_gemm", "o_voxel")
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED

    @register_to_config
    def __init__(
        self,
        resolution: int = 256,
        model_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        latent_channels: int = 32,
        num_blocks: Sequence[int] = (4, 16, 8, 4, 0),
        block_type: Sequence[str] = (
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
        ),
        up_block_type: Sequence[str] = (
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
        ),
        block_args: Sequence[dict[str, Any]] = ({}, {}, {}, {}, {}),
        voxel_margin: float = 0.5,
        use_fp16: bool = True,
        portable_tiny: bool = False,
    ) -> None:
        super().__init__()
        if not portable_tiny:
            _production_sparse_gate("TRELLIS.2 shape dual-grid decoding", dtype="float16" if use_fp16 else "float32")
        if (
            not isinstance(resolution, int)
            or isinstance(resolution, bool)
            or resolution <= 0
            or not isinstance(latent_channels, int)
            or latent_channels <= 0
        ):
            raise ValueError("resolution and latent_channels must be positive integers")
        if voxel_margin < 0:
            raise ValueError("voxel_margin must be non-negative")
        model_channels = tuple(int(value) for value in model_channels)
        num_blocks = tuple(int(value) for value in num_blocks)
        block_type = tuple(block_type)
        up_block_type = tuple(up_block_type)
        block_args = tuple(dict(value) for value in block_args)
        if not model_channels or min(model_channels) <= 0 or len(num_blocks) != len(model_channels):
            raise ValueError("model_channels and aligned num_blocks must be non-empty and positive")
        if len(block_type) != len(model_channels) or len(block_args) != len(model_channels):
            raise ValueError("block_type and block_args must align with model_channels")
        if len(up_block_type) != len(model_channels) - 1:
            raise ValueError("up_block_type must have one entry per upsampling stage")
        self.resolution = resolution
        self.latent_channels = latent_channels
        self.voxel_margin = float(voxel_margin)
        self.output_layer = nn.Linear(latent_channels, 7)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 256,
            "model_channels": [1024, 512, 256, 128, 64],
            "latent_channels": 32,
            "num_blocks": [4, 16, 8, 4, 0],
            "block_type": ["SparseConvNeXtBlock3d"] * 5,
            "up_block_type": ["SparseResBlockC2S3d"] * 4,
            "block_args": [{}, {}, {}, {}, {}],
            "voxel_margin": 0.5,
            "use_fp16": True,
            "portable_tiny": False,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 8,
            "model_channels": [8],
            "latent_channels": 4,
            "num_blocks": [0],
            "block_type": ["SparseConvNeXtBlock3d"],
            "up_block_type": [],
            "block_args": [{}],
            "voxel_margin": 0.0,
            "use_fp16": False,
            "portable_tiny": True,
        }

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        *,
        return_dict: bool = True,
    ) -> Trellis2ShapeDecoderOutput | tuple[tuple[OVoxelAsset, ...]]:
        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        if hidden_states.channels != self.latent_channels:
            raise ValueError(f"hidden_states must have {self.latent_channels} channels")
        if bool((hidden_states.coordinates[:, 1:] >= self.resolution).any()):
            raise ValueError("shape sparse coordinates fall outside the configured resolution")
        parameters = self.output_layer(hidden_states.features)
        assets = []
        for batch_index in range(hidden_states.batch_size):
            mask = hidden_states.coordinates[:, 0] == batch_index
            coordinates = hidden_states.coordinates[mask, 1:].to(dtype=torch.int64)
            values = parameters[mask]
            dual_vertices = (1 + 2 * self.voxel_margin) * torch.sigmoid(values[:, 0:3]) - self.voxel_margin
            intersected = values[:, 3:6] > 0
            split_weights = F.softplus(values[:, 6:7])
            count = coordinates.shape[0]
            assets.append(
                OVoxelAsset(
                    active_coordinates=coordinates,
                    dual_grid_vertex_offsets=dual_vertices,
                    intersection_data=intersected,
                    split_weights=split_weights,
                    base_color=torch.zeros(count, 3, device=values.device, dtype=values.dtype),
                    metallic=torch.zeros(count, 1, device=values.device, dtype=values.dtype),
                    roughness=torch.full((count, 1), 0.5, device=values.device, dtype=values.dtype),
                    opacity=torch.ones(count, 1, device=values.device, dtype=values.dtype),
                    normals=F.pad(
                        torch.zeros(count, 2, device=values.device, dtype=values.dtype),
                        (0, 1),
                        value=1,
                    ),
                    emissive=torch.zeros(count, 3, device=values.device, dtype=values.dtype),
                    grid_transform=ovoxel_grid_transform(
                        self.resolution,
                        device=values.device,
                        dtype=values.dtype,
                    ),
                    coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                    metadata={
                        "family": "trellis2",
                        "representation": "o_voxel",
                        "stage": "shape_decoder_tiny",
                        "resolution": [self.resolution] * 3,
                        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        "official_checkpoint_parity": False,
                    },
                )
            )
        output = tuple(assets)
        if not return_dict:
            return (output,)
        return Trellis2ShapeDecoderOutput(assets=output)


class Trellis2PBRSparseDecoder(Object3DModel):
    """Experimental tiny sparse PBR decoder preserving every OVoxelAsset channel."""

    family_id = "trellis2"
    component_role = "tex_slat_decoder"
    supported_object_kinds = (Object3DKind.O_VOXEL,)
    required_backends = ("flex_gemm", "o_voxel")
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED

    @register_to_config
    def __init__(
        self,
        out_channels: int = 6,
        model_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        latent_channels: int = 32,
        num_blocks: Sequence[int] = (4, 16, 8, 4, 0),
        block_type: Sequence[str] = (
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
            "SparseConvNeXtBlock3d",
        ),
        up_block_type: Sequence[str] = (
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
            "SparseResBlockC2S3d",
        ),
        block_args: Sequence[dict[str, Any]] = ({}, {}, {}, {}, {}),
        pred_subdiv: bool = False,
        use_fp16: bool = True,
        portable_tiny: bool = False,
        channel_layout: Sequence[str] = (
            "base_color",
            "metallic",
            "roughness",
            "alpha",
        ),
    ) -> None:
        super().__init__()
        if not portable_tiny:
            _production_sparse_gate("TRELLIS.2 PBR sparse decoding", dtype="float16" if use_fp16 else "float32")
        if not isinstance(latent_channels, int) or latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer")
        widths = {
            "base_color": 3,
            "metallic": 1,
            "roughness": 1,
            "alpha": 1,
            "normal": 3,
            "emissive": 3,
        }
        channel_layout = tuple(channel_layout)
        if (
            not channel_layout
            or len(set(channel_layout)) != len(channel_layout)
            or set(channel_layout).difference(widths)
        ):
            raise ValueError(f"channel_layout must contain unique values from {sorted(widths)}")
        expected_channels = sum(widths[name] for name in channel_layout)
        if out_channels != expected_channels:
            raise ValueError(f"out_channels must be {expected_channels} for channel_layout={channel_layout}")
        if pred_subdiv:
            raise ValueError("the released PBR decoder requires pred_subdiv=False and shape-guided coordinates")
        model_channels = tuple(int(value) for value in model_channels)
        num_blocks = tuple(int(value) for value in num_blocks)
        if not model_channels or len(num_blocks) != len(model_channels):
            raise ValueError("model_channels and num_blocks must be non-empty aligned sequences")
        if len(tuple(block_type)) != len(model_channels) or len(tuple(block_args)) != len(model_channels):
            raise ValueError("block_type and block_args must align with model_channels")
        if len(tuple(up_block_type)) != len(model_channels) - 1:
            raise ValueError("up_block_type must have one entry per upsampling stage")
        self.latent_channels = latent_channels
        self.channel_layout = channel_layout
        self.output_layer = nn.Linear(latent_channels, out_channels)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "out_channels": 6,
            "model_channels": [1024, 512, 256, 128, 64],
            "latent_channels": 32,
            "num_blocks": [4, 16, 8, 4, 0],
            "block_type": ["SparseConvNeXtBlock3d"] * 5,
            "up_block_type": ["SparseResBlockC2S3d"] * 4,
            "block_args": [{}, {}, {}, {}, {}],
            "pred_subdiv": False,
            "use_fp16": True,
            "portable_tiny": False,
            "channel_layout": ["base_color", "metallic", "roughness", "alpha"],
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "out_channels": 12,
            "model_channels": [8],
            "latent_channels": 4,
            "num_blocks": [0],
            "block_type": ["SparseConvNeXtBlock3d"],
            "up_block_type": [],
            "block_args": [{}],
            "pred_subdiv": False,
            "use_fp16": False,
            "portable_tiny": True,
            "channel_layout": [
                "base_color",
                "metallic",
                "roughness",
                "alpha",
                "normal",
                "emissive",
            ],
        }

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        shape_assets: Sequence[OVoxelAsset],
        *,
        return_dict: bool = True,
    ) -> Trellis2PBRDecoderOutput | tuple[tuple[OVoxelAsset, ...]]:
        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        if hidden_states.channels != self.latent_channels:
            raise ValueError(f"hidden_states must have {self.latent_channels} channels")
        shape_assets = tuple(shape_assets)
        if len(shape_assets) != hidden_states.batch_size or any(
            type(asset) is not OVoxelAsset for asset in shape_assets
        ):
            raise ValueError("shape_assets must contain one exact OVoxelAsset per sparse batch item")
        raw = self.output_layer(hidden_states.features)
        channels: dict[str, torch.Tensor] = {}
        offset = 0
        widths = {"base_color": 3, "metallic": 1, "roughness": 1, "alpha": 1, "normal": 3, "emissive": 3}
        for name in self.channel_layout:
            width = widths[name]
            value = raw[:, offset : offset + width]
            channels[name] = torch.tanh(value) if name == "normal" else torch.sigmoid(value)
            offset += width
        assets = []
        for batch_index, shape in enumerate(shape_assets):
            mask = hidden_states.coordinates[:, 0] == batch_index
            coordinates = hidden_states.coordinates[mask, 1:].to(dtype=shape.active_coordinates.dtype)
            if not torch.equal(coordinates, shape.active_coordinates):
                raise ValueError("texture SLAT coordinates must align exactly with shape O-Voxel coordinates")
            count = coordinates.shape[0]
            base_color = (
                channels.get(
                    "base_color",
                    torch.zeros(count, 3, device=raw.device, dtype=raw.dtype),
                )[mask]
                if "base_color" in channels
                else torch.zeros(count, 3, device=raw.device, dtype=raw.dtype)
            )
            metallic = (
                channels["metallic"][mask]
                if "metallic" in channels
                else torch.zeros(count, 1, device=raw.device, dtype=raw.dtype)
            )
            roughness = (
                channels["roughness"][mask]
                if "roughness" in channels
                else torch.full((count, 1), 0.5, device=raw.device, dtype=raw.dtype)
            )
            opacity = (
                channels["alpha"][mask]
                if "alpha" in channels
                else torch.ones(count, 1, device=raw.device, dtype=raw.dtype)
            )
            if "normal" in channels:
                normals = channels["normal"][mask]
                norm = torch.linalg.vector_norm(normals.float(), dim=1, keepdim=True)
                fallback = F.pad(
                    torch.zeros(count, 2, device=raw.device, dtype=raw.dtype),
                    (0, 1),
                    value=1,
                )
                normals = torch.where(norm > 1e-8, normals / norm.to(normals.dtype), fallback)
            else:
                normals = F.pad(
                    torch.zeros(count, 2, device=raw.device, dtype=raw.dtype),
                    (0, 1),
                    value=1,
                )
            emissive = (
                channels["emissive"][mask]
                if "emissive" in channels
                else torch.zeros(count, 3, device=raw.device, dtype=raw.dtype)
            )
            assets.append(
                OVoxelAsset(
                    active_coordinates=shape.active_coordinates,
                    dual_grid_vertex_offsets=shape.dual_grid_vertex_offsets,
                    dual_grid_topology=shape.dual_grid_topology,
                    intersection_data=shape.intersection_data,
                    split_weights=shape.split_weights,
                    base_color=base_color,
                    metallic=metallic,
                    roughness=roughness,
                    opacity=opacity,
                    normals=normals,
                    emissive=emissive,
                    transform=shape.transform,
                    grid_transform=shape.grid_transform,
                    coordinate_system=shape.coordinate_system,
                    extras=shape.extras,
                    metadata={
                        **shape.metadata,
                        "stage": "pbr_decoder_tiny",
                        "pbr_channel_layout": list(self.channel_layout),
                        "full_pbr_asset_channels": True,
                        "official_checkpoint_parity": False,
                    },
                )
            )
        output = tuple(assets)
        if not return_dict:
            return (output,)
        return Trellis2PBRDecoderOutput(assets=output)


__all__ = [
    "Trellis2PBRDecoderOutput",
    "Trellis2PBRSparseDecoder",
    "Trellis2ShapeDecoderOutput",
    "Trellis2ShapeDualGridDecoder",
    "Trellis2SparseStructureDecoder",
]
