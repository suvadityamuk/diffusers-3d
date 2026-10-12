# Portions of this file reproduce decoder contracts from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Sparse convolution is reimplemented in plain PyTorch; the O-Voxel extension source is not vendored.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import ovoxel_grid_transform
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import CoordinateSystem, Object3DKind, OVoxelAsset, SparseVoxelAsset
from ..trellis.decoders import TrellisSparseStructureDecoder
from ..trellis.models import TrellisLayerNorm32
from ..trellis.sparse import TrellisSparseTensor, trellis_grid_transform
from ..trellis.sparse_ops import SparseConv3d, channel_to_spatial, sparse_subdivide, submanifold_neighbors


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
        pooling: int | None = None,
    ) -> tuple[SparseVoxelAsset, ...]:
        """Decode occupancy; ``target_resolution`` or ``pooling`` (the released ``ratio``) max-pools the grid."""

        if target_resolution is not None and pooling is not None:
            raise ValueError("pass either target_resolution or pooling, not both")
        logits = self(hidden_states).sample
        if logits.shape[1] != 1:
            raise ValueError("SparseVoxelAsset conversion requires a one-channel occupancy decoder")
        decoded_resolution = logits.shape[2]
        if logits.shape[2:] != (decoded_resolution, decoded_resolution, decoded_resolution):
            raise ValueError("decoded occupancy grid must be cubic")
        if pooling is not None:
            if not isinstance(pooling, int) or isinstance(pooling, bool) or pooling <= 0:
                raise ValueError("pooling must be a positive integer")
            target_resolution = decoded_resolution // pooling
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
    """Decoded O-Voxel shapes plus the subdivision masks chosen at each upsampling stage.

    ``subdivisions[i]`` is the boolean ``(voxels_at_stage_i, 8)`` mask the ``i``-th upsampling block used;
    the texture decoder is run with the same masks so its voxels line up with the shape voxels.
    """

    assets: tuple[OVoxelAsset, ...]
    subdivisions: tuple[torch.Tensor, ...]


@dataclass
class Trellis2PBRDecoderOutput(BaseOutput):
    assets: tuple[OVoxelAsset, ...]


class Trellis2SparseConvNeXtBlock3d(nn.Module):
    """Released ``SparseConvNeXtBlock3d``: 3x3x3 submanifold conv, layer norm, MLP, residual."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm = TrellisLayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        neighbors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.conv(coordinates, features, neighbors=neighbors)
        return features + self.mlp(self.norm(hidden_states))


class Trellis2SparseResBlockC2S3d(nn.Module):
    """Released ``SparseResBlockC2S3d``: doubles the resolution by moving channels into child voxels.

    ``conv1`` produces eight channel blocks per voxel; :func:`sparse_subdivide` keeps the children selected
    by the subdivision mask (predicted by ``to_subdiv`` or supplied by the caller) and each child takes its
    own block. The residual path splits the parent's features the same way and repeats them to width.
    """

    def __init__(self, channels: int, out_channels: int, *, pred_subdiv: bool = True) -> None:
        super().__init__()
        if channels % 8 or out_channels % (channels // 8):
            raise ValueError(
                "channel-to-spatial blocks need channels divisible by 8 and out_channels by channels // 8"
            )
        self.channels = channels
        self.out_channels = out_channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = TrellisLayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = TrellisLayerNorm32(out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels * 8, 3)
        self.conv2 = SparseConv3d(out_channels, out_channels, 3)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        if pred_subdiv:
            self.to_subdiv = nn.Linear(channels, 8)

    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        *,
        subdivision: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(child_coordinates, child_features, subdivision_mask, child_neighbors)``."""

        if self.pred_subdiv:
            subdivision = self.to_subdiv(features) > 0
        elif subdivision is None:
            raise ValueError("this upsampling block predicts no subdivision; pass the shape decoder's mask")
        elif subdivision.shape != (features.shape[0], 8) or subdivision.dtype != torch.bool:
            raise ValueError("subdivision must be a boolean (voxels, 8) mask aligned with the current stage")
        hidden_states = self.conv1(coordinates, F.silu(self.norm1(features)), neighbors=neighbors)
        child_coordinates, parent_index, child_index = sparse_subdivide(coordinates, subdivision)
        if child_coordinates.shape[0] == 0:
            raise ValueError("subdivision selected no child voxels")
        hidden_states = channel_to_spatial(hidden_states, parent_index, child_index)
        residual = channel_to_spatial(features, parent_index, child_index)
        residual = residual.repeat_interleave(self.out_channels // (self.channels // 8), dim=1)
        child_neighbors = submanifold_neighbors(child_coordinates)
        hidden_states = self.conv2(child_coordinates, F.silu(self.norm2(hidden_states)), neighbors=child_neighbors)
        return child_coordinates, hidden_states + residual, subdivision, child_neighbors


_BLOCK_TYPES = {"SparseConvNeXtBlock3d": Trellis2SparseConvNeXtBlock3d}
_UP_BLOCK_TYPES = {"SparseResBlockC2S3d": Trellis2SparseResBlockC2S3d}


class _Trellis2SparseUnetDecoder(Object3DModel):
    """Shared body of the released ``SparseUnetVaeDecoder``: ``from_latent`` -> stages -> ``output_layer``.

    Stage ``i`` holds ``num_blocks[i]`` residual blocks at ``model_channels[i]`` followed (except for the
    last stage) by one upsampling block into ``model_channels[i + 1]``. State-dict keys match the released
    checkpoints (``blocks.{stage}.{index}.*``).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Trellis2SparseConvNeXtBlock3d", "Trellis2SparseResBlockC2S3d"]

    def _build_unet(
        self,
        *,
        out_channels: int,
        model_channels: Sequence[int],
        latent_channels: int,
        num_blocks: Sequence[int],
        block_type: Sequence[str],
        up_block_type: Sequence[str],
        block_args: Sequence[Mapping[str, Any]],
        pred_subdiv: bool,
        use_fp16: bool,
        use_checkpoint: bool,
    ) -> None:
        model_channels = tuple(int(value) for value in model_channels)
        num_blocks = tuple(int(value) for value in num_blocks)
        block_type = tuple(block_type)
        up_block_type = tuple(up_block_type)
        block_args = tuple(dict(value) for value in block_args)
        if not isinstance(latent_channels, int) or isinstance(latent_channels, bool) or latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer")
        if not model_channels or min(model_channels) <= 0 or len(num_blocks) != len(model_channels):
            raise ValueError("model_channels and aligned num_blocks must be non-empty and positive")
        if min(num_blocks) < 0:
            raise ValueError("num_blocks must be non-negative")
        if len(block_type) != len(model_channels) or len(block_args) != len(model_channels):
            raise ValueError("block_type and block_args must align with model_channels")
        if len(up_block_type) != len(model_channels) - 1:
            raise ValueError("up_block_type must have one entry per upsampling stage")
        unknown = set(block_type).difference(_BLOCK_TYPES) | set(up_block_type).difference(_UP_BLOCK_TYPES)
        if unknown:
            raise ValueError(
                f"unsupported block types {sorted(unknown)}; the released TRELLIS.2 decoders use "
                f"{sorted(_BLOCK_TYPES)} and {sorted(_UP_BLOCK_TYPES)}"
            )
        for args in block_args:
            if set(args).difference({"mlp_ratio", "use_checkpoint"}):
                raise ValueError("block_args may only set mlp_ratio and use_checkpoint")

        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.pred_subdiv = pred_subdiv
        self.gradient_checkpointing = use_checkpoint
        self.output_layer = nn.Linear(model_channels[-1], out_channels)
        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.blocks = nn.ModuleList()
        for stage, channels in enumerate(model_channels):
            mlp_ratio = float(block_args[stage].get("mlp_ratio", 4.0))
            stage_blocks = nn.ModuleList(
                _BLOCK_TYPES[block_type[stage]](channels, mlp_ratio=mlp_ratio) for _ in range(num_blocks[stage])
            )
            if stage < len(model_channels) - 1:
                stage_blocks.append(
                    _UP_BLOCK_TYPES[up_block_type[stage]](channels, model_channels[stage + 1], pred_subdiv=pred_subdiv)
                )
            self.blocks.append(stage_blocks)
        self.num_upsamples = len(model_channels) - 1

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)
        if use_fp16:
            self.blocks.to(dtype=torch.float16)

    def _run_stages(
        self,
        hidden_states: TrellisSparseTensor,
        guide_subdivisions: Sequence[torch.Tensor] | None,
        *,
        num_stages: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run ``from_latent`` and the first ``num_stages`` stages (all of them by default)."""

        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        if hidden_states.channels != self.latent_channels:
            raise ValueError(f"hidden_states must have {self.latent_channels} channels")
        if guide_subdivisions is not None and len(guide_subdivisions) != self.num_upsamples:
            raise ValueError(f"guide_subdivisions must contain one mask per upsampling stage ({self.num_upsamples})")
        inner_dtype = get_parameter_dtype(self.blocks)
        coordinates = hidden_states.coordinates
        features = self.from_latent(hidden_states.features).to(dtype=inner_dtype)
        neighbors = submanifold_neighbors(coordinates)
        subdivisions: list[torch.Tensor] = []
        stages = self.blocks if num_stages is None else self.blocks[:num_stages]
        for stage, stage_blocks in enumerate(stages):
            for block in stage_blocks:
                if isinstance(block, Trellis2SparseResBlockC2S3d):
                    guide = None if guide_subdivisions is None else guide_subdivisions[stage]
                    coordinates, features, subdivision, neighbors = block(
                        coordinates, features, subdivision=guide, neighbors=neighbors
                    )
                    subdivisions.append(subdivision)
                elif torch.is_grad_enabled() and self.gradient_checkpointing:
                    features = self._gradient_checkpointing_func(block, coordinates, features, neighbors)
                else:
                    features = block(coordinates, features, neighbors)
        return coordinates, features, tuple(subdivisions)

    def _run_unet(
        self,
        hidden_states: TrellisSparseTensor,
        guide_subdivisions: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        coordinates, features, subdivisions = self._run_stages(hidden_states, guide_subdivisions)
        features = F.layer_norm(features.to(dtype=hidden_states.dtype), features.shape[-1:])
        return coordinates, self.output_layer(features), subdivisions


class Trellis2ShapeDualGridDecoder(_Trellis2SparseUnetDecoder):
    """Released ``FlexiDualGridVaeDecoder``: shape SLAT -> flexible dual-grid O-Voxels.

    Each upsampling stage predicts which children to keep, so the decoder grows from the 32^3 / 64^3 latent
    grid to the 512 / 1024 output grid on its own. The seven output channels are the dual vertex offset,
    three per-axis intersection logits, and the quad split weight; turning them into a mesh needs the
    compiled O-Voxel runtime (``OVoxelBackend``), the ``.npz`` codec needs nothing.
    """

    family_id = "trellis2"
    component_role = "shape_slat_decoder"
    supported_object_kinds = (Object3DKind.O_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(
        self,
        resolution: int = 256,
        model_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        latent_channels: int = 32,
        num_blocks: Sequence[int] = (4, 16, 8, 4, 0),
        block_type: Sequence[str] = ("SparseConvNeXtBlock3d",) * 5,
        up_block_type: Sequence[str] = ("SparseResBlockC2S3d",) * 4,
        block_args: Sequence[dict[str, Any]] = ({}, {}, {}, {}, {}),
        voxel_margin: float = 0.5,
        use_fp16: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        if voxel_margin < 0:
            raise ValueError("voxel_margin must be non-negative")
        self.resolution = resolution
        self.voxel_margin = float(voxel_margin)
        self._build_unet(
            out_channels=7,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            pred_subdiv=True,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
        )

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
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "resolution": 16,
            "model_channels": [8, 8],
            "latent_channels": 4,
            "num_blocks": [1, 0],
            "block_type": ["SparseConvNeXtBlock3d"] * 2,
            "up_block_type": ["SparseResBlockC2S3d"],
            "block_args": [{}, {}],
            "voxel_margin": 0.5,
            "use_fp16": False,
        }

    @torch.no_grad()
    def upsample_coordinates(self, hidden_states: TrellisSparseTensor, upsample_times: int) -> torch.Tensor:
        """Released ``upsample``: run only the first ``upsample_times`` stages and return the grown coordinates.

        The cascade uses this to turn a low-resolution shape SLAT into the active voxels of the next stage
        without decoding it.
        """

        if not isinstance(upsample_times, int) or not 0 <= upsample_times <= self.num_upsamples:
            raise ValueError(f"upsample_times must be an integer in [0, {self.num_upsamples}]")
        coordinates, _, _ = self._run_stages(hidden_states, None, num_stages=upsample_times)
        return coordinates

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        *,
        resolution: int | None = None,
        return_dict: bool = True,
    ) -> Trellis2ShapeDecoderOutput | tuple[tuple[OVoxelAsset, ...], tuple[torch.Tensor, ...]]:
        """Decode; ``resolution`` is the output grid size (latent resolution times ``2 ** num_upsamples``)."""

        resolution = self.resolution if resolution is None else resolution
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        coordinates, values, subdivisions = self._run_unet(hidden_states, None)
        if bool((coordinates[:, 1:] >= resolution).any()):
            raise ValueError("decoded voxels fall outside the requested resolution")
        dual_vertices = (1 + 2 * self.voxel_margin) * torch.sigmoid(values[:, 0:3]) - self.voxel_margin
        intersected = values[:, 3:6] > 0
        split_weights = F.softplus(values[:, 6:7])
        assets = []
        for batch_index in range(hidden_states.batch_size):
            mask = coordinates[:, 0] == batch_index
            active = coordinates[mask, 1:].to(dtype=torch.int64)
            source = hidden_states.source_assets[batch_index] if hidden_states.source_assets is not None else None
            count = active.shape[0]
            zeros = values.new_zeros(count, 3)
            assets.append(
                OVoxelAsset(
                    active_coordinates=active,
                    dual_grid_vertex_offsets=dual_vertices[mask],
                    intersection_data=intersected[mask],
                    split_weights=split_weights[mask],
                    base_color=zeros,
                    metallic=zeros[:, :1],
                    roughness=values.new_full((count, 1), 0.5),
                    opacity=values.new_ones(count, 1),
                    normals=F.pad(zeros[:, :2], (0, 1), value=1),
                    emissive=zeros,
                    grid_transform=ovoxel_grid_transform(resolution, device=values.device, dtype=values.dtype),
                    transform=source.transform
                    if source is not None
                    else torch.eye(4, device=values.device, dtype=values.dtype),
                    coordinate_system=(
                        source.coordinate_system if source is not None else CoordinateSystem.RIGHT_HANDED_Z_UP
                    ),
                    metadata={
                        **({} if source is None else source.metadata),
                        "family": "trellis2",
                        "representation": "o_voxel",
                        "stage": "shape_decoder",
                        "resolution": [resolution] * 3,
                        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    },
                )
            )
        output = tuple(assets)
        if not return_dict:
            return (output, subdivisions)
        return Trellis2ShapeDecoderOutput(assets=output, subdivisions=subdivisions)


_PBR_CHANNEL_WIDTHS = {"base_color": 3, "metallic": 1, "roughness": 1, "alpha": 1, "normal": 3, "emissive": 3}


class Trellis2PBRSparseDecoder(_Trellis2SparseUnetDecoder):
    """Released texture ``SparseUnetVaeDecoder``: texture SLAT -> per-voxel PBR attributes.

    It predicts no subdivision of its own; the shape decoder's masks are replayed so every texture voxel
    lands on a shape voxel. Outputs are mapped from ``[-1, 1]`` to ``[0, 1]`` exactly like the released
    pipeline (``* 0.5 + 0.5``) and written onto the shape O-Voxels.
    """

    family_id = "trellis2"
    component_role = "tex_slat_decoder"
    supported_object_kinds = (Object3DKind.O_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED

    @register_to_config
    def __init__(
        self,
        out_channels: int = 6,
        model_channels: Sequence[int] = (1024, 512, 256, 128, 64),
        latent_channels: int = 32,
        num_blocks: Sequence[int] = (4, 16, 8, 4, 0),
        block_type: Sequence[str] = ("SparseConvNeXtBlock3d",) * 5,
        up_block_type: Sequence[str] = ("SparseResBlockC2S3d",) * 4,
        block_args: Sequence[dict[str, Any]] = ({}, {}, {}, {}, {}),
        pred_subdiv: bool = False,
        use_fp16: bool = True,
        use_checkpoint: bool = False,
        channel_layout: Sequence[str] = ("base_color", "metallic", "roughness", "alpha"),
    ) -> None:
        super().__init__()
        channel_layout = tuple(channel_layout)
        if (
            not channel_layout
            or len(set(channel_layout)) != len(channel_layout)
            or set(channel_layout).difference(_PBR_CHANNEL_WIDTHS)
        ):
            raise ValueError(f"channel_layout must contain unique values from {sorted(_PBR_CHANNEL_WIDTHS)}")
        expected_channels = sum(_PBR_CHANNEL_WIDTHS[name] for name in channel_layout)
        if out_channels != expected_channels:
            raise ValueError(f"out_channels must be {expected_channels} for channel_layout={channel_layout}")
        if pred_subdiv:
            raise ValueError(
                "the released PBR decoder uses pred_subdiv=False and follows the shape decoder's subdivision"
            )
        self.channel_layout = channel_layout
        self._build_unet(
            out_channels=out_channels,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            pred_subdiv=False,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
        )

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
            "channel_layout": ["base_color", "metallic", "roughness", "alpha"],
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "out_channels": 6,
            "model_channels": [8, 8],
            "latent_channels": 4,
            "num_blocks": [1, 0],
            "block_type": ["SparseConvNeXtBlock3d"] * 2,
            "up_block_type": ["SparseResBlockC2S3d"],
            "block_args": [{}, {}],
            "pred_subdiv": False,
            "use_fp16": False,
            "channel_layout": ["base_color", "metallic", "roughness", "alpha"],
        }

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        shape_assets: Sequence[OVoxelAsset],
        subdivisions: Sequence[torch.Tensor],
        *,
        return_dict: bool = True,
    ) -> Trellis2PBRDecoderOutput | tuple[tuple[OVoxelAsset, ...]]:
        shape_assets = tuple(shape_assets)
        if len(shape_assets) != hidden_states.batch_size or any(
            type(asset) is not OVoxelAsset for asset in shape_assets
        ):
            raise ValueError("shape_assets must contain one exact OVoxelAsset per sparse batch item")
        coordinates, raw, _ = self._run_unet(hidden_states, tuple(subdivisions))
        # The released pipeline maps to [0, 1] without clamping; OVoxelAsset requires the range, so clamp.
        values = (raw * 0.5 + 0.5).clamp(0, 1)
        channels: dict[str, torch.Tensor] = {}
        offset = 0
        for name in self.channel_layout:
            width = _PBR_CHANNEL_WIDTHS[name]
            channels[name] = values[:, offset : offset + width]
            offset += width
        assets = []
        for batch_index, shape in enumerate(shape_assets):
            mask = coordinates[:, 0] == batch_index
            if not torch.equal(
                coordinates[mask, 1:].to(dtype=shape.active_coordinates.dtype), shape.active_coordinates
            ):
                raise ValueError(
                    "texture voxels do not line up with the shape O-Voxels; pass the shape decoder's subdivisions"
                )
            count = shape.active_coordinates.shape[0]
            zeros = values.new_zeros(count, 3)
            normals = (
                channels["normal"][mask] * 2 - 1 if "normal" in channels else F.pad(zeros[:, :2], (0, 1), value=1)
            )
            assets.append(
                OVoxelAsset(
                    active_coordinates=shape.active_coordinates,
                    dual_grid_vertex_offsets=shape.dual_grid_vertex_offsets,
                    dual_grid_topology=shape.dual_grid_topology,
                    intersection_data=shape.intersection_data,
                    split_weights=shape.split_weights,
                    base_color=channels["base_color"][mask] if "base_color" in channels else zeros,
                    metallic=channels["metallic"][mask] if "metallic" in channels else zeros[:, :1],
                    roughness=channels["roughness"][mask]
                    if "roughness" in channels
                    else values.new_full((count, 1), 0.5),
                    opacity=channels["alpha"][mask] if "alpha" in channels else values.new_ones(count, 1),
                    normals=F.normalize(normals, dim=1, eps=1e-8),
                    emissive=channels["emissive"][mask] if "emissive" in channels else zeros,
                    transform=shape.transform,
                    grid_transform=shape.grid_transform,
                    coordinate_system=shape.coordinate_system,
                    extras=shape.extras,
                    metadata={
                        **shape.metadata,
                        "stage": "pbr_decoder",
                        "pbr_channel_layout": list(self.channel_layout),
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
    "Trellis2SparseConvNeXtBlock3d",
    "Trellis2SparseResBlockC2S3d",
    "Trellis2SparseStructureDecoder",
]
