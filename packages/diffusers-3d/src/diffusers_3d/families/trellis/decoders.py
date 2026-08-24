# Portions of this file are derived from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for object-native decoder outputs.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ModelMixin  # noqa: F401 - required by external-component loading
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn

from ...backends import BACKEND_REGISTRY, BackendCapability
from ...execution.metadata import ContributionStatus, ReviewStatus
from ...execution.models import Object3DModel
from ...objects import CoordinateSystem, GaussianSplatAsset, Object3DKind, SparseVoxelAsset
from .models import TrellisAbsolutePositionEmbedder, TrellisAttention, TrellisFeedForwardNet, TrellisLayerNorm32
from .sparse import TrellisSparseTensor, trellis_grid_transform


@dataclass
class TrellisSparseStructureDecoderOutput(BaseOutput):
    """Dense occupancy logits produced by the sparse-structure decoder."""

    sample: torch.Tensor


@dataclass
class TrellisGaussianDecoderOutput(BaseOutput):
    """One canonical Gaussian splat asset per sparse batch item."""

    assets: tuple[GaussianSplatAsset, ...]


class TrellisChannelLayerNorm32(TrellisLayerNorm32):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dimensions = hidden_states.ndim
        hidden_states = hidden_states.permute(0, *range(2, dimensions), 1).contiguous()
        hidden_states = super().forward(hidden_states)
        return hidden_states.permute(0, dimensions - 1, *range(1, dimensions - 1)).contiguous()


def _decoder_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        if channels % 32:
            raise ValueError("group normalization channels must be divisible by 32")
        return nn.GroupNorm(32, channels)
    if norm_type == "layer":
        return TrellisChannelLayerNorm32(channels)
    raise ValueError("norm_type must be 'group' or 'layer'")


class TrellisResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None, norm_type: str = "layer") -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = channels if out_channels is None else out_channels
        self.norm1 = _decoder_norm(norm_type, channels)
        self.norm2 = _decoder_norm(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(hidden_states)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return residual + self.skip_connection(hidden_states)


class TrellisUpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "conv") -> None:
        super().__init__()
        if mode not in {"conv", "nearest"}:
            raise ValueError("upsample mode must be 'conv' or 'nearest'")
        if mode == "nearest" and in_channels != out_channels:
            raise ValueError("nearest upsampling requires matching channel counts")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mode == "nearest":
            return F.interpolate(hidden_states, scale_factor=2, mode="nearest")
        hidden_states = self.conv(hidden_states)
        batch_size, channels, depth, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size,
            channels // 8,
            2,
            2,
            2,
            depth,
            height,
            width,
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return hidden_states.reshape(
            batch_size,
            channels // 8,
            depth * 2,
            height * 2,
            width * 2,
        )


class TrellisSparseStructureDecoder(Object3DModel):
    """Faithful dense TRELLIS sparse-structure occupancy decoder."""

    family_id = "trellis"
    component_role = "sparse_structure_decoder"
    supported_object_kinds = (Object3DKind.SPARSE_VOXEL,)
    required_backends = ()
    contribution_status = ContributionStatus.REVIEWED_PACKAGE
    review_status = ReviewStatus.REVIEWED
    _no_split_modules = ["TrellisResBlock3d"]

    @register_to_config
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: tuple[int, ...] | list[int] = (512, 128, 32),
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        if min(out_channels, latent_channels, num_res_blocks, num_res_blocks_middle, *channels) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[TrellisResBlock3d(channels[0], channels[0], norm_type) for _ in range(num_res_blocks_middle)]
        )
        blocks: list[nn.Module] = []
        for index, channel_count in enumerate(channels):
            blocks.extend(TrellisResBlock3d(channel_count, channel_count, norm_type) for _ in range(num_res_blocks))
            if index < len(channels) - 1:
                blocks.append(TrellisUpsampleBlock3d(channel_count, channels[index + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.out_layer = nn.Sequential(
            _decoder_norm(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )
        if use_fp16:
            self.middle_block.to(dtype=torch.float16)
            self.blocks.to(dtype=torch.float16)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "out_channels": 1,
            "latent_channels": 8,
            "num_res_blocks": 2,
            "num_res_blocks_middle": 2,
            "channels": [512, 128, 32],
            "norm_type": "layer",
            "use_fp16": True,
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        return {
            "out_channels": 1,
            "latent_channels": 2,
            "num_res_blocks": 1,
            "num_res_blocks_middle": 1,
            "channels": [8, 4],
            "norm_type": "layer",
            "use_fp16": False,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> TrellisSparseStructureDecoderOutput | tuple[torch.Tensor]:
        if hidden_states.ndim != 5 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must have shape (batch, {self.latent_channels}, depth, height, width)")
        output_dtype = hidden_states.dtype
        hidden_states = self.input_layer(hidden_states)
        inner_dtype = torch.float16 if self.use_fp16 else torch.float32
        hidden_states = hidden_states.to(dtype=inner_dtype)
        hidden_states = self.middle_block(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.out_layer(hidden_states.to(dtype=output_dtype))
        if not return_dict:
            return (hidden_states,)
        return TrellisSparseStructureDecoderOutput(sample=hidden_states)

    def decode_to_sparse_voxels(self, hidden_states: torch.Tensor) -> tuple[SparseVoxelAsset, ...]:
        """Threshold released occupancy logits at zero and emit native Z-up assets."""

        logits = self(hidden_states).sample
        if logits.shape[1] != 1:
            raise ValueError("SparseVoxelAsset conversion requires a one-channel occupancy decoder")
        resolution = logits.shape[2]
        if logits.shape[2:] != (resolution, resolution, resolution):
            raise ValueError("decoded occupancy grid must be cubic")
        assets = []
        for batch_index in range(logits.shape[0]):
            coordinates = torch.argwhere(logits[batch_index, 0] > 0).to(dtype=torch.int64)
            if coordinates.shape[0] == 0:
                raise ValueError(f"decoded sparse structure is empty for batch item {batch_index}")
            features = logits[batch_index, 0][tuple(coordinates.unbind(dim=1))].unsqueeze(1)
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
                        "family": "trellis",
                        "representation": "sparse_structure",
                        "resolution": resolution,
                        "occupancy_threshold": 0.0,
                    },
                )
            )
        return tuple(assets)


class TrellisSparseTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        use_rope: bool,
        qk_rms_norm: bool,
    ) -> None:
        super().__init__()
        self.norm1 = TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = TrellisLayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = TrellisAttention(
            channels,
            num_heads,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = TrellisFeedForwardNet(channels, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_indices: torch.Tensor,
        coordinates: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for batch_index in range(batch_size):
            positions = torch.nonzero(batch_indices == batch_index, as_tuple=False).reshape(-1)
            batch_states = hidden_states[positions].unsqueeze(0)
            batch_coordinates = coordinates[positions].unsqueeze(0)
            batch_states = batch_states + self.attn(
                self.norm1(batch_states),
                indices=batch_coordinates,
            )
            batch_states = batch_states + self.mlp(self.norm2(batch_states))
            output = output.index_copy(0, positions, batch_states.squeeze(0))
        return output


def _radical_inverse(base: int, value: int) -> float:
    result = 0.0
    inverse_base = 1.0 / base
    inverse_power = inverse_base
    while value > 0:
        result += (value % base) * inverse_power
        value //= base
        inverse_power *= inverse_base
    return result


def _hammersley_3d(index: int, count: int) -> tuple[float, float, float]:
    return index / count, _radical_inverse(2, index), _radical_inverse(3, index)


class TrellisSLatGaussianDecoder(Object3DModel):
    """Portable full-attention SLAT Gaussian parameter decoder."""

    family_id = "trellis"
    component_role = "slat-gaussian-decoder"
    supported_object_kinds = (Object3DKind.GAUSSIAN_SPLAT,)
    required_backends = ("spconv",)
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED
    _supports_gradient_checkpointing = True
    _no_split_modules = ["TrellisSparseTransformerBlock"]
    _repeated_blocks = ["TrellisSparseTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        resolution: int = 64,
        model_channels: int = 768,
        latent_channels: int = 8,
        num_blocks: int = 12,
        num_heads: int | None = 12,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        del window_size
        if attn_mode != "full":
            BACKEND_REGISTRY.select(
                BackendCapability.SPARSE_COMPUTE,
                name="spconv",
                device="cuda",
                dtype="float16" if use_fp16 else "float32",
                differentiable=True,
            )
            raise NotImplementedError(
                "official TRELLIS windowed sparse Gaussian decoding requires a separately tested production sparse "
                "backend; attn_mode='full' is the backend-free tiny path"
            )
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        if representation_config is None:
            representation_config = self.production_representation_config()
        representation_config = dict(representation_config)
        required_representation_keys = {
            "lr",
            "perturb_offset",
            "voxel_size",
            "num_gaussians",
            "3d_filter_kernel_size",
            "scaling_bias",
            "opacity_bias",
            "scaling_activation",
        }
        missing = required_representation_keys.difference(representation_config)
        if missing:
            raise ValueError(f"representation_config is missing keys: {sorted(missing)}")
        resolved_heads = model_channels // num_head_channels if num_heads is None else num_heads
        if resolved_heads <= 0 or model_channels % resolved_heads:
            raise ValueError("model_channels must be divisible by num_heads")
        num_gaussians = int(representation_config["num_gaussians"])
        if min(resolution, model_channels, latent_channels, num_blocks, num_gaussians) <= 0:
            raise ValueError("decoder dimensions must be positive")

        self.resolution = resolution
        self.model_channels = model_channels
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks
        self.num_heads = resolved_heads
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.rep_config = representation_config
        self.gradient_checkpointing = use_checkpoint

        if pe_mode == "ape":
            self.pos_embedder = TrellisAbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(latent_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                TrellisSparseTransformerBlock(
                    model_channels,
                    resolved_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=pe_mode == "rope",
                    qk_rms_norm=qk_rms_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_channels = num_gaussians * 14
        self.out_layer = nn.Linear(model_channels, self.out_channels)
        perturbation = torch.tensor(
            [_hammersley_3d(index, num_gaussians) for index in range(num_gaussians)],
            dtype=torch.float32,
        )
        perturbation = (perturbation * 2 - 1) / float(representation_config["voxel_size"])
        self.register_buffer("offset_perturbation", torch.atanh(perturbation))
        self._initialize_weights()
        if use_fp16:
            self.blocks.to(dtype=torch.float16)

    @staticmethod
    def production_representation_config() -> dict[str, Any]:
        return {
            "lr": {
                "_xyz": 1.0,
                "_features_dc": 1.0,
                "_opacity": 1.0,
                "_scaling": 1.0,
                "_rotation": 0.1,
            },
            "perturb_offset": True,
            "voxel_size": 1.5,
            "num_gaussians": 32,
            "2d_filter_kernel_size": 0.1,
            "3d_filter_kernel_size": 9e-4,
            "scaling_bias": 4e-3,
            "opacity_bias": 0.1,
            "scaling_activation": "softplus",
        }

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 64,
            "model_channels": 768,
            "latent_channels": 8,
            "num_blocks": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "attn_mode": "swin",
            "window_size": 8,
            "use_fp16": True,
            "representation_config": cls.production_representation_config(),
        }

    @classmethod
    def tiny_config(cls) -> dict[str, Any]:
        representation_config = cls.production_representation_config()
        representation_config["num_gaussians"] = 2
        representation_config["perturb_offset"] = False
        return {
            "resolution": 8,
            "model_channels": 16,
            "latent_channels": 4,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "attn_mode": "full",
            "window_size": 2,
            "use_fp16": False,
            "representation_config": representation_config,
        }

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _to_assets(
        self,
        hidden_states: TrellisSparseTensor,
        parameters: torch.Tensor,
    ) -> tuple[GaussianSplatAsset, ...]:
        count = int(self.rep_config["num_gaussians"])
        learning_rates = self.rep_config["lr"]
        assets = []
        for batch_index in range(hidden_states.batch_size):
            mask = hidden_states.coordinates[:, 0] == batch_index
            coordinates = hidden_states.coordinates[mask, 1:].to(dtype=parameters.dtype)
            values = parameters[mask].reshape(parameters[mask].shape[0], count, 14)
            raw_xyz = values[..., 0:3] * float(learning_rates["_xyz"])
            if bool(self.rep_config["perturb_offset"]):
                raw_xyz = raw_xyz + self.offset_perturbation.to(dtype=raw_xyz.dtype)
            offsets = torch.tanh(raw_xyz) / self.resolution * 0.5 * float(self.rep_config["voxel_size"])
            means = (coordinates[:, None, :] + 0.5) / self.resolution + offsets - 0.5
            features_dc = values[..., 3:6] * float(learning_rates["_features_dc"])
            raw_scaling = values[..., 6:9] * float(learning_rates["_scaling"])
            raw_rotation = values[..., 9:13] * float(learning_rates["_rotation"])
            raw_opacity = values[..., 13:14] * float(learning_rates["_opacity"])

            scaling_bias = float(self.rep_config["scaling_bias"])
            activation = self.rep_config["scaling_activation"]
            if activation == "exp":
                scales = torch.exp(raw_scaling + math.log(scaling_bias))
            elif activation == "softplus":
                inverse_bias = scaling_bias + math.log(-math.expm1(-scaling_bias))
                scales = F.softplus(raw_scaling + inverse_bias)
            else:
                raise ValueError("scaling_activation must be 'exp' or 'softplus'")
            kernel_size = float(self.rep_config["3d_filter_kernel_size"])
            scales = torch.sqrt(scales.square() + kernel_size**2)
            rotation_bias = raw_rotation.new_tensor([1.0, 0.0, 0.0, 0.0])
            quaternions = F.normalize(raw_rotation + rotation_bias, dim=-1)
            opacity_bias = float(self.rep_config["opacity_bias"])
            opacity_logits = raw_opacity + math.log(opacity_bias / (1 - opacity_bias))

            assets.append(
                GaussianSplatAsset(
                    means=means.flatten(0, 1),
                    log_scales=scales.log().flatten(0, 1),
                    quaternions_wxyz=quaternions.flatten(0, 1),
                    opacity_logits=opacity_logits.flatten(0, 1),
                    sh_coefficients=features_dc.flatten(0, 1).unsqueeze(1),
                    active_sh_degree=0,
                    coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
                    extras={
                        "trellis_raw_xyz": values[..., 0:3].flatten(0, 1),
                        "trellis_raw_scaling": values[..., 6:9].flatten(0, 1),
                        "trellis_raw_rotation": values[..., 9:13].flatten(0, 1),
                        "trellis_raw_opacity": values[..., 13:14].flatten(0, 1),
                    },
                    metadata={
                        "family": "trellis",
                        "representation": "gaussian",
                        "resolution": self.resolution,
                        "num_gaussians_per_voxel": count,
                    },
                )
            )
        return tuple(assets)

    def forward(
        self,
        hidden_states: TrellisSparseTensor,
        *,
        return_dict: bool = True,
    ) -> TrellisGaussianDecoderOutput | tuple[tuple[GaussianSplatAsset, ...]]:
        if not isinstance(hidden_states, TrellisSparseTensor):
            raise TypeError("hidden_states must be a TrellisSparseTensor")
        if hidden_states.channels != self.latent_channels:
            raise ValueError(f"hidden_states must have {self.latent_channels} feature channels")
        features = self.input_layer(hidden_states.features)
        if self.pe_mode == "ape":
            features = features + self.pos_embedder(hidden_states.coordinates[:, 1:]).to(features)
        inner_dtype = torch.float16 if self.use_fp16 else torch.float32
        features = features.to(dtype=inner_dtype)
        batch_indices = hidden_states.coordinates[:, 0]
        coordinates = hidden_states.coordinates[:, 1:]
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                features = self._gradient_checkpointing_func(
                    block,
                    features,
                    batch_indices,
                    coordinates,
                    hidden_states.batch_size,
                )
            else:
                features = block(features, batch_indices, coordinates, hidden_states.batch_size)
        features = F.layer_norm(features, features.shape[-1:])
        parameters = self.out_layer(features.to(dtype=hidden_states.dtype))
        assets = self._to_assets(hidden_states, parameters)
        if not return_dict:
            return (assets,)
        return TrellisGaussianDecoderOutput(assets=assets)


class TrellisSLatMeshDecoder(Object3DModel):
    """Capability gate for the not-yet-ported TRELLIS SLAT mesh field decoder."""

    family_id = "trellis"
    component_role = "slat-mesh-decoder"
    supported_object_kinds = (Object3DKind.MESH,)
    required_backends = ("kaolin", "spconv")
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED

    @register_to_config
    def __init__(
        self,
        resolution: int = 64,
        model_channels: int = 768,
        latent_channels: int = 8,
        num_blocks: int = 12,
        num_heads: int | None = 12,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = True,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        del num_head_channels
        if min(resolution, model_channels, latent_channels, num_blocks, window_size) <= 0:
            raise ValueError("decoder dimensions must be positive")
        if num_heads is not None and (num_heads <= 0 or model_channels % num_heads):
            raise ValueError("model_channels must be divisible by num_heads")
        if attn_mode not in {"full", "shift_window", "shift_sequence", "shift_order", "swin"}:
            raise ValueError("unsupported sparse attention mode")
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        self.resolution = resolution
        self.representation_config = {} if representation_config is None else dict(representation_config)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 64,
            "model_channels": 768,
            "latent_channels": 8,
            "num_blocks": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "attn_mode": "swin",
            "window_size": 8,
            "use_fp16": True,
            "representation_config": {"use_color": True},
        }

    def forward(self, hidden_states: TrellisSparseTensor) -> None:
        del hidden_states
        BACKEND_REGISTRY.select(
            BackendCapability.SURFACE_EXTRACTION,
            name="kaolin",
            device="cuda",
            dtype="float32",
            differentiable=True,
        )
        raise NotImplementedError(
            "the TRELLIS SLAT mesh field network is not ported; the permissive Kaolin FlexiCubes adapter alone is "
            "not sufficient to decode an official mesh checkpoint"
        )


class TrellisSLatRadianceFieldDecoder(Object3DModel):
    """Explicit future type for unsupported TRELLIS radiance-field decoding."""

    family_id = "trellis"
    component_role = "slat-radiance-field-decoder"
    supported_object_kinds = ()
    required_backends = ("diffoctreerast",)
    contribution_status = ContributionStatus.EXPERIMENTAL_HUB
    review_status = ReviewStatus.UNREVIEWED

    @register_to_config
    def __init__(
        self,
        resolution: int = 64,
        model_channels: int = 768,
        latent_channels: int = 8,
        num_blocks: int = 12,
        num_heads: int | None = 12,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = True,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        del num_head_channels
        if min(resolution, model_channels, latent_channels, num_blocks, window_size) <= 0:
            raise ValueError("decoder dimensions must be positive")
        if num_heads is not None and (num_heads <= 0 or model_channels % num_heads):
            raise ValueError("model_channels must be divisible by num_heads")
        if attn_mode not in {"full", "shift_window", "shift_sequence", "shift_order", "swin"}:
            raise ValueError("unsupported sparse attention mode")
        if pe_mode not in {"ape", "rope"}:
            raise ValueError("pe_mode must be 'ape' or 'rope'")
        self.resolution = resolution
        self.representation_config = {} if representation_config is None else dict(representation_config)

    @classmethod
    def production_config(cls) -> dict[str, Any]:
        return {
            "resolution": 64,
            "model_channels": 768,
            "latent_channels": 8,
            "num_blocks": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "attn_mode": "swin",
            "window_size": 8,
            "use_fp16": True,
            "representation_config": {"rank": 16, "dim": 8},
        }

    def forward(self, hidden_states: TrellisSparseTensor) -> None:
        del hidden_states
        raise NotImplementedError(
            "TRELLIS radiance fields do not yet have a package-native Object3D type and are intentionally unsupported"
        )


__all__ = [
    "TrellisGaussianDecoderOutput",
    "TrellisSLatGaussianDecoder",
    "TrellisSLatMeshDecoder",
    "TrellisSLatRadianceFieldDecoder",
    "TrellisSparseStructureDecoder",
    "TrellisSparseStructureDecoderOutput",
]
