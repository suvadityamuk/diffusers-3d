from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..objects import CoordinateSystem, SparseVoxelAsset
from ._optional import load_explicit_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability

SPCONV_BATCH_INDICES = "spconv_batch_indices"


class SpconvBackend:
    """Narrow, explicit bridge between ``SparseVoxelAsset`` and spconv tensors."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        root_module = load_explicit_backend(
            "spconv",
            "spconv",
            (BackendCapability.SPARSE_COMPUTE,),
            device=device,
            dtype=dtype,
            differentiable=True,
            registry=registry,
        )
        pytorch_module = getattr(root_module, "pytorch", None)
        if pytorch_module is None:
            try:
                pytorch_module = importlib.import_module("spconv.pytorch")
            except ImportError as error:
                raise RuntimeError("the selected spconv build does not provide spconv.pytorch") from error
        if not callable(getattr(pytorch_module, "SparseConvTensor", None)):
            raise RuntimeError("the selected spconv build does not expose SparseConvTensor")
        self._spconv = pytorch_module

    def to_spconv_tensors(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        spatial_shape: Sequence[int],
        batch_size: int,
    ) -> Any:
        """Build a native tensor from TRELLIS ``[batch, x, y, z]`` coordinates."""

        if not isinstance(features, torch.Tensor) or features.ndim != 2 or not features.is_floating_point():
            raise ValueError("features must be a rank-two floating-point tensor")
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.ndim != 2
            or coordinates.shape[1] != 4
            or coordinates.is_floating_point()
        ):
            raise ValueError("coordinates must have integer shape (active_voxels, 4)")
        if features.shape[0] == 0 or features.shape[0] != coordinates.shape[0]:
            raise ValueError("features and coordinates must share a non-zero active-voxel count")
        if features.device != coordinates.device:
            raise ValueError("features and coordinates must be on the same device")
        if features.device.type != self.device.type or (
            self.device.index is not None and features.device.index != self.device.index
        ):
            raise ValueError(f"SpconvBackend was configured for {self.device}, got {features.device}")
        if features.dtype is not self.dtype:
            raise ValueError(f"SpconvBackend was configured for {self.dtype}, got {features.dtype}")
        if bool((coordinates < 0).any()):
            raise ValueError("spconv coordinates must be non-negative")
        spatial_shape = tuple(int(value) for value in spatial_shape)
        if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
            raise ValueError("spatial_shape must contain three positive integers")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if int(coordinates[:, 0].max().item()) >= batch_size:
            raise ValueError("batch_size must cover all coordinate batch indices")
        shape_tensor = coordinates.new_tensor(spatial_shape)
        if bool((coordinates[:, 1:] >= shape_tensor).any()):
            raise ValueError("coordinates fall outside spatial_shape")
        return self._spconv.SparseConvTensor(
            features,
            coordinates.to(dtype=torch.int32).contiguous(),
            spatial_shape,
            batch_size,
        )

    def to_spconv(
        self,
        voxels: SparseVoxelAsset,
        *,
        spatial_shape: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> Any:
        """Convert coordinates to the TRELLIS/spconv ``[batch, x, y, z]`` layout."""

        if type(voxels) is not SparseVoxelAsset:
            raise TypeError("voxels must be an exact SparseVoxelAsset")
        voxels.validate(expensive=True)
        if bool((voxels.coordinates < 0).any()):
            raise ValueError("spconv coordinates must be non-negative")

        stored_batch_indices = voxels.extras.get(SPCONV_BATCH_INDICES)
        if stored_batch_indices is None:
            batch_indices = torch.zeros(
                voxels.coordinates.shape[0],
                device=voxels.coordinates.device,
                dtype=torch.int32,
            )
        else:
            if stored_batch_indices.ndim not in (1, 2):
                raise ValueError(f"{SPCONV_BATCH_INDICES} must be rank one or have a trailing singleton dimension")
            batch_indices = stored_batch_indices.reshape(-1)
            if batch_indices.shape[0] != voxels.coordinates.shape[0] or batch_indices.is_floating_point():
                raise ValueError(f"{SPCONV_BATCH_INDICES} must contain one integer per active voxel")
            if bool((batch_indices < 0).any()):
                raise ValueError("spconv batch indices must be non-negative")
            batch_indices = batch_indices.to(dtype=torch.int32)

        coordinates = torch.cat(
            [batch_indices[:, None], voxels.coordinates.to(dtype=torch.int32)],
            dim=1,
        ).contiguous()
        if spatial_shape is None:
            stored_shape = voxels.metadata.get("spconv_spatial_shape")
            if stored_shape is not None:
                spatial_shape = stored_shape
            else:
                spatial_shape = tuple(int(value) + 1 for value in voxels.coordinates.amax(dim=0).tolist())
        spatial_shape = tuple(int(value) for value in spatial_shape)
        if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
            raise ValueError("spatial_shape must contain three positive integers")
        inferred_batch_size = int(batch_indices.max().item()) + 1
        if batch_size is None:
            batch_size = int(voxels.metadata.get("spconv_batch_size", inferred_batch_size))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < inferred_batch_size:
            raise ValueError("batch_size must be an integer covering all batch indices")
        return self.to_spconv_tensors(
            voxels.features,
            coordinates,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )

    def from_spconv(
        self,
        native_tensor: Any,
        *,
        voxel_size: float | torch.Tensor | None = 1.0,
        grid_transform: torch.Tensor | None = None,
        coordinate_system: CoordinateSystem | str = CoordinateSystem.RIGHT_HANDED_Z_UP,
        transform: torch.Tensor | None = None,
        template: SparseVoxelAsset | None = None,
    ) -> SparseVoxelAsset:
        """Convert one spconv tensor without dropping batch or grid-shape metadata."""

        features = getattr(native_tensor, "features", None)
        indices = getattr(native_tensor, "indices", None)
        spatial_shape = getattr(native_tensor, "spatial_shape", None)
        batch_size = getattr(native_tensor, "batch_size", None)
        if not isinstance(features, torch.Tensor) or not isinstance(indices, torch.Tensor):
            raise TypeError("native_tensor must expose tensor features and indices")
        if indices.ndim != 2 or indices.shape[1] != 4 or indices.is_floating_point():
            raise ValueError("spconv indices must have integer shape (active_voxels, 4)")
        if features.ndim != 2 or features.shape[0] != indices.shape[0]:
            raise ValueError("spconv features must have shape (active_voxels, channels)")
        if not isinstance(spatial_shape, Sequence) or len(spatial_shape) != 3:
            raise ValueError("spconv spatial_shape must contain three values")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("spconv batch_size must be a positive integer")
        if features.device != indices.device:
            raise ValueError("spconv features and indices must be on the same device")
        if features.device.type != self.device.type or (
            self.device.index is not None and features.device.index != self.device.index
        ):
            raise ValueError(f"SpconvBackend was configured for {self.device}, got {features.device}")
        if features.dtype is not self.dtype:
            raise ValueError(f"SpconvBackend was configured for {self.dtype}, got {features.dtype}")

        if template is not None:
            if type(template) is not SparseVoxelAsset:
                raise TypeError("template must be an exact SparseVoxelAsset")
            template.validate(expensive=True)
            if template.coordinates.shape[0] != indices.shape[0]:
                raise ValueError("template must have the same active-voxel count as native_tensor")
            if not torch.equal(template.coordinates.to(dtype=indices.dtype), indices[:, 1:]):
                raise ValueError("template coordinates must match native_tensor indices")
            template_batch_indices = template.extras.get(SPCONV_BATCH_INDICES)
            if template_batch_indices is None:
                template_batch_indices = torch.zeros_like(indices[:, 0])
            if not torch.equal(template_batch_indices.reshape(-1).to(dtype=indices.dtype), indices[:, 0]):
                raise ValueError("template batch indices must match native_tensor indices")
            voxel_size = template.voxel_size
            grid_transform = template.grid_transform
            coordinate_system = template.coordinate_system
            transform = template.transform
            extras = dict(template.extras)
            metadata = dict(template.metadata)
            semantic_labels = template.semantic_labels
        else:
            extras = {}
            metadata = {}
            semantic_labels = None
        if (voxel_size is None) == (grid_transform is None):
            raise ValueError("exactly one of voxel_size or grid_transform must be provided")
        resolved_transform = (
            torch.eye(4, device=features.device, dtype=features.dtype) if transform is None else transform
        )
        extras[SPCONV_BATCH_INDICES] = indices[:, 0].to(dtype=torch.int64)
        metadata.update(
            {
                "spconv_spatial_shape": [int(value) for value in spatial_shape],
                "spconv_batch_size": batch_size,
            }
        )
        return SparseVoxelAsset(
            coordinates=indices[:, 1:].to(dtype=torch.int64),
            features=features,
            voxel_size=voxel_size,
            grid_transform=grid_transform,
            transform=resolved_transform,
            coordinate_system=coordinate_system,
            semantic_labels=semantic_labels,
            extras=extras,
            metadata=metadata,
        )

    def sparse_compute(
        self,
        voxels: SparseVoxelAsset,
        *,
        operation: str,
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> SparseVoxelAsset:
        """Run the bridge's deliberately small identity or feature-linear operation."""

        native_tensor = self.to_spconv(voxels)
        if operation == "identity":
            if parameters:
                raise ValueError("identity does not accept parameters")
        elif operation == "linear":
            values = {} if parameters is None else dict(parameters)
            unknown = set(values).difference({"weight", "bias"})
            if unknown or "weight" not in values:
                raise ValueError("linear requires weight and accepts only an optional bias")
            features = F.linear(native_tensor.features, values["weight"], values.get("bias"))
            replace_feature = getattr(native_tensor, "replace_feature", None)
            if callable(replace_feature):
                native_tensor = replace_feature(features)
            else:
                native_tensor = self._spconv.SparseConvTensor(
                    features,
                    native_tensor.indices,
                    native_tensor.spatial_shape,
                    native_tensor.batch_size,
                )
        else:
            raise ValueError("SpconvBackend supports only 'identity' and 'linear' operations")
        return self.from_spconv(
            native_tensor,
            template=voxels,
        )


__all__ = ["SPCONV_BATCH_INDICES", "SpconvBackend"]
