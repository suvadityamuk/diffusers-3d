from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..objects import SparseVoxelAsset
from ._optional import load_explicit_backend
from .defaults import BACKEND_REGISTRY
from .registry import BackendRegistry
from .types import BackendCapability

FLEX_GEMM_SOURCE_URL = "https://github.com/JeffreyXiang/FlexGEMM.git"
FLEX_GEMM_BATCH_INDICES = "flex_gemm_batch_indices"


class FlexGemmBackend:
    """Narrow FlexGEMM sparse-convolution and 3D grid-sampling adapter.

    FlexGEMM is source-built and upstream TRELLIS.2 does not pin it. Callers
    therefore have to record both the source revision and build identity.
    """

    def __init__(
        self,
        *,
        source_revision: str,
        build_id: str,
        source_url: str = FLEX_GEMM_SOURCE_URL,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        registry: BackendRegistry = BACKEND_REGISTRY,
    ) -> None:
        if source_url != FLEX_GEMM_SOURCE_URL:
            raise ValueError(f"FlexGEMM source_url must be {FLEX_GEMM_SOURCE_URL!r}")
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError("source_revision must record the audited FlexGEMM commit")
        if not isinstance(build_id, str) or not build_id.strip():
            raise ValueError("build_id must record the PyTorch/Triton/compiler build")
        self.source_url = source_url
        self.source_revision = source_revision
        self.build_id = build_id
        self.device = torch.device(device)
        self.dtype = dtype
        self._module = load_explicit_backend(
            "flex_gemm",
            "flex_gemm",
            (BackendCapability.SPARSE_COMPUTE,),
            device=device,
            dtype=dtype,
            differentiable=True,
            registry=registry,
        )
        declared_revision = getattr(self._module, "__source_revision__", None)
        declared_build = getattr(self._module, "__build_id__", None)
        if declared_revision is None or declared_build is None:
            raise RuntimeError(
                "the selected FlexGEMM wrapper must expose __source_revision__ and __build_id__ attestations"
            )
        if declared_revision != source_revision:
            raise RuntimeError(
                f"loaded FlexGEMM declares revision {declared_revision!r}, expected {source_revision!r}"
            )
        if declared_build != build_id:
            raise RuntimeError(f"loaded FlexGEMM declares build {declared_build!r}, expected {build_id!r}")

    def _validate_voxels(self, voxels: SparseVoxelAsset) -> torch.Tensor:
        if type(voxels) is not SparseVoxelAsset:
            raise TypeError("voxels must be an exact SparseVoxelAsset")
        voxels.validate(expensive=True)
        if voxels.device.type != self.device.type or (
            self.device.index is not None and voxels.device.index != self.device.index
        ):
            raise ValueError(f"FlexGemmBackend was configured for {self.device}, got {voxels.device}")
        if voxels.features.dtype is not self.dtype:
            raise ValueError(f"FlexGemmBackend was configured for {self.dtype}, got {voxels.features.dtype}")
        batch_indices = voxels.extras.get(FLEX_GEMM_BATCH_INDICES)
        if batch_indices is None:
            batch_indices = torch.zeros(
                voxels.coordinates.shape[0],
                1,
                dtype=voxels.coordinates.dtype,
                device=voxels.device,
            )
        else:
            batch_indices = batch_indices.reshape(-1, 1)
            if batch_indices.shape[0] != voxels.coordinates.shape[0] or batch_indices.is_floating_point():
                raise ValueError(f"{FLEX_GEMM_BATCH_INDICES} must contain one integer per active voxel")
        return torch.cat([batch_indices, voxels.coordinates], dim=1).to(dtype=torch.int32).contiguous()

    def sparse_compute(
        self,
        voxels: SparseVoxelAsset,
        *,
        operation: str,
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> SparseVoxelAsset:
        """Run the official submanifold-convolution primitive only."""

        coordinates = self._validate_voxels(voxels)
        if operation != "submanifold_conv3d":
            raise ValueError("FlexGemmBackend supports only 'submanifold_conv3d'")
        values = {} if parameters is None else dict(parameters)
        unknown = set(values).difference({"weight", "bias", "spatial_shape", "dilation"})
        if unknown or "weight" not in values:
            raise ValueError("submanifold_conv3d requires weight and accepts bias, spatial_shape, and dilation")
        weight = values["weight"]
        bias = values.get("bias")
        spatial_shape_value = values.get("spatial_shape")
        if spatial_shape_value is None:
            spatial_shape = tuple(int(item) + 1 for item in voxels.coordinates.amax(dim=0).tolist())
        else:
            spatial_shape = tuple(int(item) for item in spatial_shape_value.reshape(-1).tolist())
        batch_size = int(coordinates[:, 0].max().item()) + 1
        dilation_value = values.get("dilation")
        dilation: Sequence[int] = (
            (1, 1, 1) if dilation_value is None else tuple(int(item) for item in dilation_value.reshape(-1).tolist())
        )
        spconv = getattr(getattr(self._module, "ops", None), "spconv", None)
        function = getattr(spconv, "sparse_submanifold_conv3d", None)
        if not callable(function):
            raise RuntimeError("the selected FlexGEMM build does not expose ops.spconv.sparse_submanifold_conv3d")
        output = function(
            voxels.features,
            coordinates,
            torch.Size([batch_size, voxels.features.shape[1], *spatial_shape]),
            weight,
            bias,
            None,
            tuple(dilation),
        )
        features = output[0] if isinstance(output, tuple) else output
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError("FlexGEMM sparse convolution returned an invalid feature tensor")
        return SparseVoxelAsset(
            coordinates=voxels.coordinates,
            features=features,
            voxel_size=voxels.voxel_size,
            grid_transform=voxels.grid_transform,
            transform=voxels.transform,
            coordinate_system=voxels.coordinate_system,
            semantic_labels=voxels.semantic_labels,
            extras=voxels.extras,
            metadata={
                **voxels.metadata,
                "flex_gemm_source_revision": self.source_revision,
                "flex_gemm_build_id": self.build_id,
            },
        )

    def grid_sample_3d(
        self,
        voxels: SparseVoxelAsset,
        grid: torch.Tensor,
        *,
        shape: Sequence[int] | None = None,
        mode: str = "trilinear",
    ) -> torch.Tensor:
        """Sample aligned sparse features with FlexGEMM's released grid sampler."""

        coordinates = self._validate_voxels(voxels)
        if not isinstance(grid, torch.Tensor) or not grid.is_floating_point() or grid.shape[-1] != 3:
            raise ValueError("grid must be a floating-point tensor ending in XYZ coordinates")
        if grid.device != voxels.device:
            raise ValueError("grid and voxels must be on the same device")
        if shape is None:
            spatial_shape = tuple(int(item) + 1 for item in voxels.coordinates.amax(dim=0).tolist())
            shape = (int(coordinates[:, 0].max().item()) + 1, voxels.features.shape[1], *spatial_shape)
        function = getattr(getattr(getattr(self._module, "ops", None), "grid_sample", None), "grid_sample_3d", None)
        if not callable(function):
            raise RuntimeError("the selected FlexGEMM build does not expose ops.grid_sample.grid_sample_3d")
        return function(
            voxels.features,
            coordinates,
            shape=torch.Size(tuple(int(item) for item in shape)),
            grid=grid,
            mode=mode,
        )


__all__ = ["FLEX_GEMM_BATCH_INDICES", "FLEX_GEMM_SOURCE_URL", "FlexGemmBackend"]
