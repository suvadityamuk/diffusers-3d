from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..objects import OVoxelAsset
from .cumesh import CUMESH_SOURCE_URL, CuMeshBackend
from .defaults import BACKEND_REGISTRY
from .flex_gemm import FLEX_GEMM_SOURCE_URL, FlexGemmBackend
from .o_voxel import OVoxelBackend, OVoxelCapability, official_tensors_from_ovoxel_asset
from .registry import BackendRegistry
from .types import BackendCapability, BackendSpec


class Trellis2PBRPostprocessFacade:
    """Explicit gate for the complete O-Voxel-to-PBR-GLB research path.

    Construction and :meth:`requirements` are side-effect free. Only
    :meth:`to_glb` imports or invokes O-Voxel, CuMesh, FlexGEMM, or nvdiffrast.
    """

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        self.registry = registry

    def requirements(
        self,
        *,
        device: str | torch.device = "cuda",
        accept_nvdiffrast_research_license: bool = False,
    ) -> Mapping[str, BackendSpec]:
        if not accept_nvdiffrast_research_license:
            raise ValueError(
                "accept_nvdiffrast_research_license=True is required; nvdiffrast is restricted to "
                "non-commercial research/evaluation under its source license"
            )
        return {
            "o_voxel": self.registry.select(
                BackendCapability.NATIVE_REPRESENTATION,
                name="o_voxel",
                device=device,
                dtype=torch.float32,
                differentiable=False,
            ),
            "cumesh": self.registry.select(
                BackendCapability.GEOMETRY_PROCESSING,
                name="cumesh",
                device=device,
                dtype=torch.float32,
                differentiable=False,
            ),
            "flex_gemm": self.registry.select(
                BackendCapability.SPARSE_COMPUTE,
                name="flex_gemm",
                device=device,
                dtype=torch.float32,
                differentiable=True,
            ),
            "nvdiffrast": self.registry.select(
                BackendCapability.MESH_RASTERIZATION,
                name="nvdiffrast",
                device=device,
                dtype=torch.float32,
                differentiable=True,
                allow_research_only=True,
            ),
        }

    def to_glb(
        self,
        asset: OVoxelAsset,
        *,
        flex_gemm_source_revision: str,
        flex_gemm_build_id: str,
        cumesh_source_revision: str,
        cumesh_build_id: str,
        accept_nvdiffrast_research_license: bool = False,
        device: str | torch.device = "cuda",
        decimation_target: int = 1_000_000,
        texture_size: int = 2048,
        remesh: bool = False,
        postprocess_parameters: Mapping[str, object] | None = None,
    ) -> Any:
        """Run the upstream PBR postprocess only after all four explicit gates."""

        self.requirements(
            device=device,
            accept_nvdiffrast_research_license=accept_nvdiffrast_research_license,
        )
        # Instantiate all permissive runtime adapters to enforce source/build
        # attestations before the restricted renderer can execute.
        FlexGemmBackend(
            source_url=FLEX_GEMM_SOURCE_URL,
            source_revision=flex_gemm_source_revision,
            build_id=flex_gemm_build_id,
            device=device,
            registry=self.registry,
        )
        CuMeshBackend(
            source_url=CUMESH_SOURCE_URL,
            source_revision=cumesh_source_revision,
            build_id=cumesh_build_id,
            device=device,
            registry=self.registry,
        )
        ovoxel = OVoxelBackend(
            device=device,
            accept_nvdiffrast_research_license=accept_nvdiffrast_research_license,
            registry=self.registry,
        )
        mesh = ovoxel.to_mesh(asset)
        runtime = ovoxel._load_runtime(
            "PBR/GLB postprocess",
            capability=OVoxelCapability.NATIVE_CONVERSION,
            required_members=("postprocess.to_glb",),
        )
        coordinates, attributes = official_tensors_from_ovoxel_asset(asset, packed=False)
        channel_names = [
            name
            for name in ("base_color", "metallic", "roughness", "emissive", "alpha", "normal")
            if name in attributes
        ]
        attr_layout = {}
        feature_parts = []
        start = 0
        for name in channel_names:
            value = attributes[name]
            width = value.shape[1]
            attr_layout[name] = slice(start, start + width)
            feature_parts.append(value)
            start += width
        required = {"base_color", "metallic", "roughness", "alpha"}
        if not required.issubset(attr_layout):
            raise ValueError("PBR/GLB postprocess requires base_color, metallic, roughness, and alpha")
        metadata_resolution = asset.metadata.get("resolution")
        metadata_aabb = asset.metadata.get("aabb")
        if metadata_resolution is None or metadata_aabb is None:
            raise ValueError("PBR/GLB postprocess requires explicit resolution and aabb metadata")
        function = getattr(getattr(runtime, "postprocess", None), "to_glb", None)
        if not callable(function):
            raise RuntimeError("the selected O-Voxel runtime does not expose postprocess.to_glb")
        return function(
            vertices=mesh.vertices,
            faces=mesh.faces.to(dtype=torch.int32),
            attr_volume=torch.cat(feature_parts, dim=1).to(device),
            coords=coordinates.to(device=device, dtype=torch.int32),
            attr_layout=attr_layout,
            grid_size=metadata_resolution,
            aabb=metadata_aabb,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            **({} if postprocess_parameters is None else dict(postprocess_parameters)),
        )


OVoxelPBRPostprocessFacade = Trellis2PBRPostprocessFacade

__all__ = ["OVoxelPBRPostprocessFacade", "Trellis2PBRPostprocessFacade"]
