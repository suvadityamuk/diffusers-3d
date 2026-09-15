"""TRELLIS ``to_glb``: a decoded FlexiCubes mesh textured from renders of the matching Gaussian splats."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from ..objects import GaussianSplatAsset, MeshAsset, PBRMaterial
from .cumesh import CuMeshBackend
from .defaults import BACKEND_REGISTRY
from .gsplat import GsplatBackend
from .protocols import GeometryProcessingBackend
from .registry import BackendRegistry
from .texture_baking import bake_texture, rasterize_uv_atlas, sphere_hammersley_cameras
from .types import BackendCapability, BackendSpec
from .xatlas import XAtlasBackend


class TrellisGlbPostprocessFacade:
    """Explicit gate for the TRELLIS textured-mesh path (upstream ``postprocessing_utils.to_glb``).

    The released recipe simplifies the mesh to 5% of its faces, unwraps it with xatlas, renders the Gaussian
    splats from 100 Hammersley-distributed views, and bakes those renders into a base-colour texture. Here the
    simplification runs on CuMesh, the renders come from gsplat, and the baking is plain PyTorch
    (:mod:`.texture_baking`). Construction and :meth:`requirements` are side-effect free.
    """

    def __init__(self, *, registry: BackendRegistry = BACKEND_REGISTRY) -> None:
        self.registry = registry

    def requirements(self, *, device: str | torch.device = "cuda") -> Mapping[str, BackendSpec]:
        return {
            "cumesh": self.registry.select(
                BackendCapability.GEOMETRY_PROCESSING,
                name="cumesh",
                device=device,
                dtype=torch.float32,
                differentiable=False,
            ),
            "xatlas": self.registry.select(
                BackendCapability.GEOMETRY_PROCESSING, name="xatlas", device="cpu", differentiable=False
            ),
            "gsplat": self.registry.select(
                BackendCapability.GAUSSIAN_RASTERIZATION,
                name="gsplat",
                device=device,
                dtype=torch.float32,
                differentiable=True,
            ),
        }

    def to_textured_mesh(
        self,
        mesh: MeshAsset,
        gaussians: GaussianSplatAsset,
        *,
        device: str | torch.device = "cuda",
        simplify_ratio: float = 0.95,
        texture_size: int = 1024,
        num_views: int = 100,
        render_size: int = 1024,
        depth_tolerance: float = 0.03,
        mesh_processor: GeometryProcessingBackend | None = None,
    ) -> MeshAsset:
        """Return ``mesh`` simplified, UV-unwrapped, and carrying a baked base-colour :class:`PBRMaterial`.

        ``mesh`` and ``gaussians`` must share a coordinate system and have identity transforms (both come from one
        TRELLIS SLAT). ``simplify_ratio`` is the fraction of faces to remove (``0`` skips CuMesh);
        ``mesh_processor`` replaces CuMesh for the repair/simplify step. The result keeps TRELLIS' Z-up frame;
        call :meth:`MeshAsset.to_coordinate_system` before exporting with :class:`TrimeshBackend`.
        """

        if type(mesh) is not MeshAsset or type(gaussians) is not GaussianSplatAsset:
            raise TypeError("mesh must be a MeshAsset and gaussians a GaussianSplatAsset")
        if mesh.coordinate_system is not gaussians.coordinate_system:
            raise ValueError("mesh and gaussians must use the same coordinate system")
        identity = torch.eye(4, device=mesh.device, dtype=mesh.transform.dtype)
        if not torch.allclose(mesh.transform, identity):
            raise ValueError("mesh must have an identity transform; bake it into the vertices first")
        if not 0.0 <= simplify_ratio < 1.0:
            raise ValueError("simplify_ratio must be in [0, 1)")
        if simplify_ratio > 0 and mesh_processor is None:
            self.requirements(device=device)
        device = torch.device(device)

        # Only the geometry survives simplification and unwrapping; colours are re-baked from the splats.
        working = replace(
            mesh.to(device, torch.float32),
            colors=None,
            normals=None,
            uvs=None,
            face_material_ids=None,
            materials=(),
            extras={},
        )
        if simplify_ratio > 0:
            processor = (
                CuMeshBackend(device=device, registry=self.registry) if mesh_processor is None else mesh_processor
            )
            working = processor.process_geometry(working, operation="repair")
            target_faces = max(4, int(working.faces.shape[0] * (1.0 - simplify_ratio)))
            working = processor.process_geometry(
                working, operation="simplify", parameters={"target_faces": target_faces}
            )
        unwrapped = XAtlasBackend(registry=self.registry).process_geometry(working, operation="unwrap_uv")
        unwrapped = unwrapped.to(device, torch.float32)

        cameras = sphere_hammersley_cameras(
            num_views, image_size=render_size, coordinate_system=mesh.coordinate_system, device=device
        )
        renderer = GsplatBackend(device=device, dtype=torch.float32, registry=self.registry)
        splats = gaussians.to(device, torch.float32)
        colors, depths, alphas = [], [], []
        for start in range(0, num_views, 8):
            rig = replace(
                cameras,
                world_to_camera=cameras.world_to_camera[start : start + 8],
                intrinsics=cameras.intrinsics[start : start + 8],
                image_sizes=cameras.image_sizes[start : start + 8],
            )
            rendered = renderer.rasterize_gaussians(splats, rig)
            colors.append(rendered["color"])
            depths.append(rendered["depth"])
            alphas.append(rendered["alpha"])
        surface = rasterize_uv_atlas(unwrapped, texture_size)
        texture, observed = bake_texture(
            surface,
            cameras,
            torch.cat(colors),
            torch.cat(depths),
            torch.cat(alphas),
            texture_size=texture_size,
            depth_tolerance=depth_tolerance,
        )
        material = PBRMaterial(base_color=texture, roughness=torch.tensor(1.0, device=device))
        return replace(
            unwrapped,
            materials=(material,),
            metadata={
                **unwrapped.metadata,
                "texture": "baked-from-gaussians",
                "texture_size": texture_size,
                "num_views": num_views,
                "observed_texel_fraction": float(observed.sum()) / surface.texel_index.shape[0],
            },
        )


__all__ = ["TrellisGlbPostprocessFacade"]
