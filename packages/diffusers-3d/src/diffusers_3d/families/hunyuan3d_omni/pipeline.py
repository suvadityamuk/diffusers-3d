# Portions of this file reproduce pipeline semantics from Tencent Hunyuan3D-Omni:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni
# Revision: 4d47c0cc2bd0c4281963a7314ab330a5af36bfa8
#
# Tencent Hunyuan 3D Omni is licensed under the Tencent Hunyuan 3D Omni Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: typed image and tensor control inputs (no open3d or trimesh preprocessing), the Hunyuan3D-2.1 port's
# sampling and decoding, MeshAsset outputs.

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Sequence

import numpy as np
import torch
from transformers import Dinov2Model

from ...backends import ScikitImageBackend
from ...backends._optional import require
from ...data import ImageInput, validate_image_condition_pixels
from ...data.image_processing import _quantized_rgba
from ...execution.pipelines import ExternalComponent, Object3DPipeline
from ...objects import CoordinateSystem, Latent3DOutput, MeshAsset, Object3DPipelineOutput
from ..hunyuan3d.models import DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, Hunyuan3DDiTModel, load_dinov2
from ..hunyuan3d.pipeline import Hunyuan3DImageTo3DPipeline
from ..hunyuan3d.scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from ..hunyuan3d.vae import Hunyuan3DShapeVAE
from .models import Hunyuan3DOmniControlEncoder


def recenter_image(rgba: np.ndarray, border_ratio: float) -> np.ndarray:
    """Omni's ``ImageProcessorV2.recenter``: the Hunyuan3D-2.1 recentering, but resized with ``INTER_CUBIC``.

    ``rgba`` is ``(height, width, 4)`` uint8; returns a square uint8 RGB image composited on white.
    """

    import cv2

    height, width = rgba.shape[:2]
    rows, columns = np.nonzero(rgba[..., 3])
    if rows.size == 0:
        raise ValueError("the image alpha and mask contain no foreground")
    # Upstream excludes the last foreground row and column from the crop.
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    if bottom == top or right == left:
        raise ValueError("the image foreground must span at least two rows and two columns")
    size = max(height, width)
    scale = int(size * (1 - border_ratio)) / max(bottom - top, right - left)
    resized_height, resized_width = int((bottom - top) * scale), int((right - left) * scale)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    offset_y, offset_x = (size - resized_height) // 2, (size - resized_width) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = cv2.resize(
        rgba[top:bottom, left:right], (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    alpha = canvas[..., 3:].astype(np.float32) / 255
    return (canvas[..., :3] * alpha + 255 * (1 - alpha)).clip(0, 255).astype(np.uint8)


def remove_floaters_and_degenerate_faces(mesh):
    """Omni's ``FloaterRemover()`` then ``DegenerateFaceRemover()`` on a ``trimesh.Trimesh``, as ``inference.py`` runs.

    Connected components with fewer than 0.5% of the faces are removed with pymeshlab; the degenerate-face pass is
    a pymeshlab save and reload. Each pass round-trips through PLY files as upstream does, so trimesh re-merges
    vertices after it. Needs pymeshlab (GPL): ``pip install "diffusers-3d[meshlab]"``.
    """

    pymeshlab = require("pymeshlab", "meshlab")
    trimesh = require("trimesh", "portable")
    with tempfile.TemporaryDirectory() as directory:
        exported, saved = os.path.join(directory, "trimesh.ply"), os.path.join(directory, "pymeshlab.ply")

        def to_meshset(mesh):
            mesh.export(exported)
            meshset = pymeshlab.MeshSet()
            meshset.load_new_mesh(exported)
            return meshset

        def to_trimesh(meshset):
            meshset.save_current_mesh(saved)
            return trimesh.load(saved)

        meshset = to_meshset(mesh)
        meshset.apply_filter("compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.005)
        meshset.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
        meshset.apply_filter("meshing_remove_selected_vertices_and_faces")
        meshset = to_meshset(to_trimesh(meshset))
        reloaded = os.path.join(directory, "degenerate.ply")
        meshset.save_current_mesh(reloaded)
        meshset = pymeshlab.MeshSet()
        meshset.load_new_mesh(reloaded)
        return to_trimesh(meshset)


class Hunyuan3DOmniImageTo3DPipeline(Object3DPipeline):
    """Hunyuan3D-Omni controllable image-to-shape: the Hunyuan3D-2.1 DiT and ShapeVAE decoder, with control tokens.

    Pass exactly one control as a tensor, in the normalized frame of the output mesh (right-handed, Y up, the object
    inside ``[-1, 1]``):

    - ``point_cloud``: ``(points, 3)``. Upstream centers the source mesh's bounding box at the origin and scales its
      longest side to 1.96 (``normalize_mesh(scale=0.98)``) before taking its vertices.
    - ``voxels``: ``(points, 3)`` surface samples, normalized like ``point_cloud`` with scale 0.9999; the control
      encoder keeps the centers of the occupied cells of a 16^3 grid.
    - ``bbox``: ``(3,)`` side lengths of an origin-centered box along x, y, z; the released demos use values in
      ``(0, 1]`` with the longest side 1.
    - ``skeleton``: ``(bones, 6)`` rows ``(x1, y1, z1, x2, y2, z2)`` of bone start and end points.

    A leading batch dimension gives each image its own control; without one the control is shared. Classifier-free
    guidance zeroes only the image tokens; the control tokens condition both branches, as upstream.
    """

    # Upstream downloads Meta's DINOv2-L at run time. It loads in the pipeline's dtype (float32 unless one is given).
    external_components = {
        "image_encoder": ExternalComponent(DINOV2_LARGE_REPOSITORY, DINOV2_LARGE_REVISION, load_dinov2)
    }
    model_cpu_offload_seq = "image_encoder->control_encoder->transformer->vae"

    def __init__(
        self,
        image_encoder: Dinov2Model,
        control_encoder: Hunyuan3DOmniControlEncoder,
        transformer: Hunyuan3DDiTModel,
        vae: Hunyuan3DShapeVAE,
        scheduler: Hunyuan3DFlowMatchEulerDiscreteScheduler,
        image_size: int = 512,
        border_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        warnings.warn(
            "Hunyuan3D-Omni weights are licensed under the Tencent Hunyuan 3D Omni Community License, which does not "
            "apply in the European Union, the United Kingdom, or South Korea. You are responsible for making sure "
            "your use of it complies with that license.",
            UserWarning,
            stacklevel=2,
        )
        self.register_modules(
            image_encoder=image_encoder,
            control_encoder=control_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        self.register_to_config(image_size=image_size, border_ratio=border_ratio)

    # Sampling and decoding are those of Hunyuan3D-2.1, which Omni shares.
    encode_image = Hunyuan3DImageTo3DPipeline.encode_image
    prepare_latents = Hunyuan3DImageTo3DPipeline.prepare_latents
    denoise = Hunyuan3DImageTo3DPipeline.denoise
    decode_occupancy = Hunyuan3DImageTo3DPipeline.decode_occupancy

    def preprocess(self, image: ImageInput | list[ImageInput]) -> torch.Tensor:
        """Recenter each image and resize it to ``image_size`` like upstream; returns float32 values in [-1, 1]."""

        import cv2

        size = self.config.image_size
        images = []
        for condition in self.image_conditions(image):
            validate_image_condition_pixels(condition)
            square = recenter_image(_quantized_rgba(condition)[1], self.config.border_ratio)
            resized = torch.from_numpy(cv2.resize(square, (size, size), interpolation=cv2.INTER_CUBIC)).permute(
                2, 0, 1
            )
            images.append(resized.float() / 255 * 2 - 1)
        return torch.stack(images).to(self._execution_device)

    def encode_control(self, control: torch.Tensor, control_type: str, batch_size: int) -> torch.Tensor:
        """Control tokens for ``batch_size`` images from one shared or ``(batch, ...)`` per-image control."""

        control = torch.as_tensor(control)
        if control_type == "bbox":
            control = control.reshape(-1, 1, 3)
        if control.ndim == 2:
            control = control.unsqueeze(0)
        if control.shape[0] == 1:
            control = control.expand(batch_size, -1, -1)
        if control.shape[0] != batch_size:
            raise ValueError(f"{control_type} has {control.shape[0]} entries for {batch_size} images")
        control = control.to(device=self._execution_device, dtype=self.control_encoder.dtype)
        return self.control_encoder(control, control_type).embeddings

    def encode_conditions(
        self, images: torch.Tensor, control: torch.Tensor, control_type: str, do_classifier_free_guidance: bool
    ) -> torch.Tensor:
        """Image tokens then control tokens; the unconditional context keeps the control tokens."""

        image_tokens = self.encode_image(images, do_classifier_free_guidance=False)
        control_tokens = self.encode_control(control, control_type, images.shape[0]).to(image_tokens.dtype)
        context = torch.cat([image_tokens, control_tokens], dim=1)
        if do_classifier_free_guidance:
            unconditional = torch.cat([torch.zeros_like(image_tokens), control_tokens], dim=1)
            context = torch.cat([context, unconditional])
        return context

    @staticmethod
    def extract_mesh(
        grid: torch.Tensor, bounds: tuple[float, ...], mc_level: float, postprocess_passes: int = 1
    ) -> MeshAsset:
        """Marching cubes on one occupancy grid, scaled into ``bounds`` and merged like Omni's vanilla extractor.

        ``postprocess_passes`` times, :func:`remove_floaters_and_degenerate_faces` follows, as upstream's
        ``inference.py`` does (twice for bounding-box control).
        """

        trimesh = require("trimesh", "portable")
        mesh = ScikitImageBackend().extract_surface(
            grid, level=mc_level, gradient_direction="ascent", allow_degenerate=True
        )
        lower = torch.tensor(bounds[:3], dtype=torch.float64)
        extent = torch.tensor(bounds[3:], dtype=torch.float64) - lower
        # Unlike Hunyuan3D-2.1, Omni divides by the number of cells per axis.
        vertices = mesh.vertices.double() / (grid.shape[0] - 1) * extent + lower
        # Upstream builds a trimesh from the extractor's (descent) winding, then flips it into a second trimesh.
        mesh = trimesh.Trimesh(vertices.float().cpu().numpy(), mesh.faces.flip(1).cpu().numpy())
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1])
        for _ in range(postprocess_passes):
            mesh = remove_floaters_and_degenerate_faces(mesh)
        return MeshAsset(
            vertices=torch.from_numpy(mesh.vertices).float(),
            faces=torch.from_numpy(mesh.faces).to(torch.int64),
            coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
            metadata={"family": "hunyuan3d-omni", "octree_resolution": grid.shape[0] - 1, "mc_level": mc_level},
        )

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput | list[ImageInput],
        *,
        point_cloud: torch.Tensor | None = None,
        voxels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        skeleton: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        octree_resolution: int = 512,
        bounds: float | Sequence[float] = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        postprocess: bool = True,
        return_latents: bool = True,
        return_dict: bool = True,
    ) -> Object3DPipelineOutput | tuple[tuple[MeshAsset, ...], Latent3DOutput | None]:
        """Generate one mesh per image under one control, with the released sampler and extraction defaults.

        A negative ``guidance_scale`` disables classifier-free guidance, as upstream. ``bounds`` is a half extent
        or six ``(x_min, y_min, z_min, x_max, y_max, z_max)`` values; the occupancy grid has
        ``octree_resolution + 1`` samples per axis and is evaluated ``num_chunks`` points at a time. ``postprocess``
        runs upstream ``inference.py``'s pymeshlab clean-up (:func:`remove_floaters_and_degenerate_faces`), twice for
        ``bbox`` as upstream does; it needs ``pip install "diffusers-3d[meshlab]"``.
        """

        controls = {"skeleton": skeleton, "bbox": bbox, "voxels": voxels, "point_cloud": point_cloud}
        given = [(name, value) for name, value in controls.items() if value is not None]
        if len(given) != 1:
            raise ValueError("pass exactly one of point_cloud, voxels, bbox, or skeleton")
        control_type, control = given[0]
        bounds = (-bounds,) * 3 + (bounds,) * 3 if isinstance(bounds, (int, float)) else tuple(bounds)
        images = self.preprocess(image)
        context = self.encode_conditions(images, control, control_type, guidance_scale >= 0)
        latents = self.prepare_latents(images.shape[0], generator, latents)
        latents = self.denoise(latents, context, num_inference_steps, guidance_scale)
        # Upstream always rounds the grid points to float16 before querying.
        grids = self.decode_occupancy(latents, bounds, octree_resolution, num_chunks, query_dtype=torch.float16)
        # ``inference.py`` cleans bounding-box results in ``infer_bbox`` and again in ``postprocess``.
        passes = (2 if control_type == "bbox" else 1) if postprocess else 0
        meshes = tuple(self.extract_mesh(grid, bounds, mc_level, postprocess_passes=passes) for grid in grids)
        metadata = {"family": "hunyuan3d-omni", "control": control_type}
        latent_output = Latent3DOutput(latents=latents, metadata=metadata) if return_latents else None
        self.maybe_free_model_hooks()
        if not return_dict:
            return meshes, latent_output
        return Object3DPipelineOutput(objects=meshes, latents=latent_output, previews=None)


__all__ = ["Hunyuan3DOmniImageTo3DPipeline", "recenter_image", "remove_floaters_and_degenerate_faces"]
