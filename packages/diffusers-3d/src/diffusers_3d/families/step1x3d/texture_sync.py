# Portions of this file reproduce the SyncMVD latent texture synchronisation used by Step1X-3D:
# https://github.com/stepfun-ai/Step1X-3D (step1x3d_texture/texture_sync/project.py and step_sync.py)
# Revision: cb5ac944709c6c913109070c7b90c3447f57f3d4
# Derived from https://github.com/LIU-Yuxin/SyncMVD
#
# MIT License. Copyright (c) 2023 LIU-Yuxin.
# Modified to rasterize with the package mesh rasterizer, to bake by explicit scatter instead of differentiating a
# render, and to fill texels with a jump-flooding nearest-texel search instead of the CuPy Voronoi kernel.

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from ...backends.mesh_rasterizer import interpolate, rasterize


def nearest_valid_index(valid: torch.Tensor) -> torch.Tensor:
    """Flat index of a nearest ``True`` texel for every texel of an ``(H, W)`` mask, found by jump flooding.

    Returns ``(H * W,)`` indices, ``-1`` everywhere when nothing is valid.
    """

    height, width = valid.shape
    rows = torch.arange(height, device=valid.device)[:, None].expand(height, width)
    columns = torch.arange(width, device=valid.device)[None].expand(height, width)
    nearest = torch.where(valid, rows * width + columns, -1)
    step = 1 << (max(height, width).bit_length() - 1)
    while step:
        padded = F.pad(nearest, (step, step, step, step), value=-1)
        best = nearest
        best_distance = torch.where(best >= 0, (best // width - rows) ** 2 + (best % width - columns) ** 2, -1)
        for row_offset in (0, step, 2 * step):
            for column_offset in (0, step, 2 * step):
                candidate = padded[row_offset : row_offset + height, column_offset : column_offset + width]
                distance = (candidate // width - rows) ** 2 + (candidate % width - columns) ** 2
                closer = (candidate >= 0) & ((best_distance < 0) | (distance < best_distance))
                best = torch.where(closer, candidate, best)
                best_distance = torch.where(closer, distance, best_distance)
        nearest = best
        step //= 2
    return nearest.flatten()


def _fill(values: torch.Tensor, nearest: torch.Tensor) -> torch.Tensor:
    return values if bool((nearest < 0).all()) else values[..., nearest]


def vertex_normals(positions: torch.Tensor, faces: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Area-weighted vertex normals."""

    corners = positions[faces]
    face_normals = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
    normals = torch.zeros_like(positions).index_add_(0, faces.flatten(), face_normals.repeat_interleave(3, dim=0))
    return F.normalize(normals, dim=1, eps=eps)


class Step1X3DLatentTextureSync:
    """Keeps the six latent views consistent through a shared UV texture.

    ``world_to_clip`` ``(6, 4, 4)`` maps ``positions`` into clip space with row 0 of the image at the top; views are
    rasterized at ``image_size`` with back faces culled and sample the ``texture_size`` texture at the nearest texel
    (``v = 1`` is row 0). Baking averages each view's pixels into the texels they sample, fills every other texel
    from its nearest sampled one, and blends the views with ``visible * cos ** exp`` weights, where ``visible``
    marks the texels of faces seen at ``visibility_size`` (dilated by a texel) and ``cos`` is the baked cosine
    between the vertex normal and the direction to the camera.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        faces: torch.Tensor,
        uvs: torch.Tensor,
        world_to_clip: torch.Tensor,
        camera_positions: torch.Tensor,
        *,
        image_size: int,
        texture_size: int,
        visibility_size: int = 512,
    ) -> None:
        self.image_size, self.texture_size = image_size, texture_size
        clip = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1) @ world_to_clip.transpose(1, 2)
        normals = vertex_normals(positions, faces)
        # The clip transform flips y, so front faces turn clockwise; reversing the winding culls the back faces.
        faces = faces[:, [0, 2, 1]]
        raster = rasterize(clip, faces, (image_size, image_size), cull_backfaces=True)
        texel = (interpolate(uvs, raster, faces) * (texture_size - 1)).round().long().clamp(0, texture_size - 1)
        texel = (texture_size - 1 - texel[..., 1]) * texture_size + texel[..., 0]
        self.pixel_texel = torch.where(raster.face_index >= 0, texel, -1).flatten(1)
        self.masks = (raster.face_index >= 0)[:, None].float()

        counts = torch.zeros(len(clip), texture_size**2, device=positions.device)
        counts.scatter_add_(1, self.pixel_texel.clamp(min=0), self.masks.flatten(1))
        self.counts = counts
        self.nearest = [nearest_valid_index(count.view(texture_size, texture_size) > 0) for count in counts]

        points = interpolate(positions, raster, faces)
        normals = F.normalize(interpolate(normals, raster, faces), dim=-1, eps=1e-6)
        to_camera = F.normalize(camera_positions[:, None, None] - points, dim=-1, eps=1e-6)
        cos = (normals * to_camera).sum(dim=-1, keepdim=True).clamp(0, 1).permute(0, 3, 1, 2)
        self.cos = torch.stack([_fill(m, n) for m, n in zip(self._mean(cos), self.nearest)])

        visible_faces = rasterize(clip, faces, (visibility_size, visibility_size), cull_backfaces=True).face_index
        seen = torch.zeros(len(clip), len(faces) + 1, dtype=torch.bool, device=positions.device)
        seen.scatter_(1, visible_faces.flatten(1) + 1, True)
        uv_clip = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=1)
        uv_clip[:, 1] = -uv_clip[:, 1]
        uv_face = rasterize(uv_clip, faces, (texture_size, texture_size)).face_index.flatten()
        visible = seen[:, 1:][:, uv_face.clamp(min=0)] & (uv_face >= 0)
        visible = F.max_pool2d(visible.view(-1, 1, texture_size, texture_size).float(), 3, stride=1, padding=1)
        self.visible = visible.flatten(1)

    def _mean(self, views: torch.Tensor) -> torch.Tensor:
        """Average ``(6, C, H, W)`` pixel values into the texels they sample, giving ``(6, C, T * T)``."""

        sums = views.new_zeros(*views.shape[:2], self.texture_size**2)
        index = self.pixel_texel.clamp(min=0)[:, None].expand(-1, views.shape[1], -1)
        sums.scatter_add_(2, index, views.flatten(2) * self.masks.flatten(2))
        return sums / self.counts[:, None].clamp(min=1)

    def render(self, texture: torch.Tensor) -> torch.Tensor:
        """Sample a ``(C, T, T)`` texture into ``(6, C, H, W)`` views with a background of 1."""

        views = texture.flatten(1)[:, self.pixel_texel.clamp(min=0)].transpose(0, 1)
        views = torch.where(self.pixel_texel[:, None] >= 0, views, torch.ones_like(views))
        return views.unflatten(2, (self.image_size, self.image_size))

    def bake(self, views: torch.Tensor, exp: float) -> torch.Tensor:
        """Blend ``(6, C, H, W)`` views into a ``(C, T, T)`` texture."""

        baked = torch.stack([_fill(m, n) for m, n in zip(self._mean(views), self.nearest)])
        weights = (self.visible * self.cos[:, 0] ** exp)[:, None]
        total = weights.sum(dim=0)
        texture = (baked * weights).sum(dim=0) / (total + 1e-8)
        texture = _fill(texture, nearest_valid_index(total[0].view(self.texture_size, self.texture_size) > 0))
        return texture.unflatten(1, (self.texture_size, self.texture_size))

    def step(
        self,
        scheduler,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        texture: torch.Tensor,
        exp: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``DDPMScheduler`` posterior step taken on the texture: bake the predicted clean views, step the texture,
        and render it. Returns the next views (background 1) and texture."""

        timestep = int(timestep)
        if scheduler.config.prediction_type != "epsilon" or scheduler.config.thresholding:
            raise ValueError("texture synchronisation supports epsilon prediction without thresholding")
        previous = scheduler.previous_timestep(timestep)
        alpha_prod = scheduler.alphas_cumprod[timestep]
        alpha_prod_previous = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.one
        current_alpha = alpha_prod / alpha_prod_previous
        original = (sample.float() - (1 - alpha_prod) ** 0.5 * model_output.float()) / alpha_prod**0.5
        if scheduler.config.clip_sample:
            original = original.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
        original_coefficient = alpha_prod_previous**0.5 * (1 - current_alpha) / (1 - alpha_prod)
        current_coefficient = current_alpha**0.5 * (1 - alpha_prod_previous) / (1 - alpha_prod)
        texture = original_coefficient * self.bake(original, exp) + current_coefficient * texture
        if timestep > 0:
            noise = randn_tensor(texture.shape, generator=generator, device=texture.device, dtype=texture.dtype)
            variance = scheduler._get_variance(timestep)
            texture = texture + (variance if scheduler.variance_type == "fixed_small_log" else variance**0.5) * noise
        return self.render(texture).to(sample.dtype), texture


__all__ = ["Step1X3DLatentTextureSync", "nearest_valid_index", "vertex_normals"]
