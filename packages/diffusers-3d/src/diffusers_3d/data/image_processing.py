from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..objects import CameraRig
from .conditions import ImageCondition


def _as_unit_interval(tensor: torch.Tensor, name: str) -> torch.Tensor:
    value = tensor.detach().cpu().to(torch.float32)
    minimum = float(value.min())
    maximum = float(value.max())
    if 0.0 <= minimum and maximum <= 1.0:
        return value
    if -1.0 <= minimum and maximum <= 1.0:
        return (value + 1.0) * 0.5
    if 0.0 <= minimum and maximum <= 255.0:
        return value / 255.0
    raise ValueError(f"{name} values must be in [0, 1], [-1, 1], or [0, 255]")


class HunyuanImageProcessor:
    """Prepare one typed image condition with Hunyuan ImageProcessorV2 semantics.

    Background removal is intentionally outside this utility. Callers must
    provide an alpha channel or mask when the foreground is not the full image.
    """

    def __init__(self, size: int = 512, border_ratio: float | None = None) -> None:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("size must be a positive integer")
        self.size = size
        self.border_ratio = None if border_ratio is None else self._validate_border_ratio(border_ratio)

    @staticmethod
    def _validate_border_ratio(border_ratio: float) -> float:
        if not isinstance(border_ratio, (int, float)) or isinstance(border_ratio, bool):
            raise TypeError("border_ratio must be a number")
        value = float(border_ratio)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("border_ratio must be finite and satisfy 0 <= border_ratio < 1")
        return value

    def _update_camera(
        self,
        camera: CameraRig | None,
        *,
        top: int,
        left: int,
        crop_height: int,
        crop_width: int,
        output_top: int,
        output_left: int,
        output_height: int,
        output_width: int,
    ) -> CameraRig | None:
        if camera is None:
            return None
        scale_y = output_height / crop_height
        scale_x = output_width / crop_width
        image_transform = camera.intrinsics.new_tensor(
            [
                [scale_x, 0.0, output_left - scale_x * left],
                [0.0, scale_y, output_top - scale_y * top],
                [0.0, 0.0, 1.0],
            ]
        )
        return CameraRig(
            world_to_camera=camera.world_to_camera.detach().cpu(),
            intrinsics=(image_transform @ camera.intrinsics).detach().cpu(),
            image_sizes=torch.tensor([[self.size, self.size]], dtype=camera.image_sizes.dtype),
            coordinate_system=camera.coordinate_system,
            metadata=dict(camera.metadata),
        )

    def preprocess(
        self,
        condition: ImageCondition,
        *,
        border_ratio: float = 0.15,
    ) -> ImageCondition:
        """Crop by alpha/mask, recenter, white-composite, and normalize to ``[-1, 1]``."""

        if type(condition) is not ImageCondition:
            raise TypeError("condition must be an exact ImageCondition")
        condition.validate()
        ratio = self.border_ratio if self.border_ratio is not None else self._validate_border_ratio(border_ratio)

        image = condition.image
        if image.shape[0] == 1:
            rgb = _as_unit_interval(image, "image").expand(3, -1, -1)
            alpha = None
        elif image.shape[0] == 3:
            rgb = _as_unit_interval(image, "image")
            alpha = None
        else:
            rgb = _as_unit_interval(image[:3], "image RGB channels")
            alpha = _as_unit_interval(image[3:4], "image alpha channel")

        mask = None if condition.mask is None else condition.mask.detach().cpu().to(torch.float32)
        if alpha is not None:
            mask = alpha if mask is None else alpha * mask
        if mask is None:
            mask = torch.ones((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        mask = mask.clamp(0.0, 1.0)

        foreground = torch.nonzero(mask[0] > 0.0, as_tuple=False)
        if foreground.numel() == 0:
            raise ValueError("image alpha/mask contains no foreground pixels")
        top = int(foreground[:, 0].min())
        bottom = int(foreground[:, 0].max()) + 1
        left = int(foreground[:, 1].min())
        right = int(foreground[:, 1].max()) + 1
        crop_height = bottom - top
        crop_width = right - left

        desired_size = max(1, int(self.size * (1.0 - ratio)))
        scale = desired_size / max(crop_height, crop_width)
        output_height = max(1, int(crop_height * scale))
        output_width = max(1, int(crop_width * scale))
        output_top = (self.size - output_height) // 2
        output_left = (self.size - output_width) // 2

        cropped_rgb = rgb[:, top:bottom, left:right].unsqueeze(0)
        cropped_mask = mask[:, top:bottom, left:right].unsqueeze(0)
        resized_rgb = F.interpolate(
            cropped_rgb,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        resized_mask = F.interpolate(
            cropped_mask,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].clamp(0.0, 1.0)

        rgb_canvas = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        mask_canvas = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        row_slice = slice(output_top, output_top + output_height)
        column_slice = slice(output_left, output_left + output_width)
        rgb_canvas[:, row_slice, column_slice] = resized_rgb
        mask_canvas[:, row_slice, column_slice] = resized_mask

        composited = rgb_canvas * mask_canvas + (1.0 - mask_canvas)
        normalized = composited.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        camera = self._update_camera(
            condition.camera,
            top=top,
            left=left,
            crop_height=crop_height,
            crop_width=crop_width,
            output_top=output_top,
            output_left=output_left,
            output_height=output_height,
            output_width=output_width,
        )
        return ImageCondition(image=normalized, camera=camera, mask=mask_canvas)

    def recenter(
        self,
        condition: ImageCondition,
        *,
        border_ratio: float = 0.2,
    ) -> ImageCondition:
        """Typed equivalent of ImageProcessorV2 ``recenter``."""

        return self.preprocess(condition, border_ratio=border_ratio)

    def __call__(
        self,
        condition: ImageCondition,
        *,
        border_ratio: float = 0.15,
    ) -> ImageCondition:
        return self.preprocess(condition, border_ratio=border_ratio)


__all__ = ["HunyuanImageProcessor"]
