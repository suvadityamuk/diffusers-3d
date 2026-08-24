from __future__ import annotations

import math

import numpy as np
import torch

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
        output_top: float,
        output_left: float,
        output_height: float,
        output_width: float,
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

        rgb_array = (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        mask = (
            None
            if condition.mask is None
            else (condition.mask[0].detach().cpu().to(torch.float32).numpy() * 255.0).astype(np.uint8)
        )
        if alpha is not None:
            alpha_array = (alpha[0].numpy() * 255.0).astype(np.uint8)
            mask = (
                alpha_array
                if mask is None
                else (alpha_array.astype(np.float32) * (mask.astype(np.float32) / 255.0)).astype(np.uint8)
            )
        if mask is None:
            mask = np.full((image.shape[1], image.shape[2]), 255, dtype=np.uint8)

        foreground = np.nonzero(mask)
        if foreground[0].size == 0:
            raise ValueError("image alpha/mask contains no foreground pixels")
        top = int(foreground[0].min())
        bottom = int(foreground[0].max())
        left = int(foreground[1].min())
        right = int(foreground[1].max())
        crop_height = bottom - top
        crop_width = right - left
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("image alpha/mask foreground must span at least two rows and columns")

        square_size = max(image.shape[1:])
        desired_size = int(square_size * (1.0 - ratio))
        scale = desired_size / max(crop_height, crop_width)
        output_height = int(crop_height * scale)
        output_width = int(crop_width * scale)
        if output_height <= 0 or output_width <= 0:
            raise ValueError("border_ratio leaves no pixels for the recentered foreground")
        output_top = (square_size - output_height) // 2
        output_left = (square_size - output_width) // 2

        try:
            import cv2
        except ImportError as error:
            raise ImportError(
                'Hunyuan image preprocessing requires OpenCV. Install it with `pip install "diffusers-3d[hunyuan3d]"`.'
            ) from error

        rgba_array = np.concatenate([rgb_array, mask[..., None]], axis=-1)
        cropped_rgba = rgba_array[top:bottom, left:right]
        resized_rgba = cv2.resize(cropped_rgba, (output_width, output_height), interpolation=cv2.INTER_AREA)
        rgba_canvas = np.zeros((square_size, square_size, 4), dtype=np.uint8)
        row_slice = slice(output_top, output_top + output_height)
        column_slice = slice(output_left, output_left + output_width)
        rgba_canvas[row_slice, column_slice] = resized_rgba

        centered_mask = rgba_canvas[..., 3:].astype(np.float32) / 255.0
        composited = (rgba_canvas[..., :3] * centered_mask + 255.0 * (1.0 - centered_mask)).astype(np.uint8)
        centered_mask = (centered_mask * 255.0).astype(np.uint8)
        composited = cv2.resize(composited, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        output_mask = cv2.resize(centered_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if output_mask.ndim == 2:
            output_mask = output_mask[..., None]
        normalized = (
            torch.from_numpy(composited.copy()).permute(2, 0, 1).to(torch.float32).div(255.0).mul(2.0).sub(1.0)
        )
        mask_tensor = torch.from_numpy(output_mask.copy()).permute(2, 0, 1).to(torch.float32).div(255.0)

        final_scale = self.size / square_size
        camera = self._update_camera(
            condition.camera,
            top=top,
            left=left,
            crop_height=crop_height,
            crop_width=crop_width,
            output_top=output_top * final_scale,
            output_left=output_left * final_scale,
            output_height=output_height * final_scale,
            output_width=output_width * final_scale,
        )
        return ImageCondition(image=normalized, camera=camera, mask=mask_tensor)

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
