# Foreground crop semantics are adapted from Microsoft TRELLIS and TRELLIS.2:
# https://github.com/microsoft/TRELLIS (revision 442aa1e1afb9014e80681d3bf604e8d728a86ee7)
# https://github.com/microsoft/TRELLIS.2 (revision 75fbf0183001ed9876c8dbb35de6b68552ee08bd)
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for typed image conditions, explicit masks, and camera updates.

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from ..objects import CameraRig
from .conditions import ImageCondition


def validate_image_condition_pixels(condition: ImageCondition) -> None:
    """Validate that an exact image condition contains unit-range pixels."""

    if type(condition) is not ImageCondition:
        raise TypeError("condition must be an exact ImageCondition")
    condition.validate()
    if bool(((condition.image < 0) | (condition.image > 1)).any()):
        raise ValueError("image values must be in [0, 1]")


def _updated_camera(
    camera: CameraRig | None,
    *,
    crop_box: tuple[int, int, int, int],
    image_size: int,
) -> CameraRig | None:
    if camera is None:
        return None
    left, upper, right, lower = crop_box
    crop_width = right - left
    crop_height = lower - upper
    image_transform = camera.intrinsics.new_tensor(
        [
            [image_size / crop_width, 0.0, -left * image_size / crop_width],
            [0.0, image_size / crop_height, -upper * image_size / crop_height],
            [0.0, 0.0, 1.0],
        ]
    )
    return CameraRig(
        world_to_camera=camera.world_to_camera,
        intrinsics=image_transform.unsqueeze(0) @ camera.intrinsics,
        image_sizes=camera.image_sizes.new_full((1, 2), image_size),
        coordinate_system=camera.coordinate_system,
        metadata=camera.metadata,
    )


def _validate_preprocessing_arguments(image_size: int, foreground_scale: float) -> None:
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if (
        isinstance(foreground_scale, bool)
        or not isinstance(foreground_scale, (int, float))
        or not math.isfinite(float(foreground_scale))
        or foreground_scale <= 0
    ):
        raise ValueError("foreground_scale must be a finite positive number")


def _quantized_rgba(condition: ImageCondition) -> tuple[torch.Tensor, np.ndarray]:
    pixels = condition.image
    if pixels.shape[0] == 1:
        rgb = pixels.expand(3, -1, -1)
        alpha = pixels.new_ones((1, *pixels.shape[-2:]))
    elif pixels.shape[0] == 3:
        rgb = pixels
        alpha = pixels.new_ones((1, *pixels.shape[-2:]))
    else:
        rgb = pixels[:3]
        alpha = pixels[3:4]
    if condition.mask is not None:
        alpha = alpha * condition.mask
    rgba = torch.cat((rgb, alpha)).permute(1, 2, 0).detach().cpu().mul(255).to(torch.uint8).numpy()
    return pixels, rgba


def preprocess_image_condition(
    condition: ImageCondition,
    *,
    image_size: int,
    foreground_scale: float,
    premultiply_before_resize: bool = False,
) -> ImageCondition:
    """Prepare one typed image with pinned TRELLIS inference semantics.

    RGBA alpha and an optional mask are multiplied into one alpha channel. If
    that quantized uint8 channel contains transparency, foreground pixels are
    selected with ``alpha > 0.8 * 255`` and recentered with the supplied family
    crop scale. Plain RGB (or fully opaque RGBA) without a meaningful mask is
    treated as an already background-removed full frame; this function never
    invokes a background-removal model.

    TRELLIS resizes RGBA before premultiplication. TRELLIS.2 premultiplies its
    cropped RGBA image before the image encoder resizes RGB, represented by
    ``premultiply_before_resize=True``. Pillow performs all crops and resizes.
    """

    validate_image_condition_pixels(condition)
    _validate_preprocessing_arguments(image_size, foreground_scale)
    if not isinstance(premultiply_before_resize, bool):
        raise TypeError("premultiply_before_resize must be a bool")

    pixels, rgba = _quantized_rgba(condition)
    height, width = pixels.shape[-2:]
    alpha = rgba[:, :, 3]
    meaningful_alpha = not bool(np.all(alpha == 255))
    if meaningful_alpha:
        foreground = np.argwhere(alpha > 0.8 * 255)
        if foreground.size == 0:
            raise ValueError("alpha and mask must contain at least one foreground pixel above 0.8")
        minimum_y, minimum_x = foreground.min(axis=0).tolist()
        maximum_y, maximum_x = foreground.max(axis=0).tolist()
        center_x = (minimum_x + maximum_x) / 2
        center_y = (minimum_y + maximum_y) / 2
        crop_size = int(max(maximum_x - minimum_x, maximum_y - minimum_y) * foreground_scale)
        half_size = crop_size // 2
        pillow_box = (
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size,
        )
        crop_box = tuple(int(round(value)) for value in pillow_box)
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError("foreground extent is too small for the requested crop scale")
    else:
        pillow_box = (0, 0, width, height)
        crop_box = pillow_box

    output = Image.fromarray(rgba).crop(pillow_box)
    if premultiply_before_resize:
        output_array = np.asarray(output).astype(np.float32) / 255
        output_array = output_array[:, :, :3] * output_array[:, :, 3:4]
        output = Image.fromarray((output_array * 255).astype(np.uint8))
        output_array = np.asarray(output.resize((image_size, image_size), Image.Resampling.LANCZOS))
    else:
        output = output.resize((image_size, image_size), Image.Resampling.LANCZOS)
        output_array = np.asarray(output).astype(np.float32) / 255
        output_array = output_array[:, :, :3] * output_array[:, :, 3:4]
        output_array = (output_array * 255).astype(np.uint8)
    output_tensor = torch.from_numpy(output_array.copy()).permute(2, 0, 1)
    output_tensor = output_tensor.to(device=pixels.device, dtype=pixels.dtype).div(255)

    return ImageCondition(
        image=output_tensor,
        camera=_updated_camera(condition.camera, crop_box=crop_box, image_size=image_size),
    )


def preprocess_training_image_condition(
    condition: ImageCondition,
    *,
    image_size: int,
    foreground_scale: float,
) -> ImageCondition:
    """Prepare one typed image with pinned TRELLIS dataset semantics."""

    validate_image_condition_pixels(condition)
    _validate_preprocessing_arguments(image_size, foreground_scale)
    pixels, rgba = _quantized_rgba(condition)
    alpha = rgba[:, :, 3]
    foreground = np.nonzero(alpha)
    if foreground[0].size == 0:
        raise ValueError("alpha and mask must contain at least one nonzero foreground pixel")
    minimum_x, minimum_y = foreground[1].min(), foreground[0].min()
    maximum_x, maximum_y = foreground[1].max(), foreground[0].max()
    center_x = (minimum_x + maximum_x) / 2
    center_y = (minimum_y + maximum_y) / 2
    half_size = max(maximum_x - minimum_x, maximum_y - minimum_y) / 2
    augmented_half_size = half_size * foreground_scale
    crop_box = (
        int(center_x - augmented_half_size),
        int(center_y - augmented_half_size),
        int(center_x + augmented_half_size),
        int(center_y + augmented_half_size),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("foreground extent is too small for the requested crop scale")

    output = Image.fromarray(rgba).crop(crop_box)
    output = output.resize((image_size, image_size), Image.Resampling.LANCZOS)
    resized_alpha = np.asarray(output.getchannel(3)).astype(np.float32) / 255
    output_array = np.asarray(output.convert("RGB")).astype(np.float32) / 255
    output_array = output_array * resized_alpha[:, :, None]
    output_tensor = torch.from_numpy(output_array.copy()).permute(2, 0, 1)
    output_tensor = output_tensor.to(device=pixels.device, dtype=pixels.dtype)
    return ImageCondition(
        image=output_tensor,
        camera=_updated_camera(condition.camera, crop_box=crop_box, image_size=image_size),
    )


__all__ = [
    "preprocess_image_condition",
    "preprocess_training_image_condition",
    "validate_image_condition_pixels",
]
