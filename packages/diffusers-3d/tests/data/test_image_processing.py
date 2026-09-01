from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers_3d import (
    CameraRig,
    ImageCondition,
    Object3DValidationError,
    preprocess_image_condition,
    preprocess_training_image_condition,
)


def _rgba_tensor(height: int = 10, width: int = 12) -> torch.Tensor:
    rgba = torch.zeros(4, height, width)
    rgba[0] = 1
    rgba[1, :, ::2] = 1
    rgba[2, ::2] = 1
    rgba[3, 2:8, :5] = 1
    return rgba


def _pinned_reference(
    condition: ImageCondition,
    *,
    image_size: int,
    foreground_scale: float,
    premultiply_before_resize: bool,
) -> torch.Tensor:
    pixels = condition.image
    rgb = pixels[:3] if pixels.shape[0] != 1 else pixels.expand(3, -1, -1)
    alpha = pixels[3:4] if pixels.shape[0] == 4 else pixels.new_ones((1, *pixels.shape[-2:]))
    if condition.mask is not None:
        alpha = alpha * condition.mask
    rgba = torch.cat((rgb, alpha)).permute(1, 2, 0).mul(255).to(torch.uint8).numpy()

    alpha_array = rgba[:, :, 3]
    foreground = np.argwhere(alpha_array > 0.8 * 255)
    minimum_x, minimum_y = np.min(foreground[:, 1]), np.min(foreground[:, 0])
    maximum_x, maximum_y = np.max(foreground[:, 1]), np.max(foreground[:, 0])
    center = (minimum_x + maximum_x) / 2, (minimum_y + maximum_y) / 2
    crop_size = int(max(maximum_x - minimum_x, maximum_y - minimum_y) * foreground_scale)
    box = (
        center[0] - crop_size // 2,
        center[1] - crop_size // 2,
        center[0] + crop_size // 2,
        center[1] + crop_size // 2,
    )
    output = Image.fromarray(rgba).crop(box)
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
    return torch.from_numpy(output_array.copy()).permute(2, 0, 1).float().div(255)


def _pinned_training_reference(
    condition: ImageCondition,
    *,
    image_size: int,
    foreground_scale: float,
) -> torch.Tensor:
    pixels = condition.image
    rgb = pixels[:3] if pixels.shape[0] != 1 else pixels.expand(3, -1, -1)
    alpha = pixels[3:4] if pixels.shape[0] == 4 else pixels.new_ones((1, *pixels.shape[-2:]))
    if condition.mask is not None:
        alpha = alpha * condition.mask
    rgba = torch.cat((rgb, alpha)).permute(1, 2, 0).mul(255).to(torch.uint8).numpy()

    foreground = np.array(rgba[:, :, 3]).nonzero()
    bbox = [foreground[1].min(), foreground[0].min(), foreground[1].max(), foreground[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    half_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    augmented_half_size = half_size * foreground_scale
    box = [
        int(center[0] - augmented_half_size),
        int(center[1] - augmented_half_size),
        int(center[0] + augmented_half_size),
        int(center[1] + augmented_half_size),
    ]
    output = Image.fromarray(rgba).crop(box)
    output = output.resize((image_size, image_size), Image.Resampling.LANCZOS)
    output_alpha = torch.tensor(np.array(output.getchannel(3))).float() / 255
    output_rgb = torch.tensor(np.array(output.convert("RGB"))).permute(2, 0, 1).float() / 255
    return output_rgb * output_alpha.unsqueeze(0)


@pytest.mark.parametrize(
    ("foreground_scale", "premultiply_before_resize"),
    ((1.2, False), (1.0, True)),
    ids=("trellis", "trellis2"),
)
@pytest.mark.parametrize("mask_kind", ("rgba", "separate", "combined"))
def test_trellis_foreground_preprocessing_matches_pinned_reference(
    foreground_scale,
    premultiply_before_resize,
    mask_kind,
):
    rgba = _rgba_tensor()
    if mask_kind == "rgba":
        condition = ImageCondition(rgba)
    elif mask_kind == "separate":
        condition = ImageCondition(rgba[:3], mask=rgba[3:4])
    else:
        mask = torch.zeros(1, 10, 12)
        mask[:, 3:7, 1:4] = 1
        condition = ImageCondition(rgba, mask=mask)

    actual = preprocess_image_condition(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
        premultiply_before_resize=premultiply_before_resize,
    )
    expected = _pinned_reference(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
        premultiply_before_resize=premultiply_before_resize,
    )

    assert actual.mask is None
    assert actual.camera is None
    torch.testing.assert_close(actual.image, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("foreground_scale", "premultiply_before_resize"),
    ((1.2, False), (1.0, True)),
    ids=("trellis", "trellis2"),
)
def test_soft_alpha_inference_matches_pinned_quantization_and_operation_order(
    foreground_scale,
    premultiply_before_resize,
):
    rgba = torch.zeros(4, 7, 9)
    rgba[0] = torch.linspace(0, 1, 9).repeat(7, 1)
    rgba[1] = torch.linspace(1, 0, 7).reshape(7, 1).repeat(1, 9)
    rgba[2] = 0.75
    rgba[3, 1:6, 2:7] = torch.linspace(0.05, 1.0, 25).reshape(5, 5)
    rgba[3, 0, 0] = 204.9 / 255
    rgba[3, 6, 8] = 205 / 255
    condition = ImageCondition(rgba)

    actual = preprocess_image_condition(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
        premultiply_before_resize=premultiply_before_resize,
    )
    expected = _pinned_reference(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
        premultiply_before_resize=premultiply_before_resize,
    )

    torch.testing.assert_close(actual.image, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("foreground_scale", (1.2, 1.0), ids=("trellis", "trellis2"))
def test_training_preprocessing_matches_pinned_dataset_components(foreground_scale):
    rgba = torch.zeros(4, 9, 11)
    rgba[0] = torch.linspace(0, 1, 11).repeat(9, 1)
    rgba[1] = 0.5
    rgba[2] = torch.linspace(1, 0, 9).reshape(9, 1)
    rgba[3, 2:8, 3:9] = torch.linspace(1 / 255, 1.0, 36).reshape(6, 6)
    rgba[3, 0, 1] = 1 / 255
    condition = ImageCondition(rgba)

    actual = preprocess_training_image_condition(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
    )
    expected = _pinned_training_reference(
        condition,
        image_size=8,
        foreground_scale=foreground_scale,
    )

    assert actual.mask is None
    torch.testing.assert_close(actual.image, expected, atol=0.0, rtol=0.0)


def test_full_frame_rgb_is_resized_without_foreground_extraction():
    image = torch.zeros(3, 5, 7)
    image[0, :, :3] = 1
    image[1, :, 3:] = 1
    condition = ImageCondition(image)

    actual = preprocess_image_condition(condition, image_size=8, foreground_scale=1.2).image
    rgba = torch.cat((image, torch.ones(1, 5, 7))).permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
    expected = Image.fromarray(rgba).resize((8, 8), Image.Resampling.LANCZOS)
    expected_array = np.asarray(expected).astype(np.float32) / 255
    expected_array = expected_array[:, :, :3] * expected_array[:, :, 3:4]
    expected = torch.from_numpy((expected_array * 255).astype(np.uint8).copy()).permute(2, 0, 1).float().div(255)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_preprocessing_rejects_invalid_image_pixels_and_empty_foreground():
    with pytest.raises(Object3DValidationError, match="finite"):
        ImageCondition(torch.full((3, 4, 4), torch.nan))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        preprocess_image_condition(
            ImageCondition(torch.full((3, 4, 4), 1.01)),
            image_size=8,
            foreground_scale=1.2,
        )
    with pytest.raises(ValueError, match="foreground pixel"):
        preprocess_image_condition(
            ImageCondition(torch.ones(3, 4, 4), mask=torch.zeros(1, 4, 4)),
            image_size=8,
            foreground_scale=1.2,
        )


def test_preprocessing_updates_camera_for_outside_crop_and_resize():
    rgba = _rgba_tensor()
    camera = CameraRig(
        world_to_camera=torch.eye(4).unsqueeze(0),
        intrinsics=torch.tensor([[[6.0, 0.5, 4.0], [0.0, 8.0, 5.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[10, 12]], dtype=torch.int64),
    )

    processed = preprocess_image_condition(
        ImageCondition(rgba, camera=camera),
        image_size=8,
        foreground_scale=1.2,
    )

    # Pinned crop box (-1, 1.5, 5, 7.5) is Pillow-rounded to (-1, 2, 5, 8).
    image_transform = torch.tensor([[[8 / 6, 0.0, 8 / 6], [0.0, 8 / 6, -16 / 6], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(processed.camera.intrinsics, image_transform @ camera.intrinsics)
    assert torch.equal(processed.camera.image_sizes, torch.tensor([[8, 8]]))
    assert torch.equal(processed.camera.world_to_camera, camera.world_to_camera)
