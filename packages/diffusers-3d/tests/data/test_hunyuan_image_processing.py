from __future__ import annotations

import pytest
import torch

from diffusers_3d import CameraRig, HunyuanImageProcessor, ImageCondition


def test_hunyuan_processor_recenters_alpha_and_white_composites():
    image = torch.zeros(4, 10, 20)
    image[0, 2:8, 8:12] = 1.0
    image[3, 2:8, 8:12] = 1.0
    processor = HunyuanImageProcessor(size=32, border_ratio=0.25)

    output = processor(ImageCondition(image=image))

    assert type(output) is ImageCondition
    assert output.image.shape == (3, 32, 32)
    assert output.mask.shape == (1, 32, 32)
    foreground = torch.nonzero(output.mask[0] > 0.0, as_tuple=False)
    assert int(foreground[:, 0].min()) == 4
    assert int(foreground[:, 0].max()) == 27
    assert int(foreground[:, 1].min()) == 8
    assert int(foreground[:, 1].max()) == 22
    assert torch.equal(output.image[:, 16, 16], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.equal(output.image[:, 0, 0], torch.ones(3))
    assert float(output.image.min()) == -1.0
    assert float(output.image.max()) == 1.0


def test_hunyuan_processor_uses_explicit_mask_without_background_removal():
    image = torch.zeros(3, 12, 8)
    image[1] = 1.0
    mask = torch.zeros(1, 12, 8)
    mask[:, 4:8, 2:6] = 1.0

    output = HunyuanImageProcessor(size=20)(ImageCondition(image=image, mask=mask), border_ratio=0.2)

    foreground = torch.nonzero(output.mask[0] > 0.0, as_tuple=False)
    assert tuple((foreground.max(dim=0).values - foreground.min(dim=0).values + 1).tolist()) == (15, 15)
    assert torch.equal(output.image[:, 10, 10], torch.tensor([-1.0, 1.0, -1.0]))
    assert bool((output.image[:, 0, 0] >= 0.98).all())


def test_hunyuan_processor_normalizes_255_input_deterministically():
    image = torch.zeros(3, 5, 7)
    image[0] = 255.0
    condition = ImageCondition(image=image)
    processor = HunyuanImageProcessor(size=16, border_ratio=0.0)

    first = processor(condition)
    second = processor(condition)

    assert torch.equal(first.image, second.image)
    assert torch.equal(first.mask, second.mask)
    assert first.image.dtype is torch.float32
    assert not first.image.requires_grad
    assert bool(((first.image >= -1.0) & (first.image <= 1.0)).all())
    assert bool(((first.mask >= 0.0) & (first.mask <= 1.0)).all())


def test_hunyuan_processor_updates_camera_for_exclusive_crop_recenter_and_final_resize():
    image = torch.zeros(4, 10, 20)
    image[:3, 2:8, 8:12] = 1.0
    image[3, 2:8, 8:12] = 1.0
    camera = CameraRig(
        world_to_camera=torch.eye(4).unsqueeze(0),
        intrinsics=torch.tensor([[[100.0, 0.0, 10.0], [0.0, 120.0, 5.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[10, 20]], dtype=torch.int64),
    )

    output = HunyuanImageProcessor(size=32, border_ratio=0.25)(ImageCondition(image=image, camera=camera))

    expected_intrinsics = torch.tensor([[[480.0, 0.0, 17.6], [0.0, 576.0, 17.6], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(output.camera.intrinsics, expected_intrinsics)
    assert torch.equal(output.camera.image_sizes, torch.tensor([[32, 32]], dtype=torch.int64))


def test_hunyuan_processor_keeps_unit_alpha_separate_from_normalized_rgb():
    image = torch.full((4, 8, 8), -1.0)
    image[3].zero_()
    image[:3, 2:6, 2:6] = torch.tensor([1.0, -1.0, -1.0]).view(3, 1, 1)
    image[3, 2:6, 2:6] = 1.0

    output = HunyuanImageProcessor(size=16, border_ratio=0.5)(ImageCondition(image=image))

    assert bool((output.mask[:, 0] == 0).all())
    assert torch.equal(output.image[:, 8, 8], torch.tensor([1.0, -1.0, -1.0]))


def test_hunyuan_processor_rejects_empty_mask_and_invalid_configuration():
    condition = ImageCondition(image=torch.zeros(3, 8, 8), mask=torch.zeros(1, 8, 8))

    with pytest.raises(ValueError, match="no foreground"):
        HunyuanImageProcessor(size=16)(condition)
    with pytest.raises(ValueError, match="positive integer"):
        HunyuanImageProcessor(size=0)
    with pytest.raises(ValueError, match="border_ratio"):
        HunyuanImageProcessor(border_ratio=1.0)
