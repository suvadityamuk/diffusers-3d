import pytest
import torch

import diffusers_3d
from diffusers_3d import (
    CameraRig,
    ImageCondition,
    MultiViewCondition,
    Object3DValidationError,
    TextCondition,
)


def make_camera(count: int, height: int = 8, width: int = 6) -> CameraRig:
    return CameraRig(
        world_to_camera=torch.eye(4).expand(count, -1, -1).clone(),
        intrinsics=torch.tensor([[4.0, 0.0, 3.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]).expand(count, -1, -1).clone(),
        image_sizes=torch.tensor([[height, width]], dtype=torch.int64).expand(count, -1).clone(),
    )


def test_conditions_contract():
    TextCondition("a chair", negative_text="")
    ImageCondition(torch.zeros(3, 8, 6), camera=make_camera(1), mask=torch.ones(1, 8, 6))
    MultiViewCondition(
        torch.zeros(2, 3, 8, 6),
        cameras=make_camera(2),
        masks=torch.ones(2, 1, 8, 6),
    )

    assert diffusers_3d.TextCondition is TextCondition
    assert diffusers_3d.ImageCondition is ImageCondition
    assert diffusers_3d.MultiViewCondition is MultiViewCondition


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: TextCondition(" "),
        lambda: ImageCondition(torch.zeros(2, 8, 6)),
        lambda: ImageCondition(torch.zeros(3, 8, 6), camera=make_camera(2)),
        lambda: ImageCondition(torch.zeros(3, 8, 6), mask=torch.full((1, 8, 6), 2.0)),
        lambda: MultiViewCondition(torch.zeros(2, 3, 8, 6), cameras=make_camera(1)),
        lambda: MultiViewCondition(torch.zeros(2, 3, 8, 6), cameras=make_camera(2, height=7)),
    ],
)
def test_conditions_reject_invalid_values(constructor):
    with pytest.raises(Object3DValidationError):
        constructor()


def test_condition_to_preserves_integer_dtypes_and_gradients():
    images = torch.ones(2, 3, 8, 6, requires_grad=True)
    condition = MultiViewCondition(images, make_camera(2))
    moved = condition.to(dtype=torch.float64)

    assert moved.images.dtype is torch.float64
    assert moved.cameras.image_sizes.dtype is torch.int64
    moved.images.sum().backward()
    assert images.grad is not None
