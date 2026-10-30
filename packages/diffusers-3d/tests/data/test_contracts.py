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


def test_as_image_conditions_accepts_pil_arrays_paths_tensors_and_lists(tmp_path):
    import numpy as np
    from PIL import Image

    from diffusers_3d.data import ImageCondition, as_image_conditions

    rgba = np.zeros((6, 5, 4), dtype=np.uint8)
    rgba[2:4, 1:3] = 255
    Image.fromarray(rgba).save(tmp_path / "object.png")
    expected = torch.from_numpy(rgba).permute(2, 0, 1).float() / 255

    for value in (Image.fromarray(rgba), rgba, tmp_path / "object.png", str(tmp_path / "object.png")):
        (condition,) = as_image_conditions(value)
        assert torch.equal(condition.image, expected)
    (rgb,) = as_image_conditions(Image.fromarray(rgba[..., :3]))
    assert rgb.image.shape == (3, 6, 5)
    batch = as_image_conditions([expected, ImageCondition(image=expected), expected[None].expand(2, -1, -1, -1)])
    assert len(batch) == 4 and all(torch.equal(c.image, expected) for c in batch)
    with pytest.raises(TypeError, match="expected an image"):
        as_image_conditions([])


def test_background_remover_cuts_out_rgb_inputs_only():
    from PIL import Image

    from diffusers_3d.data import ImageCondition, remove_background

    calls = []

    def remover(image):
        calls.append(image.mode)
        alpha = Image.new("L", image.size, 0)
        alpha.paste(255, (1, 1, 3, 3))
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        return rgba

    rgb = ImageCondition(image=torch.full((3, 4, 4), 0.5))
    cut = remove_background(rgb, remover)
    assert cut.image.shape == (4, 4, 4) and calls == ["RGB"]
    assert cut.image[3, 1:3, 1:3].eq(1).all() and cut.image[3].sum() == 4
    rgba = ImageCondition(image=torch.ones(4, 4, 4))
    assert remove_background(rgba, remover).image[3].sum() == 4 and len(calls) == 2  # opaque alpha is no alpha
    rgba.image[3, 0, 0] = 0.0
    assert remove_background(rgba, remover) is rgba and len(calls) == 2
