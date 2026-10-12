import pytest
import torch

from diffusers_3d import (
    GaussianSplatAsset,
    Object3DKind,
    Object3DValidationError,
    TensorShapeError,
)


def gaussian_arguments() -> dict:
    return {
        "means": torch.zeros(2, 3),
        "log_scales": torch.zeros(2, 3),
        "quaternions_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1).clone(),
        "opacity_logits": torch.zeros(2, 1),
        "sh_coefficients": torch.zeros(2, 4, 3),
        "active_sh_degree": 1,
    }


def test_valid_gaussian_contract(gaussian):
    assert gaussian.kind is Object3DKind.GAUSSIAN_SPLAT
    assert gaussian.object_to_world is gaussian.transform
    assert gaussian.sh_coefficients.shape == (2, 4, 3)
    gaussian.validate(expensive=True)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("log_scales", torch.zeros(2, 1), "log_scales"),
        ("quaternions_wxyz", torch.zeros(2, 3), "quaternions_wxyz"),
        ("opacity_logits", torch.zeros(2, 2), "opacity_logits"),
        ("sh_coefficients", torch.zeros(2, 4, 4), "sh_coefficients"),
    ],
)
def test_gaussian_rejects_misaligned_channels(field, value, match):
    arguments = gaussian_arguments()
    arguments[field] = value
    with pytest.raises(Object3DValidationError, match=match):
        GaussianSplatAsset(**arguments)


def test_gaussian_requires_normalized_quaternions():
    arguments = gaussian_arguments()
    arguments["quaternions_wxyz"][:, 0] = 1.01
    with pytest.raises(Object3DValidationError, match="unit length"):
        GaussianSplatAsset(**arguments)

    arguments["quaternions_wxyz"][:, 0] = 1.0005
    GaussianSplatAsset(**arguments)


def test_gaussian_validates_scale_opacity_and_finite_values():
    arguments = gaussian_arguments()
    arguments["log_scales"][0, 0] = 1000
    with pytest.raises(Object3DValidationError, match="finite positive scales"):
        GaussianSplatAsset(**arguments)

    arguments = gaussian_arguments()
    arguments["opacity_logits"][0, 0] = float("nan")
    with pytest.raises(Object3DValidationError, match="finite"):
        GaussianSplatAsset(**arguments)


def test_gaussian_validates_sh_basis_and_active_degree():
    arguments = gaussian_arguments()
    arguments["sh_coefficients"] = torch.zeros(2, 3, 3)
    arguments["active_sh_degree"] = 0
    with pytest.raises(TensorShapeError, match="squared SH basis"):
        GaussianSplatAsset(**arguments)

    arguments = gaussian_arguments()
    arguments["active_sh_degree"] = 2
    with pytest.raises(Object3DValidationError, match="between 0 and 1"):
        GaussianSplatAsset(**arguments)

    arguments["active_sh_degree"] = True
    with pytest.raises(Object3DValidationError, match="integer"):
        GaussianSplatAsset(**arguments)


def test_gaussian_to_casts_floats_but_not_integer_extras(gaussian):
    moved = gaussian.to(dtype=torch.float64)
    assert moved.means.dtype is torch.float64
    assert moved.sh_coefficients.dtype is torch.float64
    assert moved.extras["labels"].dtype is torch.int64
    assert moved.active_sh_degree == gaussian.active_sh_degree
