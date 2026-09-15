from __future__ import annotations

import math

import pytest
import torch

from diffusers_3d import CameraRig, CoordinateSystem, RadianceFieldAsset
from diffusers_3d.backends.radiance_field import SH_C0, render_radiance_field, sample_radiance_field
from diffusers_3d.families.trellis.sparse import trellis_grid_transform

RESOLUTION = 8


def _asset(coordinates, density, colors, *, trivec=None):
    count = len(coordinates)
    return RadianceFieldAsset(
        coordinates=torch.tensor(coordinates),
        trivec=torch.ones(count, 1, 3, 8) if trivec is None else trivec,
        density=torch.tensor(density).reshape(count, 1),
        color_coefficients=torch.tensor(colors).reshape(count, 1, 3),
        resolution=RESOLUTION,
        grid_transform=trellis_grid_transform(RESOLUTION),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )


def _camera_above(target_xy, height=2.0, focal=64.0, size=32):
    """OpenCV camera on the +z axis looking straight down at ``target_xy``."""

    camera_to_world = torch.eye(4)
    camera_to_world[:3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    camera_to_world[:3, 3] = torch.tensor([target_xy[0], target_xy[1], height])
    intrinsics = torch.tensor([[focal, 0.0, size / 2], [0.0, focal, size / 2], [0.0, 0.0, 1.0]])
    return CameraRig(
        world_to_camera=torch.linalg.inv(camera_to_world).unsqueeze(0),
        intrinsics=intrinsics.unsqueeze(0),
        image_sizes=torch.tensor([[size, size]]),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )


def test_constant_voxel_matches_the_closed_form_transmittance():
    # Voxel (3, 3, 3) spans [-0.125, 0]^3 in object space; the centre ray crosses 1/8 world units of it.
    asset = _asset([[3, 3, 3]], [2.0], [[3.0, -3.0, 0.0]])
    cameras = _camera_above((-0.0625, -0.0625))

    rendered = render_radiance_field(asset, cameras, samples_per_cell=4)

    alpha = rendered["alpha"][0, 0, 16, 16]
    expected_alpha = 1.0 - math.exp(-float(torch.nn.functional.softplus(torch.tensor(2.0))) / RESOLUTION)
    assert abs(float(alpha) - expected_alpha) < 1e-3
    expected_color = torch.sigmoid(SH_C0 * torch.tensor([3.0, -3.0, 0.0])) * alpha
    torch.testing.assert_close(rendered["color"][0, :, 16, 16], expected_color, atol=1e-3, rtol=0)
    # Expected depth lies inside the voxel's z extent, which the camera sees at z in [2.0, 2.125].
    assert 2.0 <= float(rendered["depth"][0, 0, 16, 16]) <= 2.125
    assert float(rendered["alpha"][0, 0, 2, 2]) == 0.0 and float(rendered["depth"][0, 0, 2, 2]) == 0.0
    # 0.125 world units at focal 64 from 2 units away covers 4 pixels per side.
    assert int((rendered["alpha"][0, 0] > 0).sum()) == 16
    # Background shows through where the field is transparent.
    with_background = render_radiance_field(asset, cameras, background=(1.0, 1.0, 1.0))
    torch.testing.assert_close(with_background["color"][0, :, 2, 2], torch.ones(3))


def test_front_voxel_occludes_the_one_behind_it():
    # Two voxels stacked along z under the camera: the upper one is nearly opaque and red-ish, the lower green-ish.
    asset = _asset([[3, 3, 4], [3, 3, 3]], [40.0, 40.0], [[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]])
    cameras = _camera_above((-0.0625, -0.0625))

    rendered = render_radiance_field(asset, cameras)

    pixel = rendered["color"][0, :, 16, 16]
    assert pixel[0] > 0.9 and pixel[1] < 0.1, pixel
    assert float(rendered["alpha"][0, 0, 16, 16]) > 0.99
    # Depth sits just inside the front face of the upper voxel (object z = 0.125 -> camera z = 1.875).
    assert 1.875 <= float(rendered["depth"][0, 0, 16, 16]) < 1.93


def test_trivec_interpolation_varies_inside_a_voxel_and_extrapolates_at_the_border():
    trivec = torch.ones(1, 1, 3, 8)
    trivec[0, 0, 0] = torch.linspace(0.0, 7.0, 8)  # component grows along x, flat along y and z
    asset = _asset([[0, 0, 0]], [0.0], [[0.0, 0.0, 0.0]], trivec=trivec)
    rows = torch.zeros(4, dtype=torch.int64)
    local = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [0.75, 0.2, 0.9]])

    density, color = sample_radiance_field(asset, rows, local)

    # Cell centres sit at (i + 0.5) / 8, so x = 0.5 reads 3.5; the border reads the extrapolated 0-th segment.
    component = torch.tensor([3.5, -0.5, 7.5, 5.5])
    torch.testing.assert_close(density, torch.nn.functional.softplus(0.0 * component))
    torch.testing.assert_close(color, torch.full((4, 3), 0.5))


def test_renderer_rejects_mismatched_frames_and_batches_cameras():
    asset = _asset([[3, 3, 3]], [2.0], [[0.0, 0.0, 0.0]])
    cameras = _camera_above((-0.0625, -0.0625))
    two = CameraRig(
        world_to_camera=cameras.world_to_camera.repeat(2, 1, 1),
        intrinsics=cameras.intrinsics.repeat(2, 1, 1),
        image_sizes=torch.tensor([[32, 32], [32, 32]]),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    rendered = render_radiance_field(asset, two)
    assert rendered["color"].shape == (2, 3, 32, 32) and rendered["depth"].shape == (2, 1, 32, 32)
    torch.testing.assert_close(rendered["color"][0], rendered["color"][1])
    y_up = CameraRig(
        world_to_camera=cameras.world_to_camera,
        intrinsics=cameras.intrinsics,
        image_sizes=cameras.image_sizes,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Y_UP,
    )
    with pytest.raises(ValueError, match="same coordinate system"):
        render_radiance_field(asset, y_up)
