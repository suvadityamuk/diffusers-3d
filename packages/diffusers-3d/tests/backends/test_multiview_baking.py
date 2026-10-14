from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from diffusers_3d.backends import (
    canny,
    inpaint_texture,
    look_at_view,
    merge_views,
    orthographic_projection,
    project_view,
    rasterize,
    rasterize_texels,
    splat_view,
)

VIEWS = ((0, 0), (0, 90), (0, 180), (0, 270), (90, 0), (-90, 180))
FACE_COLORS = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
)


def _cube() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A tilted cube in MeshRender's frame with every side in its own UV cell; returns vertices, faces, uvs, sides.

    The tilt shows every view several sides at varying depth (a view of one flat side has no depth range).
    """

    corners = torch.tensor([[x, y, z] for x in (-0.3, 0.3) for y in (-0.3, 0.3) for z in (-0.3, 0.3)])
    # Outward counter-clockwise quads of a Y-up cube, mapped into MeshRender's mirrored frame (-x, z, -y).
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    yaw, pitch = math.radians(30), math.radians(20)
    rotation = torch.tensor(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    ) @ torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]]
    )
    corners = torch.stack([-corners[:, 0], corners[:, 2], -corners[:, 1]], dim=1) @ rotation.T
    vertices, faces, uvs, sides = [], [], [], []
    for side, quad in enumerate(quads):
        origin = torch.tensor([side % 3 / 3, side // 3 / 2])
        base = len(vertices)
        for local, corner in enumerate(quad):
            vertices.append(corners[corner])
            uvs.append(origin + torch.tensor([[0.04, 0.04], [0.29, 0.04], [0.29, 0.46], [0.04, 0.46]][local]))
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        sides += [side, side]
    return torch.stack(vertices), torch.tensor(faces), torch.stack(uvs), torch.tensor(sides)


def _views(vertices, faces, sides, size):
    """Renders that colour every cube side with its own colour on a white background."""

    projection = orthographic_projection(1.2)
    images = []
    for elevation, azimuth in VIEWS:
        camera = (
            torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1) @ look_at_view(elevation, azimuth, 1.45).T
        )
        face = rasterize(camera @ projection.T, faces, (size, size), align_corners=True).face_index[0]
        images.append(torch.where((face >= 0)[..., None], FACE_COLORS[sides[face.clamp(min=0)]], 1.0))
    return images


def _side_colors(vertices, faces, uvs, sides, size):
    """The expected texture: each covered texel takes its cube side's colour."""

    clip = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=1)
    clip[:, 1] = (1 - uvs[:, 1]) * 2 - 1
    face = rasterize(clip, faces, (size, size), align_corners=True).face_index[0]
    return FACE_COLORS[sides[face.clamp(min=0)]], face >= 0


def test_look_at_view_places_cameras_in_meshrender_convention():
    front = torch.linalg.inv(look_at_view(0, 0, 1.45))
    torch.testing.assert_close(front[:3, 3], torch.tensor([0.0, 1.45, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(front[:3, 2], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0)  # camera +z faces away
    torch.testing.assert_close(front[:3, 1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)  # up is +z
    top = torch.linalg.inv(look_at_view(90, 0, 2.0))
    torch.testing.assert_close(top[:3, 3], torch.tensor([0.0, 0.0, -2.0]), atol=1e-6, rtol=0)  # elevation is negated
    for elevation, azimuth in VIEWS:
        rotation = look_at_view(elevation, azimuth, 1.0)[:3, :3]
        torch.testing.assert_close(rotation @ rotation.T, torch.eye(3), atol=1e-6, rtol=0)
    projection = orthographic_projection(1.2)
    assert projection[0, 0] == projection[1, 1] == torch.tensor(2 / 1.2, dtype=torch.float32)


def test_canny_keeps_the_dark_side_of_a_step_and_matches_opencv():
    image = torch.zeros(12, 16, dtype=torch.uint8)
    image[:, 7:] = 200
    edges = canny(image)
    assert edges[:, 6].all() and edges.sum() == 12
    cv2 = pytest.importorskip("cv2")
    generator = np.random.default_rng(0)
    for array in (
        generator.integers(0, 256, (40, 50), dtype=np.uint8),
        np.kron(generator.integers(0, 4, (6, 8)) * 60, np.ones((7, 7))).astype(np.uint8),
    ):
        assert np.array_equal(canny(torch.from_numpy(array)).numpy(), cv2.Canny(array, 30, 80) > 0)


@pytest.mark.parametrize("method", ["project", "splat"])
def test_baking_rendered_views_recovers_the_texture(method):
    vertices, faces, uvs, sides = _cube()
    expected, covered = _side_colors(vertices, faces, uvs, sides, 64)
    texels = rasterize_texels(vertices, faces, uvs, 64)
    assert torch.equal(covered.flatten().nonzero()[:, 0], texels.texel_index)
    projection = orthographic_projection(1.2)
    textures, weights = [], []
    for image, (elevation, azimuth) in zip(_views(vertices, faces, sides, 96), VIEWS):
        world_to_camera = look_at_view(elevation, azimuth, 1.45)
        if method == "project":
            texture, cosine = project_view(
                image, vertices, faces, world_to_camera, projection, texels, 64, unreliable_radius=1
            )
        else:
            texture, cosine = splat_view(
                image, vertices, faces, uvs, world_to_camera, projection, 64, unreliable_radius=1
            )
        textures.append(texture)
        weights.append(cosine**4)
    texture, mask = merge_views(textures, weights)
    # Splatting also reaches texels just outside the atlas triangles; sampling only fills covered ones.
    assert method == "splat" or not (mask & ~covered).any()
    assert (mask & covered).sum() > 0.7 * covered.sum()
    close = ((texture - expected).abs().amax(dim=-1) < 1e-4)[mask & covered]
    assert close.float().mean() > 0.9


def test_merge_views_skips_a_view_already_covered_and_weights_the_rest():
    first, second, third = torch.zeros(3, 4, 4, 3), torch.ones(4, 4, 3), torch.full((4, 4, 3), 0.5)
    first_weight = torch.zeros(4, 4)
    first_weight[:2] = 1.0
    second_weight = torch.zeros(4, 4)
    second_weight[:2, :2] = 3.0
    third_weight = torch.zeros(4, 4)
    third_weight[1:3] = 1.0
    texture, mask = merge_views([first[0], second, third], [first_weight, second_weight, third_weight])
    assert torch.equal(mask, (first_weight + third_weight) > 0)
    assert torch.equal(texture[0], torch.zeros(4, 3))  # the second view is skipped
    torch.testing.assert_close(texture[1], torch.full((4, 3), 0.25))
    torch.testing.assert_close(texture[2], torch.full((4, 3), 0.5))


def test_inpaint_keeps_known_texels_and_propagates_vertex_colors_over_the_mesh():
    vertices, faces, uvs, sides = _cube()
    texture, covered = _side_colors(vertices, faces, uvs, sides, 64)
    known = covered.clone()
    known[:32] = False
    filled = inpaint_texture(torch.where(known[..., None], texture, 0.0), known, vertices, faces, uvs)
    assert torch.equal(filled[known], texture[known])
    assert torch.isfinite(filled).all() and (filled[~known] > 0).any(dim=-1).all()

    # Vertex 0 sits in a blue neighbourhood of the atlas, but its mesh neighbour (vertex 1) is red.
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uvs = torch.tensor([[0.25, 0.5], [0.875, 0.875], [0.875, 0.125]])  # texel centres of a 17x17 texture
    texture = torch.tensor([0.0, 0.0, 1.0]).expand(17, 17, 3).clone()
    known = torch.ones(17, 17, dtype=torch.bool)
    known[6:11, 2:7] = False
    texture[2, 14] = texture[14, 14] = torch.tensor([1.0, 0.0, 0.0])
    filled = inpaint_texture(texture, known, vertices, torch.tensor([[0, 1, 2]]), uvs)
    torch.testing.assert_close(filled[8, 4], torch.tensor([1.0, 0.0, 0.0]))
    assert filled[6, 2, 2] > 0.9  # the rest of the hole is filled from the blue around it


def test_baking_is_deterministic():
    vertices, faces, uvs, sides = _cube()
    texels = rasterize_texels(vertices, faces, uvs, 32)
    image = _views(vertices, faces, sides, 48)[0]
    camera = (look_at_view(0, 0, 1.45), orthographic_projection(1.2))
    for _ in range(2):
        projected = project_view(image, vertices, faces, *camera, texels, 32, unreliable_radius=1)
        splatted = splat_view(image, vertices, faces, uvs, *camera, 32, unreliable_radius=1)
        if _ == 0:
            first = projected + splatted
    assert all(torch.equal(a, b) for a, b in zip(first, projected + splatted))
