from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from test_texture_baking import _cube_with_manual_atlas, _render_cube

from diffusers_3d.backends import Rasterization, interpolate, rasterize, rasterize_mesh
from diffusers_3d.backends.texture_baking import pixel_barycentrics, sphere_hammersley_cameras

HEIGHT, WIDTH = 24, 32


def _random_clip_scene(seed: int, views: int = 2, faces: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlapping, interpenetrating triangles of both windings with perspective ``w`` and some behind the near plane."""

    generator = torch.Generator().manual_seed(seed)
    ndc = torch.rand(views, 3 * faces, 3, generator=generator) * 2.4 - 1.2
    ndc[..., 2] = torch.rand(views, 3 * faces, generator=generator) * 2.2 - 1.1
    w = torch.rand(views, 3 * faces, 1, generator=generator) * 2 + 0.5
    return torch.cat([ndc * w, w], dim=-1), torch.arange(3 * faces).reshape(faces, 3)


def _brute_force(clip, faces, align_corners, cull_backfaces):
    """Every face tested against every pixel centre, nearest quantized depth then lowest face index wins."""

    views, face_count = clip.shape[0], faces.shape[0]
    size = torch.tensor([WIDTH, HEIGHT], dtype=clip.dtype)
    ndc = clip[..., :3] / clip[..., 3:]
    xy = (ndc[..., :2] * 0.5 + 0.5) * ((size - 1) if align_corners else size) + (0.5 if align_corners else 0.0)
    face_index = torch.full((views, HEIGHT * WIDTH), -1)
    barycentrics = torch.zeros(views, HEIGHT * WIDTH, 3)
    for view in range(views):
        corners = xy[view, faces]
        everywhere = torch.zeros(face_count, 2, dtype=torch.int64)
        face, column, row, weights = pixel_barycentrics(corners, everywhere, size.long().expand(face_count, 2))
        depth = (weights * (ndc[view, faces, 2] * 0.49999 + 0.5)[face]).sum(dim=1)
        edges = corners[:, 1:] - corners[:, :1]
        front = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0] > 0
        hit = (weights >= 0).all(dim=1) & (depth >= 0) & (front[face] | (not cull_backfaces))
        key = torch.where(hit, (depth * 2**18).long() * 2**32 + face, torch.iinfo(torch.int64).max)
        dense = torch.full((face_count, HEIGHT * WIDTH), torch.iinfo(torch.int64).max)
        dense[face, row * WIDTH + column] = key
        best = dense.amin(dim=0)
        covered = best != torch.iinfo(torch.int64).max
        winner = best[covered] % 2**32
        pair = winner * HEIGHT * WIDTH + covered.nonzero()[:, 0]
        lookup = torch.full((face_count * HEIGHT * WIDTH,), -1)
        lookup[face * HEIGHT * WIDTH + row * WIDTH + column] = torch.arange(face.shape[0])
        perspective = weights[lookup[pair]] / clip[view, faces[winner], 3]
        face_index[view, covered] = winner
        barycentrics[view, covered] = perspective / perspective.sum(dim=1, keepdim=True)
    return face_index.reshape(views, HEIGHT, WIDTH), barycentrics.reshape(views, HEIGHT, WIDTH, 3)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("cull_backfaces", [False, True])
def test_rasterize_matches_brute_force(align_corners, cull_backfaces):
    clip, faces = _random_clip_scene(0)
    result = rasterize(
        clip, faces, (HEIGHT, WIDTH), align_corners=align_corners, cull_backfaces=cull_backfaces, chunk_faces=7
    )
    face_index, barycentrics = _brute_force(clip, faces, align_corners, cull_backfaces)

    assert result.face_index.shape == (2, HEIGHT, WIDTH)
    assert 0.3 < (face_index >= 0).float().mean() < 1.0  # both coverage and background are exercised
    torch.testing.assert_close(result.face_index, face_index, rtol=0, atol=0)
    torch.testing.assert_close(result.barycentrics, barycentrics)
    # Perspective-correct barycentrics reproduce the interpolated NDC point under the pixel centre.
    ndc = clip[..., :3] / clip[..., 3:]
    covered = result.face_index >= 0
    depth = interpolate(clip[..., 2:], result, faces) / interpolate(clip[..., 3:], result, faces)
    torch.testing.assert_close(result.depth[covered], depth[..., 0][covered], atol=1e-5, rtol=1e-5)
    assert bool((result.depth[covered] >= -1.0001).all()) and bool((ndc[..., 2] < -1).any())


def test_nearest_face_wins_and_exact_ties_go_to_the_lowest_index():
    # A large back triangle, a smaller front one listed later, and a flipped copy of the front one nearer by less
    # than one 2^-18 depth quantum, which ties with it.
    front = torch.tensor([[-0.5, -0.5, -0.5, 1], [0.5, -0.5, -0.5, 1], [-0.5, 0.5, -0.5, 1]])
    back = torch.tensor([[-1, -1, 0.5, 1], [3, -1, 0.5, 1], [-1, 3, 0.5, 1]])
    vertices = torch.cat([back, front, front - torch.tensor([0, 0, 1e-6, 0])])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 8, 7]])
    result = rasterize(vertices, faces, (8, 8))
    assert set(result.face_index.unique().tolist()) == {0, 1}
    assert result.face_index[0, 2, 2] == 1 and result.face_index[0, 7, 7] == 0
    torch.testing.assert_close(result.depth[0, 2, 2], torch.tensor(-0.5))
    # Reversed order: the front pair still wins, and of the tied pair the lower index.
    reversed_result = rasterize(vertices, faces.flip(0), (8, 8))
    torch.testing.assert_close(reversed_result.face_index[0, 2, 2], torch.tensor(0))
    # Interpolated attributes use their own index buffer and are zero on the background.
    values = torch.arange(6.0)[:, None]
    attribute = interpolate(values, rasterize(vertices, faces[1:2], (8, 8)), faces[1:2])
    assert attribute.shape == (1, 8, 8, 1) and attribute[0, 7, 7, 0] == 0
    torch.testing.assert_close(attribute[0, 2, 2, 0], (result.barycentrics[0, 2, 2] * values[3:, 0]).sum())


def _custom_rasterizer_reference(clip: torch.Tensor, faces: torch.Tensor, height: int, width: int):
    """Float32 NumPy restatement of ``custom_rasterizer.rasterize`` as Hunyuan3D-2.1 (82920d6) and Step1X-3D (cb5ac94)
    describe it: pixel ``(i, j)`` samples ``(j + 0.5, i + 0.5)`` with NDC mapped by ``(n * 0.5 + 0.5) * (size - 1) + 0.5``;
    a face covers it when all screen barycentrics lie in ``[0, 1]``; the integer token ``trunc(depth * 2^18)`` then
    face id decides visibility for window depth ``>= 0``; winners get screen barycentrics divided by ``w``.
    """

    one, half = np.float32(1), np.float32(0.5)
    clip = clip.numpy().astype(np.float32)
    ndc = clip[:, :3] / clip[:, 3:]
    screen = np.stack(
        [
            (ndc[:, 0] * half + half) * np.float32(width - 1) + half,
            (ndc[:, 1] * half + half) * np.float32(height - 1) + half,
            ndc[:, 2] * np.float32(0.49999) + half,
        ],
        axis=1,
    )

    def weights(a, b, c, p):
        def area(u, v, w):
            return (w[0] - u[0]) * (v[1] - u[1]) - (v[0] - u[0]) * (w[1] - u[1])

        total = area(a, b, c)
        if total == 0:
            return None
        beta, gamma = area(a, p, c) / total, area(a, b, p) / total
        return np.array([one - beta - gamma, beta, gamma], dtype=np.float32)

    token = np.full((height, width), np.iinfo(np.int64).max, dtype=np.int64)
    for face, corner_ids in enumerate(faces.tolist()):
        a, b, c = screen[corner_ids]
        for column in range(
            max(0, math.floor(min(a[0], b[0], c[0]))), min(width, math.ceil(max(a[0], b[0], c[0])) + 1)
        ):
            for row in range(
                max(0, math.floor(min(a[1], b[1], c[1]))), min(height, math.ceil(max(a[1], b[1], c[1])) + 1)
            ):
                w = weights(a, b, c, np.array([column + 0.5, row + 0.5], dtype=np.float32))
                if w is None or not ((w >= 0) & (w <= 1)).all():
                    continue
                depth = w[0] * a[2] + w[1] * b[2] + w[2] * c[2]
                if depth >= 0:
                    token[row, column] = min(token[row, column], int(depth * 2**18) * 2**32 + face)
    face_index = np.where(token == np.iinfo(np.int64).max, -1, token % 2**32)
    barycentrics = np.zeros((height, width, 3), dtype=np.float32)
    for row, column in zip(*np.nonzero(face_index >= 0)):
        corner_ids = faces[face_index[row, column]].tolist()
        a, b, c = screen[corner_ids]
        w = weights(a, b, c, np.array([column + 0.5, row + 0.5], dtype=np.float32)) / clip[corner_ids, 3]
        barycentrics[row, column] = w / w.sum()
    return torch.from_numpy(face_index), torch.from_numpy(barycentrics)


def test_align_corners_reproduces_custom_rasterizer_semantics():
    clip, faces = _random_clip_scene(1, views=1)
    clip = clip[0]
    result = rasterize(clip, faces, (HEIGHT, WIDTH), align_corners=True)
    face_index, barycentrics = _custom_rasterizer_reference(clip, faces, HEIGHT, WIDTH)

    # Formula-level rounding may only flip pixel centres that sit on an edge.
    on_edge = (result.barycentrics[0].amin(dim=-1) < 1e-5) & (result.face_index[0] >= 0)
    on_edge |= (barycentrics.amin(dim=-1) < 1e-5) & (face_index >= 0)
    assert (result.face_index[0] != face_index)[~on_edge].sum() == 0
    same = result.face_index[0] == face_index
    torch.testing.assert_close(result.barycentrics[0][same], barycentrics[same], atol=1e-5, rtol=0)


def test_rasterize_mesh_matches_an_analytic_render_of_the_cube():
    mesh = _cube_with_manual_atlas()
    cameras = sphere_hammersley_cameras(6, image_size=48, coordinate_system="right_handed_z_up")
    rendered = rasterize_mesh(mesh, cameras)
    _, depths, alphas = _render_cube(cameras, 48)

    assert rendered["depth"].shape == rendered["alpha"].shape == (6, 1, 48, 48)
    assert (rendered["alpha"] != alphas).float().mean() < 0.005  # pixel centres on the silhouette only
    both = (rendered["alpha"] > 0) & (alphas > 0)
    torch.testing.assert_close(rendered["depth"][both], depths[both], atol=1e-4, rtol=1e-4)
    # Interpolated surface points project back onto their pixel centres (perspective-correct barycentrics).
    camera_points = (
        torch.cat([mesh.vertices, torch.ones(mesh.vertices.shape[0], 1)], dim=1)
        @ cameras.world_to_camera.transpose(1, 2)
    )[..., :3]
    rasterization = Rasterization(rendered["face_index"], rendered["barycentrics"], rendered["depth"][:, 0])
    points = interpolate(camera_points, rasterization, mesh.faces)
    pixels = points @ cameras.intrinsics.transpose(1, 2)[:, None]
    pixels = pixels[..., :2] / pixels[..., 2:]
    rows, columns = torch.meshgrid(torch.arange(48) + 0.5, torch.arange(48) + 0.5, indexing="ij")
    covered = rasterization.face_index >= 0
    torch.testing.assert_close(pixels[covered], torch.stack([columns, rows], dim=-1).expand(6, -1, -1, -1)[covered])
    with pytest.raises(ValueError, match="same coordinate system"):
        rasterize_mesh(mesh.to_coordinate_system("right_handed_y_up"), cameras)


@pytest.mark.accelerated
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_agrees_with_cpu():
    clip, faces = _random_clip_scene(2, views=3, faces=400)
    cpu = rasterize(clip, faces, (96, 128))
    mps = rasterize(clip.to("mps"), faces.to("mps"), (96, 128))
    same = mps.face_index.cpu() == cpu.face_index
    assert same.float().mean() > 0.999
    torch.testing.assert_close(mps.barycentrics.cpu()[same], cpu.barycentrics[same], atol=1e-4, rtol=0)
