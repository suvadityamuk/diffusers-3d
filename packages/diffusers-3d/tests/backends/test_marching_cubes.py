"""Plain-PyTorch marching cubes. Bit-for-bit parity with torchmcubes' CPU and CUDA kernels is checked in
``tests/families/triposr/test_reference_parity.py`` against the pinned torchmcubes source."""

import torch

from diffusers_3d.backends.marching_cubes import _CORNERS, _TRIANGLES, marching_cubes, unique_rows_in_order


def _sphere(size: int, radius: float) -> torch.Tensor:
    axis = torch.linspace(-1, 1, size)
    return radius - torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij")).norm(dim=0)


def test_single_inside_corner_gives_one_triangle_at_the_edge_midpoints():
    volume = torch.ones(2, 2, 2)
    volume[0, 0, 0] = -1
    vertices, faces = marching_cubes(volume)
    # Bourke case 1 (edges 0, 8, 3) on torchmcubes' corner layout: the k, j, and i edges out of the corner.
    assert torch.equal(vertices, torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]]))
    assert torch.equal(faces, torch.tensor([[0, 1, 2]]))


def test_sphere_vertices_lie_on_the_surface_and_faces_point_below_the_threshold():
    volume = _sphere(24, 0.7)
    vertices, faces = marching_cubes(volume)
    centred = vertices / 23 * 2 - 1
    torch.testing.assert_close(centred.norm(dim=-1), torch.full((len(vertices),), 0.7), atol=0.02, rtol=0)
    assert torch.linalg.det(centred[faces]).sum() > 0
    flipped_vertices, flipped_faces = marching_cubes(-volume)
    assert torch.linalg.det((flipped_vertices / 23 * 2 - 1)[flipped_faces]).sum() < 0


def test_vertices_are_numbered_in_order_of_first_appearance():
    _, faces = marching_cubes(_sphere(12, 0.6))
    first_seen = list(dict.fromkeys(faces.flatten().tolist()))
    assert first_seen == list(range(len(first_seen)))


def test_threshold_and_empty_volumes():
    volume = _sphere(10, 0.5)
    vertices, faces = marching_cubes(volume + 0.25, threshold=0.25)
    expected_vertices, expected_faces = marching_cubes(volume)
    torch.testing.assert_close(vertices, expected_vertices, atol=1e-5, rtol=0)
    assert torch.equal(faces, expected_faces)
    vertices, faces = marching_cubes(torch.ones(4, 4, 4))
    assert vertices.shape == (0, 3) and faces.shape == (0, 3) and faces.dtype == torch.int64


def test_unique_rows_in_order():
    rows = torch.tensor([[3, 1], [0, 0], [3, 1], [2, 2], [0, 0]])
    first, inverse = unique_rows_in_order(rows)
    assert first.tolist() == [0, 1, 3]
    assert inverse.tolist() == [0, 1, 0, 2, 1]


def test_cuda_kernel_keeps_four_of_five_triangles_and_does_not_merge_vertices():
    five = [code for code, edges in enumerate(_TRIANGLES) if len(edges) == 15]
    assert len(five) == 32
    for code in five:
        volume = torch.ones(2, 2, 2)
        for bit, corner in enumerate(_CORNERS):
            if (code >> bit) & 1:
                volume[corner] = -1
        cpu_vertices, cpu_faces = marching_cubes(volume)
        vertices, faces = marching_cubes(volume, _cuda_kernel=True)
        assert cpu_faces.shape == (5, 3) and faces.shape == (4, 3)
        assert torch.equal(faces, torch.arange(12).reshape(4, 3))
        assert torch.equal(vertices, cpu_vertices[cpu_faces[:4]].reshape(-1, 3))


def test_cuda_kernel_soup_matches_the_cpu_mesh_where_no_cell_has_five_triangles():
    volume = _sphere(14, 0.6)
    cpu_vertices, cpu_faces = marching_cubes(volume)
    vertices, faces = marching_cubes(volume, _cuda_kernel=True)
    assert torch.equal(vertices, cpu_vertices[cpu_faces].reshape(-1, 3))
    assert torch.equal(faces, torch.arange(len(vertices)).reshape(-1, 3))
    vertices, faces = marching_cubes(torch.ones(4, 4, 4), _cuda_kernel=True)
    assert vertices.shape == (0, 3) and faces.shape == (0, 3) and faces.dtype == torch.int64
