import io
import math

import numpy as np
import pytest
import torch

from diffusers_3d import CoordinateSystem, GaussianSplatAsset, MeshAsset

Z_UP = CoordinateSystem.RIGHT_HANDED_Z_UP
Y_UP = CoordinateSystem.RIGHT_HANDED_Y_UP


def _quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _splats(coefficients: int = 4, coordinate_system=Y_UP) -> GaussianSplatAsset:
    generator = torch.Generator().manual_seed(0)
    quaternions = torch.randn(5, 4, generator=generator)
    return GaussianSplatAsset(
        means=torch.randn(5, 3, generator=generator),
        log_scales=torch.randn(5, 3, generator=generator),
        quaternions_wxyz=quaternions / quaternions.norm(dim=1, keepdim=True),
        opacity_logits=torch.randn(5, 1, generator=generator),
        sh_coefficients=torch.randn(5, coefficients, 3, generator=generator),
        active_sh_degree=math.isqrt(coefficients) - 1,
        coordinate_system=coordinate_system,
    )


def test_splat_ply_uses_the_reference_layout_and_round_trips():
    splats = _splats()
    buffer = io.BytesIO()
    splats.save(buffer)
    data = buffer.getvalue()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    names = [line.split()[-1] for line in data[:end].decode().splitlines() if line.startswith("property")]
    table = np.frombuffer(data, dtype=[(name, "<f4") for name in names], offset=end)
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[-8:] == ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    # f_rest is channel-major: the three red coefficients come first, then green, then blue.
    np.testing.assert_allclose(table["f_rest_1"], splats.sh_coefficients[:, 2, 0].numpy(), rtol=1e-6)
    np.testing.assert_allclose(table["f_rest_3"], splats.sh_coefficients[:, 1, 1].numpy(), rtol=1e-6)
    np.testing.assert_allclose(table["rot_0"], splats.quaternions_wxyz[:, 0].numpy(), rtol=1e-6)

    buffer.seek(0)
    loaded = GaussianSplatAsset.load(buffer)
    assert loaded.active_sh_degree == 1
    for name in ("means", "log_scales", "quaternions_wxyz", "opacity_logits", "sh_coefficients"):
        torch.testing.assert_close(getattr(loaded, name), getattr(splats, name))


def test_splats_rotate_to_y_up_with_their_covariance():
    splats = _splats(coefficients=1, coordinate_system=Z_UP)
    rotated = splats.to_coordinate_system(Y_UP)
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    torch.testing.assert_close(rotated.means, splats.means @ matrix.T)
    torch.testing.assert_close(
        _quaternion_matrix(rotated.quaternions_wxyz), matrix @ _quaternion_matrix(splats.quaternions_wxyz)
    )
    with pytest.raises(ValueError, match="higher-order"):
        _splats(coordinate_system=Z_UP).to_coordinate_system(Y_UP)


@pytest.mark.portable
def test_mesh_save_writes_y_up_without_extras_and_loads_back(tmp_path):
    pytest.importorskip("trimesh")
    mesh = MeshAsset(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        faces=torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        colors=torch.rand(4, 3),
        coordinate_system=Z_UP,
        extras={"normal_map": torch.zeros(4, 3)},
    )
    mesh.save(tmp_path / "mesh.glb")
    loaded = MeshAsset.load(tmp_path / "mesh.glb")
    assert loaded.coordinate_system is Y_UP
    torch.testing.assert_close(loaded.vertices, mesh.to_coordinate_system(Y_UP).vertices)


@pytest.mark.portable
def test_mesh_geometry_methods():
    pytest.importorskip("fast_simplification")
    pytest.importorskip("xatlas")
    angles = torch.linspace(0, 2 * math.pi, 33)[:-1]
    ring = torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], dim=1)
    vertices = torch.cat([ring, ring + torch.tensor([0.0, 0.0, 1.0])])
    count = ring.shape[0]
    faces = [[i, (i + 1) % count, count + i] for i in range(count)]
    faces += [[(i + 1) % count, count + (i + 1) % count, count + i] for i in range(count)]
    mesh = MeshAsset(vertices=vertices, faces=torch.tensor(faces), colors=torch.rand(2 * count, 3))

    simplified = mesh.simplify(16)
    assert simplified.faces.shape[0] <= 20 and simplified.colors is None
    assert mesh.repair().faces.shape[0] == mesh.faces.shape[0]
    unwrapped = mesh.unwrap_uv()
    assert unwrapped.uvs is not None and unwrapped.colors.shape[0] == unwrapped.vertices.shape[0]


@pytest.mark.portable
def test_mesh_save_follows_the_upstream_export_rotation(tmp_path):
    pytest.importorskip("trimesh")
    mesh = MeshAsset(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        faces=torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        coordinate_system=Z_UP,
        metadata={"export_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    )
    mesh.save(tmp_path / "raw.obj")
    loaded = MeshAsset.load(tmp_path / "raw.obj")
    torch.testing.assert_close(loaded.vertices, mesh.vertices)  # raw Z-up coordinates, as upstream writes them


def test_splat_save_follows_the_upstream_export_rotation():
    splats = _splats(coefficients=1, coordinate_system=Z_UP)
    rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    splats.metadata["export_rotation"] = rotation
    buffer = io.BytesIO()
    splats.save(buffer)
    buffer.seek(0)
    loaded = GaussianSplatAsset.load(buffer)
    matrix = torch.tensor(rotation)
    torch.testing.assert_close(loaded.means, splats.means @ matrix.T)
    torch.testing.assert_close(
        _quaternion_matrix(loaded.quaternions_wxyz), matrix @ _quaternion_matrix(splats.quaternions_wxyz)
    )
