from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from diffusers_3d import (
    CoordinateSystem,
    GeometryProcessingBackend,
    MeshAsset,
    PBRMaterial,
    ScikitImageBackend,
    SurfaceExtractionBackend,
    TrimeshBackend,
    XAtlasBackend,
)

pytestmark = pytest.mark.portable


@pytest.fixture
def cube() -> MeshAsset:
    return MeshAsset(
        vertices=torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        ),
        faces=torch.tensor(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [1, 2, 6],
                [1, 6, 5],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=torch.int64,
        ),
    )


def test_adapter_modules_are_lazy_until_initialization():
    script = """
import sys
import diffusers_3d
assert "trimesh" not in sys.modules
assert "skimage" not in sys.modules
assert "xatlas" not in sys.modules
assert diffusers_3d.TrimeshBackend.__name__ == "TrimeshBackend"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_trimesh_native_roundtrip_preserves_owned_channels(cube):
    pytest.importorskip("trimesh")
    backend = TrimeshBackend()
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([2.0, 3.0, 4.0])
    colors = torch.linspace(0.0, 1.0, cube.vertices.shape[0]).unsqueeze(1).expand(-1, 3)
    material = PBRMaterial(
        base_color=torch.tensor([0.2, 0.4, 0.6, 1.0]),
        metallic=torch.tensor(0.25),
        roughness=torch.tensor(0.75),
        metadata={"name": "cube"},
    )
    mesh = MeshAsset(
        vertices=cube.vertices.requires_grad_(True),
        faces=cube.faces,
        transform=transform,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        normals=torch.nn.functional.normalize(cube.vertices, dim=1),
        colors=colors,
        uvs=torch.rand(cube.vertices.shape[0], 2, generator=torch.Generator().manual_seed(0)),
        face_material_ids=torch.zeros(cube.faces.shape[0], dtype=torch.int64),
        materials=(material,),
        extras={
            "weights": torch.arange(cube.vertices.shape[0], dtype=torch.float32).unsqueeze(1),
            "face_labels": torch.arange(cube.faces.shape[0], dtype=torch.int32),
        },
        metadata={"source": "test"},
    )

    restored = backend.from_trimesh(backend.to_trimesh(mesh))

    assert restored.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert torch.equal(restored.transform, transform)
    assert torch.equal(restored.vertices, mesh.vertices.detach())
    assert torch.equal(restored.faces, mesh.faces)
    assert torch.equal(restored.normals, mesh.normals)
    assert torch.equal(restored.colors, colors)
    assert torch.allclose(restored.uvs, mesh.uvs)
    assert torch.equal(restored.face_material_ids, mesh.face_material_ids)
    assert torch.equal(restored.extras["weights"], mesh.extras["weights"])
    assert torch.equal(restored.extras["face_labels"], mesh.extras["face_labels"])
    assert restored.metadata == mesh.metadata
    assert restored.materials[0].metadata["name"] == "cube"
    assert torch.allclose(restored.materials[0].base_color, material.base_color)
    assert torch.equal(restored.materials[0].metallic, material.metallic)
    assert torch.equal(restored.materials[0].roughness, material.roughness)
    assert restored.materials[0].opacity is None
    assert not restored.vertices.requires_grad
    assert isinstance(backend, GeometryProcessingBackend)


@pytest.mark.parametrize("file_type", ["obj", "ply", "glb", "stl"])
def test_trimesh_canonical_cube_export_import(cube, tmp_path, file_type):
    pytest.importorskip("trimesh")
    backend = TrimeshBackend()
    path = tmp_path / f"cube.{file_type}"

    backend.export_mesh(cube, path)
    restored = backend.import_mesh(path)

    assert restored.faces.shape[0] == cube.faces.shape[0]
    assert torch.allclose(restored.vertices.amin(dim=0), cube.vertices.amin(dim=0))
    assert torch.allclose(restored.vertices.amax(dim=0), cube.vertices.amax(dim=0))
    assert restored.faces.dtype is torch.int64
    assert not restored.vertices.requires_grad


def test_trimesh_glb_preserves_object_transform(cube, tmp_path):
    pytest.importorskip("trimesh")
    backend = TrimeshBackend()
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([2.0, -1.0, 0.5])
    mesh = MeshAsset(cube.vertices, cube.faces, transform=transform)
    path = tmp_path / "transformed.glb"

    backend.export_mesh(mesh, path)
    restored = backend.import_mesh(path)

    assert torch.allclose(restored.transform, transform)
    with pytest.raises(ValueError, match="non-identity"):
        backend.export_mesh(mesh, tmp_path / "transformed.obj")


def test_trimesh_rejects_lossy_stl_channels(cube, tmp_path):
    pytest.importorskip("trimesh")
    mesh = MeshAsset(cube.vertices, cube.faces, colors=torch.ones(cube.vertices.shape[0], 3))

    with pytest.raises(ValueError, match="STL export cannot preserve colors"):
        TrimeshBackend().export_mesh(mesh, tmp_path / "colored.stl")


def test_trimesh_remove_unreferenced_vertices_remaps_vertex_channels(cube):
    pytest.importorskip("trimesh")
    vertices = torch.cat([cube.vertices, torch.tensor([[99.0, 99.0, 99.0]])])
    weights = torch.arange(vertices.shape[0], dtype=torch.float32).unsqueeze(1)
    mesh = MeshAsset(vertices, cube.faces, extras={"weights": weights})

    processed = TrimeshBackend().process_geometry(mesh, operation="remove_unreferenced_vertices")

    assert processed.vertices.shape[0] == cube.vertices.shape[0]
    assert torch.equal(processed.extras["weights"], weights[:-1])


def test_scikit_image_marching_cubes_topology_bounds_and_winding():
    pytest.importorskip("skimage")
    backend = ScikitImageBackend()
    axes = torch.meshgrid(*(torch.arange(17, dtype=torch.float32) for _ in range(3)), indexing="ij")
    center = torch.tensor([8.0, 8.0, 8.0]).view(3, 1, 1, 1)
    field = 25.0 - ((torch.stack(axes) - center) ** 2).sum(dim=0)
    spacing = (0.5, 1.0, 2.0)

    mesh = backend.extract_surface(field.requires_grad_(True), level=0.0, spacing=spacing)

    assert isinstance(backend, SurfaceExtractionBackend)
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert int(mesh.faces.min()) >= 0
    assert int(mesh.faces.max()) < mesh.vertices.shape[0]
    expected_min = torch.tensor([1.5, 3.0, 6.0])
    expected_max = torch.tensor([6.5, 13.0, 26.0])
    assert torch.allclose(mesh.vertices.amin(dim=0), expected_min, atol=0.1)
    assert torch.allclose(mesh.vertices.amax(dim=0), expected_max, atol=0.1)
    triangles = mesh.vertices[mesh.faces]
    face_normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    world_center = torch.tensor([4.0, 8.0, 16.0])
    outward = triangles.mean(dim=1) - world_center
    assert float((face_normals * outward).sum(dim=1).mean()) > 0.0
    assert not mesh.vertices.requires_grad


def test_xatlas_unwraps_plane_and_remaps_vertex_channels():
    pytest.importorskip("xatlas")
    plane = MeshAsset(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64),
        normals=torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1),
        colors=torch.eye(4, 3),
        extras={"weights": torch.arange(4, dtype=torch.float32).unsqueeze(1)},
        metadata={"shape": "plane"},
    )

    unwrapped = XAtlasBackend().process_geometry(plane, operation="unwrap_uv")

    assert unwrapped.faces.shape == plane.faces.shape
    assert unwrapped.uvs.shape == (unwrapped.vertices.shape[0], 2)
    assert unwrapped.colors.shape[0] == unwrapped.vertices.shape[0]
    assert unwrapped.extras["weights"].shape[0] == unwrapped.vertices.shape[0]
    assert bool(((unwrapped.uvs >= 0.0) & (unwrapped.uvs <= 1.0)).all())
    assert unwrapped.metadata == plane.metadata
    assert isinstance(XAtlasBackend(), GeometryProcessingBackend)


def test_xatlas_duplicates_cube_seams_and_remaps_channels(cube):
    pytest.importorskip("xatlas")
    colors = (cube.vertices + 1.0) * 0.5
    weights = cube.vertices.sum(dim=1)
    mesh = MeshAsset(cube.vertices, cube.faces, colors=colors, extras={"weights": weights})

    unwrapped = XAtlasBackend().process_geometry(mesh, operation="unwrap_uv")

    assert unwrapped.vertices.shape[0] > mesh.vertices.shape[0]
    assert torch.equal(unwrapped.colors, (unwrapped.vertices + 1.0) * 0.5)
    assert torch.equal(unwrapped.extras["weights"], unwrapped.vertices.sum(dim=1))


def test_xatlas_rejects_alignment_ambiguous_extra_when_vertices_duplicate():
    pytest.importorskip("xatlas")
    tetrahedron = MeshAsset(
        vertices=torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]),
        faces=torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int64),
        extras={"ambiguous": torch.arange(4, dtype=torch.float32)},
    )

    with pytest.raises(ValueError, match="alignment-ambiguous"):
        XAtlasBackend().process_geometry(tetrahedron, operation="unwrap_uv")
