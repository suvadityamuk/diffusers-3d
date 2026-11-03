from __future__ import annotations

import dataclasses
import io
import sys
import types

import numpy as np
import pytest
import torch

from diffusers_3d import (
    CoordinateSystem,
    CuMeshBackend,
    MeshAsset,
    TrimeshBackend,
)
from diffusers_3d.backends import (
    official_tensors_from_ovoxel_asset,
    ovoxel_asset_from_official,
    ovoxel_to_mesh,
    read_ovoxel_npz,
    textured_mesh_from_ovoxel,
    write_ovoxel_npz,
)
from diffusers_3d.backends.o_voxel import morton_decode_3d, morton_encode_3d
from diffusers_3d.backends.texture_baking import pixel_barycentrics
from diffusers_3d.backends.voxel_index import VoxelIndex, trilinear_sample


def _packed_official() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    coordinates = torch.tensor(
        [[7, 0, 3], [0, 0, 0], [1, 2, 3], [2, 1, 0]],
        dtype=torch.int32,
    )
    attributes = {
        "dual_vertices": torch.tensor(
            [[0, 127, 255], [255, 1, 2], [11, 22, 33], [44, 55, 66]],
            dtype=torch.uint8,
        ),
        "intersected": torch.tensor([[5], [2], [7], [0]], dtype=torch.uint8),
        "base_color": torch.tensor(
            [[1, 2, 3], [4, 5, 6], [70, 80, 90], [200, 201, 202]],
            dtype=torch.uint8,
        ),
        "metallic": torch.tensor([[7], [8], [9], [10]], dtype=torch.uint8),
        "roughness": torch.tensor([[11], [12], [13], [14]], dtype=torch.uint8),
        "alpha": torch.tensor([[15], [16], [17], [18]], dtype=torch.uint8),
        "normal": torch.tensor(
            [[19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
            dtype=torch.uint8,
        ),
        "emissive": torch.tensor(
            [[31, 32, 33], [34, 35, 36], [37, 38, 39], [40, 41, 42]],
            dtype=torch.uint8,
        ),
        "split_weight": torch.tensor([[0.25], [1.5], [12.75], [2.0]], dtype=torch.float32),
    }
    return coordinates, attributes


def test_ovoxel_official_mixed_roundtrip_preserves_all_channels_and_grid_metadata():
    coordinates, attributes = _packed_official()
    aabb = [[-2.0, -1.0, 0.5], [2.0, 3.0, 4.5]]
    asset = ovoxel_asset_from_official(
        coordinates,
        attributes,
        resolution=(8, 4, 8),
        aabb=aabb,
        packed=True,
    )

    assert asset.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    assert asset.metadata["resolution"] == [8, 4, 8]
    assert asset.metadata["aabb"] == aabb
    assert asset.metadata["dual_vertex_semantics"] == "fractional_cell_offset"
    torch.testing.assert_close(
        asset.grid_transform,
        torch.tensor(
            [
                [0.5, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert asset.intersection_data.dtype is torch.bool
    assert torch.equal(
        asset.intersection_data,
        torch.tensor([[True, False, True], [False, True, False], [True, True, True], [False, False, False]]),
    )

    restored_coordinates, restored_attributes = official_tensors_from_ovoxel_asset(asset, packed=True)
    assert torch.equal(restored_coordinates, coordinates)
    assert restored_attributes.keys() == attributes.keys()
    for name, value in attributes.items():
        assert torch.equal(restored_attributes[name], value), name


def test_ovoxel_npz_is_official_reader_compatible_and_roundtrips_without_compiled_runtime():
    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    buffer = io.BytesIO()
    write_ovoxel_npz(buffer, asset, compressed=False)

    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        assert set(data.files) == {"coord", *attributes}
        assert data["coord"].dtype == np.uint16
        assert all(data[name].dtype == np.uint8 for name in attributes if name != "split_weight")
        assert data["split_weight"].dtype == np.float32
        official_coordinates = torch.from_numpy(data["coord"]).int()
        official_attributes = {name: torch.from_numpy(value) for name, value in data.items() if name != "coord"}
        assert official_coordinates.shape == coordinates.shape
        assert official_attributes.keys() == attributes.keys()
        stored_coordinates = torch.from_numpy(data["coord"].astype(np.int64))
        expected_order = torch.from_numpy(
            np.lexsort(
                (
                    coordinates[:, 2].numpy(),
                    coordinates[:, 1].numpy(),
                    coordinates[:, 0].numpy(),
                )
            ).copy()
        )
        assert torch.equal(stored_coordinates, coordinates[expected_order])

    buffer.seek(0)
    restored = read_ovoxel_npz(buffer, resolution=8)
    restored_coordinates, restored_attributes = official_tensors_from_ovoxel_asset(restored, packed=True)
    assert torch.equal(restored_coordinates, coordinates[expected_order])
    for name, value in attributes.items():
        assert torch.equal(restored_attributes[name], value[expected_order]), name
    assert restored.metadata["resolution"] == [8, 8, 8]
    assert not restored.metadata["resolution_inferred"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_ovoxel_npz_preserves_unbounded_split_weight_dtype_and_values(dtype):
    coordinates, attributes = _packed_official()
    attributes["split_weight"] = torch.tensor([[0.125], [1.25], [17.5], [3.0]], dtype=dtype)
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    buffer = io.BytesIO()

    write_ovoxel_npz(buffer, asset, compressed=False)
    buffer.seek(0)
    restored = read_ovoxel_npz(buffer, resolution=8)

    assert restored.split_weights.dtype is dtype
    expected_order = torch.from_numpy(
        np.lexsort(
            (
                coordinates[:, 2].numpy(),
                coordinates[:, 1].numpy(),
                coordinates[:, 0].numpy(),
            )
        ).copy()
    )
    expected_split_weights = attributes["split_weight"][expected_order]
    torch.testing.assert_close(restored.split_weights, expected_split_weights, atol=0.0, rtol=0.0)
    _, restored_attributes = official_tensors_from_ovoxel_asset(restored, packed=True)
    assert restored_attributes["split_weight"].dtype is dtype
    torch.testing.assert_close(restored_attributes["split_weight"], expected_split_weights, atol=0.0, rtol=0.0)


def test_ovoxel_npz_preserves_generated_out_of_cell_dual_vertices_for_official_reader():
    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    asset.dual_grid_vertex_offsets = torch.tensor(
        [
            [-0.5, 0.0, 1.5],
            [0.25, 0.5, 0.75],
            [1.25, -0.25, 1.0],
            [0.0, 1.0, 0.5],
        ]
    )
    buffer = io.BytesIO()

    write_ovoxel_npz(buffer, asset, compressed=False)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        official_coordinates = torch.from_numpy(data["coord"]).int()
        official_attributes = {name: torch.from_numpy(value) for name, value in data.items() if name != "coord"}
        assert official_attributes["dual_vertices"].dtype is torch.float32
        torch.testing.assert_close(
            official_attributes["dual_vertices"],
            asset.dual_grid_vertex_offsets[torch.tensor([1, 2, 3, 0])],
        )

    buffer.seek(0)
    restored = read_ovoxel_npz(buffer, resolution=8)
    torch.testing.assert_close(
        restored.dual_grid_vertex_offsets,
        asset.dual_grid_vertex_offsets[torch.tensor([1, 2, 3, 0])],
    )
    direct = ovoxel_asset_from_official(
        official_coordinates,
        official_attributes,
        resolution=8,
        packed=True,
    )
    torch.testing.assert_close(
        direct.dual_grid_vertex_offsets,
        asset.dual_grid_vertex_offsets[torch.tensor([1, 2, 3, 0])],
    )


def test_ovoxel_npz_accepts_bfloat16_pipeline_outputs():
    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    asset.dual_grid_vertex_offsets = torch.tensor(
        [[-0.5, 0.0, 1.5], [0.25, 0.5, 0.75], [1.25, -0.25, 1.0], [0.0, 1.0, 0.5]]
    )
    asset = asset.to(dtype=torch.bfloat16)
    buffer = io.BytesIO()

    write_ovoxel_npz(buffer, asset, compressed=False)
    buffer.seek(0)
    restored = read_ovoxel_npz(buffer, resolution=8)
    order = torch.tensor([1, 2, 3, 0])
    torch.testing.assert_close(restored.dual_grid_vertex_offsets, asset.dual_grid_vertex_offsets.float()[order])
    torch.testing.assert_close(restored.base_color, asset.base_color.float()[order], atol=1 / 255, rtol=0)


def test_ovoxel_reader_accepts_pre_release_metadata_without_exposing_it_as_attributes():
    coordinates, attributes = _packed_official()
    buffer = io.BytesIO()
    arrays = {"coord": coordinates.numpy().astype(np.uint16)}
    arrays.update({name: value.numpy() for name, value in attributes.items()})
    arrays.update(
        {
            "__diffusers_3d_ovoxel_resolution": np.asarray([8, 8, 8], dtype=np.uint32),
            "__diffusers_3d_ovoxel_aabb": np.asarray([[-0.5] * 3, [0.5] * 3], dtype=np.float32),
            "__diffusers_3d_ovoxel_packed": np.asarray([1], dtype=np.uint8),
            "__diffusers_3d_ovoxel_layout": np.asarray('{"schema_version":2}'),
        }
    )
    np.savez(buffer, **arrays)

    buffer.seek(0)
    restored = read_ovoxel_npz(buffer)

    assert restored.metadata["resolution"] == [8, 8, 8]
    _, restored_attributes = official_tensors_from_ovoxel_asset(restored, packed=True)
    assert restored_attributes.keys() == attributes.keys()


@pytest.mark.parametrize(
    "intersection_data",
    (
        torch.tensor([[-1], [0], [0], [0]], dtype=torch.int64),
        torch.tensor([[8], [0], [0], [0]], dtype=torch.int64),
        torch.tensor([[256, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.int64),
    ),
)
def test_ovoxel_official_mapping_validates_intersection_flags_before_uint8_cast(intersection_data):
    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    asset.intersection_data = intersection_data

    with pytest.raises(ValueError, match="intersected|intersection"):
        official_tensors_from_ovoxel_asset(asset, packed=True)


@pytest.mark.parametrize(
    ("coordinates", "resolution", "message"),
    [
        (torch.tensor([[-1, 0, 0]], dtype=torch.int32), 8, "non-negative"),
        (torch.tensor([[8, 0, 0]], dtype=torch.int32), 8, "strictly below"),
        (torch.tensor([[65536, 0, 0]], dtype=torch.int64), 65537, "fit in uint16"),
    ],
)
@pytest.mark.parametrize("morton_order", [None, False, True])
def test_ovoxel_npz_rejects_invalid_coordinates_before_serialization(
    coordinates,
    resolution,
    message,
    morton_order,
):
    valid_coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(valid_coordinates, attributes, resolution=8, packed=True)
    asset.active_coordinates = coordinates
    asset.metadata["resolution"] = [resolution, resolution, resolution]

    with pytest.raises(ValueError, match=message):
        write_ovoxel_npz(io.BytesIO(), asset, morton_order=morton_order)


def test_ovoxel_npz_uint16_boundaries_use_default_lexicographic_fallback():
    _, attributes = _packed_official()
    coordinates = torch.tensor(
        [[1535, 2, 0], [1024, 1, 0], [1023, 3, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=1536, packed=True)
    buffer = io.BytesIO()

    write_ovoxel_npz(buffer, asset, compressed=False)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        assert set(data.files) == {"coord", *attributes}
        assert data["coord"].dtype == np.uint16
        assert data["coord"][:, 0].tolist() == [0, 1023, 1024, 1535]

    buffer.seek(0)
    restored = read_ovoxel_npz(buffer, resolution=1536)
    assert restored.active_coordinates[:, 0].tolist() == [0, 1023, 1024, 1535]
    assert restored.metadata["resolution"] == [1536, 1536, 1536]
    with pytest.raises(ValueError, match=r"\[0, 1023\]"):
        write_ovoxel_npz(io.BytesIO(), asset, morton_order=True)


def test_ovoxel_unpacked_mapping_and_morton_codec_are_exact():
    coordinates = torch.tensor([[0, 0, 0], [1, 2, 3], [1023, 1023, 1023]], dtype=torch.int64)
    assert torch.equal(morton_decode_3d(morton_encode_3d(coordinates)), coordinates)
    assert torch.equal(
        morton_encode_3d(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int32)),
        torch.tensor([1, 2, 4]),
    )
    with pytest.raises(ValueError, match=r"\[0, 1023\]"):
        morton_encode_3d(torch.tensor([[1024, 0, 0]]))

    unpacked = {
        "dual_vertices": torch.tensor([[0.125, 0.25, 0.5], [0.75, 0.875, 1.0]]),
        "intersected": torch.tensor([[True, False, True], [False, True, False]]),
        "base_color": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "metallic": torch.tensor([[0.2], [0.8]]),
        "roughness": torch.tensor([[0.3], [0.7]]),
        "alpha": torch.tensor([[0.9], [0.6]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "emissive": torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),
        "split_weight": torch.tensor([[0.25], [0.75]]),
    }
    asset = ovoxel_asset_from_official(coordinates[:2], unpacked, resolution=4, packed=False)
    restored_coordinates, restored = official_tensors_from_ovoxel_asset(asset, packed=False)
    assert torch.equal(restored_coordinates, coordinates[:2])
    assert restored.keys() == unpacked.keys()
    for name, value in unpacked.items():
        if name == "intersected":
            expected = torch.tensor([[5], [2]], dtype=torch.uint8)
            assert torch.equal(restored[name], expected)
        else:
            torch.testing.assert_close(restored[name], value)


def test_cumesh_cpu_fake_covers_repair_simplify_remesh_uv_and_bvh(monkeypatch):
    calls = []

    class FakeCuMesh:
        def init(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

        def read(self):
            return self.vertices, self.faces

        def remove_duplicate_faces(self):
            calls.append("remove_duplicate_faces")

        def repair_non_manifold_edges(self):
            calls.append("repair_non_manifold_edges")

        def remove_small_connected_components(self, threshold):
            calls.append(("remove_small_connected_components", threshold))

        def fill_holes(self, *, max_hole_perimeter):
            calls.append(("fill_holes", max_hole_perimeter))

        def unify_face_orientations(self):
            calls.append("unify_face_orientations")

        def simplify(self, target, *, verbose):
            calls.append(("simplify", target, verbose))

        def uv_unwrap(self, *, compute_charts_kwargs, return_vmaps, verbose):
            calls.append(("uv_unwrap", compute_charts_kwargs, return_vmaps, verbose))
            uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            return self.vertices, self.faces, uvs, torch.arange(3)

    class FakeBVH:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

        def unsigned_distance(self, points, *, return_uvw):
            return points.square().sum(dim=-1).sqrt(), return_uvw

    def remesh(vertices, faces, **kwargs):
        calls.append(("remesh", kwargs["resolution"], type(kwargs["bvh"]).__name__))
        return vertices, faces

    module = types.ModuleType("cumesh")
    module.CuMesh = FakeCuMesh
    module.cuBVH = FakeBVH
    module.remeshing = types.SimpleNamespace(remesh_narrow_band_dc=remesh)
    monkeypatch.setitem(sys.modules, "cumesh", module)
    with pytest.warns(UserWarning, match="non-commercial"):
        backend = CuMeshBackend(device="cpu")
    mesh = MeshAsset(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    repaired = backend.repair(mesh)
    simplified = backend.simplify(mesh, 1)
    remeshed = backend.remesh(mesh, 8)
    unwrapped = backend.unwrap_uv(mesh)
    assert repaired.metadata["cumesh_operation"] == "repair"
    assert simplified.metadata["cumesh_operation"] == "simplify"
    assert remeshed.metadata["cumesh_operation"] == "remesh"
    assert unwrapped.metadata["cumesh_operation"] == "uv_unwrap"
    assert unwrapped.uvs is not None
    assert torch.equal(unwrapped.extras["cumesh_vertex_map"], torch.arange(3))
    bvh = backend.build_bvh(mesh)
    assert isinstance(bvh, FakeBVH)
    assert bvh.faces.dtype is torch.int32
    distances, return_uvw = backend.unsigned_distance(mesh, torch.tensor([[3.0, 4.0, 0.0]]))
    torch.testing.assert_close(distances, torch.tensor([5.0]))
    assert not return_uvw
    assert "remove_duplicate_faces" in calls
    assert ("simplify", 1, False) in calls
    assert ("remesh", 8, "FakeBVH") in calls


def test_ovoxel_to_mesh_builds_one_vertex_per_voxel_and_two_triangles_per_closed_quad(slab_ovoxel):
    asset = slab_ovoxel

    mesh = ovoxel_to_mesh(asset)

    assert mesh.vertices.shape == (64, 3) and mesh.faces.shape == (2 * 7 * 7, 3)
    torch.testing.assert_close(mesh.vertices[:, 2], torch.full((64,), 4.5 / 8 - 0.5))
    torch.testing.assert_close(mesh.colors, torch.cat([asset.base_color, asset.opacity[:, None]], dim=1))
    assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP
    with pytest.raises(ValueError, match="resolution and aabb"):
        ovoxel_to_mesh(dataclasses.replace(asset, metadata={}))
    with pytest.raises(ValueError, match="no surface"):
        ovoxel_to_mesh(dataclasses.replace(asset, intersection_data=torch.zeros_like(asset.intersection_data)))


@pytest.mark.portable
@pytest.mark.parametrize("remesh", [False, True])
def test_textured_mesh_follows_upstream_steps_and_bakes_textures_a_viewer_reads_back(monkeypatch, slab_ovoxel, remesh):
    pytest.importorskip("trimesh")
    calls = []

    class FakeCuMesh:
        def init(self, vertices, faces):
            self.vertices, self.faces = vertices, faces

        def read(self):
            return self.vertices, self.faces

        def __getattr__(self, name):
            return lambda *args, **kwargs: calls.append((name, *args, *kwargs.values()))

        def uv_unwrap(self, *, compute_charts_kwargs, return_vmaps, verbose):
            calls.append(("uv_unwrap",))
            uvs = self.vertices[:, :2] + 0.5
            return self.vertices, self.faces, uvs, torch.arange(self.vertices.shape[0])

        def read_vertex_normals(self):
            return torch.tensor([0.0, 0.0, 1.0]).expand(self.vertices.shape[0], 3)

    class FakeBVH:
        def __init__(self, vertices, faces):
            self.triangles = vertices[faces.long()][..., :2]  # the slab is flat, so barycentrics are planar

        def unsigned_distance(self, points, *, return_uvw):
            a, b, c = self.triangles.unbind(1)
            v0, v1, v2 = b - a, c - a, points[:, None, :2] - a
            denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            w1 = (v2[..., 0] * v1[:, 1] - v1[:, 0] * v2[..., 1]) / denominator
            w2 = (v0[:, 0] * v2[..., 1] - v2[..., 0] * v0[:, 1]) / denominator
            uvw = torch.stack([1 - w1 - w2, w1, w2], dim=-1)
            face_id = uvw.amin(dim=-1).argmax(dim=1)
            return torch.zeros(points.shape[0]), face_id, uvw[torch.arange(points.shape[0]), face_id]

    def remesh_narrow_band_dc(vertices, faces, **kwargs):
        calls.append(("remesh", kwargs["resolution"]))
        return vertices, faces

    def rasterize(context, clip, faces, *, resolution):
        # nvdiffrast's output: (u, v, z/w, face id + 1), with p = u * v0 + v * v1 + (1 - u - v) * v2; row 0 at v = 0.
        size = resolution[0]
        triangles = ((clip[0, :, :2] + 1) / 2 * size)[faces.long()]
        lower = triangles.amin(dim=1).floor().clamp(min=0).long()
        upper = triangles.amax(dim=1).ceil().long().clamp(max=size)
        face, column, row, weights = pixel_barycentrics(triangles, lower, upper)
        inside = (weights >= 0).all(dim=1)
        face, column, row, weights = face[inside], column[inside], row[inside], weights[inside]
        output = torch.zeros(1, size, size, 4)
        output[0, row, column] = torch.stack([weights[:, 0], weights[:, 1], weights[:, 2] * 0, face + 1.0], dim=1)
        return output, None

    def interpolate(attributes, rast, faces):
        corners = attributes[0][faces[(rast[..., 3].long() - 1).clamp(min=0)].long()]
        u, v = rast[..., :1], rast[..., 1:2]
        values = u * corners[..., 0, :] + v * corners[..., 1, :] + (1 - u - v) * corners[..., 2, :]
        return (values * (rast[..., 3:] > 0),)

    def grid_sample_3d(features, coordinates, *, shape, grid, mode):
        assert mode == "trilinear"
        return trilinear_sample(VoxelIndex(coordinates[:, 1:], tuple(shape[2:])), features, grid[0])[None]

    module = types.ModuleType("cumesh")
    module.CuMesh, module.cuBVH = FakeCuMesh, FakeBVH
    module.remeshing = types.SimpleNamespace(remesh_narrow_band_dc=remesh_narrow_band_dc)
    monkeypatch.setitem(sys.modules, "cumesh", module)
    nvdiffrast = types.SimpleNamespace(RasterizeCudaContext=object, rasterize=rasterize, interpolate=interpolate)
    monkeypatch.setitem(sys.modules, "nvdiffrast.torch", nvdiffrast)
    monkeypatch.setitem(sys.modules, "flex_gemm.ops.grid_sample", types.SimpleNamespace(grid_sample_3d=grid_sample_3d))

    textured = textured_mesh_from_ovoxel(
        slab_ovoxel, device="cpu", decimation_target=100, texture_size=64, remesh=remesh
    )

    cleanup = [
        ("remove_duplicate_faces",),
        ("repair_non_manifold_edges",),
        ("remove_small_connected_components", 1e-5),
    ]
    fill = ("fill_holes", 3e-2)
    expected = (
        [fill, ("remesh", 8), ("simplify", 100, False)]
        if remesh
        else [
            fill,
            ("simplify", 300, False),
            *cleanup,
            fill,
            ("simplify", 100, False),
            *cleanup,
            fill,
            ("unify_face_orientations",),
        ]
    )
    assert calls == [*expected, ("uv_unwrap",), ("compute_vertex_normals",)]

    material = textured.materials[0]
    assert material.metadata["double_sided"] is not remesh
    assert textured.metadata["export_rotation"] == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    # Upstream bakes 8-bit textures.
    assert torch.equal((material.base_color * 255).round() / 255, material.base_color)
    # Read each vertex's texel the way a viewer does (OpenGL uvs: v up, row 0 at the top).
    size = material.base_color.shape[0]
    columns = (textured.uvs[:, 0] * size).long().clamp(0, size - 1)
    rows = ((1 - textured.uvs[:, 1]) * size).long().clamp(0, size - 1)
    ramp = (textured.vertices[:, :2] + 0.5) * 8 / 7 - 1 / 14
    torch.testing.assert_close(material.base_color[rows, columns, :2], ramp, atol=0.05, rtol=0)
    torch.testing.assert_close(material.metallic[rows, columns], ramp[:, 0], atol=0.05, rtol=0)
    torch.testing.assert_close(material.roughness[rows, columns], ramp[:, 1], atol=0.05, rtol=0)

    exported = TrimeshBackend().to_trimesh(textured.to_coordinate_system(CoordinateSystem.RIGHT_HANDED_Y_UP))
    assert exported.visual.material.metallicRoughnessTexture is not None
    assert exported.visual.material.doubleSided is not remesh


def test_ovoxel_asset_save_and_load_round_trip(tmp_path):
    from diffusers_3d import OVoxelAsset

    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    asset.save(tmp_path / "asset.npz")
    loaded = OVoxelAsset.load(tmp_path / "asset.npz", resolution=8)

    # The writer stores voxels sorted, so compare the voxel sets.
    def rows(value):
        return sorted(map(tuple, official_tensors_from_ovoxel_asset(value)[0].tolist()))

    assert rows(loaded) == rows(asset)
