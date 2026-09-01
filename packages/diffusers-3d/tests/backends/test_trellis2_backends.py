from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pytest
import torch

import diffusers_3d.backends._optional as optional_backends
from diffusers_3d import (
    CUMESH_SOURCE_REVISION,
    CUMESH_SOURCE_URL,
    FLEX_GEMM_BATCH_INDICES,
    FLEX_GEMM_SOURCE_REVISION,
    FLEX_GEMM_SOURCE_URL,
    BackendCapability,
    BackendLicenseClass,
    BackendRegistry,
    BackendSpec,
    BackendSupportLevel,
    BackendUnavailableError,
    CoordinateSystem,
    CuMeshBackend,
    FlexGemmBackend,
    MeshAsset,
    OVoxelBackend,
    OVoxelCapability,
    OVoxelRuntimeUnavailableError,
    SparseVoxelAsset,
    Trellis2PBRPostprocessFacade,
    morton_decode_3d,
    morton_encode_3d,
    official_tensors_from_ovoxel_asset,
    ovoxel_asset_from_official,
    ovoxel_grid_transform,
    read_ovoxel_npz,
    write_ovoxel_npz,
)


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


def test_ovoxel_npz_uses_default_lexicographic_order_and_roundtrips_without_compiled_runtime():
    coordinates, attributes = _packed_official()
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    buffer = io.BytesIO()
    write_ovoxel_npz(buffer, asset, compressed=False)

    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        assert data["coord"].dtype == np.uint16
        assert all(data[name].dtype == np.uint8 for name in attributes if name != "split_weight")
        assert data["split_weight"].dtype == np.float32
        layout = json.loads(str(data["__diffusers_3d_ovoxel_layout"].item()))
        assert layout["attributes"]["split_weight"] == {
            "dtype": "float32",
            "encoding": "nonnegative_float",
            "layout": "voxel_scalar_nonnegative",
            "shape": [1],
        }
        assert layout["coordinate_order"] == "lexicographic_xyz"
        assert not layout["morton_order"]
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
    restored = read_ovoxel_npz(buffer)
    restored_coordinates, restored_attributes = official_tensors_from_ovoxel_asset(restored, packed=True)
    assert torch.equal(restored_coordinates, coordinates[expected_order])
    for name, value in attributes.items():
        assert torch.equal(restored_attributes[name], value[expected_order]), name
    assert restored.metadata["resolution"] == [8, 8, 8]
    assert restored.metadata["coordinate_order"] == "lexicographic_xyz"
    assert not restored.metadata["resolution_inferred"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_ovoxel_npz_preserves_unbounded_split_weight_dtype_and_values(dtype):
    coordinates, attributes = _packed_official()
    attributes["split_weight"] = torch.tensor([[0.125], [1.25], [17.5], [3.0]], dtype=dtype)
    asset = ovoxel_asset_from_official(coordinates, attributes, resolution=8, packed=True)
    buffer = io.BytesIO()

    write_ovoxel_npz(buffer, asset, compressed=False)
    buffer.seek(0)
    restored = read_ovoxel_npz(buffer)

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
        layout = json.loads(str(data["__diffusers_3d_ovoxel_layout"].item()))
        assert layout["coordinate_order"] == "lexicographic_xyz"
        assert data["coord"].dtype == np.uint16
        assert data["coord"][:, 0].tolist() == [0, 1023, 1024, 1535]
        assert data["__diffusers_3d_ovoxel_resolution"].tolist() == [1536, 1536, 1536]

    buffer.seek(0)
    restored = read_ovoxel_npz(buffer)
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


def test_ovoxel_capabilities_are_independent_and_vxz_never_claims_pure_support(
    registry_factory,
    spec_factory,
):
    spec = spec_factory(
        "o_voxel",
        capabilities=(
            BackendCapability.NATIVE_REPRESENTATION,
            BackendCapability.CONVERSION,
            BackendCapability.SERIALIZATION,
        ),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cpu",),
        differentiable=False,
        import_name="o_voxel",
        distribution_name="o-voxel",
    )
    registry = registry_factory((spec,), installed=(), importable=())
    backend = OVoxelBackend(device="cpu", registry=registry)
    assert backend.supports(OVoxelCapability.SCHEMA_PACK)
    assert backend.supports(OVoxelCapability.NPZ_CODEC)
    assert not backend.supports(OVoxelCapability.NATIVE_CODEC)
    assert not backend.supports(OVoxelCapability.NATIVE_CONVERSION)
    assert not backend.supports(OVoxelCapability.NATIVE_RENDERING)
    with pytest.raises(OVoxelRuntimeUnavailableError, match="accept_nvdiffrast"):
        backend.read_vxz("not-real.vxz")

    acknowledged = OVoxelBackend(
        device="cpu",
        registry=registry,
        accept_nvdiffrast_research_license=True,
    )
    with pytest.raises(OVoxelRuntimeUnavailableError, match="separately compiled"):
        acknowledged.read_vxz("not-real.vxz")


def test_ovoxel_native_facade_delegates_to_pinned_io_dual_grid_and_renderer_api(
    monkeypatch,
    registry_factory,
    spec_factory,
):
    coordinates, attributes = _packed_official()
    calls = {}

    def read_vxz(file, num_threads=-1):
        calls["read_vxz"] = (file, num_threads)
        return coordinates, attributes

    def write_vxz(file, coord, attr, **kwargs):
        calls["write_vxz"] = (file, coord, attr, kwargs)

    module = types.ModuleType("o_voxel")
    module.io = types.SimpleNamespace(read_vxz=read_vxz, write_vxz=write_vxz)
    module.convert = types.SimpleNamespace()
    module.rasterize = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "o_voxel", module)
    spec = spec_factory(
        "o_voxel",
        capabilities=(
            BackendCapability.NATIVE_REPRESENTATION,
            BackendCapability.CONVERSION,
            BackendCapability.SERIALIZATION,
        ),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cpu",),
        differentiable=False,
        import_name="o_voxel",
        distribution_name="o-voxel",
    )
    backend = OVoxelBackend(
        device="cpu",
        registry=registry_factory((spec,)),
        accept_nvdiffrast_research_license=True,
    )
    assert backend.supports(OVoxelCapability.NATIVE_CODEC)
    assert not backend.supports(OVoxelCapability.NATIVE_CONVERSION)
    assert not backend.supports(OVoxelCapability.NATIVE_RENDERING)

    def flexible_dual_grid_to_mesh(
        coord,
        dual_vertices,
        intersected,
        split_weight,
        *,
        grid_size,
        aabb,
        train,
    ):
        calls["to_mesh"] = {
            "coord": coord,
            "dual_vertices": dual_vertices,
            "intersected": intersected,
            "split_weight": split_weight,
            "grid_size": grid_size,
            "aabb": aabb,
            "train": train,
        }
        return (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
        )

    class VoxelRenderer:
        def __init__(self, rendering_options):
            calls["renderer_options"] = rendering_options

        def render(self, **kwargs):
            calls["render"] = kwargs
            return types.SimpleNamespace(
                attr=torch.ones(3, 4, 4),
                depth=torch.ones(4, 4),
                alpha=torch.ones(4, 4),
            )

    module.convert.flexible_dual_grid_to_mesh = flexible_dual_grid_to_mesh
    module.rasterize.VoxelRenderer = VoxelRenderer
    assert backend.supports(OVoxelCapability.NATIVE_CONVERSION)
    assert backend.supports(OVoxelCapability.NATIVE_RENDERING)

    with pytest.raises(ValueError, match="does not encode.*resolution"):
        backend.read_vxz("asset.vxz")
    asset = backend.read_vxz("asset.vxz", resolution=8, num_threads=3)
    assert calls["read_vxz"] == ("asset.vxz", 3)
    assert asset.metadata["resolution"] == [8, 8, 8]

    with pytest.raises(ValueError, match="no verified lossless split_weights encoding"):
        backend.write_vxz("lossy.vxz", asset)
    asset.split_weights = None
    backend.write_vxz("restored.vxz", asset, compression="zstd")
    output_file, output_coordinates, output_attributes, output_kwargs = calls["write_vxz"]
    assert output_file == "restored.vxz"
    assert torch.equal(output_coordinates, coordinates)
    for name, value in attributes.items():
        if name == "split_weight":
            continue
        assert torch.equal(output_attributes[name], value)
    assert output_kwargs == {"compression": "zstd"}

    mesh = backend.to_mesh(asset, train=True)
    assert mesh.faces.dtype is torch.int64
    assert mesh.metadata["resolution"] == [8, 8, 8]
    assert torch.equal(
        calls["to_mesh"]["intersected"],
        torch.tensor([[True, False, True], [False, True, False], [True, True, True], [False, False, False]]),
    )
    assert calls["to_mesh"]["grid_size"] == [8, 8, 8]
    assert calls["to_mesh"]["train"]

    rendered = backend.render_voxels(
        asset,
        extrinsics=torch.eye(4),
        intrinsics=torch.eye(3),
        image_size=4,
    )
    assert set(rendered) == {"attr", "depth", "alpha"}
    assert calls["renderer_options"] == {"resolution": 4}
    assert calls["render"]["voxel_size"] == pytest.approx(1 / 8)
    torch.testing.assert_close(
        calls["render"]["position"],
        asset.active_coordinates.to(dtype=torch.float32) / 8 - 0.5,
    )

    high_coordinates = torch.tensor(
        [[1535, 2, 0], [1024, 1, 0], [1023, 3, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    asset.active_coordinates = high_coordinates
    asset.metadata["resolution"] = [1536, 1536, 1536]
    asset.grid_transform = ovoxel_grid_transform(1536)
    backend.write_vxz("high-resolution.vxz", asset)
    assert calls["write_vxz"][0] == "high-resolution.vxz"
    assert torch.equal(calls["write_vxz"][1], high_coordinates)


def test_top_level_import_does_not_import_optional_trellis2_backends():
    source_root = Path(__file__).parents[2] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import diffusers_3d; "
        "names=('o_voxel','flex_gemm','cumesh','nvdiffrast'); "
        "assert all(name not in sys.modules for name in names)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_flex_gemm_realistic_cpu_fake_adapts_sparse_operations_without_custom_attestations(
    monkeypatch,
    registry_factory,
    spec_factory,
):
    captured = {}

    def sparse_submanifold_conv3d(features, coordinates, shape, weight, bias, unused, dilation):
        captured.update(coordinates=coordinates, shape=shape, dilation=dilation, unused=unused)
        return features @ weight.T + bias

    def grid_sample_3d(features, coordinates, *, shape, grid, mode):
        captured.update(grid_coordinates=coordinates, grid_shape=shape, grid=grid, mode=mode)
        return features.sum(dim=1, keepdim=True)

    module = types.ModuleType("flex_gemm")
    module.ops = types.SimpleNamespace(
        spconv=types.SimpleNamespace(sparse_submanifold_conv3d=sparse_submanifold_conv3d),
        grid_sample=types.SimpleNamespace(grid_sample_3d=grid_sample_3d),
    )
    monkeypatch.setitem(sys.modules, "flex_gemm", module)
    spec = spec_factory(
        "flex_gemm",
        capabilities=(BackendCapability.SPARSE_COMPUTE,),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cpu",),
        dtypes=("float32",),
        differentiable=True,
        import_name="flex_gemm",
        distribution_name="flex-gemm",
    )
    backend = FlexGemmBackend(
        device="cpu",
        registry=registry_factory((spec,)),
    )
    assert backend.build_identity is None
    voxels = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.int64),
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        voxel_size=0.25,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        extras={FLEX_GEMM_BATCH_INDICES: torch.tensor([0, 1], dtype=torch.int64)},
    )
    projected = backend.sparse_compute(
        voxels,
        operation="submanifold_conv3d",
        parameters={
            "weight": torch.tensor([[2.0, -1.0]]),
            "bias": torch.tensor([0.5]),
            "dilation": torch.tensor([1, 2, 1]),
        },
    )
    torch.testing.assert_close(projected.features, torch.tensor([[0.5], [2.5]]))
    assert torch.equal(
        captured["coordinates"],
        torch.tensor([[0, 0, 1, 2], [1, 3, 2, 1]], dtype=torch.int32),
    )
    assert captured["dilation"] == (1, 2, 1)
    assert projected.metadata["flex_gemm_source_revision"] == FLEX_GEMM_SOURCE_REVISION
    assert "flex_gemm_build_identity" not in projected.metadata
    sampled = backend.grid_sample_3d(voxels, torch.zeros(1, 2, 3), mode="trilinear")
    torch.testing.assert_close(sampled, torch.tensor([[3.0], [7.0]]))
    assert captured["mode"] == "trilinear"


def test_flex_gemm_rejects_incompatible_runtime_before_extension_import(monkeypatch):
    imported = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        optional_backends.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(RuntimeError, match="available CUDA/ROCm"):
        FlexGemmBackend(device="cuda")
    assert imported == []


def test_cumesh_cpu_fake_covers_repair_simplify_remesh_uv_and_bvh(
    monkeypatch,
    registry_factory,
    spec_factory,
):
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
    spec = spec_factory(
        "cumesh",
        capabilities=(BackendCapability.GEOMETRY_PROCESSING, BackendCapability.CONVERSION),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cpu",),
        dtypes=("float32",),
        differentiable=False,
        import_name="cumesh",
        distribution_name="cumesh",
    )
    backend = CuMeshBackend(
        device="cpu",
        registry=registry_factory((spec,)),
    )
    assert backend.build_identity is None
    mesh = MeshAsset(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    repaired = backend.process_geometry(mesh, operation="repair")
    simplified = backend.process_geometry(mesh, operation="simplify", parameters={"target_faces": 1})
    remeshed = backend.process_geometry(mesh, operation="remesh", parameters={"resolution": 8})
    unwrapped = backend.process_geometry(mesh, operation="uv_unwrap")
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


@pytest.mark.parametrize(
    ("name", "source_url", "source_revision", "capabilities", "differentiable", "backend_type"),
    [
        (
            "flex_gemm",
            FLEX_GEMM_SOURCE_URL,
            FLEX_GEMM_SOURCE_REVISION,
            (BackendCapability.SPARSE_COMPUTE,),
            True,
            FlexGemmBackend,
        ),
        (
            "cumesh",
            CUMESH_SOURCE_URL,
            CUMESH_SOURCE_REVISION,
            (BackendCapability.GEOMETRY_PROCESSING, BackendCapability.CONVERSION),
            False,
            CuMeshBackend,
        ),
    ],
)
def test_source_backends_reject_wrong_pep610_source_before_import(
    monkeypatch,
    name,
    source_url,
    source_revision,
    capabilities,
    differentiable,
    backend_type,
):
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = BackendSpec(
        name=name,
        import_names=(name,),
        distribution_names=(name,),
        capabilities=frozenset(capabilities),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32"}),
        differentiable=differentiable,
        install_hint=f"Build {name} from its pinned source",
        source_url=source_url,
        source_revision=source_revision,
        requires_source_provenance=True,
    )

    class WrongDistribution:
        def read_text(self, filename):
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/untrusted/fork.git",
                    "vcs_info": {"vcs": "git", "commit_id": source_revision},
                }
            )

    registry = BackendRegistry(
        (spec,),
        module_finder=lambda candidate: ModuleSpec(candidate, loader=None),
        version_getter=lambda _: "1.0",
        distribution_getter=lambda _: WrongDistribution(),
    )
    with pytest.raises(BackendUnavailableError, match="does not match required source"):
        backend_type(device="cpu", registry=registry)
    assert name not in sys.modules


def test_pbr_facade_requires_all_backends_and_explicit_research_license(
    registry_factory,
    spec_factory,
):
    specs = (
        spec_factory(
            "o_voxel",
            capabilities=(BackendCapability.NATIVE_REPRESENTATION,),
            support_level=BackendSupportLevel.ACCELERATED,
            devices=("cpu",),
            differentiable=False,
            import_name="o_voxel",
            distribution_name="o-voxel",
        ),
        spec_factory(
            "cumesh",
            capabilities=(BackendCapability.GEOMETRY_PROCESSING,),
            support_level=BackendSupportLevel.ACCELERATED,
            devices=("cpu",),
            differentiable=False,
            import_name="cumesh",
            distribution_name="cumesh",
        ),
        spec_factory(
            "flex_gemm",
            capabilities=(BackendCapability.SPARSE_COMPUTE,),
            support_level=BackendSupportLevel.ACCELERATED,
            devices=("cpu",),
            differentiable=True,
            import_name="flex_gemm",
            distribution_name="flex-gemm",
        ),
        spec_factory(
            "nvdiffrast",
            capabilities=(BackendCapability.MESH_RASTERIZATION,),
            support_level=BackendSupportLevel.RESEARCH_ONLY,
            license_class=BackendLicenseClass.RESTRICTED,
            devices=("cpu",),
            differentiable=True,
            import_name="nvdiffrast",
            distribution_name="nvdiffrast",
        ),
    )
    facade = Trellis2PBRPostprocessFacade(registry=registry_factory(specs))
    with pytest.raises(ValueError, match="accept_nvdiffrast_research_license"):
        facade.requirements(device="cpu")
    requirements = facade.requirements(
        device="cpu",
        accept_nvdiffrast_research_license=True,
    )
    assert set(requirements) == {"o_voxel", "cumesh", "flex_gemm", "nvdiffrast"}
