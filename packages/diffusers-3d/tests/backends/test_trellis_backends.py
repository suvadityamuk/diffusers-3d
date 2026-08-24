from __future__ import annotations

import importlib
import sys
import types
from importlib.machinery import ModuleSpec

import pytest
import torch

from diffusers_3d import (
    SPCONV_BATCH_INDICES,
    BackendCapability,
    BackendLicenseClass,
    BackendRegistry,
    BackendSpec,
    BackendSupportLevel,
    BackendUnavailableError,
    CameraRig,
    CoordinateSystem,
    DiffoctreerastBackendFacade,
    GaussianRasterizerBackend,
    GaussianSplatAsset,
    GsplatBackend,
    KaolinFlexiCubesBackend,
    MipGaussianBackendFacade,
    NvdiffrastBackendFacade,
    SparseComputeBackend,
    SparseVoxelAsset,
    SpconvBackend,
    SurfaceExtractionBackend,
)


def _registry(
    name,
    import_name,
    capability,
    *,
    support_level=BackendSupportLevel.ACCELERATED,
    license_class=BackendLicenseClass.PERMISSIVE,
):
    spec = BackendSpec(
        name=name,
        import_names=(import_name,),
        distribution_names=(name,),
        capabilities=frozenset({capability}),
        support_level=support_level,
        license_class=license_class,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=f"Install {name}",
    )
    return BackendRegistry(
        (spec,),
        module_finder=lambda candidate: ModuleSpec(candidate, loader=None),
        version_getter=lambda _: "1.0",
    )


def test_gsplat_fake_module_converts_canonical_assets_and_scales_intrinsics(monkeypatch):
    captured = {}
    module = types.ModuleType("gsplat")

    def rasterization(**kwargs):
        captured.update(kwargs)
        rendered = torch.zeros(1, kwargs["height"], kwargs["width"], 4)
        rendered[..., :3] = 0.25
        rendered[..., 3] = 2.0
        alpha = torch.full((1, kwargs["height"], kwargs["width"], 1), 0.75)
        return rendered, alpha, {}

    module.rasterization = rasterization
    monkeypatch.setitem(sys.modules, "gsplat", module)
    backend = GsplatBackend(
        device="cpu",
        registry=_registry("gsplat", "gsplat", BackendCapability.GAUSSIAN_RASTERIZATION),
    )
    gaussians = GaussianSplatAsset(
        means=torch.tensor([[0.0, 0.0, 0.0]]),
        log_scales=torch.zeros(1, 3),
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.zeros(1, 1),
        sh_coefficients=torch.ones(1, 1, 3),
        active_sh_degree=0,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    cameras = CameraRig(
        world_to_camera=torch.eye(4).unsqueeze(0),
        intrinsics=torch.tensor([[[4.0, 0.0, 4.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[4, 8]], dtype=torch.int64),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    output = backend.rasterize_gaussians(gaussians, cameras, image_size=(8, 4))
    assert isinstance(backend, GaussianRasterizerBackend)
    assert output["color"].shape == (1, 3, 8, 4)
    assert output["depth"].shape == (1, 1, 8, 4)
    assert output["alpha"].shape == (1, 1, 8, 4)
    torch.testing.assert_close(captured["Ks"][0, 0], torch.tensor([2.0, 0.0, 2.0]))
    torch.testing.assert_close(captured["Ks"][0, 1], torch.tensor([0.0, 4.0, 4.0]))
    torch.testing.assert_close(captured["scales"], gaussians.log_scales.exp())
    torch.testing.assert_close(captured["opacities"], gaussians.opacity_logits.sigmoid().reshape(-1))


def test_spconv_fake_module_roundtrip_and_linear_operation(monkeypatch):
    class FakeSparseConvTensor:
        def __init__(self, features, indices, spatial_shape, batch_size):
            self.features = features
            self.indices = indices
            self.spatial_shape = spatial_shape
            self.batch_size = batch_size

        def replace_feature(self, features):
            return type(self)(features, self.indices, self.spatial_shape, self.batch_size)

    pytorch_module = types.ModuleType("spconv.pytorch")
    pytorch_module.SparseConvTensor = FakeSparseConvTensor
    root_module = types.ModuleType("spconv")
    root_module.pytorch = pytorch_module
    monkeypatch.setitem(sys.modules, "spconv", root_module)
    monkeypatch.setitem(sys.modules, "spconv.pytorch", pytorch_module)
    backend = SpconvBackend(
        device="cpu",
        registry=_registry("spconv", "spconv", BackendCapability.SPARSE_COMPUTE),
    )
    voxels = SparseVoxelAsset(
        coordinates=torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.int64),
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        voxel_size=0.25,
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
        extras={SPCONV_BATCH_INDICES: torch.tensor([0, 1], dtype=torch.int64)},
        metadata={"spconv_spatial_shape": [4, 4, 4], "spconv_batch_size": 2},
    )
    native = backend.to_spconv(voxels)
    assert native.indices.dtype is torch.int32
    assert torch.equal(native.indices, torch.tensor([[0, 0, 1, 2], [1, 3, 2, 1]], dtype=torch.int32))
    restored = backend.from_spconv(native, template=voxels)
    torch.testing.assert_close(restored.features, voxels.features)
    assert torch.equal(restored.coordinates, voxels.coordinates)
    assert torch.equal(restored.extras[SPCONV_BATCH_INDICES], torch.tensor([0, 1]))
    projected = backend.sparse_compute(
        voxels,
        operation="linear",
        parameters={"weight": torch.tensor([[1.0, -1.0]])},
    )
    torch.testing.assert_close(projected.features, torch.tensor([[-1.0], [-1.0]]))
    assert isinstance(backend, SparseComputeBackend)


def test_kaolin_fake_module_uses_only_permissive_ops_flexicubes(monkeypatch):
    class FakeFlexiCubes:
        __module__ = "kaolin.ops.conversions"

        def __init__(self, device):
            assert device == "cpu"

        def construct_voxel_grid(self, resolution):
            return torch.zeros(8, 3), torch.zeros(1, 8, dtype=torch.int64)

        def __call__(self, vertices, scalar_field, cubes, resolution, training):
            del vertices, scalar_field, cubes, resolution, training
            return (
                torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5]]),
                torch.tensor([[0, 1, 2]], dtype=torch.int64),
                torch.empty(0),
            )

    module = types.ModuleType("kaolin")
    module.ops = types.SimpleNamespace(conversions=types.SimpleNamespace(FlexiCubes=FakeFlexiCubes))
    monkeypatch.setitem(sys.modules, "kaolin", module)
    backend = KaolinFlexiCubesBackend(
        device="cpu",
        registry=_registry("kaolin", "kaolin", BackendCapability.SURFACE_EXTRACTION),
    )
    mesh = backend.extract_surface(torch.zeros(3, 3, 3), spacing=(1.0, 2.0, 3.0))
    assert isinstance(backend, SurfaceExtractionBackend)
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
    )
    assert mesh.coordinate_system is CoordinateSystem.RIGHT_HANDED_Z_UP


def test_kaolin_rejects_legacy_noncommercial_flexicubes(monkeypatch):
    class RestrictedFlexiCubes:
        __module__ = "kaolin.non_commercial.flexicubes"

    module = types.ModuleType("kaolin")
    module.ops = types.SimpleNamespace(conversions=types.SimpleNamespace(FlexiCubes=RestrictedFlexiCubes))
    monkeypatch.setitem(sys.modules, "kaolin", module)
    with pytest.raises(RuntimeError, match="Apache-2.0 kaolin.ops"):
        KaolinFlexiCubesBackend(
            device="cpu",
            registry=_registry("kaolin", "kaolin", BackendCapability.SURFACE_EXTRACTION),
        )


@pytest.mark.parametrize(
    ("backend_type", "name", "capability"),
    (
        (GsplatBackend, "gsplat", BackendCapability.GAUSSIAN_RASTERIZATION),
        (SpconvBackend, "spconv", BackendCapability.SPARSE_COMPUTE),
        (KaolinFlexiCubesBackend, "kaolin", BackendCapability.SURFACE_EXTRACTION),
    ),
)
def test_accelerated_adapters_fail_before_import_when_dependency_is_missing(
    monkeypatch,
    backend_type,
    name,
    capability,
):
    spec = BackendSpec(
        name=name,
        import_names=(name,),
        distribution_names=(name,),
        capabilities=frozenset({capability}),
        support_level=BackendSupportLevel.ACCELERATED,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32"}),
        differentiable=True,
        install_hint=f"Install {name}",
    )
    registry = BackendRegistry(
        (spec,),
        module_finder=lambda _: None,
        version_getter=lambda _: "1.0",
    )
    imported = []

    def reject_import(name, *args, **kwargs):
        imported.append(name)
        raise AssertionError("dependency selection must fail before import")

    monkeypatch.setattr(importlib, "import_module", reject_import)
    with pytest.raises(BackendUnavailableError, match=name):
        backend_type(device="cpu", registry=registry)
    assert imported == []


def test_research_facades_never_import_and_require_explicit_acknowledgement(monkeypatch):
    specs = []
    for name, capability in (
        ("nvdiffrast", BackendCapability.MESH_RASTERIZATION),
        ("diffoctreerast", BackendCapability.FIELD_RENDERING),
        ("mip_gaussian", BackendCapability.GAUSSIAN_RASTERIZATION),
    ):
        specs.append(
            BackendSpec(
                name=name,
                import_names=(name,),
                distribution_names=(name,),
                capabilities=frozenset({capability}),
                support_level=BackendSupportLevel.RESEARCH_ONLY,
                license_class=BackendLicenseClass.RESTRICTED,
                devices=frozenset({"cpu"}),
                dtypes=frozenset({"float32"}),
                differentiable=True,
                install_hint=f"Review {name} license",
            )
        )
    registry = BackendRegistry(
        specs,
        module_finder=lambda candidate: ModuleSpec(candidate, loader=None),
        version_getter=lambda _: "1.0",
    )
    imported = []
    original_import_module = importlib.import_module

    def record_import(name, *args, **kwargs):
        imported.append(name)
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record_import)
    facades = (
        (NvdiffrastBackendFacade(registry=registry), BackendCapability.MESH_RASTERIZATION),
        (DiffoctreerastBackendFacade(registry=registry), BackendCapability.FIELD_RENDERING),
        (MipGaussianBackendFacade(registry=registry), BackendCapability.GAUSSIAN_RASTERIZATION),
    )
    for facade, capability in facades:
        assert facade.status().available
        with pytest.raises(ValueError, match="accept_research_license"):
            facade.require(capability, device="cpu")
        assert facade.require(capability, device="cpu", accept_research_license=True).name == facade.name
    assert imported == []
