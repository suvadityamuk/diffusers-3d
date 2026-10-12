from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import replace

import pytest
import torch

import diffusers_3d
from diffusers_3d.backends import (
    BACKEND_REGISTRY,
    DEFAULT_BACKEND_SPECS,
    BackendCapability,
    BackendIncompatibleError,
    BackendNotFoundError,
    BackendPolicyError,
    BackendRegistry,
    BackendSupportLevel,
    BackendUnavailableError,
    create_default_backend_registry,
)


def test_selection_is_deterministic_by_support_level_then_exact_name(spec_factory, registry_factory):
    accelerated = spec_factory(
        "a-accelerated",
        capabilities=(BackendCapability.MESH_RASTERIZATION,),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cuda",),
        dtypes=("float16",),
        differentiable=True,
    )
    portable_z = spec_factory(
        "z-portable",
        capabilities=(BackendCapability.MESH_RASTERIZATION,),
        devices=("cuda",),
        dtypes=("float16",),
        differentiable=True,
    )
    portable_a = spec_factory(
        "b-portable",
        capabilities=(BackendCapability.MESH_RASTERIZATION,),
        devices=("cuda",),
        dtypes=("float16",),
        differentiable=True,
    )
    registry = registry_factory((portable_z, accelerated, portable_a))

    selected = registry.select(
        BackendCapability.MESH_RASTERIZATION,
        device="cuda:0",
        dtype=torch.float16,
        differentiable=True,
    )

    assert selected.name == "b-portable"
    assert tuple(
        spec.name
        for spec in registry.candidates(
            capability=BackendCapability.MESH_RASTERIZATION,
            device="cuda",
            dtype="float16",
            differentiable=True,
        )
    ) == ("b-portable", "z-portable", "a-accelerated")


def test_candidates_filter_capability_device_dtype_and_differentiability(spec_factory, registry_factory):
    cpu = spec_factory(
        "cpu",
        capabilities=(BackendCapability.GEOMETRY_PROCESSING,),
        devices=("cpu",),
        dtypes=("float32", "float64"),
    )
    cuda = spec_factory(
        "cuda",
        capabilities=(BackendCapability.GEOMETRY_PROCESSING, BackendCapability.MESH_RASTERIZATION),
        support_level=BackendSupportLevel.ACCELERATED,
        devices=("cuda",),
        dtypes=("float16",),
        differentiable=True,
    )
    registry = registry_factory((cpu, cuda))

    assert registry.available(device="cpu") == (cpu,)
    assert registry.candidates(device="cuda:7", dtype=torch.float16, differentiable=True) == (cuda,)
    assert registry.candidates(capability=BackendCapability.MESH_RASTERIZATION) == (cuda,)
    assert registry.candidates(device="cuda", differentiable=False) == ()


def test_research_only_backend_is_never_an_automatic_candidate(spec_factory, registry_factory):
    research = spec_factory(
        "research",
        capabilities=(BackendCapability.FIELD_RENDERING,),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        devices=("cuda",),
        differentiable=True,
    )
    registry = registry_factory((research,))

    assert registry.candidates(capability=BackendCapability.FIELD_RENDERING) == ()
    assert registry.available(capability=BackendCapability.FIELD_RENDERING) == (research,)
    with pytest.raises(BackendIncompatibleError):
        registry.select(BackendCapability.FIELD_RENDERING, device="cuda")


def test_explicit_research_selection_requires_policy_opt_in(spec_factory, registry_factory):
    research = spec_factory(
        "research",
        capabilities=(BackendCapability.FIELD_RENDERING,),
        support_level=BackendSupportLevel.RESEARCH_ONLY,
        devices=("cuda",),
        differentiable=True,
    )
    registry = registry_factory((research,))

    with pytest.raises(BackendPolicyError, match="allow_research_only=True"):
        registry.select(BackendCapability.FIELD_RENDERING, name="research", device="cuda")

    assert (
        registry.select(
            BackendCapability.FIELD_RENDERING,
            name="research",
            device="cuda",
            allow_research_only=True,
        )
        is research
    )


def test_explicit_selection_has_actionable_not_found_unavailable_and_incompatible_errors(
    spec_factory, registry_factory
):
    spec = replace(
        spec_factory(
            "renderer",
            capabilities=(BackendCapability.MESH_RASTERIZATION,),
            devices=("cuda",),
            dtypes=("float32",),
        ),
        tested_version="2.0.0",
    )
    unavailable_registry = registry_factory((spec,), installed=set(), importable=set())

    with pytest.raises(BackendNotFoundError, match="Registered backend names: renderer"):
        unavailable_registry.get("Renderer")
    with pytest.raises(BackendUnavailableError) as unavailable:
        unavailable_registry.select(
            BackendCapability.MESH_RASTERIZATION,
            name="renderer",
            device="cuda",
            dtype="float32",
        )
    assert "Install renderer-distribution" in str(unavailable.value)
    assert "tested version: 2.0.0" in str(unavailable.value)

    installed_registry = registry_factory((spec,))
    with pytest.raises(BackendIncompatibleError) as incompatible:
        installed_registry.select(
            BackendCapability.MESH_RASTERIZATION,
            name="renderer",
            device="cpu",
            dtype="float16",
        )
    assert "device='cpu'" in str(incompatible.value)
    assert "dtype='float16'" in str(incompatible.value)
    assert "devices=['cuda']" in str(incompatible.value)


def test_default_registry_factory_has_no_global_mutation_leakage(spec_factory):
    first = create_default_backend_registry()
    second = create_default_backend_registry()
    custom = spec_factory("custom")

    first.register(custom)

    assert "custom" in first
    assert "custom" not in second
    assert "custom" not in BACKEND_REGISTRY
    assert BACKEND_REGISTRY.frozen
    with pytest.raises(RuntimeError, match="read-only"):
        BACKEND_REGISTRY.register(custom)


def test_planned_default_specs_keep_import_distribution_and_policy_metadata_separate():
    specs = {spec.name: spec for spec in DEFAULT_BACKEND_SPECS}
    assert set(specs) == {
        "cumesh",
        "diffoctreerast",
        "flex_gemm",
        "gsplat",
        "kaolin",
        "mip_gaussian",
        "nvdiffrast",
        "nvdiffrec_render",
        "o_voxel",
        "scikit-image",
        "spconv",
        "trimesh",
        "utils3d",
        "xatlas",
    }
    assert specs["scikit-image"].import_names == ("skimage",)
    assert specs["scikit-image"].distribution_names == ("scikit-image",)
    assert specs["utils3d"].source_revision == "9a4eb15e4021b67b12c460c7057d642626897ec8"
    assert "EasternJournalist" in specs["utils3d"].install_hint
    assert specs["kaolin"].license_class.value == "permissive"
    assert "kaolin.non_commercial" in specs["kaolin"].install_hint
    assert specs["cumesh"].source_revision == "12289e1062f0603f2f0d0771b02e1395d247f26f"
    assert specs["cumesh"].requires_source_provenance
    assert specs["flex_gemm"].source_revision == "6dd94a859c26ee8246888502eada3dd8ad85532e"
    assert specs["flex_gemm"].requires_source_provenance
    assert specs["flex_gemm"].devices == frozenset({"cuda"})
    assert specs["o_voxel"].source_url is not None
    assert "--no-deps --no-build-isolation" in specs["o_voxel"].install_hint
    research_revisions = {
        "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",
        "nvdiffrec_render": "a3e73909a01887c8a135235ff860dd23a045cc1b",
        "diffoctreerast": "b09c20b84ec3aace4729e6e18a613112320eca3a",
        "mip_gaussian": "dda02ab5ecf45d6edb8c540d9bb65c7e451345a9",
    }
    for name, revision in research_revisions.items():
        assert specs[name].source_revision == revision
        assert specs[name].requires_source_provenance
    for name in ("utils3d", "nvdiffrast", "nvdiffrec_render", "diffoctreerast", "mip_gaussian"):
        assert specs[name].support_level is BackendSupportLevel.RESEARCH_ONLY


def test_no_installed_backend_smoke_does_not_import_optional_modules():
    watched_modules = {name.split(".", maxsplit=1)[0] for spec in DEFAULT_BACKEND_SPECS for name in spec.import_names}
    before = {name: name in sys.modules for name in watched_modules}

    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    registry = BackendRegistry(
        DEFAULT_BACKEND_SPECS,
        module_finder=lambda _: None,
        version_getter=missing_version,
    )
    report = registry.report()

    assert len(report.unavailable) == len(DEFAULT_BACKEND_SPECS)
    assert {name: name in sys.modules for name in watched_modules} == before
    assert diffusers_3d.BackendRegistry is BackendRegistry
    assert diffusers_3d.BACKEND_REGISTRY is BACKEND_REGISTRY
