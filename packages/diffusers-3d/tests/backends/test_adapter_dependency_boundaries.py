from __future__ import annotations

import importlib
import importlib.metadata

import pytest

from diffusers_3d import (
    BackendCapability,
    BackendLicenseClass,
    BackendRegistry,
    BackendSpec,
    BackendSupportLevel,
    BackendUnavailableError,
    ScikitImageBackend,
    TrimeshBackend,
    XAtlasBackend,
)


@pytest.mark.parametrize(
    ("adapter", "name", "import_name", "distribution_name", "capabilities"),
    [
        (
            TrimeshBackend,
            "trimesh",
            "trimesh",
            "trimesh",
            (
                BackendCapability.CONVERSION,
                BackendCapability.SERIALIZATION,
                BackendCapability.GEOMETRY_PROCESSING,
            ),
        ),
        (
            ScikitImageBackend,
            "scikit-image",
            "skimage",
            "scikit-image",
            (BackendCapability.SURFACE_EXTRACTION,),
        ),
        (
            XAtlasBackend,
            "xatlas",
            "xatlas",
            "xatlas",
            (BackendCapability.GEOMETRY_PROCESSING,),
        ),
    ],
)
def test_missing_dependency_fails_registry_selection_before_import(
    monkeypatch,
    adapter,
    name,
    import_name,
    distribution_name,
    capabilities,
):
    spec = BackendSpec(
        name=name,
        import_names=(import_name,),
        distribution_names=(distribution_name,),
        capabilities=frozenset(capabilities),
        support_level=BackendSupportLevel.PORTABLE,
        license_class=BackendLicenseClass.PERMISSIVE,
        devices=frozenset({"cpu"}),
        dtypes=frozenset({"float32"}),
        differentiable=False,
        install_hint=f"Install {distribution_name}",
    )

    def missing_version(candidate: str) -> str:
        raise importlib.metadata.PackageNotFoundError(candidate)

    registry = BackendRegistry(
        (spec,),
        module_finder=lambda _: None,
        version_getter=missing_version,
    )
    imported = []
    original_import_module = importlib.import_module

    def record_import(candidate, *args, **kwargs):
        imported.append(candidate)
        return original_import_module(candidate, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record_import)

    with pytest.raises(BackendUnavailableError, match=f"Backend {name!r} is unavailable"):
        adapter(registry=registry)
    assert not any(candidate == import_name or candidate.startswith(f"{import_name}.") for candidate in imported)
