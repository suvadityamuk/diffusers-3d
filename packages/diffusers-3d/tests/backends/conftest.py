from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable
from importlib.machinery import ModuleSpec

import pytest

from diffusers_3d.backends import (
    BackendCapability,
    BackendLicenseClass,
    BackendRegistry,
    BackendSpec,
    BackendSupportLevel,
)


def make_backend_spec(
    name: str,
    *,
    capabilities: Iterable[BackendCapability] = (BackendCapability.GEOMETRY_PROCESSING,),
    support_level: BackendSupportLevel = BackendSupportLevel.PORTABLE,
    license_class: BackendLicenseClass = BackendLicenseClass.PERMISSIVE,
    devices: Iterable[str] = ("cpu",),
    dtypes: Iterable[str] = ("float32",),
    differentiable: bool = False,
    import_name: str | None = None,
    distribution_name: str | None = None,
) -> BackendSpec:
    module_name = import_name or f"{name.replace('-', '_')}_module"
    dist_name = distribution_name or f"{name}-distribution"
    return BackendSpec(
        name=name,
        import_names=(module_name,),
        distribution_names=(dist_name,),
        capabilities=frozenset(capabilities),
        support_level=support_level,
        license_class=license_class,
        devices=frozenset(devices),
        dtypes=frozenset(dtypes),
        differentiable=differentiable,
        install_hint=f"Install {dist_name}",
    )


def make_fake_registry(
    specs: Iterable[BackendSpec],
    *,
    installed: Iterable[str] | None = None,
    importable: Iterable[str] | None = None,
) -> BackendRegistry:
    specs = tuple(specs)
    installed_names = {spec.name for spec in specs} if installed is None else set(installed)
    importable_names = {spec.name for spec in specs} if importable is None else set(importable)
    modules = {import_name for spec in specs if spec.name in importable_names for import_name in spec.import_names}
    versions = {
        distribution_name: f"1.0.{index}"
        for index, spec in enumerate(specs)
        if spec.name in installed_names
        for distribution_name in spec.distribution_names
    }

    def find_module(name: str) -> ModuleSpec | None:
        return ModuleSpec(name, loader=None) if name in modules else None

    def get_version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as error:
            raise importlib.metadata.PackageNotFoundError(name) from error

    return BackendRegistry(specs, module_finder=find_module, version_getter=get_version)


@pytest.fixture
def spec_factory():
    return make_backend_spec


@pytest.fixture
def registry_factory():
    return make_fake_registry
