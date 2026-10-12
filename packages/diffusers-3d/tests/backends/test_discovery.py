from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from diffusers_3d.backends import (
    BackendCapability,
    BackendRegistry,
    BackendSpec,
    BackendSupportLevel,
    discover_backend,
)


def test_backend_spec_is_deeply_immutable_and_normalizes_values(spec_factory):
    spec = spec_factory(
        "example",
        capabilities=[BackendCapability.CONVERSION],
        devices=["CPU"],
        dtypes=["torch.float32"],
    )

    assert spec.capabilities == frozenset({BackendCapability.CONVERSION})
    assert spec.devices == frozenset({"cpu"})
    assert spec.dtypes == frozenset({"float32"})
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"
    with pytest.raises(AttributeError):
        spec.devices.add("cuda")


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"name": "Invalid Name"}, ValueError),
        ({"import_names": ()}, ValueError),
        ({"capabilities": frozenset()}, ValueError),
        ({"devices": frozenset()}, ValueError),
        ({"dtypes": frozenset()}, ValueError),
        ({"differentiable": 1}, TypeError),
        ({"distribution_names": (), "source_url": None}, ValueError),
        ({"import_names": {"valid_module": "spoofed"}}, TypeError),
        ({"distribution_names": (name for name in ("valid-distribution",))}, TypeError),
        ({"capabilities": {BackendCapability.CONVERSION}}, TypeError),
        ({"devices": {"cpu"}}, TypeError),
    ],
)
def test_backend_spec_rejects_invalid_metadata(spec_factory, changes, error_type):
    values = {
        "name": "valid",
        "import_names": ("valid_module",),
        "distribution_names": ("valid-distribution",),
        "capabilities": frozenset({BackendCapability.CONVERSION}),
        "support_level": BackendSupportLevel.PORTABLE,
        "license_class": "permissive",
        "devices": frozenset({"cpu"}),
        "dtypes": frozenset({"float32"}),
        "differentiable": False,
        "install_hint": "Install valid-distribution",
    }
    values.update(changes)

    with pytest.raises(error_type):
        BackendSpec(**values)


def test_registry_rejects_duplicate_exact_name(spec_factory):
    first = spec_factory("same")
    registry = BackendRegistry((first,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec_factory("same", import_name="other_module"))


def test_discovery_uses_explicit_distribution_name(spec_factory):
    spec = spec_factory(
        "image-tools",
        import_name="image_tools_import",
        distribution_name="distribution-with-a-different-name",
    )
    requested_distributions = []

    def get_version(name: str) -> str:
        requested_distributions.append(name)
        return "2.4.1"

    status = discover_backend(
        spec,
        module_finder=lambda name: object() if name == "image_tools_import" else None,
        version_getter=get_version,
    )

    assert status.available
    assert status.version == "2.4.1"
    assert status.distribution_name == "distribution-with-a-different-name"
    assert requested_distributions == ["distribution-with-a-different-name"]


def test_discovery_rejects_colliding_distribution_with_wrong_source(spec_factory):
    source_url = "https://github.com/expected/project.git"
    source_revision = "0123456789abcdef"
    spec = replace(
        spec_factory("colliding", distribution_name="colliding-distribution"),
        source_url=source_url,
        source_revision=source_revision,
        requires_source_provenance=True,
    )

    class Distribution:
        def read_text(self, filename: str) -> str:
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/unrelated/project.git",
                    "vcs_info": {"vcs": "git", "commit_id": source_revision},
                }
            )

    status = discover_backend(
        spec,
        module_finder=lambda _: object(),
        version_getter=lambda _: "1.0.0",
        distribution_getter=lambda _: Distribution(),
    )

    assert status.installed
    assert status.importable
    assert not status.provenance_verified
    assert not status.available
    assert "does not match required source" in status.reason
    assert spec.install_hint in status.reason


def test_discovery_accepts_required_pinned_source_provenance(spec_factory):
    source_url = "https://github.com/expected/project.git"
    source_revision = "0123456789abcdef"
    spec = replace(
        spec_factory("pinned", distribution_name="pinned-distribution"),
        source_url=source_url,
        source_revision=source_revision,
        requires_source_provenance=True,
    )

    class Distribution:
        def read_text(self, filename: str) -> str:
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": source_url,
                    "vcs_info": {"vcs": "git", "commit_id": source_revision},
                }
            )

    status = discover_backend(
        spec,
        module_finder=lambda _: object(),
        version_getter=lambda _: "1.0.0",
        distribution_getter=lambda _: Distribution(),
    )

    assert status.provenance_verified
    assert status.available
    assert status.reason is None


def test_discovery_rejects_an_installed_untested_version(spec_factory):
    spec = replace(spec_factory("versioned"), tested_version="2.4.1")

    status = discover_backend(
        spec,
        module_finder=lambda _: object(),
        version_getter=lambda _: "2.4.0",
    )

    assert status.installed and status.importable
    assert not status.version_compatible
    assert not status.available
    assert "does not match required tested version" in status.reason


def test_discovery_rejects_missing_direct_url_for_required_source(spec_factory):
    spec = replace(
        spec_factory("source-build"),
        source_url="https://github.com/expected/project.git",
        source_revision="0123456789abcdef",
        requires_source_provenance=True,
    )

    class Distribution:
        def read_text(self, filename: str) -> None:
            assert filename == "direct_url.json"
            return None

    status = discover_backend(
        spec,
        module_finder=lambda _: object(),
        version_getter=lambda _: "1.0.0",
        distribution_getter=lambda _: Distribution(),
    )

    assert not status.provenance_verified
    assert not status.available
    assert "has no direct_url.json" in status.reason


def test_discovery_does_not_import_optional_package(tmp_path, monkeypatch, spec_factory):
    package_name = "fake_optional_backend_no_import"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("raise AssertionError('backend was imported')\n")
    (package_dir / "submodule.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(package_name, None)
    sys.modules.pop(f"{package_name}.submodule", None)
    spec = spec_factory(
        "no-import",
        import_name=f"{package_name}.submodule",
        distribution_name="fake-no-import-distribution",
    )

    status = discover_backend(spec, version_getter=lambda _: "1.0.0")

    assert status.available
    assert package_name not in sys.modules
    assert f"{package_name}.submodule" not in sys.modules


def test_missing_backend_diagnostics_are_actionable(spec_factory):
    spec = spec_factory("missing")

    def missing_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    status = discover_backend(spec, module_finder=lambda _: None, version_getter=missing_version)

    assert not status.available
    assert not status.installed
    assert not status.importable
    assert status.missing_import_names == ("missing_module",)
    assert status.missing_distribution_names == ("missing-distribution",)
    assert "missing_module" in status.reason
    assert "missing-distribution" in status.reason
    assert "Install missing-distribution" in status.reason


def test_discovery_report_is_an_immutable_snapshot(spec_factory, registry_factory):
    available = spec_factory("available")
    missing = spec_factory("missing")
    registry = registry_factory((available, missing), installed={"available"}, importable={"available"})

    report = registry.report()

    assert tuple(status.name for status in report.available) == ("available",)
    assert tuple(status.name for status in report.unavailable) == ("missing",)
    with pytest.raises(FrozenInstanceError):
        report.statuses = ()
