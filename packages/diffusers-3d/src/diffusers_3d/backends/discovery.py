from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import json
from collections.abc import Callable, Sequence
from importlib.machinery import ModuleSpec
from typing import Any

from .types import BackendSpec, BackendStatus

ModuleFinder = Callable[[str], ModuleSpec | None]
VersionGetter = Callable[[str], str]
DistributionGetter = Callable[[str], importlib.metadata.Distribution]


def _find_module_without_importing(name: str, finder: ModuleFinder) -> ModuleSpec | None:
    """Find a module without executing it or any of its parent packages."""

    parts = name.split(".")
    spec = finder(parts[0])
    if spec is None or len(parts) == 1:
        return spec

    locations: Sequence[str] | None = spec.submodule_search_locations
    if locations is None:
        return None
    qualified_name = parts[0]
    for part in parts[1:]:
        qualified_name = f"{qualified_name}.{part}"
        spec = importlib.machinery.PathFinder.find_spec(qualified_name, locations)
        if spec is None:
            return None
        locations = spec.submodule_search_locations
    return spec


def discover_backend(
    spec: BackendSpec,
    *,
    module_finder: ModuleFinder | None = None,
    version_getter: VersionGetter | None = None,
    distribution_getter: DistributionGetter | None = None,
) -> BackendStatus:
    """Inspect backend metadata without importing any optional backend module."""

    if not isinstance(spec, BackendSpec):
        raise TypeError("spec must be a BackendSpec")

    finder = module_finder if module_finder is not None else importlib.util.find_spec
    get_version = version_getter if version_getter is not None else importlib.metadata.version
    get_distribution = distribution_getter if distribution_getter is not None else importlib.metadata.distribution

    missing_imports = []
    import_errors = []
    for import_name in spec.import_names:
        try:
            module_spec = _find_module_without_importing(import_name, finder)
        except (ImportError, ModuleNotFoundError, ValueError) as error:
            module_spec = None
            import_errors.append(f"{import_name}: {error}")
        if module_spec is None:
            missing_imports.append(import_name)

    version = None
    distribution_name = None
    missing_distributions = []
    distribution_errors = []
    for candidate in spec.distribution_names:
        try:
            candidate_version = get_version(candidate)
        except importlib.metadata.PackageNotFoundError:
            missing_distributions.append(candidate)
            continue
        except Exception as error:
            missing_distributions.append(candidate)
            distribution_errors.append(f"{candidate}: {error}")
            continue
        if not isinstance(candidate_version, str) or not candidate_version:
            missing_distributions.append(candidate)
            distribution_errors.append(f"{candidate}: version lookup returned no version")
            continue
        distribution_name = candidate
        version = candidate_version
        break

    importable = not missing_imports
    installed = distribution_name is not None
    provenance_verified = True
    provenance_error = None
    if installed and spec.requires_source_provenance:
        assert distribution_name is not None
        provenance_verified, provenance_error = _verify_source_provenance(
            spec,
            distribution_name,
            get_distribution,
        )
    reason = _unavailable_reason(
        spec,
        importable=importable,
        installed=installed,
        provenance_verified=provenance_verified,
        provenance_error=provenance_error,
        version=version,
        missing_imports=missing_imports,
        missing_distributions=missing_distributions,
        import_errors=import_errors,
        distribution_errors=distribution_errors,
    )
    return BackendStatus(
        spec=spec,
        installed=installed,
        importable=importable,
        version=version,
        distribution_name=distribution_name,
        reason=reason,
        provenance_verified=provenance_verified,
        missing_import_names=tuple(missing_imports),
        missing_distribution_names=tuple(missing_distributions),
    )


def _normalize_source_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").lower()


def _verify_source_provenance(
    spec: BackendSpec,
    distribution_name: str,
    distribution_getter: DistributionGetter,
) -> tuple[bool, str | None]:
    assert spec.source_url is not None
    assert spec.source_revision is not None
    try:
        distribution = distribution_getter(distribution_name)
        direct_url_text = distribution.read_text("direct_url.json")
    except Exception as error:
        return False, f"could not inspect source provenance for {distribution_name}: {error}"
    if not direct_url_text:
        return False, f"{distribution_name} has no direct_url.json source provenance"
    try:
        direct_url: dict[str, Any] = json.loads(direct_url_text)
    except (TypeError, ValueError) as error:
        return False, f"{distribution_name} has invalid direct_url.json metadata: {error}"

    actual_url = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(actual_url, str) or not isinstance(vcs_info, dict):
        return False, f"{distribution_name} was not installed from the required version-controlled source"
    actual_revision = vcs_info.get("commit_id")
    if _normalize_source_url(actual_url) != _normalize_source_url(spec.source_url):
        return False, f"source URL {actual_url!r} does not match required source {spec.source_url!r}"
    if actual_revision != spec.source_revision:
        return False, f"source revision {actual_revision!r} does not match required revision {spec.source_revision!r}"
    return True, None


def _unavailable_reason(
    spec: BackendSpec,
    *,
    importable: bool,
    installed: bool,
    provenance_verified: bool,
    provenance_error: str | None,
    version: str | None,
    missing_imports: list[str],
    missing_distributions: list[str],
    import_errors: list[str],
    distribution_errors: list[str],
) -> str | None:
    if importable and installed and provenance_verified:
        return None

    details = []
    if not spec.distribution_names:
        details.append(
            "this source-only backend has no automatically verifiable distribution metadata"
            + (f" (expected source: {spec.source_url})" if spec.source_url else "")
        )
    elif not installed:
        names = ", ".join(missing_distributions)
        details.append(f"no matching distribution metadata was found for: {names}")
    elif version is not None:
        details.append(f"distribution version {version} is installed")

    if not provenance_verified and provenance_error is not None:
        details.append(provenance_error)
    if not importable:
        details.append(f"the following imports are not discoverable: {', '.join(missing_imports)}")
    if import_errors:
        details.append(f"import discovery errors: {'; '.join(import_errors)}")
    if distribution_errors:
        details.append(f"distribution metadata errors: {'; '.join(distribution_errors)}")

    details.append(
        f"supported devices: {', '.join(sorted(spec.devices))}; supported dtypes: {', '.join(sorted(spec.dtypes))}"
    )
    details.append(spec.install_hint.rstrip("."))
    if spec.tested_version is not None:
        details.append(f"tested version: {spec.tested_version}")
    if spec.tested_build is not None:
        details.append(f"tested build: {spec.tested_build}")
    return ". ".join(details) + "."


__all__ = ["DistributionGetter", "ModuleFinder", "VersionGetter", "discover_backend"]
