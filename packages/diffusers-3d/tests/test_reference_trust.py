from __future__ import annotations

import pytest

import diffusers_3d._reference as reference
from diffusers_3d._reference import ReferenceCheckoutError


def test_optional_reference_failure_returns_skip_reason(monkeypatch):
    monkeypatch.delenv("DIFFUSERS_3D_REQUIRE_REFERENCE", raising=False)
    error = ReferenceCheckoutError("optional reference is unavailable")

    assert reference.reference_unavailable(error) == str(error)


def test_required_reference_dependency_failure_cannot_become_a_skip(monkeypatch):
    monkeypatch.setenv("DIFFUSERS_3D_REQUIRE_REFERENCE", "1")

    def missing_dependency(_module_name):
        raise ImportError("missing test dependency")

    monkeypatch.setattr(reference.importlib, "import_module", missing_dependency)
    with pytest.raises(
        ReferenceCheckoutError, match="reference dependency 'missing_dependency' is unavailable"
    ) as info:
        reference.import_reference_dependency("missing_dependency")

    with pytest.raises(ReferenceCheckoutError) as required:
        reference.reference_unavailable(info.value)
    assert required.value is info.value
