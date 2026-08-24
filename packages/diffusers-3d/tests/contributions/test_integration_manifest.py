from __future__ import annotations

import dataclasses

import pytest

from diffusers_3d import IntegrationManifest3D, IntegrationManifestError


def test_strict_deterministic_roundtrip_is_deeply_immutable(tmp_path, manifest_factory):
    manifest = manifest_factory()
    path = tmp_path / "nested" / "integration.json"

    first_path = manifest.save(path)
    first_payload = path.read_bytes()
    second_path = manifest.save(path)

    assert first_path == second_path == path
    assert path.read_bytes() == first_payload
    assert IntegrationManifest3D.load(path) == manifest
    assert IntegrationManifest3D.loads(manifest.dumps()) == manifest
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.integration_id = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.upstream.revision = "b" * 40


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["workflow"].update({"unknown": True}),
        lambda data: data["components"][0]["parity"][0].update({"unknown": True}),
        lambda data: data["training"].update({"unknown": True}),
    ],
)
def test_unknown_fields_are_rejected_at_every_level(manifest_factory, mutate):
    data = manifest_factory().to_dict()
    mutate(data)

    with pytest.raises(IntegrationManifestError, match="unknown fields"):
        IntegrationManifest3D.from_dict(data)


def test_missing_fields_and_wrong_json_types_are_rejected(manifest_factory):
    data = manifest_factory().to_dict()
    del data["workflow"]
    with pytest.raises(IntegrationManifestError, match="missing fields: workflow"):
        IntegrationManifest3D.from_dict(data)

    data = manifest_factory().to_dict()
    data["components"] = "not-an-array"
    with pytest.raises(IntegrationManifestError, match="components must be a JSON array"):
        IntegrationManifest3D.from_dict(data)


def test_duplicate_fields_and_non_finite_numbers_are_rejected():
    with pytest.raises(IntegrationManifestError, match="duplicate field 'schema'"):
        IntegrationManifest3D.loads('{"schema": "first", "schema": "second"}')
    with pytest.raises(IntegrationManifestError, match="non-finite number"):
        IntegrationManifest3D.loads('{"value": NaN}')
