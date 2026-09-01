from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from diffusers_3d import (
    OBJECT3D_MODEL_INDEX_NAME,
    Object3DComponentSpec,
    Object3DMetadataError,
    Object3DModelIndex,
    Object3DSchemaError,
    ReviewStatus,
)


def test_metadata_is_immutable_canonical_and_deterministic(
    tmp_path,
    tiny_pipeline_class,
):
    metadata = tiny_pipeline_class.object3d_model_index()
    first_path = metadata.save_pretrained(tmp_path / "first")
    first_contents = first_path.read_bytes()

    loaded = Object3DModelIndex.from_pretrained(tmp_path / "first")
    second_path = loaded.save_pretrained(tmp_path / "second")

    assert loaded == metadata
    assert second_path.read_bytes() == first_contents
    assert first_contents.endswith(b"\n")
    assert first_path.stat().st_mode & 0o044 == 0o044
    with pytest.raises(FrozenInstanceError):
        metadata.family_id = "changed"  # type: ignore[misc]


def test_component_records_are_strict_immutable_and_exact():
    component = Object3DComponentSpec(
        name="denoiser",
        expected_class="installed.models.ReviewedDenoiser",
        subfolder="denoiser",
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )

    assert Object3DComponentSpec.from_dict(component.to_dict()) == component
    with pytest.raises(FrozenInstanceError):
        component.name = "changed"  # type: ignore[misc]
    with pytest.raises(Object3DMetadataError, match="unknown fields"):
        Object3DComponentSpec.from_dict({**component.to_dict(), "remote_file": "model.py"})
    with pytest.raises(Object3DMetadataError, match="Only reviewed"):
        Object3DComponentSpec(
            name="denoiser",
            expected_class="installed.models.ReviewedDenoiser",
            subfolder="denoiser",
            optional=True,
            review_status=ReviewStatus.UNREVIEWED,
            loading_eligible=True,
        )


def test_local_metadata_loading_supports_subfolder(tmp_path, tiny_pipeline_class):
    metadata = tiny_pipeline_class.object3d_model_index()
    metadata.save_pretrained(tmp_path / "variant")

    loaded = Object3DModelIndex.from_pretrained(tmp_path, subfolder="variant")

    assert loaded == metadata


def test_hub_metadata_loading_uses_public_config_only_download(
    tmp_path,
    tiny_pipeline_class,
    monkeypatch,
):
    metadata_path = tiny_pipeline_class.object3d_model_index().save_pretrained(tmp_path)
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(metadata_path)

    monkeypatch.setattr(
        "diffusers_3d.execution.metadata.hf_hub_download",
        fake_hf_hub_download,
    )

    loaded = Object3DModelIndex.from_pretrained(
        "organization/object3d-model",
        revision="exact-revision",
        subfolder="pipeline",
        cache_dir=tmp_path / "cache",
        token="test-token",
        local_files_only=True,
    )

    assert loaded == tiny_pipeline_class.object3d_model_index()
    assert calls == [
        {
            "repo_id": "organization/object3d-model",
            "filename": OBJECT3D_MODEL_INDEX_NAME,
            "revision": "exact-revision",
            "subfolder": "pipeline",
            "cache_dir": tmp_path / "cache",
            "token": "test-token",
            "local_files_only": True,
        }
    ]


def test_metadata_rejects_unknown_fields_invalid_json_and_schema(
    tmp_path,
    tiny_pipeline_class,
):
    metadata = tiny_pipeline_class.object3d_model_index()
    unknown = metadata.to_dict()
    unknown["remote_python_file"] = "pipeline.py"

    with pytest.raises(Object3DMetadataError, match="unknown fields"):
        Object3DModelIndex.from_dict(unknown)
    with pytest.raises(Object3DSchemaError, match="Unsupported"):
        replace(metadata, schema_version=1)
    spoofed = metadata.to_dict()
    spoofed["task_ids"] = {"image-to-3d": "spoofed"}
    with pytest.raises(Object3DMetadataError, match="JSON array"):
        Object3DModelIndex.from_dict(spoofed)
    malformed_components = metadata.to_dict()
    malformed_components["components"] = {"denoiser": "spoofed"}
    with pytest.raises(Object3DMetadataError, match="components must be a JSON array"):
        Object3DModelIndex.from_dict(malformed_components)

    invalid_directory = tmp_path / "invalid"
    invalid_directory.mkdir()
    (invalid_directory / OBJECT3D_MODEL_INDEX_NAME).write_text("{not-json")
    with pytest.raises(Object3DMetadataError, match="Invalid JSON"):
        Object3DModelIndex.from_pretrained(invalid_directory)


def test_metadata_rejects_duplicate_component_names_and_subfolders(tiny_pipeline_class):
    component = Object3DComponentSpec(
        name="first",
        expected_class="installed.models.First",
        subfolder="first",
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )
    duplicate_name = Object3DComponentSpec(
        name="first",
        expected_class="installed.models.Second",
        subfolder="second",
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )
    duplicate_subfolder = Object3DComponentSpec(
        name="second",
        expected_class="installed.models.Second",
        subfolder="first",
        optional=False,
        review_status=ReviewStatus.REVIEWED,
        loading_eligible=True,
    )
    metadata = tiny_pipeline_class.object3d_model_index()

    with pytest.raises(Object3DMetadataError, match="duplicate names"):
        replace(metadata, components=(component, duplicate_name))
    with pytest.raises(Object3DMetadataError, match="duplicate subfolders"):
        replace(metadata, components=(component, duplicate_subfolder))
