import json
from dataclasses import replace

import pytest

from diffusers_3d import (
    TRAINING_MANIFEST_NAME,
    FullFineTune,
    TrainingManifest3D,
    TrainingManifestError,
    TrainingManifestMismatchError,
    trainable_parameter_hash,
)


class ExactTarget:
    pass


class ExactExample:
    pass


def make_manifest(**kwargs) -> TrainingManifest3D:
    arguments = {
        "target_type": ExactTarget,
        "example_type": ExactExample,
        "family_id": "manifest-family",
        "recipe_id": "manifest-objective",
        "recipe_version": "1.0",
        "strategy": FullFineTune(("decoder", "denoiser")),
        "base_model": "tests/base-object-3d",
        "revision": "abc123",
        "trainable_parameter_names": ("denoiser.z", "denoiser.a"),
        "objective_config": {"sigma_min": 1e-5, "stage": "shape"},
        "training_config": {
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "seed": 0,
        },
    }
    arguments.update(kwargs)
    return TrainingManifest3D.create(**arguments)


def test_manifest_save_load_is_atomic_deterministic_and_hashed(tmp_path):
    manifest = make_manifest()
    path = manifest.save(tmp_path)
    first_bytes = path.read_bytes()
    manifest.save(tmp_path)

    assert path.name == TRAINING_MANIFEST_NAME
    assert path.read_bytes() == first_bytes
    assert not tuple(tmp_path.glob(f".{TRAINING_MANIFEST_NAME}.*.tmp"))
    assert TrainingManifest3D.load(tmp_path) == manifest
    assert manifest.trainable_parameter_names == ("denoiser.a", "denoiser.z")
    assert manifest.trainable_parameter_hash == trainable_parameter_hash(("denoiser.a", "denoiser.z"))
    assert trainable_parameter_hash(("denoiser.z", "denoiser.a")) == manifest.trainable_parameter_hash
    assert json.loads(first_bytes)["components"] == ["decoder", "denoiser"]
    assert json.loads(first_bytes)["example_type"].endswith(".ExactExample")
    assert path.stat().st_mode & 0o044 == 0o044


def test_manifest_resume_requires_an_exact_match(tmp_path):
    manifest = make_manifest()
    manifest.save(tmp_path)
    loaded = TrainingManifest3D.load(tmp_path)
    loaded.validate_resume(manifest)

    mismatch = make_manifest(revision="different")
    with pytest.raises(TrainingManifestMismatchError, match="revision"):
        loaded.validate_resume(mismatch)
    mismatch = make_manifest(objective_config={"sigma_min": 0.1, "stage": "shape"})
    with pytest.raises(TrainingManifestMismatchError, match="objective_config"):
        loaded.validate_resume(mismatch)
    mismatch = make_manifest(
        training_config={"gradient_accumulation_steps": 1, "learning_rate": 2e-4, "seed": 0}
    )
    with pytest.raises(TrainingManifestMismatchError, match="training_config"):
        loaded.validate_resume(mismatch)


def test_manifest_rejects_hash_tampering_and_unknown_fields():
    manifest = make_manifest()
    with pytest.raises(TrainingManifestError, match="hash"):
        replace(manifest, trainable_parameter_hash="0" * 64)

    data = manifest.to_dict()
    data["unknown"] = True
    with pytest.raises(TrainingManifestError, match="unknown"):
        TrainingManifest3D.from_dict(data)

    data = manifest.to_dict()
    data["schema_version"] = True
    with pytest.raises(TrainingManifestError, match="schema"):
        TrainingManifest3D.from_dict(data)

    data = manifest.to_dict()
    data["components"] = {"decoder": "spoofed", "denoiser": "spoofed"}
    with pytest.raises(TrainingManifestError, match="JSON array"):
        TrainingManifest3D.from_dict(data)
