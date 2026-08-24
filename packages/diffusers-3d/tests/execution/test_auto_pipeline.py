from __future__ import annotations

import pytest
from diffusers import DiffusionPipeline

from diffusers_3d import (
    AutoPipelineFor3D,
    AutoPipelineForImageTo3D,
    AutoPipelineForTextTo3D,
    Object3DLoadingError,
    Object3DPipelineRegistration,
    Object3DPipelineRegistry,
    Object3DRegistrationError,
    Object3DTaskError,
)


class GenericDiffusionPipeline(DiffusionPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.register_to_config()


def reviewed_registry(pipeline_class):
    return Object3DPipelineRegistry(
        (
            Object3DPipelineRegistration(
                pipeline_class,
                pipeline_class.object3d_model_index(),
            ),
        )
    ).freeze()


def test_auto_loader_reads_metadata_then_delegates_to_exact_registered_class(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    metadata_directory = tmp_path / "variant"
    dispatch_pipeline_class.object3d_model_index().save_pretrained(metadata_directory)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        reviewed_registry(dispatch_pipeline_class),
    )

    pipeline = AutoPipelineFor3D.from_pretrained(
        tmp_path,
        subfolder="variant",
        task="text-to-3d",
        revision="exact-revision",
        cache_dir=tmp_path / "cache",
        token="token",
        local_files_only=True,
        torch_dtype="float32",
    )

    assert type(pipeline) is dispatch_pipeline_class
    assert pipeline.loaded_from == tmp_path
    assert pipeline.loaded_kwargs == {
        "cache_dir": tmp_path / "cache",
        "local_files_only": True,
        "revision": "exact-revision",
        "subfolder": "variant",
        "token": "token",
        "torch_dtype": "float32",
        "trust_remote_code": False,
    }


def test_auto_loader_requires_task_when_metadata_is_ambiguous(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    dispatch_pipeline_class.object3d_model_index().save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        reviewed_registry(dispatch_pipeline_class),
    )

    with pytest.raises(Object3DTaskError, match="multiple tasks"):
        AutoPipelineFor3D.from_pretrained(tmp_path)

    assert dispatch_pipeline_class.load_calls == []


def test_task_specific_auto_loaders_enforce_task_constraints(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    dispatch_pipeline_class.object3d_model_index().save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        reviewed_registry(dispatch_pipeline_class),
    )

    image_pipeline = AutoPipelineForImageTo3D.from_pretrained(tmp_path)
    text_pipeline = AutoPipelineForTextTo3D.from_pretrained(tmp_path)

    assert type(image_pipeline) is dispatch_pipeline_class
    assert type(text_pipeline) is dispatch_pipeline_class
    with pytest.raises(Object3DTaskError, match="only supports"):
        AutoPipelineForTextTo3D.from_pretrained(tmp_path, task="image-to-3d")


def test_task_specific_auto_loader_rejects_pipeline_without_its_task(
    tmp_path,
    tiny_pipeline_class,
    monkeypatch,
):
    tiny_pipeline_class.object3d_model_index().save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        reviewed_registry(tiny_pipeline_class),
    )

    with pytest.raises(Object3DTaskError, match="not declared"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path)


def test_auto_loader_rejects_remote_code_before_any_component_loading(
    tmp_path,
    dispatch_pipeline_class,
):
    with pytest.raises(Object3DLoadingError, match="trust_remote_code=True"):
        AutoPipelineFor3D.from_pretrained(tmp_path, trust_remote_code=True)

    assert dispatch_pipeline_class.load_calls == []


def test_unknown_object3d_metadata_fails_before_concrete_loading(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    dispatch_pipeline_class.object3d_model_index().save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        Object3DPipelineRegistry().freeze(),
    )

    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        AutoPipelineFor3D.from_pretrained(tmp_path, task="text-to-3d")

    assert dispatch_pipeline_class.load_calls == []


def test_generic_diffusion_pipeline_is_rejected_before_from_pretrained(
    tmp_path,
    monkeypatch,
):
    GenericDiffusionPipeline().save_pretrained(tmp_path)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generic component loading must not run")

    monkeypatch.setattr(
        GenericDiffusionPipeline,
        "from_pretrained",
        classmethod(fail_if_called),
    )

    with pytest.raises(Object3DLoadingError, match="Expected object3d_model_index.json"):
        AutoPipelineFor3D.from_pretrained(tmp_path)

    assert not called
