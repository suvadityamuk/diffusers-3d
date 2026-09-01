from __future__ import annotations

import json

import pytest
from diffusers import DiffusionPipeline

from diffusers_3d import (
    AutoPipelineFor3D,
    AutoPipelineForImageTo3D,
    AutoPipelineForTextTo3D,
    Object3DLoadingError,
    Object3DMetadataError,
    Object3DPipelineRegistration,
    Object3DPipelineRegistry,
    Object3DRegistrationError,
    Object3DTaskError,
    TrellisDinov2Conditioner,
    TrellisFlowEulerScheduler,
    TrellisImageTo3DPipeline,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
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


def tiny_trellis_pipeline():
    return TrellisImageTo3DPipeline(
        conditioner=TrellisDinov2Conditioner(**TrellisDinov2Conditioner.tiny_config()),
        sparse_structure_flow_model=TrellisSparseStructureFlowModel(**TrellisSparseStructureFlowModel.tiny_config()),
        sparse_structure_decoder=TrellisSparseStructureDecoder(**TrellisSparseStructureDecoder.tiny_config()),
        sparse_structure_scheduler=TrellisFlowEulerScheduler(),
    )


def update_model_index(directory, update):
    path = directory / DiffusionPipeline.config_name
    model_index = json.loads(path.read_text(encoding="utf-8"))
    update(model_index)
    path.write_text(json.dumps(model_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_auto_loader_reads_metadata_then_delegates_to_exact_registered_class(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    metadata_directory = tmp_path / "variant"
    dispatch_pipeline_class().save_pretrained(metadata_directory)
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
    assert pipeline.loaded_from == metadata_directory
    assert pipeline.loaded_kwargs == {
        "local_files_only": True,
        "torch_dtype": "float32",
        "trust_remote_code": False,
    }


def test_auto_loader_requires_task_when_metadata_is_ambiguous(
    tmp_path,
    dispatch_pipeline_class,
    monkeypatch,
):
    dispatch_pipeline_class().save_pretrained(tmp_path)
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
    dispatch_pipeline_class().save_pretrained(tmp_path)
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
    tiny_pipeline_class().save_pretrained(tmp_path)
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
    dispatch_pipeline_class().save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoPipelineFor3D,
        "_registry",
        Object3DPipelineRegistry().freeze(),
    )

    with pytest.raises(Object3DRegistrationError, match="no exact reviewed"):
        AutoPipelineFor3D.from_pretrained(tmp_path, task="text-to-3d")

    assert dispatch_pipeline_class.load_calls == []


def test_remote_loader_downloads_precise_snapshot_then_loads_installed_components(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    converted = repository / "variant"
    tiny_trellis_pipeline().save_pretrained(converted)
    hub_download_calls = []
    snapshot_calls = []

    def fake_hf_hub_download(**kwargs):
        hub_download_calls.append(kwargs)
        return str(converted / "object3d_model_index.json")

    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(repository)

    monkeypatch.setattr("diffusers_3d.execution.metadata.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        "diffusers_3d.execution.auto_pipeline.huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    loaded = AutoPipelineForImageTo3D.from_pretrained(
        "organization/converted-trellis",
        revision="immutable-revision",
        subfolder="variant",
        cache_dir=tmp_path / "cache",
        token="test-token",
        local_files_only=True,
    )

    assert type(loaded) is TrellisImageTo3DPipeline
    assert hub_download_calls == [
        {
            "repo_id": "organization/converted-trellis",
            "filename": "object3d_model_index.json",
            "revision": "immutable-revision",
            "subfolder": "variant",
            "cache_dir": tmp_path / "cache",
            "token": "test-token",
            "local_files_only": True,
        }
    ]
    assert snapshot_calls == [
        {
            "repo_id": "organization/converted-trellis",
            "revision": "immutable-revision",
            "cache_dir": tmp_path / "cache",
            "token": "test-token",
            "local_files_only": True,
            "allow_patterns": [
                "variant/model_index.json",
                "variant/object3d_model_index.json",
                "variant/conditioner/*.json",
                "variant/conditioner/*.safetensors",
                "variant/conditioner/*.bin",
                "variant/conditioner/*.flashpack",
                "variant/sparse_structure_decoder/*.json",
                "variant/sparse_structure_decoder/*.safetensors",
                "variant/sparse_structure_decoder/*.bin",
                "variant/sparse_structure_decoder/*.flashpack",
                "variant/sparse_structure_flow_model/*.json",
                "variant/sparse_structure_flow_model/*.safetensors",
                "variant/sparse_structure_flow_model/*.bin",
                "variant/sparse_structure_flow_model/*.flashpack",
                "variant/sparse_structure_scheduler/*.json",
                "variant/sparse_structure_scheduler/*.safetensors",
                "variant/sparse_structure_scheduler/*.bin",
                "variant/sparse_structure_scheduler/*.flashpack",
            ],
        }
    ]


@pytest.mark.parametrize(
    "tampered_tuple",
    (
        ["attacker.module", "TrellisDinov2Conditioner"],
        ["diffusers_3d.families.trellis.conditioner", "AttackerConditioner"],
    ),
)
def test_local_loader_rejects_tampered_component_module_or_class(tmp_path, tampered_tuple):
    tiny_trellis_pipeline().save_pretrained(tmp_path)
    update_model_index(tmp_path, lambda model_index: model_index.__setitem__("conditioner", tampered_tuple))

    with pytest.raises(Object3DLoadingError, match="expected exact reviewed tuple"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_local_loader_rejects_missing_or_malformed_required_component(tmp_path):
    missing = tmp_path / "missing"
    malformed = tmp_path / "malformed"
    tiny_trellis_pipeline().save_pretrained(missing)
    tiny_trellis_pipeline().save_pretrained(malformed)
    update_model_index(missing, lambda model_index: model_index.pop("conditioner"))
    update_model_index(malformed, lambda model_index: model_index.__setitem__("conditioner", {"class": "spoofed"}))

    with pytest.raises(Object3DLoadingError, match="missing required component"):
        AutoPipelineForImageTo3D.from_pretrained(missing, local_files_only=True)
    with pytest.raises(Object3DMetadataError, match="exact two-item JSON array"):
        AutoPipelineForImageTo3D.from_pretrained(malformed, local_files_only=True)


def test_local_loader_accepts_optional_none_and_round_trips_reviewed_pipeline(tmp_path):
    original = tiny_trellis_pipeline()
    original.save_pretrained(tmp_path)

    loaded = AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)

    assert type(loaded) is TrellisImageTo3DPipeline
    assert loaded.slat_flow_model is None
    assert loaded.slat_scheduler is None
    assert loaded.gaussian_decoder is None
    assert loaded.object3d_model_index() == original.object3d_model_index()


def test_local_loader_rejects_unexpected_experimental_component_after_load(tmp_path, monkeypatch):
    tiny_trellis_pipeline().save_pretrained(tmp_path)
    loaded_pipeline = tiny_trellis_pipeline()
    loaded_pipeline.slat_flow_model = object()
    monkeypatch.setattr(
        TrellisImageTo3DPipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: loaded_pipeline),
    )

    with pytest.raises(Object3DLoadingError, match="unexpected experimental component"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_local_loader_rejects_swapped_installed_component_class(tmp_path, monkeypatch):
    tiny_trellis_pipeline().save_pretrained(tmp_path)
    loaded_pipeline = tiny_trellis_pipeline()
    monkeypatch.setattr(
        TrellisImageTo3DPipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: loaded_pipeline),
    )
    monkeypatch.setattr(
        "diffusers_3d.families.trellis.conditioner.TrellisDinov2Conditioner",
        TrellisSparseStructureFlowModel,
    )

    with pytest.raises(Object3DLoadingError, match="does not resolve to that exact class"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_local_loader_rejects_missing_required_component_after_load(tmp_path, monkeypatch):
    tiny_trellis_pipeline().save_pretrained(tmp_path)
    loaded_pipeline = tiny_trellis_pipeline()
    loaded_pipeline.conditioner = None
    monkeypatch.setattr(
        TrellisImageTo3DPipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: loaded_pipeline),
    )

    with pytest.raises(Object3DLoadingError, match="missing required component"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


def test_auto_loader_rejects_experimental_model_index_component_before_loading(tmp_path):
    tiny_trellis_pipeline().save_pretrained(tmp_path)
    expected_class = "diffusers_3d.families.trellis.models.TrellisSLatFlowModel"
    module_name, _, class_name = expected_class.rpartition(".")
    update_model_index(
        tmp_path,
        lambda model_index: model_index.__setitem__("slat_flow_model", [module_name, class_name]),
    )

    with pytest.raises(Object3DLoadingError, match="not eligible for automatic loading"):
        AutoPipelineForImageTo3D.from_pretrained(tmp_path, local_files_only=True)


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
