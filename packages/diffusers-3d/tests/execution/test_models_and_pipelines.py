from __future__ import annotations

import json

import torch
from diffusers import ConfigMixin, DiffusionPipeline, ModelMixin, ModularPipeline

import diffusers_3d
from diffusers_3d import (
    OBJECT3D_API_VERSION,
    OBJECT3D_MODEL_INDEX_NAME,
    OBJECT3D_SCHEMA_VERSION,
    ModularObject3DPipeline,
    Object3DKind,
    Object3DModel,
    Object3DModelIndex,
    Object3DPipeline,
)


def test_nominal_model_and_pipeline_markers_expose_versioned_contracts(
    tiny_model_class,
    tiny_pipeline_class,
):
    assert issubclass(Object3DModel, ModelMixin)
    assert issubclass(Object3DModel, ConfigMixin)
    assert issubclass(Object3DPipeline, DiffusionPipeline)
    assert issubclass(ModularObject3DPipeline, ModularPipeline)
    assert tiny_model_class.api_version == OBJECT3D_API_VERSION
    assert tiny_model_class.schema_version == OBJECT3D_SCHEMA_VERSION
    assert tiny_model_class.family_id == "tiny-family"
    assert tiny_model_class.component_role == "denoiser"
    assert tiny_model_class.supported_object_kinds == (Object3DKind.MESH,)
    assert tiny_pipeline_class.family_id == "tiny-family"
    assert tiny_pipeline_class.task_ids == ("text-to-3d",)
    assert not hasattr(Object3DPipeline, "generate_3d")
    assert not hasattr(ModularObject3DPipeline, "generate_3d")
    assert diffusers_3d.Object3DModel is Object3DModel
    assert diffusers_3d.Object3DPipeline is Object3DPipeline


def test_tiny_exact_model_config_and_weights_round_trip(tmp_path, tiny_model_class):
    torch.manual_seed(0)
    model = tiny_model_class(hidden_size=3)
    inputs = torch.randn(2, 3)
    expected = model(inputs)

    model.save_pretrained(tmp_path)
    loaded = tiny_model_class.from_pretrained(tmp_path)

    assert type(loaded) is tiny_model_class
    assert loaded.config.hidden_size == 3
    torch.testing.assert_close(loaded(inputs), expected)
    assert loaded.object3d_metadata() == model.object3d_metadata()


def test_componentless_pipeline_preserves_diffusers_save_and_writes_sidecar(
    tmp_path,
    tiny_pipeline_class,
):
    pipeline = tiny_pipeline_class()

    pipeline.save_pretrained(tmp_path)

    assert (tmp_path / DiffusionPipeline.config_name).is_file()
    assert (tmp_path / OBJECT3D_MODEL_INDEX_NAME).is_file()
    metadata = Object3DModelIndex.from_pretrained(tmp_path)
    assert metadata == tiny_pipeline_class.object3d_model_index()
    assert metadata.output_representations == ("triangle-mesh",)
    assert metadata.object_kinds == (Object3DKind.MESH,)
    assert metadata.required_backends == ("trimesh",)

    model_index = json.loads((tmp_path / DiffusionPipeline.config_name).read_text())
    assert model_index["_class_name"] == tiny_pipeline_class.__name__
    loaded = tiny_pipeline_class.from_pretrained(tmp_path)
    assert type(loaded) is tiny_pipeline_class
