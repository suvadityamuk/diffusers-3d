import json

import pytest
import torch
from diffusers.configuration_utils import register_to_config

from diffusers_3d import (
    AutoPipelineFor3D,
    AutoPipelineForImageTo3D,
    AutoPipelineForMeshTexturing,
    AutoPipelineForTextTo3D,
    Hunyuan3DPaintPipeline,
    Object3DModel,
    TrellisTextTo3DPipeline,
)


class _TinyModel(Object3DModel):
    @register_to_config
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, hidden_size)


class _WiderModel(_TinyModel):
    @register_to_config
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__(hidden_size)
        self.extra = torch.nn.Linear(hidden_size, hidden_size)


def test_model_round_trips_and_raises_on_missing_weights(tmp_path):
    model = _TinyModel()
    model.save_pretrained(tmp_path)
    loaded = _TinyModel.from_pretrained(tmp_path)
    assert torch.equal(loaded.projection.weight, model.projection.weight)
    with pytest.raises(ValueError, match="has no weights for _WiderModel parameters"):
        _WiderModel.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "auto_class, class_name, expected",
    [
        (AutoPipelineFor3D, "TrellisTextTo3DPipeline", TrellisTextTo3DPipeline),
        (AutoPipelineForTextTo3D, "TrellisTextTo3DPipeline", TrellisTextTo3DPipeline),
        (AutoPipelineForMeshTexturing, "Hunyuan3DPaintPipeline", Hunyuan3DPaintPipeline),
    ],
)
def test_auto_pipeline_resolves_the_class_named_in_model_index(tmp_path, auto_class, class_name, expected):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    assert auto_class.pipeline_class(tmp_path) is expected


@pytest.mark.parametrize("class_name", ["TrellisTextTo3DPipeline", "StableDiffusionPipeline"])
def test_task_auto_pipeline_rejects_other_pipelines(tmp_path, class_name):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    with pytest.raises(ValueError, match="AutoPipelineForImageTo3D cannot load"):
        AutoPipelineForImageTo3D.pipeline_class(tmp_path)
