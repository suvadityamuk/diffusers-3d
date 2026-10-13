from __future__ import annotations

import json

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
from transformers import BitImageProcessorPil, Dinov2Model

from diffusers_3d import TripoSGImageTo3DPipeline, TripoSGTransformerModel, TripoSGVAEDecoder
from diffusers_3d.families.triposg.conversion import main


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def released_snapshot(tmp_path, tiny_triposg_components):
    """The VAST-AI/TripoSG folder layout: upstream class names, a full VAE with encoder, and RectifiedFlowScheduler."""

    root = tmp_path / "TripoSG"
    _write_json(root / "model_index.json", {"_class_name": "TripoSGPipeline", "_diffusers_version": "0.30.3"})
    transformer = tiny_triposg_components["transformer"]
    _write_json(
        root / "transformer" / "config.json",
        {"_class_name": "TripoSGDiTModel", **TripoSGTransformerModel.tiny_config()},
    )
    save_file(transformer.state_dict(), root / "transformer" / "diffusion_pytorch_model.safetensors")
    vae = tiny_triposg_components["vae"]
    _write_json(
        root / "vae" / "config.json",
        {
            "_class_name": "Tripo2VAEModel",
            "embedding_type": "frequency",
            "width_encoder": 8,
            "num_layers_encoder": 1,
            **TripoSGVAEDecoder.tiny_config(),
        },
    )
    vae_state = {
        key if key.startswith("post_quant.") else f"decoder.{key}": value for key, value in vae.state_dict().items()
    }
    vae_state.update({"encoder.proj_in.weight": torch.zeros(8, 18), "quant.weight": torch.zeros(8, 8)})
    save_file(vae_state, root / "vae" / "diffusion_pytorch_model.safetensors")
    _write_json(
        root / "scheduler" / "scheduler_config.json",
        {
            "_class_name": "RectifiedFlowScheduler",
            "num_train_timesteps": 1000,
            "shift": 1,
            "use_dynamic_shifting": False,
        },
    )
    tiny_triposg_components["image_encoder"].save_pretrained(root / "image_encoder_dinov2")
    # The released preprocessor config, with the tiny encoder's sizes.
    _write_json(
        root / "feature_extractor_dinov2" / "preprocessor_config.json",
        {
            "crop_size": {"height": 28, "width": 28},
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_processor_type": "BitImageProcessor",
            "image_std": [0.229, 0.224, 0.225],
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {"shortest_edge": 32},
        },
    )
    return root


def test_conversion_writes_a_loadable_pipeline_with_unchanged_weights(
    released_snapshot, tmp_path, tiny_triposg_components
):
    output = tmp_path / "converted"
    assert main([str(released_snapshot), str(output)]) == 0
    pipeline = TripoSGImageTo3DPipeline.from_pretrained(output, local_files_only=True)

    assert (output / "object3d_model_index.json").is_file()
    assert json.loads((output / "model_index.json").read_text())["image_encoder"] == ["transformers", "Dinov2Model"]
    assert type(pipeline.scheduler) is FlowMatchEulerDiscreteScheduler
    assert type(pipeline.image_encoder) is Dinov2Model
    assert type(pipeline.feature_extractor) is BitImageProcessorPil
    assert pipeline.feature_extractor.crop_size == {"height": 28, "width": 28}
    assert pipeline.feature_extractor.size == {"shortest_edge": 32}
    for name in ("transformer", "vae", "image_encoder"):
        expected = tiny_triposg_components[name].state_dict()
        actual = getattr(pipeline, name).state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_conversion_rejects_the_scribble_variant(released_snapshot, tmp_path):
    config_path = released_snapshot / "transformer" / "config.json"
    config = json.loads(config_path.read_text())
    config.update(use_cross_attention_2=True, cross_attention_2_dim=8)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scribble"):
        main([str(released_snapshot), str(tmp_path / "converted")])
