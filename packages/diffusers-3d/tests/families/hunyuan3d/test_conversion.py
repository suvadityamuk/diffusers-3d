from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file

from diffusers_3d import AutoPipelineForImageTo3D, Hunyuan3DImageToShapePipeline
from diffusers_3d.families.hunyuan3d.conversion import convert_hunyuan3d_checkpoint

pytestmark = pytest.mark.integration


def _aggregate_state(conditioner, denoiser, vae):
    state = {f"model.{key}": value for key, value in denoiser.state_dict().items()}
    state.update({f"vae.{key}": value for key, value in vae.state_dict().items()})
    state["vae.encoder.unsupported_test_weight"] = vae.post_kl.weight.new_zeros(1)
    state.update(
        {
            f"conditioner.main_image_encoder.model.{key.removeprefix('model.')}": value
            for key, value in conditioner.state_dict().items()
        }
    )
    return state


def _aggregate_config(conditioner, denoiser, vae):
    return {
        "model": {"params": dict(denoiser.config)},
        "vae": {"params": dict(vae.config)},
        "conditioner": {
            "params": {
                "main_image_encoder": {
                    "kwargs": {
                        "config": conditioner.model.config.to_dict(),
                        "image_size": conditioner.image_size,
                        "use_cls_token": conditioner.use_cls_token,
                    }
                }
            }
        },
        "scheduler": {"params": {"num_train_timesteps": 1000}},
        "image_processor": {"params": {"size": 8, "border_ratio": 0.0}},
    }


def test_synthetic_aggregate_conversion(tmp_path, tiny_hunyuan_components):
    conditioner, denoiser, vae, _ = tiny_hunyuan_components()
    checkpoint = tmp_path / "aggregate.safetensors"
    save_file(_aggregate_state(conditioner, denoiser, vae), checkpoint)
    output = convert_hunyuan3d_checkpoint(
        checkpoint,
        tmp_path / "converted",
        config=_aggregate_config(conditioner, denoiser, vae),
    )

    assert (output / "denoiser" / "config.json").is_file()
    assert (output / "vae" / "config.json").is_file()
    assert (output / "conditioner" / "config.json").is_file()
    assert (output / "scheduler" / "scheduler_config.json").is_file()
    report = json.loads((output / "hunyuan3d_conversion.json").read_text())
    assert report["decode_only_vae"] is True
    assert report["unsupported_vae_keys"] == ["encoder.unsupported_test_weight"]
    loaded = AutoPipelineForImageTo3D.from_pretrained(output, local_files_only=True)
    assert type(loaded) is Hunyuan3DImageToShapePipeline


def test_converter_strictly_rejects_missing_model_key(tmp_path, tiny_hunyuan_components):
    conditioner, denoiser, vae, _ = tiny_hunyuan_components()
    state = _aggregate_state(conditioner, denoiser, vae)
    del state["model.x_embedder.weight"]
    checkpoint = tmp_path / "missing.safetensors"
    save_file(state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        convert_hunyuan3d_checkpoint(
            checkpoint,
            tmp_path / "converted",
            config=_aggregate_config(conditioner, denoiser, vae),
        )
