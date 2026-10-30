from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

import diffusers_3d.execution.lora as lora


def _sparse_structure_inputs(model):
    config = model.config
    torch.manual_seed(1)
    hidden_states = torch.randn(1, config.in_channels, *([config.resolution] * 3))
    return hidden_states, torch.tensor([500.0]), torch.randn(1, 4, config.cond_channels)


def test_missing_peft_raises_install_hint(monkeypatch, tiny_trellis_pipeline):
    monkeypatch.setattr(lora, "USE_PEFT_BACKEND", False)

    with pytest.raises(ImportError, match=r"pip install \"peft>="):
        tiny_trellis_pipeline.load_lora_weights({})


def test_load_lora_weights_adapts_both_flow_models_and_unloads(
    tmp_path, tiny_trellis_full_pipeline, make_diffusers_lora
):
    from peft.utils import get_peft_model_state_dict

    pipeline = tiny_trellis_full_pipeline
    flow = pipeline.sparse_structure_flow_model
    with torch.no_grad():
        for parameter in flow.out_layer.parameters():
            parameter.normal_(0, 0.1)  # tiny init zeroes the output layer, which would hide the adapter
    inputs = _sparse_structure_inputs(flow)
    with torch.no_grad():
        base = flow(*inputs)[0]

    components = ["sparse_structure_flow_model", "slat_flow_model"]
    state_dict, donors = make_diffusers_lora(pipeline, components)
    save_file(state_dict, tmp_path / "pytorch_lora_weights.safetensors")
    pipeline.load_lora_weights(tmp_path, adapter_name="style")

    for name in components:
        loaded = get_peft_model_state_dict(getattr(pipeline, name), adapter_name="style")
        expected = get_peft_model_state_dict(donors[name])
        assert loaded.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(loaded[key], expected[key])
    with torch.no_grad():
        adapted = flow(*inputs)[0]
        torch.testing.assert_close(adapted, donors["sparse_structure_flow_model"](*inputs)[0])
    assert not torch.allclose(adapted, base)

    pipeline.unload_lora_weights()
    with torch.no_grad():
        torch.testing.assert_close(flow(*inputs)[0], base)


def test_rejects_non_diffusers_keys_and_unloaded_components(tiny_trellis_pipeline, make_diffusers_lora):
    pytest.importorskip("peft")
    state_dict, _ = make_diffusers_lora(tiny_trellis_pipeline, ["sparse_structure_flow_model"])

    renamed = {
        key.replace("sparse_structure_flow_model.", "transformer.", 1): value for key, value in state_dict.items()
    }
    with pytest.raises(ValueError, match="Only Diffusers-format LoRA checkpoints"):
        tiny_trellis_pipeline.load_lora_weights(renamed)

    slat_only = {
        key.replace("sparse_structure_flow_model.", "slat_flow_model.", 1): value for key, value in state_dict.items()
    }
    with pytest.raises(ValueError, match="slat_flow_model"):
        tiny_trellis_pipeline.load_lora_weights(slat_only)

    with pytest.raises(ValueError, match="every key must be a LoRA weight"):
        tiny_trellis_pipeline.load_lora_weights({"sparse_structure_flow_model.out_layer.weight": torch.zeros(1)})


def _perturbed_flow(pipeline):
    flow = pipeline.sparse_structure_flow_model
    with torch.no_grad():
        for parameter in flow.out_layer.parameters():
            parameter.normal_(0, 0.1)
    return flow


def test_safetensors_metadata_carries_lora_alpha(tmp_path, tiny_trellis_full_pipeline, make_diffusers_lora):
    pipeline = tiny_trellis_full_pipeline
    flow = _perturbed_flow(pipeline)
    state_dict, donors = make_diffusers_lora(pipeline, ["sparse_structure_flow_model"], rank=2, alpha=4)
    config = donors["sparse_structure_flow_model"].peft_config["default"].to_dict()
    metadata = {f"sparse_structure_flow_model.{key}": value for key, value in config.items()}
    save_file(
        state_dict,
        tmp_path / "pytorch_lora_weights.safetensors",
        metadata={"lora_adapter_metadata": json.dumps(metadata, default=sorted)},
    )

    pipeline.load_lora_weights(tmp_path)

    inputs = _sparse_structure_inputs(flow)
    with torch.no_grad():
        torch.testing.assert_close(flow(*inputs)[0], donors["sparse_structure_flow_model"](*inputs)[0])


def test_adapter_controls_work_when_only_some_flow_models_are_adapted(tiny_trellis_full_pipeline, make_diffusers_lora):
    pipeline = tiny_trellis_full_pipeline
    flow = _perturbed_flow(pipeline)
    inputs = _sparse_structure_inputs(flow)
    with torch.no_grad():
        base = flow(*inputs)[0]
    state_dict, donors = make_diffusers_lora(pipeline, ["sparse_structure_flow_model"])
    pipeline.load_lora_weights(state_dict, adapter_name="style")
    with torch.no_grad():
        adapted = donors["sparse_structure_flow_model"](*inputs)[0]

        pipeline.set_adapters("style", adapter_weights=0.0)
        torch.testing.assert_close(flow(*inputs)[0], base)
        pipeline.set_adapters("style", adapter_weights=1.0)
        torch.testing.assert_close(flow(*inputs)[0], adapted)

        pipeline.fuse_lora()
        torch.testing.assert_close(flow(*inputs)[0], adapted)
        pipeline.unfuse_lora()
        torch.testing.assert_close(flow(*inputs)[0], adapted)
        pipeline.disable_lora()
        torch.testing.assert_close(flow(*inputs)[0], base)
