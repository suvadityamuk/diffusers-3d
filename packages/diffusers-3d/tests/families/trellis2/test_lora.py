from __future__ import annotations

import pytest
import torch

import diffusers_3d.execution.lora as lora


def test_missing_peft_raises_install_hint(monkeypatch, tiny_trellis2_pipeline):
    monkeypatch.setattr(lora, "USE_PEFT_BACKEND", False)

    with pytest.raises(ImportError, match=r"pip install \"peft>="):
        tiny_trellis2_pipeline.load_lora_weights({})


def test_load_lora_weights_adapts_every_slat_flow_model(tiny_trellis2_full_pipeline, make_diffusers_lora):
    from peft.utils import get_peft_model_state_dict

    pipeline = tiny_trellis2_full_pipeline
    components = list(pipeline._lora_loadable_modules)
    state_dict, donors = make_diffusers_lora(pipeline, components)
    pipeline.load_lora_weights(state_dict)

    for name in components:
        loaded = get_peft_model_state_dict(getattr(pipeline, name), adapter_name="default_0")
        expected = get_peft_model_state_dict(donors[name])
        assert loaded.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(loaded[key], expected[key])

    flow = pipeline.sparse_structure_flow_model
    config = flow.config
    torch.manual_seed(1)
    inputs = (
        torch.randn(1, config.in_channels, *([config.resolution] * 3)),
        torch.tensor([500.0]),
        torch.randn(1, 4, config.cond_channels),
    )
    with torch.no_grad():
        torch.testing.assert_close(flow(*inputs)[0], donors["sparse_structure_flow_model"](*inputs)[0])
    assert pipeline.get_list_adapters() == {name: ["default_0"] for name in components}
