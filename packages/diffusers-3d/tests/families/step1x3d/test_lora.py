from __future__ import annotations

import copy

import pytest
import torch

import diffusers_3d.execution.lora as lora


def test_missing_peft_raises_install_hint(monkeypatch, tiny_step1x3d_pipeline):
    monkeypatch.setattr(lora, "USE_PEFT_BACKEND", False)

    with pytest.raises(ImportError, match=r"pip install \"peft>="):
        tiny_step1x3d_pipeline.load_lora_weights({})


def test_load_lora_weights_adapts_the_transformer(tiny_step1x3d_pipeline):
    peft = pytest.importorskip("peft")
    from peft.utils import get_peft_model_state_dict

    pipeline = tiny_step1x3d_pipeline
    donor = copy.deepcopy(pipeline.transformer)
    torch.manual_seed(0)
    donor.add_adapter(
        peft.LoraConfig(
            r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights=False
        )
    )
    state_dict = {f"transformer.{key}": value for key, value in get_peft_model_state_dict(donor).items()}
    pipeline.load_lora_weights(state_dict)

    config = pipeline.transformer.config
    torch.manual_seed(1)
    inputs = (
        torch.randn(1, 5, config.in_channels),
        torch.tensor([500.0]),
        torch.randn(1, 3, config.cross_attention_dim),
    )
    with torch.no_grad():
        torch.testing.assert_close(pipeline.transformer(*inputs).sample, donor(*inputs).sample)
    assert pipeline.get_list_adapters() == {"transformer": ["default_0"]}
