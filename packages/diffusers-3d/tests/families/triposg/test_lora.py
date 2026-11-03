from __future__ import annotations

import copy

import pytest
import torch


def test_load_lora_weights_adapts_the_transformer(tiny_triposg_pipeline):
    peft = pytest.importorskip("peft")
    from peft.utils import get_peft_model_state_dict

    pipeline = tiny_triposg_pipeline
    donor = copy.deepcopy(pipeline.transformer)
    torch.manual_seed(0)
    donor.add_adapter(
        peft.LoraConfig(
            r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"], init_lora_weights=False
        )
    )
    state_dict = {f"transformer.{key}": value for key, value in get_peft_model_state_dict(donor).items()}
    pipeline.load_lora_weights(state_dict)

    generator = torch.Generator().manual_seed(1)
    inputs = (
        torch.randn(1, 5, 4, generator=generator),
        torch.tensor([500.0]),
        torch.randn(1, 3, 12, generator=generator),
    )
    with torch.no_grad():
        torch.testing.assert_close(pipeline.transformer(*inputs).sample, donor(*inputs).sample)
    assert pipeline.get_list_adapters() == {"transformer": ["default_0"]}
