from __future__ import annotations

import torch


def test_load_lora_weights_adapts_the_transformer(tiny_hunyuan3d_pipeline, make_diffusers_lora):
    from peft.utils import get_peft_model_state_dict

    pipeline = tiny_hunyuan3d_pipeline
    state_dict, donors = make_diffusers_lora(pipeline, ["transformer"])
    pipeline.load_lora_weights(state_dict)

    loaded = get_peft_model_state_dict(pipeline.transformer, adapter_name="default_0")
    expected = get_peft_model_state_dict(donors["transformer"])
    assert loaded.keys() == expected.keys() and loaded
    inputs = (torch.randn(1, 6, 8), torch.tensor([0.5]), torch.randn(1, 5, 24))
    with torch.no_grad():
        torch.testing.assert_close(pipeline.transformer(*inputs).sample, donors["transformer"](*inputs).sample)
    assert pipeline.get_list_adapters() == {"transformer": ["default_0"]}
