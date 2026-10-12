from __future__ import annotations

import pytest
import torch
from transformers import CLIPTextModel

from diffusers_3d import TrellisClipTextConditioner

pytestmark = pytest.mark.integration


def test_clip_text_conditioner_matches_released_encode_text_semantics(tiny_text_conditioner):
    """Upstream: ``tokenizer(text, max_length=77, padding="max_length", truncation=True)`` -> ``last_hidden_state``."""

    conditioner = tiny_text_conditioner
    prompts = ["a red chair", "chair " * 20]
    ids = conditioner.tokenize(prompts)
    assert ids.shape == (2, conditioner.max_length)
    # Padded with the end-of-text token, truncated to max_length with the end-of-text token last.
    assert ids[0, 0] == ids[1, 0] == conditioner.tokenizer.bos_token_id
    assert ids[0, -1] == conditioner.tokenizer.pad_token_id
    assert ids[1, -1] == conditioner.tokenizer.eos_token_id

    with torch.no_grad():
        embeddings = conditioner(prompts).embeddings
        reference = conditioner.model(input_ids=ids).last_hidden_state
        from_ids = conditioner(ids, return_dict=False)[0]
        unconditional = conditioner.unconditional_embedding(2)
        empty = conditioner([""]).embeddings
    assert embeddings.shape == (2, conditioner.max_length, 12)
    torch.testing.assert_close(embeddings, reference)
    torch.testing.assert_close(from_ids, reference)
    torch.testing.assert_close(unconditional, empty.expand(2, -1, -1))
    assert not torch.allclose(unconditional[0], embeddings[0])


def test_clip_text_conditioner_save_load_round_trips_tokenizer_and_from_clip(tmp_path, tiny_text_conditioner):
    conditioner = tiny_text_conditioner
    conditioner.save_pretrained(tmp_path / "pipeline" / "conditioner")
    assert (tmp_path / "pipeline" / "conditioner" / "tokenizer").is_dir()

    loaded = TrellisClipTextConditioner.from_pretrained(tmp_path / "pipeline", subfolder="conditioner")
    direct = TrellisClipTextConditioner.from_pretrained(tmp_path / "pipeline" / "conditioner")
    with torch.no_grad():
        expected = conditioner(["a red chair"]).embeddings
        torch.testing.assert_close(loaded(["a red chair"]).embeddings, expected)
        torch.testing.assert_close(direct(["a red chair"]).embeddings, expected)

    # A raw Transformers CLIP text checkpoint folder (weights + tokenizer) is accepted by from_clip_pretrained.
    conditioner.model.save_pretrained(tmp_path / "clip")
    conditioner.tokenizer.save_pretrained(tmp_path / "clip")
    from_clip = TrellisClipTextConditioner.from_clip_pretrained(str(tmp_path / "clip"), max_length=8)
    assert type(from_clip.model) is CLIPTextModel
    assert not any(parameter.requires_grad for parameter in from_clip.parameters())
    with torch.no_grad():
        torch.testing.assert_close(from_clip(["a red chair"]).embeddings, expected)

    # Weights alone still load; tokenization then fails loudly instead of guessing a vocabulary.
    bare = TrellisClipTextConditioner(**TrellisClipTextConditioner.tiny_config())
    bare.save_pretrained(tmp_path / "bare")
    reloaded = TrellisClipTextConditioner.from_pretrained(tmp_path / "bare")
    assert reloaded.tokenizer is None
    with pytest.raises(RuntimeError, match="no tokenizer is attached"):
        reloaded.tokenize(["a"])


def test_clip_text_conditioner_rejects_bad_inputs(tiny_text_conditioner):
    with pytest.raises(TypeError, match="sequence of strings"):
        tiny_text_conditioner.tokenize("a red chair")
    with pytest.raises(ValueError, match="input ids must have shape"):
        tiny_text_conditioner(torch.zeros(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="max_length must not exceed"):
        TrellisClipTextConditioner(**{**TrellisClipTextConditioner.tiny_config(), "max_length": 9})
    with pytest.raises(ValueError, match="batch_size"):
        tiny_text_conditioner.unconditional_embedding(0)
