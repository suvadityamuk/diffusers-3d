from __future__ import annotations

import json

import pytest
from transformers import CLIPTokenizer


@pytest.fixture(scope="session")
def tiny_clip_tokenizer(tmp_path_factory) -> CLIPTokenizer:
    """A real byte-level CLIP BPE tokenizer over a-z with a few merges, small enough for the tiny text encoder."""

    root = tmp_path_factory.mktemp("tiny-clip-tokenizer")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for code in range(ord("a"), ord("z") + 1):
        vocab[chr(code)] = len(vocab)
        vocab[chr(code) + "</w>"] = len(vocab)
    for token in ("ch", "re", "red</w>", "ai", "air</w>", "chair</w>"):
        vocab[token] = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\nc h\nr e\nre d</w>\na i\nai r</w>\nch air</w>\n")
    return CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
