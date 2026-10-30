import json

import httpx
import pytest
import torch
from accelerate import init_empty_weights
from diffusers import ModelMixin  # noqa: F401 - Diffusers looks component base classes up in their module
from diffusers.configuration_utils import register_to_config
from huggingface_hub import constants
from huggingface_hub.errors import GatedRepoError

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d import Object3DModel
from diffusers_3d.execution.pipelines import ExternalComponent, Object3DPipeline


class _Encoder(Object3DModel):
    @register_to_config
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))


class _Pipeline(Object3DPipeline):
    external_components = {}

    def __init__(self, image_encoder: _Encoder, head: _Encoder) -> None:
        super().__init__()
        self.register_modules(image_encoder=image_encoder, head=head)


def test_external_components_are_never_saved_and_load_from_their_repository(tmp_path, monkeypatch):
    calls = []

    def load(repository, *, revision, dtype, **hub_kwargs):
        calls.append((repository, revision, dtype, hub_kwargs.get("local_files_only")))
        return _Encoder(width=3)

    monkeypatch.setattr(
        _Pipeline, "external_components", {"image_encoder": ExternalComponent("org/encoder", "abc123", load)}
    )
    _Pipeline(image_encoder=_Encoder(width=3), head=_Encoder()).save_pretrained(tmp_path)
    assert not (tmp_path / "image_encoder").exists() and (tmp_path / "head").is_dir()
    assert json.loads((tmp_path / "model_index.json").read_text())["image_encoder"][1] == "_Encoder"

    loaded = _Pipeline.from_pretrained(tmp_path, local_files_only=True, dtype={"image_encoder": torch.float16})
    assert calls == [("org/encoder", "abc123", torch.float16, True)]
    assert loaded.image_encoder.config.width == 3
    passed = _Encoder(width=3)
    assert _Pipeline.from_pretrained(tmp_path, image_encoder=passed).image_encoder is passed and len(calls) == 1


def test_converters_save_empty_weight_shells_without_downloading_them(tmp_path, monkeypatch):
    def load(repository, *, revision, dtype, **hub_kwargs):
        return _Encoder(width=3)

    monkeypatch.setattr(
        _Pipeline, "external_components", {"image_encoder": ExternalComponent("org/encoder", "abc123", load)}
    )
    with init_empty_weights():
        shell = _Encoder(width=3)
    assert shell.weight.is_meta
    _Pipeline(image_encoder=shell, head=_Encoder()).save_pretrained(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["head", "model_index.json"]
    assert _Pipeline.from_pretrained(tmp_path, local_files_only=True).image_encoder.config.width == 3


class _TokenizerPipeline(Object3DPipeline):
    external_components = {}

    def __init__(self, tokenizer, head: _Encoder) -> None:
        super().__init__()
        self.register_modules(tokenizer=tokenizer, head=head)


def test_external_tokenizers_are_skipped_on_save(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    monkeypatch.setattr(
        _TokenizerPipeline,
        "external_components",
        {"tokenizer": ExternalComponent("org/tokenizer", "abc123", lambda *args, **kwargs: tokenizer)},
    )
    _TokenizerPipeline(tokenizer=tokenizer, head=_Encoder()).save_pretrained(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["head", "model_index.json"]
    assert json.loads((tmp_path / "model_index.json").read_text())["tokenizer"] == [
        "transformers",
        type(tokenizer).__name__,
    ]


def test_gated_external_components_explain_how_to_get_access(monkeypatch):
    def deny(repository, token=None):
        request = httpx.Request("GET", f"https://huggingface.co/{repository}")
        raise GatedRepoError("gated", response=httpx.Response(403, request=request))

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(pipelines.huggingface_hub, "auth_check", deny)
    component = ExternalComponent("org/gated", "abc123", lambda *args, **kwargs: None)
    with pytest.raises(PermissionError, match="Accept its license at https://huggingface.co/org/gated") as error:
        component("image_encoder", "SomePipeline", None, {})
    assert "hf auth login" in str(error.value) and "pass image_encoder=..." in str(error.value)
