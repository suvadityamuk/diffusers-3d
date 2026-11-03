from __future__ import annotations

import json

import pytest
import torch
import yaml
from _configs import tiny_config
from safetensors.torch import load_file
from transformers import ViTConfig, ViTModel

from diffusers_3d.families.triposr.conversion import interpolate_position_embeddings, main
from diffusers_3d.families.triposr.models import TripoSRTransformerModel, TripoSRTriplaneDecoder
from diffusers_3d.families.triposr.pipeline import TripoSRImageTo3DPipeline

from .configs import upstream_config, upstream_vit_config


@pytest.fixture
def released_folder(tmp_path):
    """The stabilityai/TripoSR layout: ``config.yaml`` and a ``model.ckpt`` with legacy Transformers ViT names."""

    torch.manual_seed(0)
    vit = ViTModel(ViTConfig(**upstream_vit_config()))
    vit.save_pretrained(tmp_path / "vit")
    transformer = TripoSRTransformerModel(**tiny_config(TripoSRTransformerModel))
    decoder = TripoSRTriplaneDecoder(**tiny_config(TripoSRTriplaneDecoder))
    checkpoint = {
        f"image_tokenizer.model.{key}": value for key, value in load_file(tmp_path / "vit/model.safetensors").items()
    }
    for key, value in transformer.state_dict().items():
        prefix = "tokenizer." if key == "embeddings" else "post_processor." if key.startswith("upsample.") else ""
        checkpoint[prefix + key] = value
    checkpoint.update({f"decoder.{key}": value for key, value in decoder.state_dict().items()})
    root = tmp_path / "TripoSR"
    root.mkdir()
    torch.save(checkpoint, root / "model.ckpt")
    config = upstream_config(str(tmp_path / "vit"))
    config["backbone"]["in_channels"] = "${tokenizer.num_channels}"
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return root, {"image_encoder": vit, "transformer": transformer, "decoder": decoder}


def test_conversion_writes_a_loadable_pipeline_with_unchanged_weights(released_folder, tmp_path):
    root, expected = released_folder
    output = tmp_path / "converted"
    assert main([str(root), str(output)]) == 0
    pipeline = TripoSRImageTo3DPipeline.from_pretrained(output, local_files_only=True)

    assert json.loads((output / "model_index.json").read_text())["image_encoder"] == ["transformers", "ViTModel"]
    assert type(pipeline.image_encoder) is ViTModel
    assert pipeline.image_encoder.config.image_size == 32
    for name in ("transformer", "decoder", "image_encoder"):
        reference = expected[name].state_dict()
        actual = getattr(pipeline, name).state_dict()
        assert actual.keys() == reference.keys()
        if name == "image_encoder":
            position = "embeddings.position_embeddings"
            assert actual[position].shape == (1, 17, 12)
            assert torch.equal(actual.pop(position), interpolate_position_embeddings(reference.pop(position), 4))
        assert all(torch.equal(actual[key], reference[key]) for key in reference)


def test_position_embeddings_are_unchanged_at_the_native_grid():
    embeddings = torch.randn(1, 5, 12)
    assert interpolate_position_embeddings(embeddings, 2) is embeddings


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("renderer", "density_activation", "trunc_exp"), ("renderer", "feature_reduction", "mean")],
)
def test_conversion_rejects_unported_renderer_settings(released_folder, tmp_path, section, key, value):
    root, _ = released_folder
    config = yaml.safe_load((root / "config.yaml").read_text())
    config[section][key] = value
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        main([str(root), str(tmp_path / "converted")])
