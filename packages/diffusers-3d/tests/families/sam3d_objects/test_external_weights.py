"""SAM 3D Objects' frozen DINOv2 and MoGe load from Meta's and Ruicheng's repositories.

The release's four DINOv2 copies equal ``facebook/dinov2-with-registers-large`` tensor for tensor; this checks it
against the earlier published conversion of those copies (1.2 GB each) and MoGe (1.3 GB).
"""

from __future__ import annotations

import os

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers_3d import Sam3DObjectsPipeline
from diffusers_3d.families.moge.conversion import MOGE_WEIGHTS_REPOSITORY, convert_moge_checkpoint

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.skipif(
        os.environ.get("DIFFUSERS_3D_GATED_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_GATED_REAL_WEIGHTS=1"
    ),
]
# The conversion published before DINOv2 loaded from its repository: ss_generator's DINOv2 copy, converted.
_PUBLISHED = ("suvadityamuk/sam-3d-objects-diffusers-3d", "4210613b0eda348675ae0d50235ff017bb8cef38")


def _external(name: str):
    return Sam3DObjectsPipeline.external_components[name](name, "Sam3DObjectsPipeline", None, {})


@torch.no_grad()
def test_dinov2_from_its_repository_equals_the_released_copies():
    ours = _external("image_encoder")
    published = load_file(hf_hub_download(*_PUBLISHED[:1], "image_encoder/model.safetensors", revision=_PUBLISHED[1]))
    actual = ours.state_dict()
    # Transformers renames the attention projections on load; compare the tensors themselves.
    assert ours.dtype is torch.float32 and len(actual) == len(published) == 440
    assert sorted(tensor.sum().item() for tensor in actual.values()) == sorted(
        tensor.sum().item() for tensor in published.values()
    )
    reloaded = type(ours).from_pretrained(None, config=ours.config, state_dict=published).state_dict()
    assert all(torch.equal(actual[k], reloaded[k]) for k in actual)


@torch.no_grad()
def test_moge_converts_at_load_like_the_converter_did():
    ours = _external("depth_estimator")
    expected = convert_moge_checkpoint(MOGE_WEIGHTS_REPOSITORY).state_dict()
    actual = ours.state_dict()
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)
