"""TripoSplat's gated DINOv3 and Flux.2 VAE load from their original repositories, equal to the release's copies.

Needs Hub access to facebook/dinov3-vith16plus-pretrain-lvd1689m and black-forest-labs/FLUX.2-dev (about 5 GB).
"""

from __future__ import annotations

import os

import pytest
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusers_3d.families.triposplat import TripoSplatImageTo3DPipeline, TripoSplatImageVAE
from diffusers_3d.families.triposplat.conversion import (
    DINOV3_VITH16_CONFIG,
    TRIPOSPLAT_WEIGHTS_REPOSITORY,
    TRIPOSPLAT_WEIGHTS_REVISION,
    convert_dinov3_state_dict,
    convert_triposplat_vae_state_dict,
)
from diffusers_3d.families.triposplat.pipeline import UPSTREAM_DTYPES

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.skipif(
        os.environ.get("DIFFUSERS_3D_GATED_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_GATED_REAL_WEIGHTS=1"
    ),
]


def _released(path: str) -> dict[str, torch.Tensor]:
    return load_file(hf_hub_download(TRIPOSPLAT_WEIGHTS_REPOSITORY, path, revision=TRIPOSPLAT_WEIGHTS_REVISION))


def _external(name: str):
    component = TripoSplatImageTo3DPipeline.external_components[name]
    return component(name, "TripoSplatImageTo3DPipeline", UPSTREAM_DTYPES[name], {})


@torch.no_grad()
def test_dinov3_from_its_repository_is_the_released_copy(tmp_path):
    """The release's ``clip_vision`` DINOv3 is the facebook checkpoint cast to bfloat16, which loading reproduces."""

    ours = _external("image_encoder")
    # The same attention kernel as the loaded model, so the forward passes compare like for like.
    config = DINOv3ViTConfig(**DINOV3_VITH16_CONFIG, attn_implementation=ours.config._attn_implementation)
    with init_empty_weights():
        released = DINOv3ViTModel(config)
    released.load_state_dict(
        convert_dinov3_state_dict(_released("clip_vision/dino_v3_vit_h.safetensors")), strict=True, assign=True
    )
    # What the pipeline embedded before: the release's copy saved and loaded back in bfloat16, which keeps the
    # RoPE frequencies (a non-persistent buffer) in float32.
    released.save_pretrained(tmp_path)
    released = DINOv3ViTModel.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    assert ours.embeddings is not None and ours.rope_embeddings.inv_freq.dtype is torch.float32
    expected = released.state_dict()
    actual = ours.state_dict()
    assert ours.dtype is torch.bfloat16 and actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    pixels = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    assert torch.equal(ours(pixel_values=pixels).last_hidden_state, released(pixel_values=pixels).last_hidden_state)


@torch.no_grad()
def test_flux2_vae_from_its_repository_is_the_released_copy():
    """The release's ``vae/flux2-vae.safetensors`` is FLUX.2-dev's VAE; TripoSplat runs its encoder in bfloat16."""

    ours = _external("vae")
    with init_empty_weights():
        released = TripoSplatImageVAE()
    state_dict = convert_triposplat_vae_state_dict(_released("vae/flux2-vae.safetensors"))
    released.load_state_dict(state_dict, strict=True, assign=True)
    released = released.to(torch.bfloat16).eval()

    def config(model):
        return {key: list(value) if isinstance(value, tuple) else value for key, value in model.config.items()}

    assert {key: value for key, value in config(ours).items() if not key.startswith("_")} == {
        key: value for key, value in config(released).items() if not key.startswith("_")
    }
    assert ours.bn.eps == 1e-5 and not ours.config.clamp_logvar
    expected, actual = released.state_dict(), ours.state_dict()
    assert ours.dtype is torch.bfloat16 and actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    images = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16) * 2 - 1
    assert torch.equal(
        ours(images, generator=torch.Generator().manual_seed(1)),
        released(images, generator=torch.Generator().manual_seed(1)),
    )
