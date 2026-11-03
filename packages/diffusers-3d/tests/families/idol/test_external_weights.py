"""IDOL loads the gated DINOv3 ViT-H/16+ and FLUX.2-dev VAE from their repositories, as its converter used to embed them.

Needs Hub access to facebook/dinov3-vith16plus-pretrain-lvd1689m and black-forest-labs/FLUX.2-dev (about 4 GB).
"""

from __future__ import annotations

import os

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import DINOv3ViTModel

from diffusers_3d.families.idol import IdolImageTo3DPipeline
from diffusers_3d.families.idol.pipeline import UPSTREAM_DTYPES
from diffusers_3d.families.trellis2.models import DINOV3_REVISIONS, DINOV3_VITH16PLUS_REPO
from diffusers_3d.families.triposplat.models import FLUX2_REPO, FLUX2_REVISION, convert_triposplat_vae_state_dict

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.skipif(
        os.environ.get("DIFFUSERS_3D_GATED_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_GATED_REAL_WEIGHTS=1"
    ),
]


def _external(name: str):
    component = IdolImageTo3DPipeline.external_components[name]
    return component(name, "IdolImageTo3DPipeline", UPSTREAM_DTYPES[name], {})


@torch.no_grad()
def test_dinov3_from_its_repository_is_what_the_converter_embedded(tmp_path):
    revision = DINOV3_REVISIONS[DINOV3_VITH16PLUS_REPO]
    DINOv3ViTModel.from_pretrained(DINOV3_VITH16PLUS_REPO, revision=revision).save_pretrained(tmp_path)
    embedded = DINOv3ViTModel.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    ours = _external("image_encoder")
    actual, expected = ours.state_dict(), embedded.state_dict()
    assert ours.dtype is torch.bfloat16
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)
    pixels = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    assert torch.equal(ours(pixel_values=pixels).last_hidden_state, embedded(pixel_values=pixels).last_hidden_state)


@torch.no_grad()
def test_flux2_vae_from_its_repository_keeps_idols_settings():
    ours = _external("vae")
    weights = hf_hub_download(FLUX2_REPO, "vae/diffusion_pytorch_model.safetensors", revision=FLUX2_REVISION)
    expected = {k: v.to(torch.bfloat16) for k, v in convert_triposplat_vae_state_dict(load_file(weights)).items()}
    actual = ours.state_dict()
    # AutoencoderKLFlux2's batch-norm epsilon from the VAE config and its clamped posterior log-variance.
    assert ours.dtype is torch.bfloat16 and ours.bn.eps == 1e-4 and ours.config.clamp_logvar
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)
