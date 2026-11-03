"""Pixal3D's third-party NAF upsampler and MoGe-2 load from their original sources, equal to what was converted before.

Downloads valeo.ai's NAF release (2.6 MB), Ruicheng/moge-2-vitl (1.3 GB), and the earlier published conversion's
``feature_upsampler`` folder.
"""

from __future__ import annotations

import os

import pytest
import torch
from huggingface_hub import snapshot_download

from diffusers_3d import Pixal3DImageTo3DPipeline
from diffusers_3d.families.moge.conversion import MOGE2_WEIGHTS_REPOSITORY, convert_moge_checkpoint
from diffusers_3d.families.pixal3d import Pixal3DNAFUpsampler

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.skipif(
        os.environ.get("DIFFUSERS_3D_GATED_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_GATED_REAL_WEIGHTS=1"
    ),
]
# The Pixal3D conversion published before NAF loaded from its release (converted from naf_release.pth).
_PUBLISHED = ("suvadityamuk/Pixal3D-diffusers-3d", "54b95e908afc92f8e4e28e804781af35558df39f")


def _external(name: str):
    return Pixal3DImageTo3DPipeline.external_components[name](name, "Pixal3DImageTo3DPipeline", None, {})


@torch.no_grad()
def test_naf_from_its_release_equals_the_earlier_conversion(tmp_path):
    folder = snapshot_download(_PUBLISHED[0], revision=_PUBLISHED[1], allow_patterns=["feature_upsampler/*"])
    published = Pixal3DNAFUpsampler.from_pretrained(folder, subfolder="feature_upsampler")
    ours = _external("feature_upsampler")
    actual, expected = ours.state_dict(), published.state_dict()
    public = {key: value for key, value in ours.config.items() if not key.startswith("_")}
    assert ours.dtype is torch.float32 and public == {
        k: v for k, v in published.config.items() if not k.startswith("_")
    }
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)


@torch.no_grad()
def test_moge2_converts_at_load_like_the_converter_did():
    ours = _external("depth_estimator")
    expected = convert_moge_checkpoint(MOGE2_WEIGHTS_REPOSITORY).state_dict()
    actual = ours.state_dict()
    assert ours.dtype is torch.float32
    assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)


_DECODERS = {
    "sparse_structure_decoder": ("ckpts/ss_dec_conv3d_16l8_fp16", "microsoft/TRELLIS-image-large"),
    "shape_slat_decoder": ("ckpts/shape_dec_next_dc_f16c32_fp16", "microsoft/TRELLIS.2-4B"),
    "pbr_decoder": ("ckpts/tex_dec_next_dc_f16c32_fp16", "microsoft/TRELLIS.2-4B"),
}


@pytest.mark.parametrize("name", list(_DECODERS))
@torch.no_grad()
def test_decoders_are_trellis_files_and_load_like_the_earlier_conversion(name):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from diffusers_3d.families.pixal3d.conversion import PIXAL3D_WEIGHTS_REVISION

    stem, repository = _DECODERS[name]
    component = Pixal3DImageTo3DPipeline.external_components[name]
    assert component.repository == repository
    # The Pixal3D release's file is the TRELLIS release's, tensor for tensor.
    released = load_file(
        hf_hub_download("TencentARC/Pixal3D", f"{stem}.safetensors", revision=PIXAL3D_WEIGHTS_REVISION)
    )
    original = load_file(hf_hub_download(repository, f"{stem}.safetensors", revision=component.revision))
    assert released.keys() == original.keys()
    assert all(released[k].dtype == original[k].dtype and torch.equal(released[k], original[k]) for k in released)

    ours = _external(name)
    folder = snapshot_download(_PUBLISHED[0], revision=_PUBLISHED[1], allow_patterns=[f"{name}/*"])
    published = type(ours).from_pretrained(folder, subfolder=name)
    actual, expected = ours.state_dict(), published.state_dict()
    assert actual.keys() == expected.keys()
    assert all(actual[k].dtype == expected[k].dtype and torch.equal(actual[k], expected[k]) for k in expected)
