"""TRELLIS.2 and Pixal3D load the gated DINOv3 ViT-L/16 from its repository, as their converters used to embed it.

Needs Hub access to facebook/dinov3-vitl16-pretrain-lvd1689m (about 1.2 GB).
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import DINOv3ViTModel

from diffusers_3d import Pixal3DImageTo3DPipeline, Trellis2ImageTo3DPipeline
from diffusers_3d.families.trellis2.models import DINOV3_REVISIONS, DINOV3_VITL16_REPO

pytestmark = [
    pytest.mark.real_weights,
    pytest.mark.skipif(
        os.environ.get("DIFFUSERS_3D_GATED_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_GATED_REAL_WEIGHTS=1"
    ),
]


@torch.no_grad()
def test_dinov3_from_its_repository_is_what_the_converters_embedded(tmp_path):
    # The converters saved ``DINOv3ViTModel.from_pretrained`` of the pinned repository; pipelines loaded it back.
    DINOv3ViTModel.from_pretrained(DINOV3_VITL16_REPO, revision=DINOV3_REVISIONS[DINOV3_VITL16_REPO]).save_pretrained(
        tmp_path
    )
    pixels = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    for pipeline, dtype, expected_dtype in (
        (Trellis2ImageTo3DPipeline, None, torch.float32),
        (Trellis2ImageTo3DPipeline, torch.bfloat16, torch.bfloat16),
        # Pixal3D keeps DINOv3 in float32 under a bfloat16 load (its ``from_pretrained`` passes that dtype).
        (Pixal3DImageTo3DPipeline, torch.float32, torch.float32),
    ):
        component = pipeline.external_components["image_encoder"]
        assert (component.repository, component.revision) == (DINOV3_VITL16_REPO, DINOV3_REVISIONS[DINOV3_VITL16_REPO])
        ours = component("image_encoder", pipeline.__name__, dtype, {})
        embedded = DINOv3ViTModel.from_pretrained(tmp_path, dtype=expected_dtype).eval()
        assert ours.dtype is expected_dtype
        actual, expected = ours.state_dict(), embedded.state_dict()
        assert actual.keys() == expected.keys() and all(torch.equal(actual[k], expected[k]) for k in expected)
        values = pixels.to(expected_dtype)
        assert torch.equal(
            ours(pixel_values=values).last_hidden_state, embedded(pixel_values=values).last_hidden_state
        )


@torch.no_grad()
def test_sparse_structure_decoder_from_trellis_image_large_equals_the_published_conversion():
    """TRELLIS.2 references TRELLIS-image-large's decoder; converted on load, it equals what the converter wrote."""

    from huggingface_hub import snapshot_download

    from diffusers_3d.families.trellis2 import Trellis2SparseStructureDecoder

    component = Trellis2ImageTo3DPipeline.external_components["sparse_structure_decoder"]
    ours = component("sparse_structure_decoder", "Trellis2ImageTo3DPipeline", None, {})
    # The TRELLIS.2 conversion published before the decoder loaded from TRELLIS-image-large.
    folder = snapshot_download(
        "suvadityamuk/TRELLIS.2-4B-diffusers-3d",
        revision="b9ded7089ac8dcbc8c3e9a4367625194cc243ade",
        allow_patterns=["sparse_structure_decoder/*"],
    )
    published = Trellis2SparseStructureDecoder.from_pretrained(folder, subfolder="sparse_structure_decoder")
    actual, expected = ours.state_dict(), published.state_dict()
    assert type(ours) is Trellis2SparseStructureDecoder
    assert actual.keys() == expected.keys()
    assert all(actual[k].dtype == expected[k].dtype and torch.equal(actual[k], expected[k]) for k in expected)
