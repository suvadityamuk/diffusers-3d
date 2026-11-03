from __future__ import annotations

import json
import sys

import httpx
import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler
from huggingface_hub import constants
from huggingface_hub.errors import GatedRepoError
from PIL import Image

import diffusers_3d.execution.pipelines as pipelines
from diffusers_3d.data import ImageCondition, as_image_conditions
from diffusers_3d.data.background import (
    BackgroundRemoval,
    RembgBackgroundRemover,
    downscale_longest_side,
    lacks_alpha,
    lacks_alpha_channel,
    remove_background,
    resize_shortest_side,
)
from diffusers_3d.data.birefnet import RMBG2_REPO, RMBG2_REVISION, BiRefNet, BiRefNetBackgroundRemover
from diffusers_3d.data.rmbg import RMBG14_REPO, BriaRMBG, RMBG14BackgroundRemover
from diffusers_3d.execution.pipelines import Object3DPipeline

TINY_BIREFNET = {"embed_dim": 8, "depths": (2, 2, 2, 2), "num_heads": (1, 1, 2, 2), "window_size": 4}


class _Remover:
    def __init__(self):
        self.calls, self.devices = [], []

    def to(self, device):
        self.devices.append(torch.device(device))

    def __call__(self, image):
        self.calls.append(image.size)
        alpha = Image.new("L", image.size, 0)
        alpha.paste(255, (1, 1, 3, 3))
        image = image.convert("RGBA")
        image.putalpha(alpha)
        return image


def _pipeline(**removal):
    built = []

    def factory():
        built.append(_Remover())
        return built[-1]

    class Pipeline(Object3DPipeline):
        background_removal = BackgroundRemoval(remover=factory, **removal)

        def __init__(self, scheduler):
            super().__init__()
            self.register_modules(scheduler=scheduler)

    return Pipeline(DDPMScheduler()), built


def test_removal_runs_by_default_where_the_trigger_fires_and_builds_the_remover_once():
    pipeline, built = _pipeline(trigger=lacks_alpha)
    cut_out = torch.ones(4, 6, 8)
    cut_out[3, 0, 0] = 0
    assert pipeline.image_conditions(cut_out)[0].image is not None and built == []
    (rgb,) = pipeline.image_conditions(torch.full((3, 6, 8), 0.5))
    pipeline.image_conditions(torch.ones(4, 6, 8))  # a fully opaque alpha is no alpha
    assert rgb.image.shape == (4, 6, 8) and rgb.image[3].sum() == 4
    assert len(built) == 1 and built[0].calls == [(8, 6), (8, 6)] and built[0].devices == [torch.device("cpu")] * 2
    assert pipeline.background_remover is built[0]


def test_resize_runs_before_the_remover_and_disable_and_enable_override():
    pipeline, built = _pipeline(trigger=lacks_alpha_channel, resize=lambda image: downscale_longest_side(image, 4))
    (condition,) = pipeline.image_conditions(torch.full((3, 6, 8), 0.5))
    assert condition.image.shape == (4, 3, 4) and built[0].calls == [(4, 3)]
    assert pipeline.image_conditions(torch.ones(4, 6, 8))[0].image.shape == (4, 6, 8)  # opaque RGBA: no trigger

    pipeline.disable_background_removal()
    assert pipeline.image_conditions(torch.full((3, 6, 8), 0.5))[0].image.shape == (3, 6, 8)
    other = _Remover()
    pipeline.enable_background_removal(other)
    pipeline.image_conditions(torch.full((3, 6, 8), 0.5))
    assert other.calls == [(4, 3)] and len(built[0].calls) == 1
    pipeline.enable_background_removal()
    pipeline.image_conditions(torch.full((3, 6, 8), 0.5))
    assert len(built) == 2 and built[1].calls == [(4, 3)]


def test_opt_in_removal_and_conditions_with_a_mask_or_camera():
    pipeline, built = _pipeline(automatic=False, resize=lambda image: image.resize((4, 3)))
    rgb = torch.full((3, 6, 8), 0.5)
    assert pipeline.image_conditions(rgb)[0].image.shape == (3, 6, 8) and built == []
    pipeline.enable_background_removal()
    masked = ImageCondition(image=rgb, mask=torch.ones(1, 6, 8))
    assert pipeline.image_conditions(masked)[0] is masked and built == []

    from diffusers_3d import CameraRig

    camera = CameraRig(
        world_to_camera=torch.eye(4)[None],
        intrinsics=torch.tensor([[[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]]]),
        image_sizes=torch.tensor([[6, 8]]),
    )
    with pytest.raises(ValueError, match="invalidate its camera"):
        pipeline.image_conditions(ImageCondition(image=rgb, camera=camera))


def test_the_remover_is_not_saved_with_the_pipeline(tmp_path):
    pipeline, built = _pipeline()
    pipeline.image_conditions(torch.full((3, 6, 8), 0.5))
    pipeline.save_pretrained(tmp_path)
    assert "background_remover" not in json.loads((tmp_path / "model_index.json").read_text())


@pytest.mark.parametrize(
    "build, repository",
    [
        (lambda: BiRefNetBackgroundRemover(repository=RMBG2_REPO, revision=RMBG2_REVISION), RMBG2_REPO),
        (RMBG14BackgroundRemover, RMBG14_REPO),
    ],
)
def test_gated_remover_weights_explain_how_to_get_access(monkeypatch, build, repository):
    def deny(repository, token=None):
        request = httpx.Request("GET", f"https://huggingface.co/{repository}")
        raise GatedRepoError("gated", response=httpx.Response(403, request=request))

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(pipelines.huggingface_hub, "auth_check", deny)
    with pytest.raises(PermissionError, match=f"Accept its license at https://huggingface.co/{repository}"):
        build()


def test_trellis2_and_pixal3d_cut_out_with_rmbg2_by_default():
    from diffusers_3d import Pixal3DImageTo3DPipeline, Trellis2ImageTo3DPipeline

    for pipeline in (Trellis2ImageTo3DPipeline, Pixal3DImageTo3DPipeline):
        removal = pipeline.background_removal
        assert removal.automatic and removal.resize_in_source_mode and removal.resize is downscale_longest_side
        assert removal.remover.keywords == {"resize": "pil", "repository": RMBG2_REPO, "revision": RMBG2_REVISION}


def test_conditions_remember_the_source_mode_and_apply_exif_orientation(tmp_path):
    image = Image.new("RGB", (6, 4), (255, 0, 0))
    image.paste((0, 0, 255), (0, 0, 1, 1))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    image.save(tmp_path / "rotated.png", exif=exif)
    for source in (tmp_path / "rotated.png", Image.open(tmp_path / "rotated.png")):
        (condition,) = as_image_conditions(source)
        assert condition.image.shape == (3, 6, 4) and condition.source_mode == "RGB"
        assert condition.image[2, 0, -1] > 0.9  # the blue corner moved to the top right
    assert as_image_conditions(torch.ones(3, 2, 2))[0].source_mode is None
    la = as_image_conditions(Image.new("LA", (2, 2), (9, 0)))[0]
    assert la.source_mode == "LA" and la.image.shape[0] == 4
    # Equality ignores the source mode.
    assert la == ImageCondition(image=la.image)


def test_alpha_of_other_modes_is_ignored_like_upstream():
    transparent = Image.new("RGBA", (4, 4), (10, 20, 30, 0))
    assert not lacks_alpha(as_image_conditions(transparent)[0])
    assert lacks_alpha(as_image_conditions(transparent.convert("LA"))[0])
    palette = transparent.convert("RGB").convert("P")
    palette.info["transparency"] = 0
    (condition,) = as_image_conditions(palette)
    assert condition.image.shape[0] == 4 and lacks_alpha(condition)
    # Tensors carry no mode: their alpha counts.
    assert not lacks_alpha(ImageCondition(image=condition.image))


def test_resizing_in_the_source_mode_matches_pillow_on_the_loaded_image():
    rng = np.random.default_rng(0)
    rgb = Image.fromarray(rng.integers(0, 256, (30, 50, 3), dtype=np.uint8))
    la = Image.fromarray(rng.integers(0, 256, (30, 50, 2), dtype=np.uint8), "LA")
    for image in (rgb.convert("P"), la, rgb.convert("L"), rgb):
        received = []

        def remover(received_image):
            received.append(received_image)
            return received_image.convert("RGBA")

        (condition,) = as_image_conditions(image)
        remove_background(
            condition, remover, trigger=lambda _: True, resize=resize_shortest_side, resize_in_source_mode=True
        )
        expected = resize_shortest_side(image).convert("RGB")
        assert received[0].mode == "RGB" and np.array_equal(np.asarray(received[0]), np.asarray(expected))


def test_rmbg14_remover_keeps_the_largest_object_box_without_small_specks():
    class Stub(BriaRMBG):
        def forward(self, x):
            # Two objects (the larger on the left), plus a 10x10 speck in the corner, at the 1024 network size.
            probability = torch.zeros(1, 1, *x.shape[2:])
            probability[..., 100:700, 50:400] = 1.0
            probability[..., 300:500, 600:800] = 0.9
            probability[..., 1000:1010, 1000:1010] = 1.0
            return probability

    image = Image.new("RGB", (512, 512), (200, 100, 50))
    result = RMBG14BackgroundRemover(Stub())(image)
    alpha = np.asarray(result.getchannel("A"))
    assert result.mode == "RGBA" and result.size == (512, 512)
    assert set(np.unique(alpha)) == {0, 255}
    rows, columns = np.nonzero(alpha)
    # Only the largest object's box survives; the second object and the speck are cleared.
    assert 45 <= rows.min() and rows.max() <= 355 and 20 <= columns.min() and columns.max() <= 205
    with pytest.raises(ValueError, match="pure black"):
        RMBG14BackgroundRemover(Stub())(Image.new("RGB", (64, 64)))


def test_rembg_remover_names_its_session_and_forwards_remove_kwargs(monkeypatch):
    from types import ModuleType, SimpleNamespace

    calls = []
    rembg = ModuleType("rembg")
    rembg.new_session = lambda model_name="bria-rmbg": SimpleNamespace(name=model_name)
    rembg.remove = lambda image, session=None, **kwargs: calls.append((session.name, kwargs)) or image
    monkeypatch.setitem(sys.modules, "rembg", rembg)
    image = Image.new("RGB", (4, 4))
    RembgBackgroundRemover()(image)
    RembgBackgroundRemover("u2net", bgcolor=[255, 255, 255, 0])(image)
    assert calls == [("u2net", {}), ("u2net", {"bgcolor": [255, 255, 255, 0]})]


@pytest.mark.parametrize("resize", ["pil", "tensor"])
def test_birefnet_remover_returns_rgba_of_the_input_size(resize):
    torch.manual_seed(0)
    remover = BiRefNetBackgroundRemover(BiRefNet(**TINY_BIREFNET), resize=resize).to("cpu")
    image = Image.new("RGB", (37, 21), (10, 200, 30))
    result = remover(image)
    assert result.mode == "RGBA" and result.size == (37, 21) and image.mode == "RGB"
    assert result.getchannel("A").getextrema()[1] <= 255


def test_rmbg2_patch_order_is_its_remote_codes_column_major_split():
    from diffusers_3d.data.birefnet import _image_to_patches_in_order

    def get_patches_batch(x, p):  # briaai/RMBG-2.0 birefnet.py, Decoder.get_patches_batch
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                patches_x += [p.unsqueeze(0) for p in torch.split(column_x, split_size_or_sections=_size_h, dim=-2)]
            patches_batch.append(torch.cat(patches_x, dim=1))
        return torch.cat(patches_batch, dim=0)

    image, reference = torch.randn(2, 3, 32, 48), torch.empty(1, 1, 8, 12)
    assert torch.equal(_image_to_patches_in_order(image, reference, "columns"), get_patches_batch(image, reference))
    assert not torch.equal(
        _image_to_patches_in_order(image, reference, "channels"), get_patches_batch(image, reference)
    )
