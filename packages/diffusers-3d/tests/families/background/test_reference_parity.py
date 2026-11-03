"""Background removal against the pinned upstream code of each family.

The BiRefNet network and both upstream wrappers are compared bit for bit with tiny random weights. Each family's
trigger, pre-removal resize, and remover call are compared by running upstream's own preprocessing with a recording
remover. ``DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS=1`` also compares the released weights against the Hub remote code.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from PIL import Image
from torch import nn

from diffusers_3d.data import as_image_conditions
from diffusers_3d.data.background import (
    RembgBackgroundRemover,
    downscale_longest_side,
    has_ambiguous_alpha,
    lacks_alpha,
    remove_background,
)
from diffusers_3d.data.birefnet import BIREFNET_REPO, BIREFNET_REVISION, BiRefNet, BiRefNetBackgroundRemover

pytestmark = pytest.mark.reference_parity

_CHECKOUTS = {
    "idol": ("IDOL", "/tmp/idol-sdk", "4702a7f00850e34b6e83a2170cc123c3e9570744", "speridlabs/idol-sdk"),
    "triposplat": (
        "TRIPOSPLAT",
        "/tmp/TripoSplat",
        "d8db9e018b413dd9c4a9fe22463781bf98e8e68d",
        "VAST-AI-Research/TripoSplat",
    ),
    "trellis": ("TRELLIS", "/tmp/TRELLIS", "442aa1e1afb9014e80681d3bf604e8d728a86ee7", "microsoft/TRELLIS"),
    "trellis2": ("TRELLIS2", "/tmp/TRELLIS.2", "75fbf0183001ed9876c8dbb35de6b68552ee08bd", "microsoft/TRELLIS.2"),
    "pixal3d": ("PIXAL3D", "/tmp/Pixal3D", "f7cf38429b0bd264f1995f0f8743a88b1c728b94", "TencentARC/Pixal3D"),
    "triposr": ("TRIPOSR", "/tmp/TripoSR", "107cefdc244c39106fa830359024f6a2f1c78871", "VAST-AI-Research/TripoSR"),
    "step1x3d": ("STEP1X3D", "/tmp/Step1X-3D", "cb5ac944709c6c913109070c7b90c3447f57f3d4", "stepfun-ai/Step1X-3D"),
    "triposg": ("TRIPOSG", "/tmp/TripoSG", "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c", "VAST-AI-Research/TripoSG"),
    "hunyuan3d": (
        "HUNYUAN3D",
        "/tmp/Hunyuan3D-2.1",
        "82920d643c0dc2f7bfd7255f45f62d386edfe60c",
        "Tencent-Hunyuan/Hunyuan3D-2.1",
    ),
}
# The tiny Swin: 2 blocks per stage so shifted windows run, and 4-pixel windows that need padding at the last stage.
_TINY = {"embed_dim": 16, "depths": (2, 2, 2, 2), "num_heads": (1, 2, 2, 4), "window_size": 4}


def _source(name: str, *paths: str) -> Path:
    env, default, revision, repository = _CHECKOUTS[name]
    root = Path(os.environ.get(f"DIFFUSERS_3D_{env}_REFERENCE_ROOT", default))
    try:
        validate_reference_checkout(
            root,
            expected_revision=revision,
            expected_repository=f"https://github.com/{repository}.git",
            expected_paths=paths,
        )
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
    return root


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def _pinned(path: Path, names: tuple[str, ...], namespace: dict, class_name: str | None = None) -> dict:
    """Compile top-level definitions (or one class's methods) of a pinned file into ``namespace``."""

    body = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = []
    for node in body:
        targets = [target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)]
        if getattr(node, "name", None) in names or any(target in names for target in targets):
            node.decorator_list = [] if hasattr(node, "decorator_list") else None
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def _randomize(module: nn.Module, seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in module.state_dict().items():
            if name.endswith("running_var"):
                value.copy_(torch.rand(value.shape, generator=generator) + 0.5)
            elif value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator) * 0.2)
    return module.eval()


def _tiny_upstream(module: ModuleType) -> nn.Module:
    """Upstream's BiRefNet assembled from its own blocks with the tiny Swin (its constructor hard-codes Swin-L)."""

    backbone = module._SwinLarge.__new__(module._SwinLarge)
    nn.Module.__init__(backbone)
    backbone.num_layers, backbone.embed_dim = 4, _TINY["embed_dim"]
    backbone.patch_embed = module._SwinPatchEmbed(patch_size=4, in_channels=3, embed_dim=_TINY["embed_dim"])
    backbone.num_features = [_TINY["embed_dim"] * 2**index for index in range(4)]
    backbone.layers = nn.ModuleList(
        module._SwinBasicLayer(
            dim=backbone.num_features[index],
            depth=_TINY["depths"][index],
            num_heads=_TINY["num_heads"][index],
            window_size=_TINY["window_size"],
            downsample=index < 3,
        )
        for index in range(4)
    )
    for index, features in enumerate(backbone.num_features):
        backbone.add_module(f"norm{index}", nn.LayerNorm(features))
    channels = tuple(2 * features for features in reversed(backbone.num_features))
    model = module.BiRefNet.__new__(module.BiRefNet)
    nn.Module.__init__(model)
    model._CHANNELS = channels
    model.bb = backbone
    model.squeeze_module = nn.Sequential(module._BasicDecBlk(channels[0] + sum(channels[1:]), channels[0]))
    model.decoder = module._BiRefNetDecoder(channels=channels)
    return model


def _tiny_pair(module: ModuleType, seed: int = 0) -> tuple[nn.Module, BiRefNet]:
    upstream = _randomize(_tiny_upstream(module), seed)
    ours = BiRefNet(**_TINY).eval()
    state_dict = upstream.state_dict()
    ours.load_state_dict({key: value for key, value in state_dict.items() if key in ours.state_dict()}, strict=True)
    return upstream, ours


def _idol_birefnet() -> ModuleType:
    root = _source("idol", "idol/models/birefnet.py", "idol/preprocess.py")
    import_reference_dependency("torchvision.ops")
    return _module("background_reference_idol_birefnet", root / "idol/models/birefnet.py")


def _triposplat_model() -> ModuleType:
    root = _source("triposplat", "model.py", "triposplat.py")
    return _module("background_reference_triposplat_model", root / "model.py")


def _photo(width: int, height: int, seed: int = 0, mode: str = "RGB") -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 4), dtype=np.uint8)
    if mode == "RGBA":
        pixels[..., 3] = 255
    return Image.fromarray(pixels[..., : len(mode)], mode)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Upstream IDOL forward without its final sigmoid, the way the Hub model returns its prediction."""

    x1, x2, x3, x4 = model._forward_enc(x)
    return model.decoder(x, x1, x2, x3, model.squeeze_module(x4))


@pytest.mark.parametrize("size", [(128, 192), (96, 128)])
def test_birefnet_network_matches_idol_and_triposplat_bit_for_bit(size):
    # 96 rows reach an odd 3-row stage in the half-size pass: TripoSplat pads it, IDOL's port cannot run it.
    for module in (_triposplat_model(), _idol_birefnet()) if size[0] % 64 == 0 else (_triposplat_model(),):
        upstream, ours = _tiny_pair(module)
        x = torch.randn(1, 3, *size, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            assert torch.equal(ours(x), upstream(x))


def test_birefnet_tensor_wrapper_matches_idol_and_triposplat():
    image = _photo(150, 110)
    for module in (_idol_birefnet(), _triposplat_model()):
        upstream, ours = _tiny_pair(module)
        expected = upstream.remove_background(image.copy())
        result = BiRefNetBackgroundRemover(ours, resize="tensor")(image)
        assert result.mode == "RGBA" and np.array_equal(np.asarray(result), np.asarray(expected))
    assert image.mode == "RGB"


def test_birefnet_pil_wrapper_matches_trellis2_pixal3d_and_step1x3d_texture(monkeypatch):
    """TRELLIS.2 and Pixal3D ship the same ``rembg/BiRefNet.py``; Step1X-3D's texture ``remove_bg`` is its body."""

    upstream, ours = _tiny_pair(_idol_birefnet())
    image = _photo(150, 110)
    expected = []
    for name in ("trellis2", "pixal3d"):
        package = "trellis2" if name == "trellis2" else "pixal3d"
        path = _source(name, f"{package}/pipelines/rembg/BiRefNet.py") / package / "pipelines/rembg/BiRefNet.py"
        wrapper_module = _module(f"background_reference_{name}_birefnet", path)
        monkeypatch.setattr(
            wrapper_module.AutoModelForImageSegmentation,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(eval=lambda: None, __call__=None),
        )
        wrapper = wrapper_module.BiRefNet()
        wrapper.model = lambda x: [_logits(upstream, x)]
        # The wrapper hard-codes ``.to("cuda")``; run it on the CPU.
        original_to = torch.Tensor.to
        monkeypatch.setattr(
            torch.Tensor, "to", lambda self, *a, **k: original_to(self, *("cpu" if v == "cuda" else v for v in a), **k)
        )
        expected.append(np.asarray(wrapper(image.copy())))
        monkeypatch.setattr(torch.Tensor, "to", original_to)

    step1x = _source("step1x3d", "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py")
    path = step1x / "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py"
    namespace = _pinned(
        path,
        ("remove_bg",),
        {"torch": torch, "transforms": wrapper_module.transforms},
        "Step1X3DTexturePipeline",
    )
    expected.append(
        np.asarray(
            namespace["remove_bg"](
                None, image.copy(), lambda x: [_logits(upstream, x)], wrapper.transform_image, "cpu"
            )
        )
    )

    result = np.asarray(BiRefNetBackgroundRemover(ours, resize="pil")(image))
    for reference in expected:
        assert np.array_equal(result, reference)


class _Recorder:
    """A remover that records the exact image it is given and returns a fixed central foreground."""

    def __init__(self) -> None:
        self.inputs: list[tuple[str, tuple[int, int], bytes]] = []
        self.rgb_inputs: list[tuple[tuple[int, int], bytes]] = []

    def __call__(self, image: Image.Image, **kwargs) -> Image.Image:
        self.inputs.append((image.mode, image.size, image.tobytes()))
        self.rgb_inputs.append((image.size, image.convert("RGB").tobytes()))
        alpha = Image.new("L", image.size, 0)
        alpha.paste(255, (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4))
        result = image.convert("RGB")
        result.putalpha(alpha)
        return result

    remove_background = __call__


def _stub_rembg(monkeypatch, recorder: _Recorder) -> list:
    """A ``rembg`` module whose ``remove`` records its call. ``new_session()`` defaults to ``u2net`` as in 2.0.65."""

    calls = []
    rembg = ModuleType("rembg")
    rembg.new_session = lambda model_name="u2net", **kwargs: SimpleNamespace(model_name=model_name)

    def remove(image, session=None, **kwargs):
        calls.append((session.model_name, kwargs))
        return recorder(image)

    rembg.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", rembg)
    return calls


def _ours(image: Image.Image, remover, **removal) -> list:
    """Our recorded remover inputs for ``image`` loaded the way the pipelines load it."""

    for condition in as_image_conditions(image):
        remove_background(condition, remover, **removal)


def _declared(pipeline_name: str) -> dict:
    """The trigger and resize a pipeline class declares for its background removal."""

    import diffusers_3d

    removal = getattr(diffusers_3d, pipeline_name).background_removal
    return {
        "trigger": removal.trigger,
        "resize": removal.resize,
        "resize_in_source_mode": removal.resize_in_source_mode,
    }


_INPUTS = {
    "large rgb": lambda: _photo(1500, 900),
    "small rgb": lambda: _photo(300, 200, seed=1),
    "opaque rgba": lambda: _photo(1300, 1100, seed=2, mode="RGBA"),
    "grey": lambda: _photo(400, 300, seed=3).convert("L"),
    "cut-out rgba": lambda: Image.fromarray(np.asarray(_Recorder()(_photo(500, 400, seed=4)))),
    "palette": lambda: _photo(500, 400, seed=5).convert("P"),
    "large palette": lambda: _photo(1500, 900, seed=7).convert("P"),
    "cut-out la": lambda: _Recorder()(_photo(500, 400, seed=6)).convert("LA"),
    "large cut-out la": lambda: _Recorder()(_photo(1300, 1100, seed=8)).convert("LA"),
}
# Upstream only reads alpha from mode "RGBA", so it re-cuts LA (and palette + tRNS) images; TRELLIS.2, Pixal3D,
# IDOL, and TripoSplat resize the loaded image in its own mode first (NEAREST for palettes, premultiplied for LA).
_CASES = list(_INPUTS)


@pytest.mark.parametrize("name", _CASES)
def test_trellis_runs_rembg_u2net_after_its_removal_branch_downscale(monkeypatch, name):
    root = _source("trellis", "trellis/pipelines/trellis_image_to_3d.py")
    upstream, recorder = _Recorder(), _Recorder()
    calls = _stub_rembg(monkeypatch, upstream)
    namespace = {"np": np, "Image": Image, "rembg": sys.modules["rembg"]}
    path = root / "trellis/pipelines/trellis_image_to_3d.py"
    preprocess = _pinned(path, ("preprocess_image",), namespace, "TrellisImageTo3DPipeline")["preprocess_image"]
    preprocess(SimpleNamespace(rembg_session=None), _INPUTS[name]())

    ours_calls = _stub_rembg(monkeypatch, recorder)
    _ours(_INPUTS[name](), RembgBackgroundRemover("u2net"), trigger=lacks_alpha, resize=downscale_longest_side)
    assert recorder.inputs == upstream.inputs and ours_calls == calls
    assert bool(calls) == (name != "cut-out rgba") and all(call == ("u2net", {}) for call in calls)


@pytest.mark.parametrize("family", ["trellis2", "pixal3d"])
@pytest.mark.parametrize("name", _CASES)
def test_trellis2_and_pixal3d_remove_after_the_longest_side_downscale(family, name):
    package, file_name, class_name = {
        "trellis2": ("trellis2", "trellis2_image_to_3d.py", "Trellis2ImageTo3DPipeline"),
        "pixal3d": ("pixal3d", "pixal3d_image_to_3d.py", "Pixal3DImageTo3DPipeline"),
    }[family]
    path = _source(family, f"{package}/pipelines/{file_name}") / package / "pipelines" / file_name
    upstream, recorder = _Recorder(), _Recorder()
    preprocess = _pinned(path, ("preprocess_image",), {"np": np, "Image": Image}, class_name)["preprocess_image"]
    preprocess(SimpleNamespace(low_vram=False, rembg_model=upstream), _INPUTS[name]())
    pipeline = {"trellis2": "Trellis2ImageTo3DPipeline", "pixal3d": "Pixal3DImageTo3DPipeline"}[family]
    _ours(_INPUTS[name](), recorder, **_declared(pipeline))
    assert recorder.inputs == upstream.inputs and bool(upstream.inputs) == (name != "cut-out rgba")


@pytest.mark.parametrize("family", ["idol", "triposplat"])
@pytest.mark.parametrize("name", _CASES)
def test_idol_and_triposplat_remove_after_the_shortest_side_resize(family, name):
    upstream, recorder = _Recorder(), _Recorder()
    if family == "idol":
        root = _source("idol", "idol/preprocess.py")
        _module("background_reference_idol_preprocess", root / "idol/preprocess.py").preprocess_image(
            _INPUTS[name](), upstream
        )
    else:
        path = _source("triposplat", "triposplat.py") / "triposplat.py"
        namespace = {
            "np": np,
            "Image": Image,
            "ImageFilter": __import__("PIL.ImageFilter").ImageFilter,
            "torch": torch,
        }
        _pinned(path, ("_CANVAS_SIZE", "_image_to_pil", "preprocess_image"), namespace)
        namespace["preprocess_image"](_INPUTS[name](), upstream)
    pipeline = {"idol": "IdolImageTo3DPipeline", "triposplat": "TripoSplatImageTo3DPipeline"}[family]
    _ours(_INPUTS[name](), recorder, **_declared(pipeline))
    assert recorder.inputs == upstream.inputs and bool(upstream.inputs) == (name != "cut-out rgba")


@pytest.mark.parametrize("name", _CASES)
def test_triposr_and_step1x3d_geometry_run_rembg_on_the_full_size_input(monkeypatch, name):
    """Step1X-3D asks rembg 2.0.65 (its pin) for ``bria``, which that version resolves to the u2net session."""

    triposr = _source("triposr", "tsr/utils.py") / "tsr/utils.py"
    step1x = _source("step1x3d", "step1x3d_geometry/models/pipelines/pipeline_utils.py")
    step1x = step1x / "step1x3d_geometry/models/pipelines/pipeline_utils.py"
    upstream, recorder = _Recorder(), _Recorder()
    calls = _stub_rembg(monkeypatch, upstream)
    namespace = _pinned(triposr, ("remove_background",), {"rembg": sys.modules["rembg"], "Any": object})
    namespace["remove_background"](_INPUTS[name](), sys.modules["rembg"].new_session())
    import PIL.Image

    namespace = _pinned(step1x, ("preprocess_image",), {"PIL": PIL, "List": list, "Union": object})
    namespace["preprocess_image"](_INPUTS[name]())

    ours_calls = _stub_rembg(monkeypatch, recorder)
    for _ in range(2):
        _ours(_INPUTS[name](), RembgBackgroundRemover("u2net"), trigger=lacks_alpha)
    # Upstream hands rembg the image in its own mode, ours in RGB; rembg's output is the same for both (see below).
    assert recorder.rgb_inputs == upstream.rgb_inputs and len(calls) == len(ours_calls)
    assert bool(calls) == (name != "cut-out rgba")
    assert calls[::2] == ours_calls[::2] and all(call[0] == "bria" for call in calls[1::2])


def test_rembg_cuts_out_rgb_like_the_opaque_rgba_grey_and_palette_images_upstream_passes():
    rembg = import_reference_dependency("rembg")

    class Session:
        def predict(self, image, *args, **kwargs):
            return [Image.fromarray(np.asarray(image.convert("RGB"))[..., 0])]

    for image in (_INPUTS["opaque rgba"](), _INPUTS["grey"](), _INPUTS["palette"]()):
        (condition,) = as_image_conditions(image)
        rgb = condition.image[:3].mul(255).round().to(torch.uint8).expand(3, -1, -1).permute(1, 2, 0).numpy()
        for kwargs in ({}, {"bgcolor": [255, 255, 255, 0]}):
            expected = np.asarray(rembg.remove(image, session=Session(), **kwargs))
            assert np.array_equal(
                np.asarray(rembg.remove(Image.fromarray(rgb), session=Session(), **kwargs)), expected
            )


def test_hunyuan3d_opt_in_remover_is_upstreams_rembg_call(monkeypatch):
    path = _source("hunyuan3d", "hy3dshape/hy3dshape/rembg.py") / "hy3dshape/hy3dshape/rembg.py"
    upstream, recorder = _Recorder(), _Recorder()
    calls = _stub_rembg(monkeypatch, upstream)
    sys.modules.pop("background_reference_hunyuan3d_rembg", None)
    _module("background_reference_hunyuan3d_rembg", path).BackgroundRemover()(_photo(300, 200))
    ours_calls = _stub_rembg(monkeypatch, recorder)
    RembgBackgroundRemover("u2net", bgcolor=[255, 255, 255, 0])(_photo(300, 200))
    assert ours_calls == calls == [("u2net", {"bgcolor": [255, 255, 255, 0]})] and recorder.inputs == upstream.inputs


def test_triposg_trigger_is_upstreams_is_valid_alpha():
    cv2 = import_reference_dependency("cv2")
    path = _source("triposg", "scripts/image_process.py") / "scripts/image_process.py"
    load_image = next(
        node for node in ast.parse(path.read_text(encoding="utf-8")).body if getattr(node, "name", "") == "load_image"
    )
    is_valid_alpha = next(node for node in load_image.body if getattr(node, "name", "") == "is_valid_alpha")
    namespace = {"cv2": cv2, "np": np, "torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[is_valid_alpha], type_ignores=[])), str(path), "exec"),
        namespace,
    )

    rng = np.random.default_rng(0)
    alphas = [rng.integers(0, 256, (40, 50), dtype=np.uint8) for _ in range(4)]
    for low, high, count in ((12, 244, 20), (13, 244, 20), (12, 243, 20), (12, 244, 19), (0, 255, 2000)):
        alpha = np.full((40, 50), 128, np.uint8)  # 1% of 2000 pixels is 20
        alpha.flat[:count], alpha.flat[-count:] = low, high
        alphas.append(alpha)
    for alpha in alphas:
        rgba = np.concatenate([rng.integers(0, 256, (40, 50, 3), dtype=np.uint8), alpha[..., None]], -1)
        (condition,) = as_image_conditions(rgba)
        assert has_ambiguous_alpha(condition) == (not namespace["is_valid_alpha"](alpha))
    (rgb,) = as_image_conditions(rgba[..., :3])
    assert has_ambiguous_alpha(rgb)


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS=1"
)
def test_released_birefnet_matches_the_hub_remote_code_and_idol():
    """Step1X-3D's texture stage runs the Hub remote code; IDOL its own port with its bundled copy of the weights."""

    transformers = import_reference_dependency("transformers")
    for dependency in ("timm", "kornia", "einops"):
        import_reference_dependency(dependency)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ours = BiRefNet.from_pretrained()
    # Step1X-3D pins transformers 4.48, which loads remote code in float32 (5.x would pick the file's float16).
    hub = transformers.AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_REPO, revision=BIREFNET_REVISION, trust_remote_code=True, dtype=torch.float32
    ).eval()
    step1x = _source("step1x3d", "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py")
    path = step1x / "step1x3d_texture/pipelines/step1x_3d_texture_synthesis_pipeline.py"
    from torchvision import transforms

    remove_bg = _pinned(path, ("remove_bg",), {"torch": torch, "transforms": transforms}, "Step1X3DTexturePipeline")
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    idol = _idol_birefnet().BiRefNet()
    weights = hf_hub_download(
        "speridlabs/idol", "birefnet.safetensors", revision="2731c2291fc65a797c5cb627a5e5526fc1490902"
    )
    idol.load_state_dict(load_file(weights))
    triposr = _source("triposr", "examples/captured.jpeg")
    image = Image.open(triposr / "examples/captured.jpeg").convert("RGB")

    x = transform(image)[None]
    with torch.no_grad():
        assert torch.equal(ours(x), hub(x)[-1].sigmoid())
    expected = remove_bg["remove_bg"](None, image.copy(), hub, transform, "cpu")
    assert np.array_equal(np.asarray(BiRefNetBackgroundRemover(ours, resize="pil")(image)), np.asarray(expected))
    expected = idol.eval().remove_background(image.copy())
    assert np.array_equal(np.asarray(BiRefNetBackgroundRemover(ours, resize="tensor")(image)), np.asarray(expected))


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS=1"
)
def test_trellis2_rmbg2_remover_matches_the_pinned_wrapper_and_hub_remote_code(monkeypatch):
    """TRELLIS.2 and Pixal3D run ``rembg/BiRefNet.py`` with ``briaai/RMBG-2.0`` (gated): needs access on the Hub."""

    transformers = import_reference_dependency("transformers")
    for dependency in ("timm", "kornia", "einops"):
        import_reference_dependency(dependency)
    from diffusers_3d import Trellis2ImageTo3DPipeline
    from diffusers_3d.data.birefnet import RMBG2_REPO, RMBG2_REVISION

    path = _source("trellis2", "trellis2/pipelines/rembg/BiRefNet.py") / "trellis2/pipelines/rembg/BiRefNet.py"
    wrapper_module = _module("background_reference_trellis2_birefnet_real", path)
    load = transformers.AutoModelForImageSegmentation.from_pretrained
    # The wrapper loads RMBG-2.0 unpinned; pin it, in the float32 its weights are stored in.
    monkeypatch.setattr(
        wrapper_module.AutoModelForImageSegmentation,
        "from_pretrained",
        lambda name, **kwargs: load(name, revision=RMBG2_REVISION, dtype=torch.float32, **kwargs),
    )
    wrapper = wrapper_module.BiRefNet(RMBG2_REPO)
    original_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to", lambda self, *a, **k: original_to(self, *("cpu" if v == "cuda" else v for v in a), **k)
    )
    triposr = _source("triposr", "examples/captured.jpeg")
    image = Image.open(triposr / "examples/captured.jpeg").convert("RGB")
    expected = np.asarray(wrapper(image.copy()))
    monkeypatch.setattr(torch.Tensor, "to", original_to)

    remover = Trellis2ImageTo3DPipeline.background_removal.remover().to("cpu")
    assert np.array_equal(np.asarray(remover(image)), expected)


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS") != "1", reason="set DIFFUSERS_3D_BIREFNET_REAL_WEIGHTS=1"
)
def test_triposg_rmbg14_removal_matches_pinned_load_image(monkeypatch, tmp_path):
    """TripoSG's ``load_image`` with ``briaai/RMBG-1.4`` on an image above its 2000 px limit, on the CPU."""

    cv2 = import_reference_dependency("cv2")
    import_reference_dependency("skimage")
    from diffusers_3d.data.background import has_ambiguous_alpha
    from diffusers_3d.data.rmbg import RMBG14_REVISION, BriaRMBG, RMBG14BackgroundRemover
    from diffusers_3d.families.triposg.pipeline import _downscale_to_2000, prepare_triposg_image

    root = _source("triposg", "scripts/image_process.py", "scripts/briarmbg.py")
    sys.path.insert(0, str(root / "scripts"))
    try:
        image_process = _module("background_reference_triposg_image_process", root / "scripts/image_process.py")
        briarmbg = _module("background_reference_triposg_briarmbg", root / "scripts/briarmbg.py")
    finally:
        sys.path.remove(str(root / "scripts"))
    from huggingface_hub import snapshot_download

    weights = snapshot_download(
        "briaai/RMBG-1.4", revision=RMBG14_REVISION, allow_patterns=["*.json", "*.safetensors"]
    )
    upstream = briarmbg.BriaRMBG.from_pretrained(weights).eval()
    ours = BriaRMBG.from_pretrained()
    assert upstream.state_dict().keys() == ours.state_dict().keys()

    photo = Image.open(_source("triposr", "examples/captured.jpeg") / "examples/captured.jpeg").convert("RGB")
    path = tmp_path / "photo.png"
    photo.resize((2400, 1800), Image.Resampling.BICUBIC).save(path)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    expected = image_process.prepare_image(str(path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=upstream)

    (condition,) = as_image_conditions(path)
    condition = remove_background(
        condition, RMBG14BackgroundRemover(ours), trigger=has_ambiguous_alpha, resize=_downscale_to_2000
    )
    assert cv2 is not None and np.array_equal(np.asarray(prepare_triposg_image(condition)), np.asarray(expected))
