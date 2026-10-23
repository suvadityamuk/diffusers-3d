from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers_3d import (
    HUNYUAN3D_REFERENCE_REVISION,
    Hunyuan3DDiTModel,
    Hunyuan3DFlowMatchEulerDiscreteScheduler,
    Hunyuan3DShapeVAE,
    ImageCondition,
)
from diffusers_3d._reference import ReferenceCheckoutError, reference_unavailable, validate_reference_checkout
from diffusers_3d.families.hunyuan3d.conversion import convert_transformer_state_dict, convert_vae_state_dict

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_HUNYUAN3D_REFERENCE_ROOT", "/tmp/Hunyuan3D-2.1"))
REFERENCE_REPOSITORY = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
REFERENCE_PATHS = (
    "hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py",
    "hy3dshape/hy3dshape/models/autoencoders/model.py",
    "hy3dshape/hy3dshape/models/autoencoders/surface_extractors.py",
    "hy3dshape/hy3dshape/models/autoencoders/volume_decoders.py",
    "hy3dshape/hy3dshape/models/conditioner.py",
    "hy3dshape/hy3dshape/models/denoisers/hunyuandit.py",
    "hy3dshape/hy3dshape/models/denoisers/moe_layers.py",
    "hy3dshape/hy3dshape/pipelines.py",
    "hy3dshape/hy3dshape/preprocessors.py",
    "hy3dshape/hy3dshape/schedulers.py",
)
REFERENCE_PACKAGE = "_diffusers_3d_hunyuan3d_reference"
# Released options this port fixes, passed explicitly because the upstream constructor defaults differ.
RELEASED_TRANSFORMER_OPTIONS = {
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": False,
    "with_decoupled_ca": False,
    "use_attention_pooling": False,
    "use_pos_emb": False,
}
RELEASED_VAE_OPTIONS = {"include_pi": False, "qkv_bias": False, "qk_norm": True, "num_encoder_layers": 1}


class _NoTimer(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reference_unavailable(error: ReferenceCheckoutError) -> None:
    try:
        reason = reference_unavailable(error)
    except ReferenceCheckoutError as required_error:
        pytest.fail(str(required_error), pytrace=False)
    pytest.skip(reason)


def _package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _reference(module: str) -> types.ModuleType:
    """Import a pinned ``hy3dshape`` module without its package ``__init__`` files or optional training deps."""

    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=HUNYUAN3D_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
    except ReferenceCheckoutError as error:
        _reference_unavailable(error)
    if REFERENCE_PACKAGE not in sys.modules:
        source = REFERENCE_ROOT / "hy3dshape" / "hy3dshape"
        _package(REFERENCE_PACKAGE, source)
        _package(f"{REFERENCE_PACKAGE}.models", source / "models")
        _package(f"{REFERENCE_PACKAGE}.models.denoisers", source / "models" / "denoisers")
        # Timing and checkpoint-download helpers only; the timer is used both as decorator and context manager.
        utils = types.ModuleType(f"{REFERENCE_PACKAGE}.utils")
        utils.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
        utils.synchronize_timer = lambda *args, **kwargs: _NoTimer()
        utils.smart_load_model = None
        sys.modules[utils.__name__] = utils
        # moe_layers.py imports timm's ViT layers without using them; stub them only when timm is missing.
        try:
            importlib.import_module("timm.models.vision_transformer")
        except ImportError:
            timm = types.ModuleType("timm.models.vision_transformer")
            timm.PatchEmbed = timm.Attention = timm.Mlp = None
            sys.modules.setdefault("timm", types.ModuleType("timm"))
            sys.modules.setdefault("timm.models", types.ModuleType("timm.models"))
            sys.modules.setdefault(timm.__name__, timm)
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(f"{REFERENCE_PACKAGE}.{module}")
    except ImportError as error:
        _reference_unavailable(ReferenceCheckoutError(f"pinned reference dependency unavailable: {error}"))


def _randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator) * 0.2)
    return module.eval()


def _reference_transformer():
    config = Hunyuan3DDiTModel.tiny_config()
    reference_type = _reference("models.denoisers.hunyuandit").HunYuanDiTPlain
    reference = _randomize(reference_type(**config, **RELEASED_TRANSFORMER_OPTIONS, text_len=5), seed=0)
    model = Hunyuan3DDiTModel(**config).eval()
    model.load_state_dict(convert_transformer_state_dict(reference.state_dict(), config["num_heads"]), strict=True)
    return reference, model


def _reference_vae():
    config = Hunyuan3DShapeVAE.tiny_config()
    reference_type = _reference("models.autoencoders").ShapeVAE
    reference = _randomize(reference_type(**config, **RELEASED_VAE_OPTIONS), seed=1)
    vae = Hunyuan3DShapeVAE(**config).eval()
    vae.load_state_dict(convert_vae_state_dict(reference.state_dict(), config["heads"]), strict=True)
    return reference, vae


def test_tiny_transformer_matches_pinned_reference():
    reference, model = _reference_transformer()
    generator = torch.Generator().manual_seed(2)
    latents = torch.randn(2, 6, 8, generator=generator)
    timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 5, 24, generator=generator)
    with torch.no_grad():
        expected = reference(latents, timestep, {"main": context})
        actual = model(latents, timestep, context).sample
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiny_vae_decoder_matches_pinned_reference():
    reference, vae = _reference_vae()
    generator = torch.Generator().manual_seed(3)
    latents = torch.randn(2, 6, 8, generator=generator)
    points = torch.rand(2, 50, 3, generator=generator) * 2 - 1
    with torch.no_grad():
        expected_tokens = reference.decode(latents)
        expected = reference.geo_decoder(queries=points, latents=expected_tokens).squeeze(-1)
        actual_tokens = vae(latents).sample
        actual = vae.query_occupancy(actual_tokens, points)
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_scheduler_matches_pinned_reference():
    reference = _reference("schedulers").FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    # Upstream's pipeline passes ``np.linspace(0, 1, steps)`` as custom sigmas.
    reference.set_timesteps(sigmas=np.linspace(0, 1, 50))
    scheduler.set_timesteps(50)
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)


def _rgba_image(case: str) -> np.ndarray:
    """A smooth-edged foreground on transparency, larger (shrunk) or smaller (enlarged) than the target box."""

    height, width, radius_y, radius_x = (900, 1000, 445, 490) if case == "shrink" else (300, 260, 70, 49)
    rows, columns = np.mgrid[:height, :width].astype(np.float32)
    distance = np.hypot((rows - height / 2) / radius_y, (columns - width / 2) / radius_x)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = (rows / height * 255).astype(np.uint8)
    rgba[..., 1] = (columns / width * 255).astype(np.uint8)
    rgba[..., 2] = ((np.sin(rows / 7) * np.cos(columns / 11) + 1) * 127).astype(np.uint8)
    rgba[..., 3] = (np.clip((1.0 - distance) * 8, 0, 1) * 255).astype(np.uint8)
    return rgba


@pytest.mark.parametrize("case", ["shrink", "enlarge"])
def test_preprocessing_matches_pinned_opencv_reference(case, tiny_hunyuan3d_pipeline):
    reference = _reference("preprocessors").ImageProcessorV2(size=512, border_ratio=0.15)
    rgba = _rgba_image(case)
    expected = reference(Image.fromarray(rgba))["image"][0]
    pipeline = tiny_hunyuan3d_pipeline
    pipeline.register_to_config(image_size=512)
    actual = pipeline.preprocess(ImageCondition(image=torch.from_numpy(rgba).permute(2, 0, 1).float() / 255))[0]
    # OpenCV's vectorized fixed-point rounding (and its platform-specific SIMD paths) can differ from the NumPy port
    # by one level on rare pixels: 14 of 786,432 values (shrink) and 1 (enlarge) with OpenCV 5.0 on arm64 macOS.
    levels = ((actual - expected) * 127.5).abs().round()
    assert levels.max() <= 1
    assert (levels > 0).float().mean() < 1e-2


def test_tiny_pipeline_matches_pinned_reference_end_to_end(tiny_hunyuan3d_pipeline):
    reference_transformer, transformer = _reference_transformer()
    reference_vae, vae = _reference_vae()
    pipeline = tiny_hunyuan3d_pipeline
    pipeline.transformer, pipeline.vae = transformer, vae
    conditioner_config = pipeline.image_encoder.config.to_dict()
    reference_conditioner = _reference("models.conditioner").SingleImageEncoder(
        main_image_encoder={
            "type": "DinoImageEncoder",
            "kwargs": {"config": conditioner_config, "image_size": 28, "use_cls_token": True},
        }
    )
    reference_conditioner.main_image_encoder.model.load_state_dict(pipeline.image_encoder.state_dict())
    reference_pipeline = _reference("pipelines").Hunyuan3DDiTFlowMatchingPipeline(
        vae=reference_vae,
        model=reference_transformer,
        scheduler=_reference("schedulers").FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        conditioner=reference_conditioner,
        image_processor=None,
        device="cpu",
        dtype=torch.float32,
    )
    condition = ImageCondition(image=torch.from_numpy(_rgba_image("enlarge")).permute(2, 0, 1).float() / 255)
    images = pipeline.preprocess(condition)
    arguments = {"num_inference_steps": 4, "octree_resolution": 24, "num_chunks": 1000}
    expected_latents = reference_pipeline(
        image=images, mask=images, generator=torch.Generator().manual_seed(4), output_type="latent", **arguments
    )
    output = pipeline(condition, generator=torch.Generator().manual_seed(4), **arguments)
    torch.testing.assert_close(output.latents.latents, expected_latents, atol=1e-6, rtol=1e-6)

    # Decode upstream's latents with both stacks. A grid value within rounding of the iso-level can change mesh
    # topology across BLAS builds, so extraction and export are compared exactly on upstream's own grid.
    bounds = (-1.01,) * 3 + (1.01,) * 3
    with torch.no_grad():
        decoded = reference_vae(1.0 / reference_vae.scale_factor * expected_latents)
        expected_grid = reference_vae.volume_decoder(
            decoded, reference_vae.geo_decoder, bounds=1.01, octree_resolution=24, num_chunks=1000, enable_pbar=False
        )
        grid = pipeline.decode_occupancy(expected_latents, bounds, octree_resolution=24, num_chunks=1000)
    torch.testing.assert_close(grid, expected_grid, atol=1e-5, rtol=1e-5)
    extracted = reference_vae.surface_extractor(expected_grid, mc_level=0.0, bounds=1.01, octree_resolution=24)
    expected_mesh = _reference("pipelines").export_to_trimesh(extracted)[0]
    mesh = pipeline.extract_mesh(expected_grid[0], bounds, mc_level=0.0)
    assert torch.equal(mesh.faces, torch.from_numpy(np.asarray(expected_mesh.faces)).to(torch.int64))
    assert torch.equal(mesh.vertices, torch.from_numpy(np.asarray(expected_mesh.vertices)).float())
