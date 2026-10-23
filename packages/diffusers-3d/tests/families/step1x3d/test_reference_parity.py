from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import os
import sys
import types
import typing
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import ModelMixin
from PIL import Image
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from diffusers_3d import (
    ImageCondition,
    Step1X3DFlowMatchEulerScheduler,
    Step1X3DImageTo3DPipeline,
    Step1X3DShapeDecoder,
    Step1X3DTransformer1DModel,
    preprocess_step1x3d_image,
)
from diffusers_3d._reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from diffusers_3d.families.step1x3d.conversion import (
    STEP1X3D_REFERENCE_REVISION,
    convert_step1x3d_transformer_state_dict,
    convert_step1x3d_vae_state_dict,
)
from diffusers_3d.families.step1x3d.vae import step1x3d_grid_points

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_STEP1X3D_REFERENCE_ROOT", "/tmp/Step1X-3D"))
REFERENCE_REPOSITORY = "https://github.com/stepfun-ai/Step1X-3D.git"
REFERENCE_PATHS = (
    "step1x3d_geometry/models/attention.py",
    "step1x3d_geometry/models/attention_processor.py",
    "step1x3d_geometry/models/transformers/flux_transformer_1d.py",
    "step1x3d_geometry/models/autoencoders/michelangelo_autoencoder.py",
    "step1x3d_geometry/models/autoencoders/transformers/attention.py",
    "step1x3d_geometry/models/autoencoders/transformers/perceiver_1d.py",
    "step1x3d_geometry/models/autoencoders/transformers/utils.py",
    "step1x3d_geometry/models/autoencoders/volume_decoders.py",
    "step1x3d_geometry/models/autoencoders/surface_extractors.py",
    "step1x3d_geometry/models/conditional_encoders/base.py",
    "step1x3d_geometry/models/conditional_encoders/dinov2_encoder.py",
    "step1x3d_geometry/models/conditional_encoders/dinov2_with_registers/modeling_dinov2_with_registers.py",
    "step1x3d_geometry/models/pipelines/pipeline.py",
    "step1x3d_geometry/models/pipelines/pipeline_utils.py",
    "step1x3d_geometry/utils/checkpoint.py",
)
_REFERENCE: types.SimpleNamespace | None = None

TINY_TRANSFORMER = {
    "input_channels": 4,
    "width": 16,
    "layers": 1,
    "num_single_layers": 2,
    "num_heads": 2,
    "condition_dim": 12,
    "use_visual_condition": True,
    "visual_condition_dim": 12,
}
TINY_VAE = {
    "use_downsample": True,
    "num_latents": 8,
    "point_feats": 3,
    "embed_dim": 4,
    "num_freqs": 2,
    "include_pi": False,
    "width": 16,
    "heads": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 2,
    "qkv_bias": False,
    "qk_norm": False,
    "use_ln_post": True,
    "use_flash": True,
    "use_checkpoint": False,
    # Released default. Below ``min_resolution`` (63) it evaluates one dense linspace grid, like ours.
    "volume_decoder_type": "hierarchical",
}
TINY_DINO = {
    "model_type": "dinov2_with_registers",
    "hidden_size": 12,
    "num_hidden_layers": 1,
    "num_attention_heads": 3,
    "mlp_ratio": 2,
    "image_size": 28,
    "patch_size": 14,
    "num_register_tokens": 2,
}
DINO_PROCESSOR = {
    "image_processor_type": "BitImageProcessor",
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "rescale_factor": 1 / 255,
    "size": {"shortest_edge": 256},
}


def _validate_reference() -> None:
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=STEP1X3D_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        for dependency in ("einops", "skimage", "trimesh"):
            import_reference_dependency(dependency)
    except ReferenceCheckoutError as error:
        try:
            reason = reference_unavailable(error)
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
        pytest.skip(reason)


def _module(name: str, path: Path | None = None, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    module.__dict__.update(attributes)
    return module


@dataclasses.dataclass
class _Config:
    weights: str | None = None


class _BaseModule(ModelMixin):
    """Stand-in for upstream's omegaconf-backed ``BaseModule``: parse the dataclass config, then ``configure``."""

    Config = _Config

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        self.cfg = self.Config(**(cfg or {}))
        self.configure()

    def configure(self) -> None:
        pass


def _passthrough(*args, **kwargs):
    return lambda function: function


@contextlib.contextmanager
def _upstream_import_environment():
    """Expose the pinned ``step1x3d_geometry`` sources while stubbing training-only and CUDA-only imports.

    Package ``__init__`` files are bypassed (they import pytorch_lightning, every encoder, and the training
    systems). ``timm``, ``pymeshlab``, and ``rembg`` are imported but never called on the inference path.
    The vendored DINOv2 module predates Transformers 5, so the helpers it imports are shimmed while it loads.
    """

    root = REFERENCE_ROOT / "step1x3d_geometry"
    typing_names = {name: getattr(typing, name) for name in typing.__all__}
    stubs = {
        "step1x3d_geometry": _module("step1x3d_geometry", root, register=lambda name: lambda cls: cls),
        "step1x3d_geometry.utils": _module("step1x3d_geometry.utils", root / "utils"),
        "step1x3d_geometry.utils.typing": _module("step1x3d_geometry.utils.typing", **typing_names),
        "step1x3d_geometry.utils.base": _module("step1x3d_geometry.utils.base", BaseModule=_BaseModule),
        "step1x3d_geometry.utils.config": _module(
            "step1x3d_geometry.utils.config", ExperimentConfig=None, load_config=None
        ),
        "step1x3d_geometry.utils.misc": _module(
            "step1x3d_geometry.utils.misc", get_world_size=lambda: 1, get_device=lambda: "cpu"
        ),
        "timm": _module("timm"),
        "timm.models": _module("timm.models"),
        "timm.models.vision_transformer": _module("timm.models.vision_transformer", Attention=None),
        "pymeshlab": _module("pymeshlab", MeshSet=None),
        "rembg": _module("rembg"),
    }
    for package in (
        "models",
        "models/autoencoders",
        "models/autoencoders/transformers",
        "models/transformers",
        "models/pipelines",
        "models/conditional_encoders",
        "models/conditional_encoders/dinov2",
        "models/conditional_encoders/dinov2_with_registers",
    ):
        name = "step1x3d_geometry." + package.replace("/", ".")
        stubs[name] = _module(name, root / package)
    for unused in ("t5_encoder", "label_encoder"):
        name = f"step1x3d_geometry.models.conditional_encoders.{unused}"
        stubs[name] = _module(name, T5Encoder=None, LabelEncoder=None)

    import diffusers.loaders
    import transformers.pytorch_utils
    import transformers.utils

    shims = {
        diffusers.loaders: {
            name: type(name, (), {})
            for name in (
                "FluxIPAdapterMixin",
                "FluxLoraLoaderMixin",
                "FromSingleFileMixin",
                "TextualInversionLoaderMixin",
            )
        },
        transformers.pytorch_utils: {"find_pruneable_heads_and_indices": None, "prune_linear_layer": None},
        transformers.utils: dict.fromkeys(
            (
                "add_code_sample_docstrings",
                "add_start_docstrings",
                "add_start_docstrings_to_model_forward",
                "replace_return_docstrings",
            ),
            _passthrough,
        ),
    }
    added = [(module, name) for module, values in shims.items() for name in values if not hasattr(module, name)]
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    for module, name in added:
        setattr(module, name, shims[module][name])
    try:
        yield
    finally:
        for module, name in added:
            delattr(module, name)
        for name in [name for name in sys.modules if name.startswith("step1x3d_geometry")]:
            del sys.modules[name]
        for name, value in saved.items():
            if value is not None:
                sys.modules[name] = value
            else:
                sys.modules.pop(name, None)


def _load_pinned_reference() -> types.SimpleNamespace:
    global _REFERENCE
    _validate_reference()
    if _REFERENCE is None:
        with _upstream_import_environment():
            prefix = "step1x3d_geometry.models."
            autoencoder = importlib.import_module(prefix + "autoencoders.michelangelo_autoencoder")
            _REFERENCE = types.SimpleNamespace(
                FluxDenoiser=importlib.import_module(prefix + "transformers.flux_transformer_1d").FluxDenoiser,
                MichelangeloAutoencoder=autoencoder.MichelangeloAutoencoder,
                volume_decoders=importlib.import_module(prefix + "autoencoders.volume_decoders"),
                MCSurfaceExtractor=importlib.import_module(
                    prefix + "autoencoders.surface_extractors"
                ).MCSurfaceExtractor,
                Dinov2Encoder=importlib.import_module(prefix + "conditional_encoders.dinov2_encoder").Dinov2Encoder,
                preprocess_image=importlib.import_module(prefix + "pipelines.pipeline_utils").preprocess_image,
                Step1X3DGeometryPipeline=importlib.import_module(
                    prefix + "pipelines.pipeline"
                ).Step1X3DGeometryPipeline,
            )
    return _REFERENCE


def _reference_vae(reference, seed: int = 0):
    torch.manual_seed(seed)
    vae = reference.MichelangeloAutoencoder(TINY_VAE).eval()
    # Centre the tiny field on zero so marching cubes finds a surface.
    with torch.no_grad():
        grid = reference.volume_decoders.VanillaVolumeDecoder()(
            vae.decode(torch.randn(1, 8, 4)), vae.query, bounds=1.05, octree_resolution=8, enable_pbar=False
        )[0]
        vae.decoder.output_proj.bias -= grid.median()
    return vae


def _converted_vae(reference_vae) -> Step1X3DShapeDecoder:
    vae = Step1X3DShapeDecoder(
        num_latents=8, latent_channels=4, width=16, num_attention_heads=2, num_layers=2, num_freqs=2
    ).eval()
    state, _ = convert_step1x3d_vae_state_dict(reference_vae.state_dict(), num_attention_heads=2)
    vae.load_state_dict(state, strict=True)
    return vae


def _reference_dinov2_encoder(reference, tmp_path: Path):
    from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil

    folder = tmp_path / "tiny-dinov2-reg"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(TINY_DINO))
    (folder / "preprocessor_config.json").write_text(json.dumps(DINO_PROCESSOR))
    torch.manual_seed(1)
    encoder = reference.Dinov2Encoder({"image_size": 28, "dino_type": str(folder)}).eval()
    # The pinned Transformers 4.48 resolved this repository to the PIL ``BitImageProcessor``.
    encoder.image_preprocess_dino = BitImageProcessorPil(**DINO_PROCESSOR)
    encoder.dino_model.get_head_mask = lambda head_mask, layers: [None] * layers
    # The vendored upstream module keeps the Hub's DINOv2 key layout, which Transformers maps onto its own.
    image_encoder, loading_info = Dinov2WithRegistersModel.from_pretrained(
        None,
        config=Dinov2WithRegistersConfig.from_dict(TINY_DINO),
        state_dict={key.removeprefix("dino_model."): value for key, value in encoder.state_dict().items()},
        output_loading_info=True,
    )
    assert not any(loading_info.values())
    return encoder, image_encoder.eval()


def _rgba_image() -> Image.Image:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(30, 22, 4), dtype=np.uint8)
    pixels[..., 3] = 0
    pixels[5:24, 4:17, 3] = 255
    pixels[5:8, 4:17, 3] = 128
    return Image.fromarray(pixels, "RGBA")


def test_flow_transformer_matches_pinned_reference():
    reference = _load_pinned_reference()
    torch.manual_seed(0)
    upstream = reference.FluxDenoiser(TINY_TRANSFORMER).eval()
    ours = Step1X3DTransformer1DModel(
        in_channels=4, width=16, num_layers=1, num_single_layers=2, num_attention_heads=2, cross_attention_dim=12
    ).eval()
    ours.load_state_dict(convert_step1x3d_transformer_state_dict(upstream.state_dict()), strict=True)
    latents, context = torch.randn(2, 8, 4), torch.randn(2, 5, 12)
    timestep = torch.tensor([1000.0, 250.5])
    with torch.no_grad():
        expected = upstream(latents, timestep, visual_condition=context, return_dict=False)[0]
        actual = ours(latents, timestep, context).sample
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_shape_decoder_and_dense_grid_match_pinned_reference():
    reference = _load_pinned_reference()
    upstream = _reference_vae(reference)
    ours = _converted_vae(upstream)
    torch.manual_seed(2)
    latents, points = torch.randn(1, 8, 4), torch.rand(1, 50, 3) * 2 - 1
    with torch.no_grad():
        tokens = upstream.decode(latents)
        torch.testing.assert_close(ours.decode(latents), tokens, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            ours.query(points, ours.decode(latents)), upstream.query(points, tokens), atol=0.0, rtol=0.0
        )

        vanilla = reference.volume_decoders.VanillaVolumeDecoder()(
            tokens, upstream.query, bounds=1.05, octree_resolution=16, num_chunks=1000, enable_pbar=False
        )[0]
        dense = ours.decode_grid(latents, bounds=1.05, resolution=16, chunk_size=1000)
        torch.testing.assert_close(dense, vanilla, atol=0.0, rtol=0.0)

        # The released hierarchical decoder evaluates a near-surface band (|logit| < 0.95 or a sign change) and
        # leaves the rest NaN; the dense grid must agree wherever it evaluated (fine levels place points at
        # index * step rather than linspace, hence the tolerance). Steeper logits make that band a real subset.
        for projection in (upstream.decoder.output_proj, ours.proj_out):
            projection.weight *= 40
            projection.bias *= 40
        hierarchical = reference.volume_decoders.HierarchicalVolumeDecoder()(
            upstream.decode(latents), upstream.query, bounds=1.05, octree_resolution=128, enable_pbar=False
        )
        dense = ours.decode_grid(latents, bounds=1.05, resolution=128)
        evaluated = torch.isfinite(hierarchical)
        assert 0.01 < evaluated.float().mean() < 0.6
        torch.testing.assert_close(dense[evaluated], hierarchical[evaluated], atol=2e-5, rtol=1e-5)


def test_surface_extraction_matches_pinned_reference_bitwise(tiny_step1x3d_pipeline, sphere_field, monkeypatch):
    reference = _load_pinned_reference()
    field = sphere_field(32) + 0.013 * torch.sin(7 * step1x3d_grid_points(1.05, 32)).sum(-1).view(1, 33, 33, 33)
    monkeypatch.setattr(tiny_step1x3d_pipeline.vae, "decode_grid", lambda latents, **kwargs: field)
    expected = reference.MCSurfaceExtractor()(field, mc_level=0.0, bounds=1.05, octree_resolution=32)[0]
    actual = tiny_step1x3d_pipeline.extract_meshes(torch.zeros(1, 8, 4), octree_resolution=32)[0]
    assert torch.equal(actual.vertices, expected.verts)
    assert torch.equal(actual.faces, expected.faces)


def test_conditioner_and_preprocessing_match_pinned_reference(tmp_path, tiny_step1x3d_components):
    reference = _load_pinned_reference()
    encoder, image_encoder = _reference_dinov2_encoder(reference, tmp_path)
    pipeline = Step1X3DImageTo3DPipeline(**{**tiny_step1x3d_components, "image_encoder": image_encoder})
    ours_pixels = []
    image_encoder.register_forward_pre_hook(lambda module, args: ours_pixels.append(args[0]))
    image = _rgba_image()
    framed = reference.preprocess_image(image, foreground_ratio=0.95)
    with torch.no_grad():
        expected = encoder.encode_image(framed)
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        images = preprocess_step1x3d_image(ImageCondition(pixels), image_size=28)[None]
        actual, negative = pipeline.encode_conditioning(images)
        reference_pixels = encoder.image_preprocess_dino.preprocess(
            framed, return_tensors="pt", do_rescale=True, do_resize=True, size=28, crop_size=28
        ).pixel_values
    torch.testing.assert_close(ours_pixels[0], reference_pixels, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.equal(negative, encoder.empty_image_embeds)


def test_image_to_mesh_pipeline_matches_pinned_reference(tmp_path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    reference = _load_pinned_reference()
    encoder, image_encoder = _reference_dinov2_encoder(reference, tmp_path)
    torch.manual_seed(3)
    transformer = reference.FluxDenoiser(TINY_TRANSFORMER).eval()
    vae = _reference_vae(reference, seed=4)
    upstream = reference.Step1X3DGeometryPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        transformer=transformer,
        visual_encoder=encoder,
        caption_encoder=None,
        label_encoder=None,
        visual_eature_extractor=None,
    )
    ours_transformer = Step1X3DTransformer1DModel(
        in_channels=4, width=16, num_layers=1, num_single_layers=2, num_attention_heads=2, cross_attention_dim=12
    ).eval()
    ours_transformer.load_state_dict(convert_step1x3d_transformer_state_dict(transformer.state_dict()), strict=True)
    ours = Step1X3DImageTo3DPipeline(
        image_encoder=image_encoder,
        transformer=ours_transformer,
        vae=_converted_vae(vae),
        scheduler=Step1X3DFlowMatchEulerScheduler(),
    )
    image_path = tmp_path / "input.png"
    _rgba_image().save(image_path)
    pixels = torch.from_numpy(np.asarray(_rgba_image()).copy()).permute(2, 0, 1).float() / 255
    arguments = {"guidance_scale": 7.5, "num_inference_steps": 4}

    expected_latents = upstream(
        str(image_path), generator=torch.Generator().manual_seed(2025), output_type="latent", **arguments
    ).mesh
    expected_mesh = upstream(
        str(image_path),
        generator=torch.Generator().manual_seed(2025),
        output_type="raw",
        octree_resolution=16,
        bounds=1.05,
        **arguments,
    ).mesh[0]
    actual = ours(
        ImageCondition(pixels),
        generator=torch.Generator().manual_seed(2025),
        octree_resolution=16,
        **arguments,
    )
    torch.testing.assert_close(actual.latents.latents, expected_latents, atol=0.0, rtol=0.0)
    assert torch.equal(actual.objects[0].faces, expected_mesh.faces)
    assert torch.equal(actual.objects[0].vertices, expected_mesh.verts)
