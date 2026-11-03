from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image

from diffusers_3d import ImageCondition
from diffusers_3d.families.idol import (
    IDOL_REFERENCE_REVISION,
    IdolImageTo3DPipeline,
    decode_idol_camera,
    prepare_idol_image,
)
from diffusers_3d.families.idol.conversion import convert_idol_decoder_state_dict, convert_idol_transformer_state_dict
from diffusers_3d.families.idol.pipeline import float16_blocks
from diffusers_3d.families.triposplat import TripoSplatGaussianDecoder, TripoSplatTransformerModel

from .configs import tiny_decoder_config, tiny_transformer_config, tiny_upstream_config

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_IDOL_REFERENCE_ROOT", "/tmp/idol-sdk"))
REFERENCE_REPOSITORY = "https://github.com/speridlabs/idol-sdk.git"
_PACKAGE = "idol_reference"


def _load_reference() -> SimpleNamespace:
    """Import the pinned upstream ``idol`` package under a private name (it needs tensordict and plyfile)."""

    if _PACKAGE not in sys.modules:
        try:
            validate_reference_checkout(
                REFERENCE_ROOT,
                expected_revision=IDOL_REFERENCE_REVISION,
                expected_repository=REFERENCE_REPOSITORY,
                expected_paths=("idol/__init__.py", "idol/models/dit.py", "idol/models/decoder.py"),
            )
            for dependency in ("tensordict", "plyfile", "torchvision.ops"):
                import_reference_dependency(dependency)
        except ReferenceCheckoutError as error:
            try:
                pytest.skip(reference_unavailable(error))
            except ReferenceCheckoutError as required_error:
                pytest.fail(str(required_error), pytrace=False)
        package = REFERENCE_ROOT / "idol"
        spec = importlib.util.spec_from_file_location(
            _PACKAGE, package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        spec.loader.exec_module(module)
    modules = ("camera", "conditioning", "gaussian", "preprocess", "sampler", "models.dit", "models.decoder")
    return SimpleNamespace(**{name.split(".")[-1]: sys.modules[f"{_PACKAGE}.{name}"] for name in modules})


def _float32(module: torch.nn.Module) -> torch.nn.Module:
    """Undo upstream's hard-coded float16 transformer blocks, for CPU float32 comparisons."""

    module.float().dtype = torch.float32
    return module.eval()


def _randomize(module: torch.nn.Module, seed: int, std: float = 0.2) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return module


def _upstream_denoiser(reference, *, float16_blocks=False):
    args = tiny_upstream_config()["denoiser"]["args"]
    denoiser = reference.dit.LatentSeqMMFlowModel(**args).eval()
    denoiser = _randomize(denoiser if float16_blocks else _float32(denoiser), seed=10)
    # The release stores TripoSplat's scrambled Sobol positions.
    positions = torch.quasirandom.SobolEngine(3, scramble=True, seed=123).draw(args["q_token_length"])
    denoiser.pos_pe.copy_(positions[None])
    return denoiser


def _upstream_decoders(reference, *, float16_blocks=False):
    config = tiny_upstream_config()
    octree = reference.decoder.OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]).eval()
    gaussian = reference.decoder.GaussianFixedlenDecoder(**config["decoder_gs"]["args"]).eval()
    if not float16_blocks:
        octree, gaussian = _float32(octree), _float32(gaussian)
    _randomize(octree, seed=11), _randomize(gaussian, seed=12)
    # The release stores the per-Gaussian offset pattern and base offset scale TripoSplat computes.
    released = TripoSplatGaussianDecoder(**tiny_decoder_config())
    gaussian.points_offset_perturbation.copy_(released.offset_perturbation)
    gaussian.base_offset_scale.copy_(released.base_offset_scale)
    return octree, gaussian


def _converted_transformer(upstream, *, half_blocks=False) -> TripoSplatTransformerModel:
    model = TripoSplatTransformerModel(**tiny_transformer_config()).eval()
    model.load_state_dict(convert_idol_transformer_state_dict(upstream.state_dict()), strict=True)
    return float16_blocks(model) if half_blocks else model


def _converted_decoder(octree, gaussian, *, half_blocks=False) -> TripoSplatGaussianDecoder:
    model = TripoSplatGaussianDecoder(**tiny_decoder_config()).eval()
    model.load_state_dict(convert_idol_decoder_state_dict(octree.state_dict(), gaussian.state_dict()), strict=True)
    return float16_blocks(model) if half_blocks else model


def _rgba(size=(30, 40)) -> np.ndarray:
    """An RGBA image with a soft-edged opaque blob off centre."""

    width, height = size
    rgba = np.random.default_rng(3).integers(0, 256, (height, width, 4), dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[8:31, 4:20, 3] = 255
    rgba[7, 4:20, 3] = rgba[31, 4:20, 3] = 128
    return rgba


def _condition(rgba: np.ndarray) -> ImageCondition:
    return ImageCondition(image=torch.from_numpy(rgba).permute(2, 0, 1).float() / 255)


class _Flux2VAE:
    """``diffusers.AutoencoderKLFlux2``'s ``encode``/``bn``/``config`` around a TripoSplatImageVAE.

    This Diffusers build has no AutoencoderKLFlux2; its ``encode`` is ``encoder -> quant_conv`` wrapped in a
    ``DiagonalGaussianDistribution``, which the shim reproduces with the same modules.
    """

    def __init__(self, vae):
        self.vae, self.bn, self.config = vae, vae.bn, SimpleNamespace(batch_norm_eps=vae.bn.eps)

    def encode(self, images):
        moments = self.vae.quant_conv(self.vae.encoder(images))
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(moments))


@pytest.mark.parametrize("half_blocks", [False, True])
def test_transformer_matches_upstream_denoiser(half_blocks):
    """``half_blocks`` keeps upstream's own float16 block linears, as the default load does."""

    reference = _load_reference()
    upstream = _upstream_denoiser(reference, float16_blocks=half_blocks)
    model = _converted_transformer(upstream, half_blocks=half_blocks)
    config = tiny_transformer_config()
    generator = torch.Generator().manual_seed(19)
    latents = torch.randn(2, config["num_latent_tokens"], config["in_channels"], generator=generator)
    camera = torch.randn(2, 1, config["camera_channels"], generator=generator)
    context = torch.randn(2, 7, config["cross_attention_dim"], generator=generator)
    context_2 = torch.randn(2, 7, config["cross_attention_dim_2"], generator=generator)
    timestep = torch.tensor([1000.0, 250.0])
    tensordict = import_reference_dependency("tensordict")
    with torch.no_grad():
        expected = upstream(
            tensordict.TensorDict({"latent": latents, "camera": camera}, batch_size=2),
            timestep,
            tensordict.TensorDict({"feature1": context, "feature2": context_2}, batch_size=2),
        )
        actual = model(latents, camera, timestep, context, context_2)
    torch.testing.assert_close(actual.sample, expected["latent"], atol=0, rtol=0)
    torch.testing.assert_close(actual.camera, expected["camera"], atol=0, rtol=0)


@pytest.mark.parametrize("half_blocks", [False, True])
def test_octree_sampler_gaussian_decoder_and_activations_match_upstream(half_blocks):
    reference = _load_reference()
    octree, gaussian = _upstream_decoders(reference, float16_blocks=half_blocks)
    model = _converted_decoder(octree, gaussian, half_blocks=half_blocks)
    config = tiny_decoder_config()
    latents = torch.randn(1, 6, config["latent_channels"], generator=torch.Generator().manual_seed(20))
    num_points = 7
    with torch.no_grad():
        # Upstream draws from the global RNG; a CPU generator seeded alike yields the same stream.
        torch.manual_seed(21)
        expected_points = reference.decoder.OctreeProbabilityFixedlenDecoder.sample(
            octree, latents, num_points=num_points, level=config["octree_depth"]
        )
        points = model.sample_points(latents, num_points, torch.Generator().manual_seed(21))
        expected_features = gaussian(points, latents)
        features = model.gaussian_features(points, latents)
        expected = gaussian.to_representation(points, expected_features)[0]
        asset = model(latents, num_points * config["gaussians_per_token"], generator=torch.Generator().manual_seed(21))
        asset = asset.assets[0]
    torch.testing.assert_close(points, expected_points, atol=0, rtol=0)
    torch.testing.assert_close(features, expected_features, atol=0, rtol=0)
    # Upstream ``Gaussian`` holds activated values; the asset holds log scales and opacity logits.
    torch.testing.assert_close(asset.means, expected.xyz, atol=0, rtol=0)
    torch.testing.assert_close(asset.log_scales.exp(), expected.scaling)
    torch.testing.assert_close(asset.quaternions_wxyz, expected.rotation, atol=0, rtol=0)
    torch.testing.assert_close(torch.sigmoid(asset.opacity_logits), expected.opacity)
    torch.testing.assert_close(asset.sh_coefficients, expected.features_dc, atol=0, rtol=0)


def test_saved_ply_is_upstream_export_turned_upright(tmp_path):
    """Upstream writes ``(x, -z, y)``, upside down in Y-up viewers; ``GaussianSplatAsset.save`` writes ``(x, z, -y)``.

    The two differ by a half turn about x: ``(x, y, z) -> (x, -y, -z)`` and ``q -> (0, 1, 0, 0) * q``.
    """

    reference = _load_reference()
    plyfile = import_reference_dependency("plyfile")
    octree, gaussian = _upstream_decoders(reference)
    model = _converted_decoder(octree, gaussian)
    latents = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(22))
    with torch.no_grad():
        points = model.sample_points(latents, 5, torch.Generator().manual_seed(23))
        expected = gaussian.to_representation(points, gaussian(points, latents))[0]
        asset = model._assets(points, model.gaussian_features(points, latents))[0]
    expected.save_ply(str(tmp_path / "upstream.ply"))
    asset.save(tmp_path / "ours.ply")
    upstream, ours = (plyfile.PlyData.read(str(tmp_path / f"{name}.ply"))["vertex"] for name in ("upstream", "ours"))
    for name in ("x", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"):
        np.testing.assert_allclose(ours[name], upstream[name], atol=1e-5, rtol=1e-5, err_msg=name)
    for name in ("y", "z"):
        np.testing.assert_allclose(ours[name], -upstream[name], atol=1e-6, rtol=0, err_msg=name)
    w, x, y, z = (np.asarray(upstream[f"rot_{index}"]) for index in range(4))
    turned = np.stack([-x, w, -z, y], axis=1)
    ours_rotation = np.stack([np.asarray(ours[f"rot_{index}"]) for index in range(4)], axis=1)
    np.testing.assert_allclose(np.abs((ours_rotation * turned).sum(1)), 1.0, atol=1e-5)


@pytest.mark.parametrize("mode", ["synthetic", "real"])
def test_image_preparation_matches_upstream_preprocess(mode):
    reference = _load_reference()
    rgba = _rgba()
    margin = reference.preprocess.CROP_MARGIN[mode]
    expected = reference.preprocess.preprocess_image(Image.fromarray(rgba, "RGBA"), None, 64, margin)
    actual = prepare_idol_image(_condition(rgba), image_size=64, crop_margin=margin)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_image_tokens_match_upstream_conditioning(tiny_idol_components):
    reference = _load_reference()
    # Upstream feeds the Flux.2 VAE bfloat16 pixels; a bfloat16 VAE makes both sides run the same operations.
    components = {**tiny_idol_components, "vae": tiny_idol_components["vae"].to(torch.bfloat16)}
    pipeline = IdolImageTo3DPipeline(**components)
    encoder = object.__new__(reference.conditioning.ImageEncoder)
    encoder.device, encoder.dinov3, encoder.flux2_vae = (
        torch.device("cpu"),
        pipeline.image_encoder,
        _Flux2VAE(pipeline.vae),
    )
    rgba = _rgba()
    prepared = reference.preprocess.preprocess_image(Image.fromarray(rgba, "RGBA"), None, 32, 1.2)
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None]
    torch.manual_seed(24)
    expected = encoder(pixels)
    image_embeds, latent_embeds = pipeline.encode_image(
        _condition(rgba), image_size=32, generator=torch.Generator().manual_seed(24)
    )
    torch.testing.assert_close(image_embeds, expected["feature1"], atol=0, rtol=0)
    torch.testing.assert_close(latent_embeds, expected["feature2"], atol=0, rtol=0)


def test_pipeline_sampling_matches_upstream_guided_shifted_euler(tiny_idol_components):
    """Upstream ``sample_flow`` with its defaults (shift 3, ``cfg_strength`` 3, sequential guidance passes)."""

    reference = _load_reference()
    tensordict = import_reference_dependency("tensordict")
    upstream = _upstream_denoiser(reference)
    pipeline = IdolImageTo3DPipeline(**{**tiny_idol_components, "transformer": _converted_transformer(upstream)})
    pipeline.set_progress_bar_config(disable=True)
    config = tiny_transformer_config()
    generator = torch.Generator().manual_seed(25)
    latents = torch.randn(1, config["num_latent_tokens"], config["in_channels"], generator=generator)
    camera = torch.randn(1, 1, config["camera_channels"], generator=generator)
    condition = _condition(_rgba())
    image_embeds, latent_embeds = pipeline.encode_image(
        condition, image_size=32, generator=torch.Generator().manual_seed(26)
    )
    cond = tensordict.TensorDict({"feature1": image_embeds, "feature2": latent_embeds}, batch_size=1)
    noise = tensordict.TensorDict({"latent": latents, "camera": camera}, batch_size=1)
    with torch.no_grad():
        expected = reference.sampler.sample_flow(
            upstream,
            noise,
            cond,
            cond.apply(torch.zeros_like),
            steps=3,
            rescale_t=3.0,
            cfg_strength=3.0,
            verbose=False,
        )
    output = pipeline(
        condition,
        latents=latents,
        camera_latents=camera,
        num_inference_steps=3,
        num_gaussians=8,
        image_size=32,
        generator=torch.Generator().manual_seed(26),
    )
    # The stock scheduler steps float32 sigmas (timestep 599.9999 for upstream's 600.0) where upstream steps float64
    # ones, and guidance runs as one batch here; both drift about 5e-5 after 3 steps. A wrong guidance mapping
    # (``cfg_strength`` 3 read as ``guidance_scale`` 3) differs by order 1.
    torch.testing.assert_close(output.latents.latents, expected["latent"], atol=2e-4, rtol=1e-4)
    expected_camera = reference.camera.decode_camera(expected["camera"][0, 0])
    assert decode_idol_camera(expected["camera"][0, 0]) == expected_camera
    for name, value in output.objects[0].metadata["camera"].items():
        # The radius is 1 / predicted inverse distance, which magnifies the drift above.
        assert value == pytest.approx(expected_camera[name], rel=1e-3), name
