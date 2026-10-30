from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from _configs import tiny_config
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from PIL import Image

from diffusers_3d import Hunyuan3DOmniImageTo3DPipeline, ImageCondition
from diffusers_3d.families.hunyuan3d.conversion import convert_transformer_state_dict, convert_vae_state_dict
from diffusers_3d.families.hunyuan3d.models import Hunyuan3DDiTModel
from diffusers_3d.families.hunyuan3d.scheduler import Hunyuan3DFlowMatchEulerDiscreteScheduler
from diffusers_3d.families.hunyuan3d.vae import Hunyuan3DShapeVAE
from diffusers_3d.families.hunyuan3d_omni.conversion import (
    HUNYUAN3D_OMNI_REFERENCE_REVISION,
    convert_control_encoder_state_dict,
)

from .conftest import make_tiny_omni_pipeline

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_HUNYUAN3D_OMNI_REFERENCE_ROOT", "/tmp/Hunyuan3D-Omni"))
REFERENCE_REPOSITORY = "https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git"
REFERENCE_PATHS = (
    "hy3dshape/models/autoencoders/inference_utils/extract_geometry_vanilla.py",
    "hy3dshape/models/autoencoders/shape_vae.py",
    "hy3dshape/models/conditioners/omni_encoder.py",
    "hy3dshape/models/denoisers/hunyuandit.py",
    "hy3dshape/models/diffusion/transport/transport.py",
    "hy3dshape/pipelines/pipeline_generation_sit_omni.py",
    "hy3dshape/preprocessors.py",
    "inference.py",
)
# Released options this port fixes, passed explicitly because the upstream constructor defaults differ.
RELEASED_TRANSFORMER_OPTIONS = {
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": False,
    "with_decoupled_ca": False,
    "use_attention_pooling": False,
    "use_pos_emb": False,
}
CONTROLS = {
    "skeleton": ("pose", torch.linspace(-0.6, 0.7, 5 * 6).reshape(1, 5, 6)),
    "bbox": ("bbox", torch.tensor([[[0.8, 0.64, 1.0]]])),
    "voxels": ("voxel", torch.linspace(-0.95, 0.9, 40 * 3).reshape(1, 40, 3).sin()),
    "point_cloud": ("point", torch.linspace(-0.9, 0.9, 7 * 3).reshape(1, 7, 3)),
}


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


def _instantiate_from_config(config, **kwargs):
    module, name = config["target"].rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**{**kwargs, **config.get("params", {})})


def _reference(module: str) -> types.ModuleType:
    """Import a pinned ``hy3dshape`` module without its package ``__init__`` files or training-only deps."""

    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=HUNYUAN3D_OMNI_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
    except ReferenceCheckoutError as error:
        _reference_unavailable(error)
    source = REFERENCE_ROOT / "hy3dshape"
    if getattr(sys.modules.get("hy3dshape"), "__path__", None) != [str(source)]:
        # Upstream imports itself as ``hy3dshape``; the package __init__ files pull in Lightning and every extractor.
        for name in (
            "",
            ".models",
            ".models.conditioners",
            ".models.denoisers",
            ".models.autoencoders",
            ".models.autoencoders.inference_utils",
            ".models.diffusion",
            ".pipelines",
        ):
            _package(f"hy3dshape{name}", source / name.lstrip(".").replace(".", "/"))
        # misc.py imports OmegaConf only to read config files.
        misc = types.ModuleType("hy3dshape.models.utils.misc")
        misc.instantiate_from_config = _instantiate_from_config
        misc.get_config_from_file = None
        sys.modules[misc.__name__] = misc
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
        return importlib.import_module(f"hy3dshape.{module}")
    except ImportError as error:
        _reference_unavailable(ReferenceCheckoutError(f"pinned reference dependency unavailable: {error}"))


def _randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator) * 0.2)
    return module.eval()


@pytest.fixture
def pipeline():
    return make_tiny_omni_pipeline()


@pytest.fixture
def reference_encoder(pipeline, tmp_path):
    """Upstream's ``OmniEncoder`` around the tiny DINOv2, with the tiny control encoder's weights."""

    pipeline.image_encoder.save_pretrained(tmp_path / "dinov2")
    config = pipeline.control_encoder.config
    reference = _reference("models.conditioners.omni_encoder").OmniEncoder(
        image_encoder={
            "target": "hy3dshape.models.conditioners.omni_encoder.DinoImageEncoder",
            "params": {"version": str(tmp_path / "dinov2"), "image_size": 28, "use_cls_token": True},
        },
        width=config.width,
        voxel_resolution=config.voxel_resolution,
        num_freqs=config.num_freqs,
        include_pi=False,
    )
    reference.disable_drop = True
    for name, value in reference.state_dict().items():
        if not name.startswith("image_encoder."):
            torch.nn.init.normal_(value, std=0.5, generator=torch.Generator().manual_seed(len(name)))
    state_dict = {key: value for key, value in reference.state_dict().items() if not key.startswith("image_encoder.")}
    pipeline.control_encoder.load_state_dict(convert_control_encoder_state_dict(state_dict), strict=True)
    return reference.eval()


@pytest.fixture
def reference_transformer(pipeline):
    config = tiny_config(Hunyuan3DDiTModel)
    reference_type = _reference("models.denoisers.hunyuandit").HunYuanDiTPlain
    reference = _randomize(reference_type(**config, **RELEASED_TRANSFORMER_OPTIONS, text_len=5), seed=0)
    converted = convert_transformer_state_dict(reference.state_dict(), config["num_heads"])
    pipeline.transformer.load_state_dict(converted, strict=True)
    return reference


@pytest.fixture
def reference_vae(pipeline):
    config = tiny_config(Hunyuan3DShapeVAE)
    reference = _reference("models.autoencoders.shape_vae").ShapeVAE(
        device=None,
        dtype=None,
        num_latents=config["num_latents"],
        train_num_latents_range=[config["num_latents"]],
        train_num_latents_probs=[1.0],
        downsample_ratio=1,
        point_feats=3,
        sharpedge_label=True,
        embed_dim=config["embed_dim"],
        num_freqs=config["num_freqs"],
        include_pi=False,
        width=config["width"],
        heads=config["heads"],
        num_encoder_layers=1,
        num_decoder_layers=config["num_decoder_layers"],
        qkv_bias=False,
        qk_norm=True,
        pc_size=config["num_latents"],
    )
    reference = _randomize(reference, seed=1)
    reference.scale_factor = config["scale_factor"]
    pipeline.vae.load_state_dict(convert_vae_state_dict(reference.state_dict(), config["heads"]), strict=True)
    return reference


@pytest.mark.parametrize("control_type", list(CONTROLS))
def test_tiny_conditioning_matches_pinned_reference(control_type, pipeline, reference_encoder):
    upstream_name, control = CONTROLS[control_type]
    images = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.no_grad():
        expected = reference_encoder(image=images, **{upstream_name: control})["cond"]
        actual = pipeline.encode_conditions(images, control, control_type, do_classifier_free_guidance=False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiny_transformer_matches_pinned_reference(pipeline, reference_transformer):
    generator = torch.Generator().manual_seed(2)
    latents = torch.randn(2, 6, 8, generator=generator)
    timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 9, 24, generator=generator)
    with torch.no_grad():
        expected = reference_transformer(latents, timestep, {"cond": context})
        actual = pipeline.transformer(latents, timestep, context).sample
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiny_vae_decoder_matches_pinned_reference(pipeline, reference_vae):
    generator = torch.Generator().manual_seed(3)
    latents = torch.randn(2, 6, 8, generator=generator)
    points = torch.rand(2, 50, 3, generator=generator) * 2 - 1
    with torch.no_grad():
        expected_tokens = reference_vae.decode(latents)
        expected = reference_vae.query_geometry(points, expected_tokens)
        actual_tokens = pipeline.vae(latents).sample
        actual = pipeline.vae.query_occupancy(actual_tokens, points)
    torch.testing.assert_close(actual_tokens, expected_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_scheduler_matches_the_pinned_euler_transport_sampler():
    transport = _reference("models.diffusion.transport")
    sampler = transport.Sampler(transport.create_transport(path_type="Linear", prediction="velocity"))
    sample_fn = sampler.sample_ode(num_steps=6, sampling_method="euler")

    def velocity(sample, timestep):
        return torch.cos(3 * timestep)[:, None] * sample + timestep[:, None]

    start = torch.linspace(-1, 1, 2 * 4).reshape(2, 4)
    expected = sample_fn(start, lambda sample, timestep: velocity(sample, timestep))[-1]
    scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(6)
    sample = start
    for timestep in scheduler.timesteps:
        sample = scheduler.step(velocity(sample, (timestep / 1000).expand(2)), timestep, sample).prev_sample
    torch.testing.assert_close(sample, expected, atol=1e-6, rtol=1e-6)


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
def test_preprocessing_matches_pinned_opencv_reference(case, pipeline):
    rgba = _rgba_image(case)
    expected = _reference("preprocessors").ImageProcessorV2(size=512)(Image.fromarray(rgba))[0]
    pipeline.register_to_config(image_size=512)
    actual = pipeline.preprocess(ImageCondition(image=torch.from_numpy(rgba).permute(2, 0, 1).float() / 255))[0]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("control_type", ["point_cloud", "skeleton"])
def test_tiny_pipeline_matches_pinned_reference_end_to_end(
    control_type, pipeline, reference_encoder, reference_transformer, reference_vae
):
    transport = _reference("models.diffusion.transport")
    reference_pipeline = _reference("pipelines.pipeline_generation_sit_omni").Hunyuan3DOmniSiTFlowMatchingPipeline(
        vae=types.SimpleNamespace(latent_shape=(6, 8)),
        model=reference_transformer,
        scheduler=transport.Sampler(transport.create_transport(path_type="Linear", prediction="velocity")),
        cond_encoder=reference_encoder,
        device="cpu",
        dtype=torch.float32,
    )
    upstream_name, control = CONTROLS[control_type]
    condition = ImageCondition(image=torch.from_numpy(_rgba_image("enlarge")).permute(2, 0, 1).float() / 255)
    images = pipeline.preprocess(condition)
    expected_latents = reference_pipeline(
        image=images,
        num_inference_steps=4,
        guidance_scale=4.5,
        generator=torch.Generator().manual_seed(4),
        output_type="latent",
        **{upstream_name: control},
    )["shapes"][0]
    # The pymeshlab clean-up is compared on its own below.
    arguments = {"num_inference_steps": 4, "octree_resolution": 24, "num_chunks": 1000, "postprocess": False}
    output = pipeline(condition, generator=torch.Generator().manual_seed(4), **{control_type: control[0]}, **arguments)
    torch.testing.assert_close(output.latents.latents, expected_latents, atol=1e-5, rtol=1e-5)

    # Upstream always rounds the grid points to float16 before querying. Compare the occupancy at those points, then
    # run upstream's own extractor and face flip on this port's grid so extraction and scaling compare exactly.
    bounds = (-1.01,) * 3 + (1.01,) * 3
    extractor = _reference("models.autoencoders.inference_utils.extract_geometry_vanilla")
    points, _, _ = extractor.VanillaGeometryExtractor(torch.device("cpu")).generate_dense_grid_points(
        np.array(bounds[:3]), np.array(bounds[3:]), octree_resolution=24
    )
    with torch.no_grad():
        decoded = reference_vae.decode(expected_latents / reference_vae.scale_factor)
        expected_grid = reference_vae.query_geometry(torch.from_numpy(points).half()[None], decoded)
        grid = pipeline.decode_occupancy(
            expected_latents, bounds, octree_resolution=24, num_chunks=1000, query_dtype=torch.float16
        )
    torch.testing.assert_close(grid, expected_grid.view(1, 25, 25, 25), atol=1e-5, rtol=1e-5)
    _assert_vanilla_extraction_matches(extractor, grid[0], bounds)


def _assert_vanilla_extraction_matches(extractor, grid, bounds):
    chunks = iter(grid.reshape(1, -1).split(1000, dim=1))
    [(vertices, faces)], _ = extractor.extract_geometry_vanilla(
        device=torch.device("cpu"),
        geometric_func=lambda queries: next(chunks),
        bounds=1.01,
        octree_resolution=grid.shape[0] - 1,
        num_chunks=1000,
        disable_tqdm=True,
        mc_level=0.0,
    )
    expected_mesh = _reference("pipelines.utils").export_to_trimesh(
        types.SimpleNamespace(mesh_v=vertices, mesh_f=faces)
    )
    mesh = Hunyuan3DOmniImageTo3DPipeline.extract_mesh(grid, bounds, mc_level=0.0, postprocess_passes=0)
    assert torch.equal(mesh.faces, torch.from_numpy(np.asarray(expected_mesh.faces)).to(torch.int64))
    assert torch.equal(mesh.vertices, torch.from_numpy(np.asarray(expected_mesh.vertices)).float())


def test_vertex_merging_matches_pinned_vanilla_extractor():
    """Samples exactly on the level make marching cubes emit duplicate vertices, which upstream's trimesh merges."""

    extractor = _reference("models.autoencoders.inference_utils.extract_geometry_vanilla")
    axis = torch.linspace(-1, 1, 25)
    grid = 0.6 - torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij")).norm(dim=0)
    grid[:, :, 12] = 0.0
    _assert_vanilla_extraction_matches(extractor, grid, (-1.01,) * 3 + (1.01,) * 3)
    measure = import_reference_dependency("skimage.measure")
    vertices = measure.marching_cubes(grid.numpy(), 0.0, method="lewiner")[0]
    assert len(np.unique(vertices, axis=0)) < len(vertices), "the field must exercise vertex merging"


@pytest.mark.parametrize("passes", [1, 2])
def test_clean_up_matches_pinned_inference_script(passes):
    """``FloaterRemover`` then ``DegenerateFaceRemover``, once (``postprocess``) or twice (``infer_bbox``)."""

    import_reference_dependency("pymeshlab")
    postprocessors = _reference("postprocessors")
    extractor = _reference("models.autoencoders.inference_utils.extract_geometry_vanilla")
    axis = torch.linspace(-1, 1, 49)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    # A wavy body, a blob over 0.5% of the faces that stays, and an 8-face octahedron around one sample that goes.
    body = 0.6 - points.norm(dim=-1) + 0.05 * torch.sin(9 * points).sum(-1)
    kept = 0.06 - (points - torch.tensor([0.8, 0.8, 0.0])).norm(dim=-1)
    removed = 0.02 - (points - points[16, 40, 24]).norm(dim=-1)
    grid = torch.stack([body, kept, removed]).amax(dim=0)
    chunks = iter(grid.reshape(1, -1).split(1000, dim=1))
    [(vertices, faces)], _ = extractor.extract_geometry_vanilla(
        device=torch.device("cpu"),
        geometric_func=lambda queries: next(chunks),
        bounds=1.01,
        octree_resolution=48,
        num_chunks=1000,
        disable_tqdm=True,
        mc_level=0.0,
    )
    expected = _reference("pipelines.utils").export_to_trimesh(types.SimpleNamespace(mesh_v=vertices, mesh_f=faces))
    for _ in range(passes):
        expected = postprocessors.DegenerateFaceRemover()(postprocessors.FloaterRemover()(expected))
    bounds = (-1.01,) * 3 + (1.01,) * 3
    raw = Hunyuan3DOmniImageTo3DPipeline.extract_mesh(grid, bounds, mc_level=0.0, postprocess_passes=0)
    once = Hunyuan3DOmniImageTo3DPipeline.extract_mesh(grid, bounds, mc_level=0.0)
    mesh = Hunyuan3DOmniImageTo3DPipeline.extract_mesh(grid, bounds, mc_level=0.0, postprocess_passes=passes)
    assert len(raw.faces) - len(once.faces) == 8, "only the octahedron is a floater"
    assert torch.equal(mesh.faces, torch.from_numpy(np.asarray(expected.faces)).to(torch.int64))
    assert torch.equal(mesh.vertices, torch.from_numpy(np.asarray(expected.vertices)).float())


def test_default_octree_resolution_matches_the_pinned_inference_script():
    """Upstream's ``inference.py`` passes ``octree_resolution=512`` for every control; the port defaults to it."""

    import inspect
    import re

    _reference("pipelines.utils")
    script = (REFERENCE_ROOT / "inference.py").read_text(encoding="utf-8")
    values = set(re.findall(r"octree_resolution=(\d+)", script))
    default = inspect.signature(Hunyuan3DOmniImageTo3DPipeline.__call__).parameters["octree_resolution"].default
    assert values == {str(default)} == {"512"}
