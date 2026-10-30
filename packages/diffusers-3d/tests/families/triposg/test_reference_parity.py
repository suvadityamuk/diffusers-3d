from __future__ import annotations

import importlib
import os
import sys
import types
import typing
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
from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers_3d import (
    TRIPOSG_REFERENCE_REVISION,
    TripoSGImageTo3DPipeline,
    TripoSGTransformerModel,
    TripoSGVAEDecoder,
)

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRIPOSG_REFERENCE_ROOT", "/tmp/TripoSG"))
REFERENCE_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSG.git"
REFERENCE_PATHS = (
    "triposg/inference_utils.py",
    "triposg/models/attention_processor.py",
    "triposg/models/embeddings.py",
    "triposg/models/autoencoders/autoencoder_kl_triposg.py",
    "triposg/models/transformers/triposg_transformer.py",
    "triposg/pipelines/pipeline_triposg.py",
    "triposg/schedulers/scheduling_rectified_flow.py",
    "scripts/image_process.py",
)
BOUNDS = 1.005
_REFERENCE: types.SimpleNamespace | None = None


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _load_reference() -> types.SimpleNamespace:
    """Import the pinned TripoSG sources on CPU.

    ``diso`` (CUDA dual marching cubes, replaced per test), ``trimesh`` (only wraps the output), and
    the jaxtyping/omegaconf/typeguard type-alias module are stubbed. ``expand_edge_region_fast`` hard-codes CUDA,
    so that single allocation is redirected to the input's device.
    """

    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=TRIPOSG_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        for dependency in ("einops", "scipy.ndimage", "skimage.measure"):
            import_reference_dependency(dependency)
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)

    source_root = REFERENCE_ROOT / "triposg"
    # Upstream modules keep their own references to the stubs, so the global names are restored afterwards.
    stubbed = ("diso", "trimesh", "triposg.utils.typing")
    saved = {name: sys.modules.get(name) for name in stubbed}
    try:
        _module("diso", DiffDMC=None)
        _module("trimesh", Trimesh=lambda vertices, faces: (vertices, faces))
        _module("triposg", __path__=[str(source_root)])
        _module("triposg.utils.typing", **{name: getattr(typing, name) for name in typing.__all__})
        path = source_root / "inference_utils.py"
        source = path.read_text(encoding="utf-8")
        assert source.count("device='cuda'") == 1
        inference_utils = _module("triposg.inference_utils", __file__=str(path))
        patched = source.replace("device='cuda'", "device=edge_coords.device")
        exec(compile(patched, str(path), "exec"), vars(inference_utils))

        _REFERENCE = types.SimpleNamespace(
            inference_utils=inference_utils,
            hierarchical_extract_geometry=inference_utils.hierarchical_extract_geometry,
            transformer=importlib.import_module("triposg.models.transformers").TripoSGDiTModel,
            vae=importlib.import_module("triposg.models.autoencoders").TripoSGVAEModel,
            scheduler=importlib.import_module("triposg.schedulers").RectifiedFlowScheduler,
            pipeline=importlib.import_module("triposg.pipelines.pipeline_triposg").TripoSGPipeline,
        )
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return _REFERENCE


def _reference_transformer(reference):
    torch.manual_seed(10)
    model = reference.transformer(**tiny_config(TripoSGTransformerModel)).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return model


def _reference_vae(reference):
    config = tiny_config(TripoSGVAEDecoder)
    torch.manual_seed(11)
    model = reference.vae(width_encoder=8, num_layers_encoder=1, **config).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    with torch.no_grad():
        model.decoder.proj_out.weight.mul_(50)
    return model


def _converted(model_type, reference_model):
    state_dict = {
        key.removeprefix("decoder."): value
        for key, value in reference_model.state_dict().items()
        if not key.startswith(("encoder.", "quant."))
    }
    config = {key: value for key, value in reference_model.config.items() if key in tiny_config(model_type)}
    model = model_type(**config).eval()
    model.load_state_dict(state_dict, strict=True)
    return model


def test_transformer_matches_pinned_reference():
    reference = _load_reference()
    upstream = _reference_transformer(reference)
    model = _converted(TripoSGTransformerModel, upstream)
    generator = torch.Generator().manual_seed(12)
    latents = torch.randn(2, 9, 4, generator=generator)
    context = torch.randn(2, 5, 12, generator=generator)
    timestep = torch.tensor([980.0, 20.0])
    with torch.no_grad():
        expected = upstream(latents, timestep, encoder_hidden_states=context).sample
        actual = model(latents, timestep, context).sample
    assert torch.equal(actual, expected)


def test_vae_decoder_matches_pinned_reference_with_chunked_queries():
    reference = _load_reference()
    upstream = _reference_vae(reference)
    model = _converted(TripoSGVAEDecoder, upstream)
    generator = torch.Generator().manual_seed(13)
    latents = torch.randn(1, 6, 4, generator=generator)
    points = torch.rand(1, 50, 3, generator=generator) * 2 - 1
    with torch.no_grad():
        expected = upstream.decode(latents, sampled_points=points, num_chunks=16).sample
        actual = model(latents, points, num_chunks=16).sample
    assert torch.equal(actual, expected)


def test_stock_flow_match_scheduler_reproduces_rectified_flow_steps():
    reference = _load_reference()
    upstream = reference.scheduler()
    upstream.set_timesteps(7)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(sigmas=[1.0 - index / 7 for index in range(7)])
    assert torch.equal(scheduler.timesteps, upstream.timesteps)
    generator = torch.Generator().manual_seed(14)
    expected = actual = torch.randn(2, 5, 4, generator=generator)
    for timestep in upstream.timesteps:
        velocity = torch.randn(2, 5, 4, generator=generator)
        expected = upstream.step(velocity, timestep, expected).prev_sample
        actual = scheduler.step(-velocity, timestep, actual).prev_sample
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("quantized", [False, True])
def test_hierarchical_extraction_matches_pinned_reference_on_an_analytic_field(quantized):
    """A steep sphere at 128 -> 256 cells exercises float16 index truncation over a realistic band. Quantizing it
    puts samples exactly on the level, so marching cubes emits duplicate vertices that trimesh merges."""

    reference = _load_reference()
    trimesh = import_reference_dependency("trimesh")

    def field(points: torch.Tensor) -> torch.Tensor:
        values = 50 * (0.5 - points.float().norm(dim=-1, keepdim=True))
        return (values.round() if quantized else values).to(points.dtype)

    # Extraction only calls ``self.vae``; stand the analytic field in for the decoder.
    host = TripoSGImageTo3DPipeline.__new__(TripoSGImageTo3DPipeline)
    vars(host)["vae"] = lambda latents, points, num_chunks: types.SimpleNamespace(sample=field(points.to(latents)))
    (expected_vertices, expected_faces), *_ = reference.hierarchical_extract_geometry(
        lambda points: field(points.float()),
        torch.device("cpu"),
        bounds=(-BOUNDS,) * 3 + (BOUNDS,) * 3,
        dense_octree_depth=7,
        hierarchical_octree_depth=8,
    )
    mesh = TripoSGImageTo3DPipeline.extract_mesh(
        host, torch.zeros(1, 1, 1), bounds=BOUNDS, dense_octree_depth=7, hierarchical_octree_depth=8
    )
    # Upstream's pipeline and inference script both wrap the samples in a processed trimesh.
    expected = trimesh.Trimesh(expected_vertices.astype(np.float32), np.ascontiguousarray(expected_faces))
    assert (len(expected.vertices) < len(expected_vertices)) == quantized
    assert torch.equal(mesh.vertices, torch.from_numpy(expected.vertices).float())
    assert torch.equal(mesh.faces, torch.from_numpy(expected.faces))


class _RecordingDiffDMC(torch.nn.Module):
    """CPU stand-in for DISO's ``DiffDMC``: records the signed distance and meshes it with marching cubes."""

    calls: list[torch.Tensor] = []

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        assert dtype == torch.float32

    def forward(self, sdf, deform=None, return_quads=False, normalize=True):
        assert deform is None and not return_quads and not normalize
        type(self).calls.append(sdf.clone())
        measure = import_reference_dependency("skimage.measure")
        vertices, faces, _, _ = measure.marching_cubes(torch.nan_to_num(sdf, nan=1.0).numpy(), 0.0)
        return torch.from_numpy(vertices.copy()), torch.from_numpy(faces.copy())


def _flash_extract(reference, monkeypatch, upstream_vae, host, latents, octree_depth):
    """Upstream ``flash_extract_geometry`` and ``extract_mesh(use_flash_decoder=True)``, both through the stub."""

    trimesh = import_reference_dependency("trimesh")
    _RecordingDiffDMC.calls = []
    monkeypatch.setattr(reference.inference_utils, "DiffDMC", _RecordingDiffDMC)
    monkeypatch.setitem(sys.modules, "diso", types.SimpleNamespace(DiffDMC=_RecordingDiffDMC))
    (vertices, faces), *_ = reference.inference_utils.flash_extract_geometry(
        latents, upstream_vae, bounds=(-BOUNDS,) * 3 + (BOUNDS,) * 3, octree_depth=octree_depth
    )
    assert vertices is not None
    mesh = TripoSGImageTo3DPipeline.extract_mesh(
        host, latents, bounds=BOUNDS, flash_octree_depth=octree_depth, use_flash_decoder=True
    )
    expected_sdf, actual_sdf = _RecordingDiffDMC.calls
    assert torch.equal(actual_sdf.isnan(), expected_sdf.isnan())
    assert torch.equal(torch.nan_to_num(actual_sdf), torch.nan_to_num(expected_sdf))
    # Upstream's pipeline wraps the samples in a processed trimesh.
    expected = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
    assert torch.equal(mesh.vertices, torch.from_numpy(expected.vertices).float())
    assert torch.equal(mesh.faces, torch.from_numpy(expected.faces))
    return expected_sdf


@pytest.mark.parametrize("octree_depth", [7, 8])
def test_flash_extraction_matches_pinned_reference_on_an_analytic_field(octree_depth, monkeypatch):
    """Upstream's default FlashVDM grid: 63 -> 126 (-> 252) cells, with the twice-dilated candidates and the
    signed distance handed to ``DiffDMC``. A ``DiffDMC`` stand-in meshes both grids alike."""

    reference = _load_reference()

    queries = {"upstream": [], "port": []}

    def field(points: torch.Tensor, latents: torch.Tensor, side: str) -> torch.Tensor:
        # The query shapes of every decoder call, which fix the GPU kernels and hence the CUDA results.
        queries[side].append(tuple(points.shape))
        return (50 * (0.5 - points.float().norm(dim=-1, keepdim=True))).to(latents.dtype).expand(*points.shape[:2], 1)

    upstream_vae = types.SimpleNamespace(
        decoder=types.SimpleNamespace(set_topk=lambda topk: None),
        decode=lambda latents, points: types.SimpleNamespace(sample=field(points, latents, "upstream")),
    )
    host = TripoSGImageTo3DPipeline.__new__(TripoSGImageTo3DPipeline)
    vars(host)["vae"] = lambda latents, points, num_chunks: types.SimpleNamespace(
        sample=field(points, latents, "port")
    )
    sdf = _flash_extract(reference, monkeypatch, upstream_vae, host, torch.zeros(1, 1, 1), octree_depth)
    assert queries["port"] == queries["upstream"]
    assert queries["port"][0] == (2, 16**3, 3) and len(queries["port"]) > 32 + octree_depth - 6
    assert sdf.shape == ((63 * 2 ** (octree_depth - 6) + 1),) * 3
    assert sdf.isnan().any() and (sdf < 0).any() and (sdf > 0).any()


def test_flash_extraction_matches_pinned_reference_with_the_vae_decoder(monkeypatch):
    """The tiny decoder through upstream's batched 4^3 coarse blocks and 6^3-binned refined queries."""

    reference = _load_reference()
    upstream_vae = _reference_vae(reference)
    upstream_vae.set_flash_decoder()
    host = TripoSGImageTo3DPipeline.__new__(TripoSGImageTo3DPipeline)
    vars(host)["vae"] = _converted(TripoSGVAEDecoder, upstream_vae)
    latents = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(16))
    sdf = _flash_extract(reference, monkeypatch, upstream_vae, host, latents, 7)
    assert sdf.isnan().any() and (sdf < 0).any() and (sdf > 0).any()


def test_pipeline_matches_pinned_reference_end_to_end(tiny_triposg_components, rgba_condition):
    """Same DINOv2 pixels, generator, and weights through upstream's ``use_flash_decoder=False`` path."""

    reference = _load_reference()
    upstream_transformer = _reference_transformer(reference)
    upstream_vae = _reference_vae(reference)
    components = {
        **tiny_triposg_components,
        "transformer": _converted(TripoSGTransformerModel, upstream_transformer),
        "vae": _converted(TripoSGVAEDecoder, upstream_vae),
    }
    pipeline = TripoSGImageTo3DPipeline(**components)
    upstream = reference.pipeline(
        vae=upstream_vae,
        transformer=upstream_transformer,
        scheduler=reference.scheduler(),
        image_encoder_dinov2=components["image_encoder"],
        feature_extractor_dinov2=components["feature_extractor"],
    )
    pipeline.set_progress_bar_config(disable=True)
    upstream.set_progress_bar_config(disable=True)
    arguments = {"num_inference_steps": 4, "num_tokens": 6, "dense_octree_depth": 4, "hierarchical_octree_depth": 5}
    (expected_vertices, expected_faces), *_ = upstream(
        pipeline.preprocess(rgba_condition),
        generator=torch.Generator().manual_seed(15),
        bounds=(-BOUNDS,) * 3 + (BOUNDS,) * 3,
        use_flash_decoder=False,
        **arguments,
    ).samples
    mesh = pipeline(rgba_condition, generator=torch.Generator().manual_seed(15), bounds=BOUNDS, **arguments).objects[0]
    assert torch.equal(mesh.vertices, torch.from_numpy(expected_vertices))
    assert torch.equal(mesh.faces, torch.from_numpy(expected_faces).long())


def _reference_load_image():
    """Upstream's ``scripts/image_process.load_image`` with its ``.cuda()`` transfers removed."""

    _load_reference()
    for dependency in ("cv2", "skimage.morphology", "torchvision"):
        import_reference_dependency(dependency)
    path = REFERENCE_ROOT / "scripts" / "image_process.py"
    module = types.ModuleType("_diffusers_3d_triposg_image_process")
    exec(compile(path.read_text(encoding="utf-8").replace(".cuda()", ""), str(path), "exec"), vars(module))
    return module.load_image


@pytest.mark.parametrize("size", [(300, 220), (2600, 1900)])
def test_image_preparation_matches_pinned_reference(size, tmp_path):
    """Crop, composite, and pad; images over 2000 pixels are first shrunk with ``INTER_AREA``."""

    from PIL import Image

    from diffusers_3d import ImageCondition
    from diffusers_3d.families.triposg.pipeline import prepare_triposg_image

    load_image = _reference_load_image()
    height, width = size
    rows, columns = np.mgrid[:height, :width].astype(np.float32)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = (rows / height * 255).astype(np.uint8)
    rgba[..., 1] = (columns / width * 255).astype(np.uint8)
    rgba[..., 2] = ((np.sin(rows / 7) * np.cos(columns / 11) + 1) * 127).astype(np.uint8)
    distance = np.hypot(rows / height - 0.45, columns / width - 0.55) * 4
    rgba[..., 3] = (np.clip((1.0 - distance) * 6, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgba).save(tmp_path / "input.png")

    expected = load_image(str(tmp_path / "input.png"), bg_color=np.array([1.0, 1.0, 1.0]))
    expected = (expected.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    actual = prepare_triposg_image(ImageCondition(torch.from_numpy(rgba).permute(2, 0, 1).float() / 255))
    assert np.array_equal(np.asarray(actual), expected)
