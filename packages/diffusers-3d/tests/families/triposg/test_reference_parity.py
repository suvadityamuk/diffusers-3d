from __future__ import annotations

import importlib
import os
import sys
import types
import typing
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers_3d import (
    TRIPOSG_REFERENCE_REVISION,
    TripoSGImageTo3DPipeline,
    TripoSGTransformerModel,
    TripoSGVAEDecoder,
)
from diffusers_3d._reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
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

    ``diso`` (CUDA dual marching cubes, only used by the FlashVDM path), ``trimesh`` (only wraps the output), and
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
    model = reference.transformer(**TripoSGTransformerModel.tiny_config()).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return model


def _reference_vae(reference):
    config = TripoSGVAEDecoder.tiny_config()
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
    config = {key: value for key, value in reference_model.config.items() if key in model_type.tiny_config()}
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


def test_hierarchical_extraction_matches_pinned_reference_on_an_analytic_field():
    """A steep sphere at 128 -> 256 cells exercises float16 index truncation over a realistic band."""

    reference = _load_reference()

    def field(points: torch.Tensor) -> torch.Tensor:
        return (50 * (0.5 - points.float().norm(dim=-1, keepdim=True))).to(points.dtype)

    # ``extract_mesh`` only touches ``self.vae``; stand the analytic field in for the decoder.
    host = types.SimpleNamespace(
        vae=lambda latents, points, num_chunks: types.SimpleNamespace(sample=field(points.to(latents))),
    )
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
    assert torch.equal(mesh.vertices, torch.from_numpy(expected_vertices))
    assert torch.equal(mesh.faces, torch.from_numpy(expected_faces).long())


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
