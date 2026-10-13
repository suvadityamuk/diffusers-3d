from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers_3d import (
    PIXAL3D_REFERENCE_REVISION,
    CameraRig,
    ImageCondition,
    MultiViewCondition,
    Pixal3DNAFUpsampler,
    Pixal3DSLatFlowModel,
    Pixal3DSparseStructureFlowModel,
    TrellisSparseTensor,
)
from diffusers_3d._reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)
from diffusers_3d.families.pixal3d.conditioner import camera_distance_from_fov, front_view_camera
from diffusers_3d.families.pixal3d.conversion import NAF_REFERENCE_REVISION, _naf_block_key, _rope_phases
from diffusers_3d.families.pixal3d.pipeline import _STAGE_CONDITIONING

from .conftest import TINY_IMAGE_SIZE, tiny_pipeline

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_PIXAL3D_REFERENCE_ROOT", "/tmp/Pixal3D"))
NAF_ROOT = Path(os.environ.get("DIFFUSERS_3D_NAF_REFERENCE_ROOT", "/tmp/NAF"))
PROJECTION_SOURCE = "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"
REFERENCE_PATHS = (
    "inference.py",
    "inference_mv.py",
    "pixal3d/pipelines/pixal3d_image_to_3d.py",
    "pixal3d/models/sparse_structure_flow.py",
    "pixal3d/models/structured_latent_flow.py",
    "pixal3d/modules/attention/proj_attention.py",
    "pixal3d/modules/sparse/attention/proj_attention.py",
    PROJECTION_SOURCE,
)
NAF_PATHS = ("src/model/naf.py", "src/layers/attentions.py", "src/layers/convolutions.py", "src/layers/rope.py")
PACKAGE = "_diffusers_3d_pixal3d_reference"
DENSE_CELLS = torch.stack(torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
SPARSE_CELLS = torch.tensor([[0, 0, 0], [1, 5, 2], [7, 7, 7], [3, 0, 6], [5, 5, 1]])


def _unavailable(error: ReferenceCheckoutError) -> None:
    try:
        reason = reference_unavailable(error)
    except ReferenceCheckoutError as required_error:
        pytest.fail(str(required_error), pytrace=False)
    pytest.skip(reason)


def _validate(root: Path, revision: str, repository: str, paths: tuple[str, ...]) -> None:
    try:
        validate_reference_checkout(
            root, expected_revision=revision, expected_repository=repository, expected_paths=paths
        )
    except ReferenceCheckoutError as error:
        _unavailable(error)


def _load_module(name: str, path: Path, *, package: bool = False):
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _definitions(path: str, names: set[str], namespace: dict, *, owner: str | None = None) -> dict:
    """Execute only the named top-level (or ``owner`` class) definitions, skipping heavy module imports."""

    body = ast.parse((REFERENCE_ROOT / path).read_text(encoding="utf-8")).body
    if owner is not None:
        body = next(node for node in body if getattr(node, "name", None) == owner).body
    body = [node for node in body if getattr(node, "name", None) in names]
    for node in body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def reference():
    _validate(REFERENCE_ROOT, PIXAL3D_REFERENCE_REVISION, "https://github.com/TencentARC/Pixal3D.git", REFERENCE_PATHS)
    # Pixal3D ships plain SDPA backends for both the dense and the sparse attention.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    source = REFERENCE_ROOT / "pixal3d"
    _package(PACKAGE, source)
    modules = _package(f"{PACKAGE}.modules", source / "modules")
    _package(f"{PACKAGE}.models", source / "models")
    _package(f"{PACKAGE}.utils", source / "utils")
    try:
        for name in ("utils", "spatial", "norm"):
            _load_module(f"{PACKAGE}.modules.{name}", source / "modules" / f"{name}.py")
        modules.sparse = _load_module(
            f"{PACKAGE}.modules.sparse", source / "modules" / "sparse" / "__init__.py", package=True
        )
        for name in ("attention", "transformer"):
            _load_module(f"{PACKAGE}.modules.{name}", source / "modules" / name / "__init__.py", package=True)
        _load_module(f"{PACKAGE}.utils.elastic_utils", source / "utils" / "elastic_utils.py")
        _load_module(f"{PACKAGE}.models.sparse_elastic_mixin", source / "models" / "sparse_elastic_mixin.py")
        dense = _load_module(f"{PACKAGE}.models.sparse_structure_flow", source / "models" / "sparse_structure_flow.py")
        sparse = _load_module(
            f"{PACKAGE}.models.structured_latent_flow", source / "models" / "structured_latent_flow.py"
        )
    except (ImportError, RuntimeError) as error:
        _unavailable(ReferenceCheckoutError(f"optional pinned reference dependency unavailable: {error}"))
    return types.SimpleNamespace(
        SparseStructureFlowModel=dense.SparseStructureFlowModel,
        SLatFlowModel=sparse.SLatFlowModel,
        SparseTensor=modules.sparse.SparseTensor,
    )


@pytest.fixture(scope="module")
def reference_naf():
    """Upstream NAF, its ``cutlass-fna`` neighborhood attention served by NATTEN's CPU Flex backend."""

    _validate(NAF_ROOT, NAF_REFERENCE_REVISION, "https://github.com/valeoai/NAF.git", NAF_PATHS)
    try:
        natten = import_reference_dependency("natten")
        import_reference_dependency("einops")
    except ReferenceCheckoutError as error:
        _unavailable(error)
    # NAF imports itself as the top-level ``src`` package; expose it only while loading.
    saved = {name: module for name, module in sys.modules.items() if name == "src" or name.startswith("src.")}
    try:
        _package("src", NAF_ROOT / "src")
        _load_module("src.layers", NAF_ROOT / "src" / "layers" / "__init__.py", package=True)
        attentions = sys.modules["src.layers.attentions"]
        naf = _load_module(f"{PACKAGE}_naf", NAF_ROOT / "src" / "model" / "naf.py")
    finally:
        for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(saved)

    def flex_na2d(query, key, value, kernel_size, dilation, stride=1, backend=None):
        # Flex needs equal Q/K and V head dims: attend zero-padded Q-sized chunks of V and drop the padding.
        width = query.shape[-1]
        chunks = []
        for chunk in value.split(width, dim=-1):
            padded = F.pad(chunk, (0, width - chunk.shape[-1]))
            output = natten.na2d(query, key, padded, kernel_size=kernel_size, dilation=dilation, backend="flex-fna")
            chunks.append(output[..., : chunk.shape[-1]])
        return torch.cat(chunks, dim=-1)

    attentions.na2d = flex_na2d
    return naf.NAF


def _randomize_state(module: torch.nn.Module, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # Parameters only: NAF's RoPE periods are a derived buffer.
        for value in module.parameters():
            value.copy_(torch.randn(value.shape, generator=generator, dtype=value.dtype) * 0.2)


def _random_naf(reference_naf, model: Pixal3DNAFUpsampler, *, seed: int) -> torch.nn.Module:
    config = model.config
    reference = reference_naf(
        dim=config.dim,
        heads_attn=config.num_heads,
        heads_rope=config.rope_heads,
        kernel_size=config.kernel_size,
        img_layers=config.encoder_layers,
    ).eval()
    _randomize_state(reference, seed=seed)
    state = {key: value for key, value in reference.state_dict().items() if not key.endswith("rope.periods")}
    model.load_state_dict({_naf_block_key(key): value for key, value in state.items()}, strict=True)
    return reference


def test_tiny_projected_flows_match_pinned_reference(reference):
    config = Pixal3DSparseStructureFlowModel.tiny_config()
    torch.manual_seed(0)
    reference_dense = reference.SparseStructureFlowModel(**config, image_attn_mode="proj").eval()
    _randomize_state(reference_dense, seed=1)
    dense = Pixal3DSparseStructureFlowModel(**config).eval()
    state = reference_dense.state_dict()
    # The released dense checkpoint stores RoPE phases; the converter drops them after this exact check.
    torch.testing.assert_close(state.pop("rope_phases"), _rope_phases(dense), atol=0.0, rtol=0.0)
    dense.load_state_dict(state, strict=True)
    assert tuple(dense.state_dict()) == tuple(state)

    generator = torch.Generator().manual_seed(2)
    latents = torch.randn(2, 2, 2, 2, 2, generator=generator)
    timesteps = torch.rand(2, generator=generator) * 1000
    context = (torch.randn(2, 3, 12, generator=generator), torch.randn(2, 8, 12, generator=generator))
    with torch.no_grad():
        expected = reference_dense(latents, timesteps, context)
        actual = dense(latents, timesteps, context).sample
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    config = Pixal3DSLatFlowModel.tiny_config(texture=True)
    torch.manual_seed(3)
    reference_sparse = reference.SLatFlowModel(**config, image_attn_mode="proj").eval()
    _randomize_state(reference_sparse, seed=4)
    sparse = Pixal3DSLatFlowModel(**config).eval()
    sparse.load_state_dict(reference_sparse.state_dict(), strict=True)
    assert tuple(sparse.state_dict()) == tuple(reference_sparse.state_dict())

    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 7, 0, 5], [1, 2, 1, 0], [1, 7, 7, 7]])
    features = [torch.randn(5, channels, generator=generator) for channels in (4, 4, 24)]
    reference_inputs = [reference.SparseTensor(value, coordinates.int()) for value in features]
    inputs = [TrellisSparseTensor(coordinates, value) for value in features]
    global_tokens = torch.randn(2, 3, 12, generator=generator)
    with torch.no_grad():
        expected = reference_sparse(
            reference_inputs[0], timesteps, (global_tokens, reference_inputs[2]), concat_cond=reference_inputs[1]
        )
        actual = sparse(inputs[0], timesteps, (global_tokens, inputs[2]), concat_cond=inputs[1]).sample
    torch.testing.assert_close(actual.features, expected.feats, atol=1e-6, rtol=1e-5)


def test_naf_upsampler_matches_pinned_reference(reference_naf):
    model = Pixal3DNAFUpsampler(dim=32, num_heads=2, rope_heads=2, kernel_size=3, encoder_layers=1).eval()
    reference = _random_naf(reference_naf, model, seed=5)
    generator = torch.Generator().manual_seed(6)
    # Clamped border windows, integer dilations 2 and 4, and the guide-downsizing branch (40 > 4 * 8).
    for image_size, feature_size, output_size in ((10, 5, 10), (16, 4, 16), (40, 4, 8)):
        image = torch.rand(2, 3, image_size, image_size, generator=generator)
        features = torch.randn(2, 40, feature_size, feature_size, generator=generator)
        with torch.no_grad():
            expected = reference(image, features, (output_size, output_size))
            actual = model(image, features, (output_size, output_size)).sample
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


class _TransformersV4Layout(torch.nn.Module):
    """The pinned extractor walks ``model.layer`` (Transformers 4.x); share the 5.x modules under that name."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embeddings, self.rope_embeddings, self.layer = model.embeddings, model.rope_embeddings, model.model.layer
        self.config = model.config


def _reference_extractors(pipeline, reference_naf) -> tuple[dict, torch.nn.Module]:
    try:
        transforms = import_reference_dependency("torchvision.transforms")
    except ReferenceCheckoutError as error:
        _unavailable(error)
    model = _TransformersV4Layout(pipeline.image_encoder)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "np": np,
        "Image": Image,
        "transforms": transforms,
        "DINOv3ViTModel": types.SimpleNamespace(from_pretrained=lambda name: model),
    }
    exec("from typing import *", namespace)
    names = {
        "project_points_to_image_batch",
        "sample_features",
        "ProjGrid",
        "ProjGridMV",
        "compute_relative_calc_mat",
        "DinoV3ProjFeatureExtractor",
        "DinoV3ProjMultiViewFeatureExtractor",
    }
    return _definitions(PROJECTION_SOURCE, names, namespace), _random_naf(
        reference_naf, pipeline.feature_upsampler, seed=7
    )


def _assert_stage(pipeline, views, encoded, extractor_type, naf, stage, grid, cells, images, **inputs):
    """Compare one stage's global tokens and pixel-aligned features with the pinned extractor at ``cells``."""

    image_scale, upsample_scale = _STAGE_CONDITIONING[stage]
    side = TINY_IMAGE_SIZE * image_scale
    extractor = extractor_type(
        model_name="dinov3",
        image_size=side,
        grid_resolution=grid,
        use_naf_upsample=upsample_scale is not None,
        naf_target_size=TINY_IMAGE_SIZE * (upsample_scale or 1),
    )
    extractor.naf_model = naf
    with torch.no_grad():
        expected_global, expected = extractor(images[side], **inputs)
        actual = pipeline.project_views(views, encoded, stage, [cells], grid_resolution=grid)[0]
    torch.testing.assert_close(encoded[side][0], expected_global, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual, expected.reshape(grid, grid, grid, -1)[tuple(cells.T)], atol=1e-6, rtol=1e-5)


def test_single_view_projection_conditioning_matches_pinned_reference(reference_naf):
    inference = _definitions("inference.py", {"compute_f_pixels", "distance_from_fov"}, {"torch": torch})
    preprocess = _definitions(
        "pixal3d/pipelines/pixal3d_image_to_3d.py",
        {"preprocess_image"},
        {"Image": Image, "np": np},
        owner="Pixal3DImageTo3DPipeline",
    )["preprocess_image"]
    pipeline = tiny_pipeline()
    reference, naf = _reference_extractors(pipeline, reference_naf)

    rgba = np.zeros((21, 26, 4), dtype=np.uint8)
    rgba[..., :3] = np.random.default_rng(8).integers(0, 256, (21, 26, 3), dtype=np.uint8)
    rgba[3:18, 4:22, 3] = np.linspace(40, 255, 15 * 18).astype(np.uint8).reshape(15, 18)
    camera_angle_x = 0.6
    distance = inference["distance_from_fov"](
        camera_angle_x, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, 511]), 1.0, 512
    )["distance_from_x"]
    assert camera_distance_from_fov(camera_angle_x) == distance

    sides = (TINY_IMAGE_SIZE, 2 * TINY_IMAGE_SIZE)
    condition = ImageCondition(torch.from_numpy(rgba).permute(2, 0, 1).float() / 255)
    views = pipeline.prepare_views(condition, image_sizes=sides, camera_angle_x=camera_angle_x)
    # Upstream crops with ``preprocess_image`` and the extractor LANCZOS-resizes the RGB result.
    cropped = preprocess(object(), Image.fromarray(rgba))
    images = {}
    for side in sides:
        resized = np.asarray(cropped.resize((side, side), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        images[side] = torch.from_numpy(resized).permute(2, 0, 1)[None]
        torch.testing.assert_close(views.images[side][:, 0], images[side], atol=0.0, rtol=0.0)
    torch.testing.assert_close(views.camera_to_world[0, 0], front_view_camera(torch.tensor([distance]))[0])

    encoded = {side: pipeline.encode_views(views.images[side]) for side in sides}
    inputs = {
        "camera_angle_x": torch.tensor([camera_angle_x]),
        "distance": torch.tensor([distance]),
        "mesh_scale": torch.tensor([1.0]),
    }
    extractor = reference["DinoV3ProjFeatureExtractor"]
    for stage, grid, cells in (
        ("sparse_structure", 4, DENSE_CELLS),
        ("512", 4, DENSE_CELLS),
        ("1024", 8, SPARSE_CELLS),
        ("texture", 8, SPARSE_CELLS),
    ):
        _assert_stage(pipeline, views, encoded, extractor, naf, stage, grid, cells, images, **inputs)


def test_multi_view_projection_conditioning_matches_pinned_reference(reference_naf):
    to_cond_tensor = _definitions("inference_mv.py", {"to_cond_tensor"}, {"torch": torch, "np": np, "Image": Image})[
        "to_cond_tensor"
    ]
    pipeline = tiny_pipeline(multiview=True)
    reference, naf = _reference_extractors(pipeline, reference_naf)

    rng = np.random.default_rng(9)
    rgba = rng.integers(0, 256, (3, 24, 24, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(rng.random((3, 24, 24)) < 0.3, 0, rgba[..., 3])
    fovs = torch.tensor([0.35, 0.5, 0.42])
    camera_to_world = []
    for azimuth, elevation in ((0.0, 0.0), (90.0, 15.0), (200.0, -10.0)):
        yaw, pitch = math.radians(azimuth), math.radians(elevation)
        turn = torch.eye(4)
        turn[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        tilt = torch.eye(4)
        tilt[1:3, 1:3] = torch.tensor([[math.cos(pitch), -math.sin(pitch)], [math.sin(pitch), math.cos(pitch)]])
        camera_to_world.append(turn @ tilt @ front_view_camera(torch.tensor(1.7)))
    camera_to_world = torch.stack(camera_to_world)
    intrinsics = torch.eye(3).repeat(3, 1, 1)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 12 / torch.tan(fovs / 2)
    intrinsics[:, :2, 2] = 12
    cameras = CameraRig(
        # Blender cameras look down -z with +y up; package cameras are OpenCV (+z forward, +y down).
        world_to_camera=torch.linalg.inv(camera_to_world * torch.tensor([1.0, -1.0, -1.0, 1.0])),
        intrinsics=intrinsics,
        image_sizes=torch.full((3, 2), 24),
    )
    condition = MultiViewCondition(images=torch.from_numpy(rgba).permute(0, 3, 1, 2).float() / 255, cameras=cameras)
    sides = (TINY_IMAGE_SIZE, 2 * TINY_IMAGE_SIZE)
    views = pipeline.prepare_views([condition], image_sizes=sides)
    images = {}
    for side in sides:
        images[side] = torch.stack([to_cond_tensor(Image.fromarray(view), side) for view in rgba])[None]
        torch.testing.assert_close(views.images[side], images[side], atol=0.0, rtol=0.0)

    encoded = {side: pipeline.encode_views(views.images[side]) for side in sides}
    inputs = {
        "camera_angle_x": fovs[None],
        "distance": camera_to_world[None, :, :3, 3].norm(dim=-1),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": camera_to_world[None],
    }
    extractor = reference["DinoV3ProjMultiViewFeatureExtractor"]
    for stage, grid, cells in (("sparse_structure", 4, DENSE_CELLS), ("texture", 8, SPARSE_CELLS)):
        _assert_stage(pipeline, views, encoded, extractor, naf, stage, grid, cells, images, **inputs)
