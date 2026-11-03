from __future__ import annotations

import ast
import contextlib
import importlib
import os
import sys
import types
from functools import partial
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from _configs import tiny_config
from _reference import (
    ReferenceCheckoutError,
    import_reference_dependency,
    reference_unavailable,
    validate_reference_checkout,
)

from diffusers_3d import (
    SAM3D_REFERENCE_REVISION,
    Sam3DConditionEmbedder,
    Sam3DObjectsPipeline,
    Sam3DSparseStructureFlowModel,
    SparseVoxelAsset,
    preprocess_sam3d_condition,
    trellis_grid_transform,
)
from diffusers_3d.families.sam3d_objects.conversion import (
    convert_condition_embedder_state,
    convert_dinov2_encoder,
    convert_sparse_structure_flow_state,
)
from diffusers_3d.families.sam3d_objects.pipeline import _downsample_structure, _prune_interior
from diffusers_3d.objects import CoordinateSystem

from .conftest import tiny_dinov2_config, tiny_sam3d_pipeline

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_SAM3D_REFERENCE_ROOT", "/tmp/sam-3d-objects"))
REFERENCE_REPOSITORY = "https://github.com/facebookresearch/sam-3d-objects.git"
_PACKAGE = "sam3d_objects"
REFERENCE_PATHS = tuple(
    f"{_PACKAGE}/{path}"
    for path in (
        "data/utils.py",
        "data/dataset/tdfy/img_and_mask_transforms.py",
        "data/dataset/tdfy/img_processing.py",
        "data/dataset/tdfy/pose_target.py",
        "data/dataset/tdfy/preprocessor.py",
        "data/dataset/tdfy/transforms_3d.py",
        "model/backbone/dit/embedder/dino.py",
        "model/backbone/dit/embedder/embedder_fuser.py",
        "model/backbone/dit/embedder/pointmap.py",
        "model/backbone/generator/classifier_free_guidance.py",
        "model/backbone/generator/flow_matching/model.py",
        "model/backbone/generator/shortcut/model.py",
        "model/backbone/tdfy_dit/models/mm_latent.py",
        "model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py",
        "model/layers/llama3/ff.py",
        "pipeline/inference_utils.py",
    )
)
_REFERENCE: types.SimpleNamespace | None = None


def _reference_unavailable(error: ReferenceCheckoutError) -> None:
    try:
        reason = reference_unavailable(error)
    except ReferenceCheckoutError as required_error:
        pytest.fail(str(required_error), pytrace=False)
    pytest.skip(reason)


# --- Stand-ins -----------------------------------------------------------------------------------------------------
#
# Logging and config helpers are stubbed. ``pytorch3d.transforms`` is replaced by a row-vector ``Transform3d`` with
# PyTorch3D's composition order plus its quaternion conversions, which is all the pose and pointmap code uses. The
# torch-hub DINOv2 backbone becomes a small module with the hub state layout (the hub-to-Transformers equivalence is
# Transformers' own conversion). ``optree`` and ``timm`` are the real packages.


class _Transform3d:
    def __init__(self, dtype=torch.float32, device="cpu", matrix=None):
        self._matrix = torch.eye(4, dtype=dtype, device=device)[None] if matrix is None else matrix.reshape(-1, 4, 4)

    def get_matrix(self):
        return self._matrix

    def compose(self, other):
        return _Transform3d(matrix=self._matrix @ other.get_matrix())

    def _then(self, values, assign):
        values = torch.as_tensor(values, dtype=self._matrix.dtype)
        matrix = torch.eye(4, dtype=values.dtype).repeat(values.shape[0], 1, 1)
        assign(matrix, values)
        return self.compose(_Transform3d(matrix=matrix))

    def scale(self, scale):
        return self._then(
            torch.as_tensor(scale).reshape(-1, 3), lambda m, v: m[:, :3, :3].diagonal(dim1=1, dim2=2).copy_(v)
        )

    def rotate(self, rotation):
        return self._then(torch.as_tensor(rotation).reshape(-1, 3, 3), lambda m, v: m[:, :3, :3].copy_(v))

    def translate(self, translation):
        return self._then(torch.as_tensor(translation).reshape(-1, 3), lambda m, v: m[:, 3, :3].copy_(v))

    def inverse(self):
        return _Transform3d(matrix=torch.inverse(self._matrix))

    def transform_points(self, points):
        batched = points[None] if points.ndim == 2 else points
        homogeneous = torch.cat([batched, torch.ones_like(batched[..., :1])], dim=-1) @ self._matrix
        out = homogeneous[..., :3] / homogeneous[..., 3:]
        return out[0] if points.ndim == 2 else out

    def to(self, *args, **kwargs):
        return self


def _quaternion_to_matrix(quaternion):
    r, i, j, k = quaternion.unbind(-1)
    two_s = 2.0 / (quaternion * quaternion).sum(-1)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )  # fmt: skip
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_to_quaternion(matrix):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(matrix.shape[:-2] + (9,)).unbind(-1)
    q_abs = torch.sqrt(
        torch.stack(
            [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
        ).clamp(min=0)
    )
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    ) / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = F.one_hot(q_abs.argmax(dim=-1), 4) > 0.5
    return candidates[best, :].reshape(matrix.shape[:-2] + (4,))


class _LayerScale(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim))


class _HubAttention(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        query, key, value = self.qkv(x).reshape(*x.shape[:2], 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(query, key, value).transpose(1, 2).flatten(2))


class _HubBlock(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6), torch.nn.LayerNorm(dim, eps=1e-6)
        self.attn = _HubAttention(dim, heads)
        self.mlp = torch.nn.Module()
        self.mlp.fc1, self.mlp.fc2 = torch.nn.Linear(dim, 2 * dim), torch.nn.Linear(2 * dim, dim)
        self.ls1, self.ls2 = _LayerScale(dim), _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1.gamma * self.attn(self.norm1(x))
        return x + self.ls2.gamma * self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class _HubDinov2(torch.nn.Module):
    """Tiny ``dinov2_vitl14_reg``-layout backbone (8x8 input, 4x4 patches, two registers)."""

    embed_dim = 12

    def __init__(self, **kwargs):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(3, 12, 4, stride=4)
        self.patch_embed.patch_size = (4, 4)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 12))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 12))
        self.register_tokens = torch.nn.Parameter(torch.zeros(1, 2, 12))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 5, 12))
        self.blocks = torch.nn.ModuleList([_HubBlock(12, 3)])
        self.norm = torch.nn.LayerNorm(12, eps=1e-6)

    def forward_features(self, images):
        x = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        x = torch.cat([x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]], dim=1)
        for block in self.blocks:
            x = block(x)
        normed = self.norm(x)
        return {"x_norm_clstoken": normed[:, 0], "x_norm_patchtokens": normed[:, 3:], "x_prenorm": x}


@contextlib.contextmanager
def _stubbed_imports():
    noop = lambda *args, **kwargs: None  # noqa: E731
    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(info=noop, warning=noop, debug=noop, error=noop)
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(to_container=lambda value: value)
    transforms = types.ModuleType("pytorch3d.transforms")
    transforms.Transform3d = _Transform3d
    transforms.quaternion_to_matrix = _quaternion_to_matrix
    transforms.matrix_to_quaternion = _matrix_to_quaternion
    transforms.Rotate = transforms.Translate = transforms.Scale = transforms.axis_angle_to_quaternion = None
    pytorch3d = types.ModuleType("pytorch3d")
    pytorch3d.transforms = transforms
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
    fakes = {
        "loguru": loguru,
        "omegaconf": omegaconf,
        "astor": types.ModuleType("astor"),
        "matplotlib": matplotlib,
        "matplotlib.pyplot": matplotlib.pyplot,
        "pytorch3d": pytorch3d,
        "pytorch3d.transforms": transforms,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _extract(path: Path, names: set[str], namespace: dict) -> dict:
    """Execute only the named top-level functions and assignments of a module that cannot be imported."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) in names for target in node.targets))
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def _load_reference() -> types.SimpleNamespace:
    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=SAM3D_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        import_reference_dependency("optree")
        import_reference_dependency("timm.models.vision_transformer")
    except ReferenceCheckoutError as error:
        _reference_unavailable(error)

    os.environ["ATTN_BACKEND"] = "sdpa"
    source = REFERENCE_ROOT / _PACKAGE
    # Register the root and the heavy ``models`` package without running their ``__init__`` imports.
    for name, path in (
        (_PACKAGE, source),
        (f"{_PACKAGE}.model.backbone.tdfy_dit.models", source / "model/backbone/tdfy_dit/models"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    sys.modules[f"{_PACKAGE}.model.backbone.tdfy_dit.modules.sparse"] = None  # spconv-only; upstream tolerates absence

    def module(name):
        return importlib.import_module(f"{_PACKAGE}.{name}")

    with _stubbed_imports():
        flow = module("model.backbone.tdfy_dit.models.mot_sparse_structure_flow")
        latents = module("model.backbone.tdfy_dit.models.mm_latent")
        fuser = module("model.backbone.dit.embedder.embedder_fuser")
        dino = module("model.backbone.dit.embedder.dino")
        pointmap = module("model.backbone.dit.embedder.pointmap")
        guidance = module("model.backbone.generator.classifier_free_guidance")
        flow_matching = module("model.backbone.generator.flow_matching.model")
        shortcut = module("model.backbone.generator.shortcut.model")
        transforms = module("data.dataset.tdfy.img_and_mask_transforms")
        processing = module("data.dataset.tdfy.img_processing")
        preprocessor = module("data.dataset.tdfy.preprocessor")
        pose_target = module("data.dataset.tdfy.pose_target")
    utilities = _extract(
        source / "pipeline/inference_utils.py",
        {
            "ROTATION_6D_MEAN",
            "ROTATION_6D_STD",
            "pose_decoder",
            "prune_sparse_structure",
            "downsample_sparse_structure",
        },
        {
            "torch": torch,
            "matrix_to_quaternion": _matrix_to_quaternion,
            "PoseTargetConverter": pose_target.PoseTargetConverter,
        },
    )
    _REFERENCE = types.SimpleNamespace(
        flow=flow,
        latents=latents,
        fuser=fuser,
        dino=dino,
        pointmap=pointmap,
        guidance=guidance,
        flow_matching=flow_matching,
        shortcut=shortcut,
        transforms=transforms,
        processing=processing,
        preprocessor=preprocessor,
        utilities=utilities,
    )
    return _REFERENCE


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in module.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator) * 0.2)


# Upstream names the rotation modality ``6drotation_normalized`` and its merged pose stream after it.
_UPSTREAM_NAMES = {"rotation": "6drotation_normalized"}


def _upstream_names(values: dict) -> dict:
    return {_UPSTREAM_NAMES.get(name, name): value for name, value in values.items()}


def _flow_pair(reference, seed: int):
    config = tiny_config(Sam3DSparseStructureFlowModel)
    width = config["model_channels"]
    mapping = {
        "6drotation_normalized": reference.latents.Latent(
            6, width, reference.latents.LearntPositionEmbedder(width, 1)
        ),
        "scale": reference.latents.Latent(3, width, reference.latents.LearntPositionEmbedder(width, 1)),
        "shape": reference.latents.Latent(
            config["in_channels"], width, reference.latents.ShapePositionEmbedder(width, config["resolution"], 1)
        ),
        "translation": reference.latents.Latent(3, width, reference.latents.LearntPositionEmbedder(width, 1)),
        "translation_scale": reference.latents.Latent(1, width, reference.latents.LearntPositionEmbedder(width, 1)),
    }
    upstream = reference.flow.SparseStructureFlowTdfyWrapper(
        latent_mapping=mapping,
        latent_share_transformer={
            "6drotation_normalized": ["6drotation_normalized", "translation", "scale", "translation_scale"]
        },
        in_channels=config["in_channels"],
        model_channels=width,
        cond_channels=config["cond_channels"],
        out_channels=config["in_channels"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        pe_mode="ape",
        qk_rms_norm=True,
        resolution=config["resolution"],
        is_shortcut_model=True,
        force_zeros_cond=True,
    ).eval()
    _randomize(upstream, seed)
    model = Sam3DSparseStructureFlowModel(**config).eval()
    model.load_state_dict(convert_sparse_structure_flow_state(upstream.state_dict()), strict=True)
    return upstream, model


def test_sparse_structure_flow_matches_pinned_reference():
    reference = _load_reference()
    upstream, model = _flow_pair(reference, seed=1)
    generator = torch.Generator().manual_seed(2)
    latents = {name: torch.randn(shape, generator=generator) for name, shape in model.latent_shapes(2).items()}
    timesteps = torch.tensor([120.0, 870.0])
    shortcut = torch.tensor([0.0])
    context = torch.randn(2, 6, model.config.cond_channels, generator=generator)
    with torch.no_grad():
        for cfg in (False, True):
            expected = upstream(_upstream_names(latents), timesteps, context, d=shortcut, cfg=cfg)
            actual = model(latents, timesteps, torch.zeros_like(context) if cfg else context, shortcut).sample
            for name, value in _upstream_names(actual).items():
                torch.testing.assert_close(value, expected[name], atol=1e-6, rtol=1e-5)


def _embedder_pair(reference, *, prenorm: bool, seed: int):
    pointmap_config = {"embed_dim": 16, "input_size": 8, "patch_size": 4, "num_heads": 16, "mlp_ratio": 2.0}
    with mock.patch.object(torch.hub, "load", lambda **kwargs: _HubDinov2()):
        dinos = [
            reference.dino.Dino(input_size=8, dino_model="dinov2_vitl14_reg", prenorm_features=prenorm)
            for _ in range(2)
        ]
    embedders = [
        (dinos[0], [("image", "cropped"), ("rgb_image", "full")]),
        (dinos[1], [("mask", "cropped"), ("rgb_image_mask", "full")]),
    ]
    if not prenorm:
        embedders.append(
            (
                reference.pointmap.PointPatchEmbed(input_size=8, patch_size=4, embed_dim=16, remap_output="linear"),
                [("pointmap", "cropped"), ("rgb_pointmap", "full")],
            )
        )
    upstream = reference.fuser.EmbedderFuser(embedders, projection_net_hidden_dim_multiplier=1.0).eval()
    _randomize(upstream, seed)
    config = {
        **tiny_config(Sam3DConditionEmbedder),
        "prenorm_features": prenorm,
        "pointmap_config": None if prenorm else pointmap_config,
    }
    model = Sam3DConditionEmbedder(**config).eval()
    state = upstream.state_dict()
    model.load_state_dict(convert_condition_embedder_state(state), strict=True)
    encoders = [
        convert_dinov2_encoder(
            {
                key.split(".backbone.", 1)[1]: value
                for key, value in state.items()
                if key.startswith(f"module_list.{index}.backbone.")
            },
            tiny_dinov2_config(),
        )
        for index in range(2)
    ]
    return upstream, model, encoders


def test_condition_embedders_match_pinned_reference():
    reference = _load_reference()
    generator = torch.Generator().manual_seed(3)
    images = torch.rand(2, 2, 3, 8, 8, generator=generator)
    masks = (torch.rand(2, 2, 1, 8, 8, generator=generator) > 0.4).float()
    pointmaps = torch.randn(2, 2, 3, 8, 8, generator=generator)
    pointmaps[:, :, :, 1, 2] = float("nan")
    inputs = {"image": images[:, 0], "rgb_image": images[:, 1], "mask": masks[:, 0], "rgb_image_mask": masks[:, 1]}
    pointmap_inputs = {"pointmap": pointmaps[:, 0], "rgb_pointmap": pointmaps[:, 1]}
    for prenorm in (False, True):
        upstream, model, (image_encoder, mask_encoder) = _embedder_pair(reference, prenorm=prenorm, seed=4)
        features = "prenorm" if prenorm else "normed"
        with torch.no_grad():
            tokens = [
                Sam3DObjectsPipeline.encode_views(encoder, views)[features]
                for encoder, views in ((image_encoder, images), (mask_encoder, masks))
            ]
            expected = upstream(**inputs, **({} if prenorm else pointmap_inputs))
            torch.testing.assert_close(model(*tokens, pointmaps).embeddings, expected, atol=1e-5, rtol=1e-5)
            if not prenorm:
                # Missing pointmaps are upstream's deterministic modality drop.
                upstream.force_drop_modalities = ["pointmap", "rgb_pointmap"]
                torch.testing.assert_close(
                    model(*tokens).embeddings, upstream(**inputs, **pointmap_inputs), atol=1e-5, rtol=1e-5
                )


def test_preprocessing_matches_pinned_reference():
    reference = _load_reference()
    from torchvision.transforms import Compose, Resize

    transforms, processing = reference.transforms, reference.processing

    def view(interpolation=None):
        resize = Resize(518) if interpolation is None else Resize(518, interpolation=interpolation)
        return Compose([partial(processing.pad_to_square_centered), resize])

    crop = partial(transforms.crop_around_mask_with_padding, box_size_factor=1.2, padding_factor=0.0)
    structure = reference.preprocessor.PreProcessor(
        img_mask_joint_transform=[],
        img_mask_pointmap_joint_transform=[partial(transforms.resize_all_to_same_size), crop],
        img_transform=view(),
        mask_transform=view(0),
        pointmap_transform=view(0),
        normalize_pointmap=True,
        pointmap_normalizer=transforms.ObjectCentricSSI(use_scene_scale=True, allow_scale_and_shift_override=True),
    )
    latent = reference.preprocessor.PreProcessor(
        img_mask_joint_transform=[crop], img_transform=view(), mask_transform=view(0)
    )

    generator = torch.Generator().manual_seed(5)
    image = torch.rand(3, 37, 29, generator=generator)
    mask = torch.zeros(1, 37, 29)
    mask[:, 2:30, 15:28] = 1.0  # near the right edge, so the enlarged crop leaves the image
    pointmap = torch.randn(3, 37, 29, generator=generator)
    pointmap[2] = pointmap[2].abs() + 1.5
    pointmap[:, 10:12, 16:20] = float("nan")
    # MoGe marks its masked-out points with infinity; flipped to PyTorch3D axes, x and y become -inf.
    pointmap[:, 3:8, 20:24] = torch.tensor([-float("inf"), -float("inf"), float("inf")])[:, None, None]

    expected = structure._process_image_mask_pointmap_mess(image, mask, pointmap)
    expected_latent = latent._process_image_and_mask_mess(image, mask)
    actual = preprocess_sam3d_condition(image, mask, pointmap)
    for name, (cropped, full) in {
        "images": ("image", "rgb_image"),
        "masks": ("mask", "rgb_image_mask"),
        "pointmaps": ("pointmap", "rgb_pointmap"),
    }.items():
        torch.testing.assert_close(actual[name][0], expected[cropped], atol=1e-5, rtol=1e-5, equal_nan=True)
        torch.testing.assert_close(actual[name][1], expected[full], atol=1e-5, rtol=1e-5, equal_nan=True)
    torch.testing.assert_close(actual["images"][0], expected_latent["image"], atol=0, rtol=0)
    torch.testing.assert_close(actual["masks"][0], expected_latent["mask"], atol=0, rtol=0)
    torch.testing.assert_close(actual["pointmap_scale"], expected["pointmap_scale"], atol=0, rtol=0)
    torch.testing.assert_close(actual["pointmap_shift"], expected["pointmap_shift"], atol=0, rtol=0)


def test_sparse_structure_sampler_layout_and_structure_match_pinned_reference():
    reference = _load_reference()
    upstream_flow, model = _flow_pair(reference, seed=6)
    generator_model = reference.shortcut.ShortCut(
        reverse_fn=reference.guidance.ClassifierFreeGuidanceWithExternalUnconditionalProbability(
            backbone=upstream_flow,
            p_unconditional=0.1,
            strength=7.0,
            unconditional_handling="add_flag",
            interval=[0, 500],
        ),
        no_shortcut=True,
        batch_mode=True,
        inference_steps=6,
        rescale_t=3,
        sigma_min=0.0,
        time_scale=1000.0,
    ).eval()
    pipeline = tiny_sam3d_pipeline(include_slat=False)
    pipeline.sparse_structure_flow_model = model
    context = torch.randn(1, 6, model.config.cond_channels, generator=torch.Generator().manual_seed(7))

    torch.manual_seed(8)
    noise = pipeline.prepare_sparse_structure_latents(1)
    torch.manual_seed(8)
    expected_noise = generator_model._generate_noise(_upstream_names(model.latent_shapes(1)), "cpu")
    for name, value in _upstream_names(noise).items():
        torch.testing.assert_close(value, expected_noise[name], atol=0, rtol=0)

    torch.manual_seed(8)
    with torch.no_grad():
        expected = generator_model(_upstream_names(model.latent_shapes(1)), "cpu", context)
        actual = pipeline.sample_sparse_structure(
            noise,
            context,
            num_inference_steps=6,
            guidance_scale=7.0,
            guidance_interval=(0.5, 1.0),
            rescale_t=3.0,
        )
    # Upstream builds this schedule in float64 NumPy, the TRELLIS scheduler in float32 (one ulp of ``t * 1000``).
    for name, value in _upstream_names(actual).items():
        torch.testing.assert_close(value, expected[name], atol=2e-4, rtol=2e-4)

    scale, shift = torch.tensor([[1.7, 1.7, 1.7]]), torch.tensor([[0.1, -0.2, 2.5]])
    upstream_layout = reference.utilities["pose_decoder"]("ScaleShiftInvariant")(
        _upstream_names(actual), scene_scale=scale, scene_shift=shift
    )
    layout = pipeline.decode_layout(actual, pointmap_scale=scale, pointmap_shift=shift)[0]
    torch.testing.assert_close(
        layout["rotation"], _quaternion_to_matrix(upstream_layout["rotation"])[0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(layout["translation"], upstream_layout["translation"][0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layout["scale"].expand(3), upstream_layout["scale"][0], atol=1e-5, rtol=1e-5)

    # Surface pruning on a solid block, and halving of a structure above the voxel budget.
    block = torch.stack(torch.meshgrid(*[torch.arange(5)] * 3, indexing="ij"), dim=-1).reshape(-1, 3) + 7
    batched = F.pad(block, (1, 0))
    torch.testing.assert_close(_prune_interior(batched), reference.utilities["prune_sparse_structure"](batched))
    large = torch.stack(torch.meshgrid(*[torch.arange(36)] * 3, indexing="ij"), dim=-1).reshape(-1, 3) + 3
    expected_coordinates, expected_factor = reference.utilities["downsample_sparse_structure"](F.pad(large, (1, 0)))
    coordinates, factor = _downsample_structure(large)
    assert factor == expected_factor == 2
    torch.testing.assert_close(coordinates.int(), expected_coordinates[:, 1:])


class _SharedSLatBackbone(torch.nn.Module):
    """Upstream ``SLatFlowModelTdfyWrapper`` calling convention around the package SLAT flow (TRELLIS parity)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, cond, coords, cfg=False):
        from diffusers_3d import TrellisSparseTensor

        latents = TrellisSparseTensor(torch.as_tensor(coords).long(), x[0])
        return self.model(latents, t, cond * 0 if cfg else cond).sample.features[None]


def test_slat_sampler_matches_pinned_reference():
    reference = _load_reference()
    pipeline = tiny_sam3d_pipeline()
    resolution = pipeline.slat_flow_model.config.resolution
    coordinates = torch.tensor([[0, 1, 2], [3, 3, 1], [4, 0, 5], [7, 7, 7], [2, 6, 0]])
    structure = SparseVoxelAsset(
        coordinates=coordinates,
        features=torch.ones(5, 1),
        grid_transform=trellis_grid_transform(resolution),
        coordinate_system=CoordinateSystem.RIGHT_HANDED_Z_UP,
    )
    sampler = reference.flow_matching.FlowMatching(
        reverse_fn=reference.guidance.ClassifierFreeGuidance(
            backbone=_SharedSLatBackbone(pipeline.slat_flow_model),
            p_unconditional=0.0,
            strength=1.0,
            unconditional_handling="add_flag",
            interval=[0, 500],
        ),
        inference_steps=5,
        rescale_t=1,
        sigma_min=0.0,
        time_scale=1000.0,
    ).eval()
    context = torch.randn(
        1, 4, pipeline.slat_flow_model.config.cond_channels, generator=torch.Generator().manual_seed(9)
    )
    batched = F.pad(coordinates, (1, 0)).numpy()
    torch.manual_seed(10)
    with torch.no_grad():
        expected = sampler((1, 5, pipeline.slat_flow_model.config.in_channels), "cpu", context, batched)[0]
        expected = expected * torch.tensor(pipeline.config.slat_std) + torch.tensor(pipeline.config.slat_mean)
        torch.manual_seed(10)
        actual = pipeline.sample_slat(
            (structure,),
            context,
            generator=None,
            num_inference_steps=5,
            guidance_scale=1.0,
            guidance_interval=(0.5, 1.0),
            rescale_t=1.0,
        )
    torch.testing.assert_close(actual.features, expected, atol=1e-5, rtol=1e-5)
