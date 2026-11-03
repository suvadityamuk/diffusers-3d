from __future__ import annotations

import importlib
import math
import os
import re
import sys
import types
from functools import lru_cache, partial
from pathlib import Path

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
from transformers import ViTConfig
from transformers.models.vit import modeling_vit

from diffusers_3d import ImageCondition
from diffusers_3d.backends import marching_cubes as marching_cubes_module
from diffusers_3d.backends.marching_cubes import marching_cubes
from diffusers_3d.families.triposr import pipeline as pipeline_module
from diffusers_3d.families.triposr.conversion import (
    TRIPOSR_REFERENCE_REVISION,
    convert_triposr_components,
    interpolate_position_embeddings,
)
from diffusers_3d.families.triposr.pipeline import TripoSRImageTo3DPipeline, prepare_triposr_image

from .configs import upstream_config, upstream_vit_config

pytestmark = pytest.mark.reference_parity

REFERENCE_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRIPOSR_REFERENCE_ROOT", "/tmp/TripoSR"))
REFERENCE_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSR"
TORCHMCUBES_ROOT = Path(os.environ.get("DIFFUSERS_3D_TORCHMCUBES_REFERENCE_ROOT", "/tmp/torchmcubes"))
# TripoSR installs torchmcubes from its unpinned default branch; this was its head at TripoSR's pinned revision.
TORCHMCUBES_REVISION = "3381600ddc3d2e4d74222f8495866be5fafbace4"
TORCHMCUBES_REPOSITORY = "https://github.com/tatsy/torchmcubes"
REFERENCE_PATHS = (
    "tsr/system.py",
    "tsr/utils.py",
    "tsr/models/isosurface.py",
    "tsr/models/nerf_renderer.py",
    "tsr/models/network_utils.py",
    "tsr/models/tokenizers/image.py",
    "tsr/models/tokenizers/triplane.py",
    "tsr/models/transformer/transformer_1d.py",
)
_REFERENCE: types.SimpleNamespace | None = None


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _parse_c_table(source: str, name: str) -> list[int]:
    body = source[source.index("{", source.index(name)) :]
    depth = 0
    for end, character in enumerate(body):
        depth += (character == "{") - (character == "}")
        if depth == 0:
            break
    return [int(token, 0) for token in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body[:end])]


@lru_cache(maxsize=2)
def _torchmcubes_tables(source_path: str = "cxx/mcubes_cpu.cpp") -> tuple[list[int], list[list[int]]]:
    """``edgeTable`` and ``triTable`` read from the pinned torchmcubes CPU (or CUDA) source."""

    try:
        validate_reference_checkout(
            TORCHMCUBES_ROOT,
            expected_revision=TORCHMCUBES_REVISION,
            expected_repository=TORCHMCUBES_REPOSITORY,
            expected_paths=("cxx/mcubes_cpu.cpp", "cxx/mcubes_cuda.cu"),
        )
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
    source = (TORCHMCUBES_ROOT / source_path).read_text()
    triangles = _parse_c_table(source, "triTable[256][16]")
    return _parse_c_table(source, "edgeTable[256]"), [triangles[row : row + 16] for row in range(0, 4096, 16)]


def _torchmcubes_cpu(volume: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Line-by-line float32 transcription of torchmcubes' ``mcubes_cpu`` (``Polygonise``, ``VertexInterp``, and
    ``mcubes_method`` in ``cxx/mcubes_cpu.cpp``), run single-threaded. Vertices are ``(x, y, z)`` for
    ``volume[z][y][x]``, as torchmcubes returns them."""

    edge_table, tri_table = _torchmcubes_tables()
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    vol, isolevel, epsilon = volume.numpy().astype(np.float32), np.float32(threshold), np.float32(1.0e-4)

    def vertex_interp(p1, p2, valp1, valp2):
        if abs(isolevel - valp1) < epsilon:
            return p1
        if abs(isolevel - valp2) < epsilon:
            return p2
        if abs(valp1 - valp2) < epsilon:
            return p1
        mu = (isolevel - valp1) / (valp2 - valp1)
        return tuple(np.float32(a + mu * (b - a)) for a, b in zip(p1, p2))

    all_vertices = []
    for z in range(vol.shape[0] - 1):
        for y in range(vol.shape[1] - 1):
            for x in range(vol.shape[2] - 1):
                p, val = [None] * 8, [None] * 8
                for i, slot in enumerate((0, 1, 4, 5, 3, 2, 7, 6)):
                    dx, dy, dz = i % 2, (i // 2) % 2, i // 4
                    p[slot] = (np.float32(x + dx), np.float32(y + dy), np.float32(z + dz))
                    val[slot] = vol[z + dz, y + dy, x + dx]
                cubeindex = sum(1 << i for i in range(8) if val[i] < isolevel)
                if edge_table[cubeindex] == 0:
                    continue
                vertlist = [
                    vertex_interp(p[a], p[b], val[a], val[b]) if edge_table[cubeindex] & (1 << edge) else None
                    for edge, (a, b) in enumerate(edges)
                ]
                row = tri_table[cubeindex]
                all_vertices += [vertlist[row[i]] for i in range(row.index(-1))]
    unique_vertices, vertices, faces = {}, [], []
    for vertex in all_vertices:
        unique_vertices.setdefault(vertex, len(vertices))
        if unique_vertices[vertex] == len(vertices):
            vertices.append(vertex)
        faces.append(unique_vertices[vertex])
    return torch.tensor(vertices, dtype=torch.float32).reshape(-1, 3), torch.tensor(faces).reshape(-1, 3).int()


def _torchmcubes_cuda(volume: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Line-by-line float32 transcription of torchmcubes' ``mcubes_cuda`` (``mcubes_cuda_kernel``, ``vertexInterp``,
    and ``compaction`` in ``cxx/mcubes_cuda.cu``): at most four triangles per cell, three fresh vertices each, cells
    compacted in ``volume[z][y][x]`` order, and one all-zero triangle when there are none."""

    edge_table, tri_table = _torchmcubes_tables("cxx/mcubes_cuda.cu")
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    vol, threshold, epsilon = volume.numpy().astype(np.float32), np.float32(threshold), np.float32(1.0e-4)

    def vertex_interp(isolevel, p1, p2, valp1, valp2):
        if abs(isolevel - valp1) < epsilon:
            return p1
        if abs(isolevel - valp2) < epsilon:
            return p2
        if abs(valp1 - valp2) < epsilon:
            return p1
        mu = (isolevel - valp1) / (valp2 - valp1)
        return tuple(np.float32(a + mu * (b - a)) for a, b in zip(p1, p2))

    vertices = []
    for iz in range(vol.shape[0] - 1):
        for iy in range(vol.shape[1] - 1):
            for ix in range(vol.shape[2] - 1):
                cubeindex, bits, val, p = 0, 0, [None] * 8, [None] * 8
                index_table = (0, 1, 4, 5, 3, 2, 7, 6)
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            jx, jy, jz = ix + i, iy + j, iz + k
                            if vol[jz, jy, jx] < threshold:
                                cubeindex |= 1 << index_table[bits]
                            val[index_table[bits]] = vol[jz, jy, jx]
                            p[index_table[bits]] = (np.float32(jx), np.float32(jy), np.float32(jz))
                            bits += 1
                vertlist = [
                    vertex_interp(threshold, p[a], p[b], val[a], val[b])
                    if edge_table[cubeindex] & (1 << edge)
                    else None
                    for edge, (a, b) in enumerate(edges)
                ]
                for i in range(4):
                    if tri_table[cubeindex][i * 3] >= 0:
                        vertices += [vertlist[tri_table[cubeindex][i * 3 + k]] for k in range(3)]
    ntris = max(1, len(vertices) // 3)
    vertices = torch.tensor(vertices, dtype=torch.float32).reshape(-1, 3)
    vertices = torch.cat((vertices, vertices.new_zeros(3 * ntris - len(vertices), 3)))
    return vertices, torch.arange(3 * ntris, dtype=torch.int32).reshape(-1, 3)


def _marching_cubes(level: torch.Tensor, threshold: float):
    """Stand-in for ``torchmcubes.marching_cubes`` (a compiled extension): :func:`_torchmcubes_cpu`, or
    :func:`_torchmcubes_cuda` while ``_REFERENCE.cuda_kernel`` is set."""

    _REFERENCE.levels.append(level.clone())
    return (_torchmcubes_cuda if _REFERENCE.cuda_kernel else _torchmcubes_cpu)(level, threshold)


def _trimesh(**kwargs):
    """The real ``trimesh.Trimesh`` (default processing), recording the arrays upstream passes it."""

    try:
        trimesh = import_reference_dependency("trimesh")
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)
    _REFERENCE.trimesh_inputs.append(kwargs)
    return trimesh.Trimesh(**kwargs)


def _load_reference() -> types.SimpleNamespace:
    """Import the pinned TripoSR model sources on CPU.

    ``torchmcubes`` (compiled marching cubes) is replaced by :func:`_torchmcubes_cpu`, a transcription of its pinned
    CPU kernel; ``rembg`` and ``imageio`` only serve background removal and video, and are stubbed. ``trimesh`` is
    real (its vertex merging is the same in 4.0.5, which upstream pins). ``omegaconf`` is real: upstream modules
    parse their configs with it.
    """

    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    try:
        validate_reference_checkout(
            REFERENCE_ROOT,
            expected_revision=TRIPOSR_REFERENCE_REVISION,
            expected_repository=REFERENCE_REPOSITORY,
            expected_paths=REFERENCE_PATHS,
        )
        for dependency in ("einops", "omegaconf"):
            import_reference_dependency(dependency)
    except ReferenceCheckoutError as error:
        try:
            pytest.skip(reference_unavailable(error))
        except ReferenceCheckoutError as required_error:
            pytest.fail(str(required_error), pytrace=False)

    # Upstream modules keep their own references to the stubs, so the global names are restored afterwards.
    stubbed = ("imageio", "rembg", "torchmcubes", "trimesh", "tsr")
    saved = {name: sys.modules.get(name) for name in stubbed}
    try:
        _module("imageio")
        _module("rembg")
        _module("torchmcubes", marching_cubes=_marching_cubes)
        _module("trimesh", Trimesh=_trimesh)
        _module("tsr", __path__=[str(REFERENCE_ROOT / "tsr")])
        system = importlib.import_module("tsr.system")
        for path in REFERENCE_PATHS:
            importlib.import_module(path.removesuffix(".py").replace("/", "."))
        utils = importlib.import_module("tsr.utils")
        image_tokenizer = importlib.import_module("tsr.models.tokenizers.image")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    _REFERENCE = types.SimpleNamespace(
        TSR=system.TSR,
        resize_foreground=utils.resize_foreground,
        image_tokenizer=image_tokenizer,
        levels=[],
        trimesh_inputs=[],
        cuda_kernel=False,
    )
    return _REFERENCE


def _interpolate_pos_encoding_4_35(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``ViTEmbeddings.interpolate_pos_encoding`` from transformers v4.35.0, which TripoSR pins:
    https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/vit/modeling_vit.py
    (``self.config.patch_size`` there is ``self.patch_size`` on the transformers 5 module)."""

    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return self.position_embeddings
    class_pos_embed = self.position_embeddings[:, 0]
    patch_pos_embed = self.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]
    h0 = height // self.patch_size
    w0 = width // self.patch_size
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


@pytest.fixture
def upstream_model(tmp_path, monkeypatch):
    """A tiny upstream ``TSR`` with its own random initialization, run with transformers 4.35 ViT interpolation."""

    reference = _load_reference()
    vit_config = ViTConfig(**upstream_vit_config())
    vit_config.save_pretrained(tmp_path)
    # The tokenizer fetches ``config.json`` of the ViT it names from the Hub; serve the tiny one instead.
    monkeypatch.setattr(
        reference.image_tokenizer, "hf_hub_download", lambda repo_id, filename: str(tmp_path / filename)
    )
    monkeypatch.setattr(modeling_vit.ViTEmbeddings, "interpolate_pos_encoding", _interpolate_pos_encoding_4_35)
    config = upstream_config(str(tmp_path))
    torch.manual_seed(20)
    model = reference.TSR(config).eval()
    with torch.no_grad():
        # Untrained ViT weights barely depend on the image; widen them so the image actually steers the triplanes.
        for parameter in model.image_tokenizer.parameters():
            parameter.mul_(10)
    return model, config, vit_config


def _rgba(height: int, width: int, seed: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    rgba = torch.randint(0, 256, (height, width, 4), generator=generator, dtype=torch.uint8)
    rgba[..., 3] = 0
    rgba[height // 4 : height // 2 + 3, width // 3 : width - 2, 3] = torch.randint(
        1, 256, (height // 2 + 3 - height // 4, width - 2 - width // 3), generator=generator, dtype=torch.uint8
    )
    return rgba.numpy()


def _upstream_preprocess(rgba: np.ndarray, foreground_ratio: float = 0.85) -> Image.Image:
    """``run.py``: ``resize_foreground``, then a grey composite saved as an 8-bit image."""

    image = _load_reference().resize_foreground(Image.fromarray(rgba, "RGBA"), foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    return Image.fromarray((image * 255.0).astype(np.uint8))


def _condition(rgba: np.ndarray) -> ImageCondition:
    return ImageCondition(image=torch.from_numpy(rgba).permute(2, 0, 1).float() / 255)


@pytest.mark.parametrize(("height", "width", "ratio"), [(37, 50, 0.85), (64, 23, 0.6)])
def test_preprocessing_matches_pinned_resize_foreground_and_composite(height, width, ratio):
    rgba = _rgba(height, width, seed=height)
    expected = torch.from_numpy(np.asarray(_upstream_preprocess(rgba, ratio))).permute(2, 0, 1).float() / 255
    assert torch.equal(prepare_triposr_image(_condition(rgba), foreground_ratio=ratio), expected)


def test_baked_position_embeddings_match_transformers_4_35_interpolation():
    torch.manual_seed(21)
    embeddings = modeling_vit.ViTEmbeddings(ViTConfig(**upstream_vit_config()))
    expected = _interpolate_pos_encoding_4_35(embeddings, torch.zeros(1, 17, 12), 32, 32)
    assert torch.equal(interpolate_position_embeddings(embeddings.position_embeddings, 4), expected)


@pytest.mark.parametrize("cuda_kernel", [False, True], ids=["cpu-kernel", "cuda-kernel"])
def test_converted_pipeline_matches_pinned_reference_end_to_end(upstream_model, monkeypatch, cuda_kernel):
    """Same image, weights, and chunking through upstream ``TSR.forward`` and ``TSR.extract_mesh``.

    Upstream runs with a torchmcubes transcription (CPU or CUDA kernel; the port follows the same kernel) and the real
    ``trimesh``. The triplanes, density grid, merged vertices, faces, and vertex colours (before and after trimesh's
    8-bit rounding) are compared exactly.
    """

    model, config, vit_config = upstream_model
    reference = _load_reference()
    _torchmcubes_tables()
    monkeypatch.setattr(reference, "cuda_kernel", cuda_kernel)
    monkeypatch.setattr(pipeline_module, "marching_cubes", partial(marching_cubes, _cuda_kernel=cuda_kernel))
    pipeline = TripoSRImageTo3DPipeline(**convert_triposr_components(config, model.state_dict(), vit_config))
    rgba = _rgba(48, 40, seed=22)
    model.renderer.set_chunk_size(1000)
    resolution = 16

    with torch.no_grad():
        scene_codes = model([_upstream_preprocess(rgba)], device="cpu")
        # Centre the threshold on the untrained density so the surface is non-trivial.
        axis = torch.linspace(0, 1, resolution)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) * 1.74 - 0.87
        threshold = model.renderer.query_triplane(model.decoder, grid, scene_codes[0])["density_act"].median().item()
        reference.levels.clear()
        reference.trimesh_inputs.clear()
        (expected,) = model.extract_mesh(scene_codes, True, resolution=resolution, threshold=threshold)

    output = pipeline(_condition(rgba), mc_resolution=resolution, mc_threshold=threshold, chunk_size=1000)
    mesh = output.objects[0]
    assert torch.equal(output.latents.latents, scene_codes)
    triplanes = output.latents.latents
    field = pipeline.decoder(triplanes, grid[None].to(triplanes), chunk_size=1000).density - threshold
    assert torch.equal(field.flatten(), reference.levels[0].flatten())
    # torchmcubes splits some vertices on edges shared by differently ordered cells; trimesh merges most back.
    (raw,) = reference.trimesh_inputs
    assert raw["vertices"].shape[0] > expected.vertices.shape[0] > 0
    assert torch.equal(mesh.vertices.double(), torch.from_numpy(expected.vertices))
    assert torch.equal(mesh.faces, torch.from_numpy(expected.faces))
    first_raw_row = {}
    for row, vertex in enumerate(map(tuple, raw["vertices"].tolist())):
        first_raw_row.setdefault(vertex, row)
    rows = [first_raw_row[vertex] for vertex in map(tuple, mesh.vertices.tolist())]
    assert torch.equal(mesh.colors, torch.from_numpy(raw["vertex_colors"][rows]))
    rounded = (mesh.colors.clamp(0, 1) * 255).round().to(torch.uint8)
    assert torch.equal(rounded, torch.from_numpy(expected.visual.vertex_colors[:, :3]))


def test_marching_cubes_tables_match_pinned_torchmcubes():
    _load_reference()
    edge_table, tri_table = _torchmcubes_tables()
    assert _torchmcubes_tables("cxx/mcubes_cuda.cu") == (edge_table, tri_table)
    assert [tuple(row[: row.index(-1)]) for row in tri_table] == list(marching_cubes_module._TRIANGLES)
    # The CUDA kernel keeps four triangles in the 32 cases with five.
    assert sum(row.index(-1) == 15 for row in tri_table) == 32
    # The backend derives each case's crossed edges from the corner signs; torchmcubes tabulates them.
    corner_bits = [
        sum(
            1 << edge for edge, (a, b) in enumerate(marching_cubes_module._EDGES) if (code >> a) & 1 != (code >> b) & 1
        )
        for code in range(256)
    ]
    assert corner_bits == edge_table


def _single_cell(code: int) -> torch.Tensor:
    """A 2x2x2 volume whose corners are inside (negative) exactly where ``code`` has their bit."""

    magnitudes = torch.rand(8, generator=torch.Generator().manual_seed(code)) + 0.2
    volume = torch.zeros(2, 2, 2)
    for bit, corner in enumerate(marching_cubes_module._CORNERS):
        volume[corner] = -magnitudes[bit] if (code >> bit) & 1 else magnitudes[bit]
    return volume


def _snapping_field(seed: int) -> torch.Tensor:
    """A random field where many samples are exact zeros, equal neighbours, or within the 1e-4 snapping distance."""

    generator = torch.Generator().manual_seed(seed)
    shape = torch.randint(2, 9, (3,), generator=generator).tolist()
    volume = torch.randn(shape, generator=generator)
    special = torch.tensor([-1.0, -0.5, -5e-5, 0.0, 5e-5, 0.25, 0.25 + 5e-5, 1.0])
    replace = torch.rand(shape, generator=generator) < 0.4
    volume[replace] = special[torch.randint(0, 8, (int(replace.sum()),), generator=generator)]
    return volume


@pytest.mark.parametrize(
    ("volumes", "threshold"),
    [
        pytest.param([_single_cell(code) for code in range(256)], 0.0, id="all-256-cells"),
        pytest.param([_snapping_field(seed) for seed in range(12)], 0.0, id="snapping-fields"),
        pytest.param([_snapping_field(seed) for seed in range(12, 20)], 0.25, id="snapping-fields-threshold"),
    ],
)
@pytest.mark.parametrize("cuda_kernel", [False, True], ids=["cpu-kernel", "cuda-kernel"])
def test_marching_cubes_matches_torchmcubes_transcriptions(volumes, threshold, cuda_kernel):
    _load_reference()
    for volume in volumes:
        expected_vertices, expected_faces = (_torchmcubes_cuda if cuda_kernel else _torchmcubes_cpu)(volume, threshold)
        vertices, faces = marching_cubes(volume, threshold, _cuda_kernel=cuda_kernel)
        if cuda_kernel and vertices.shape[0] == 0:
            # Without a crossing the CUDA kernel returns one all-zero triangle; the port returns an empty mesh.
            assert torch.equal(expected_vertices, torch.zeros(3, 3)) and expected_faces.tolist() == [[0, 1, 2]]
            continue
        assert torch.equal(vertices, expected_vertices.flip(-1))
        assert torch.equal(faces, expected_faces.long())


@pytest.mark.parametrize(
    ("view_kwargs", "height", "width"),
    [
        ({"n_views": 4}, 12, 10),
        ({"n_views": 5, "elevation_deg": 25.0, "camera_distance": 2.2, "fovy_deg": 30.0}, 9, 9),
    ],
)
def test_render_views_matches_pinned_tsr_render(upstream_model, view_kwargs, height, width):
    model, config, vit_config = upstream_model
    pipeline = TripoSRImageTo3DPipeline(**convert_triposr_components(config, model.state_dict(), vit_config))
    model.renderer.set_chunk_size(1000)
    samples = config["renderer"]["num_samples_per_ray"]
    with torch.no_grad():
        scene_codes = model([_upstream_preprocess(_rgba(40, 48, seed=23))] * 2, device="cpu")
        expected = model.render(scene_codes, height=height, width=width, return_type="pt", **view_kwargs)
    num_views = view_kwargs.pop("n_views")
    frames = pipeline.render_views(
        scene_codes,
        num_views=num_views,
        height=height,
        width=width,
        num_samples_per_ray=samples,
        chunk_size=1000,
        **view_kwargs,
    )
    assert frames.shape == (2, num_views, height, width, 3)
    # Upstream raises when some rays miss the box, so every ray of these cameras hits it.
    assert torch.equal(frames, torch.stack([torch.stack(views) for views in expected]))
