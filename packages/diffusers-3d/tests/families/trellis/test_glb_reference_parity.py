"""CPU parity of the TRELLIS ``to_glb`` port's camera and mesh helpers with pinned TRELLIS and utils3d.

The CUDA stages (nvdiffrast, mip-splatting, pyvista, pymeshfix, igraph) are compared end to end on a GPU instead.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from _reference import ReferenceCheckoutError, reference_unavailable, validate_reference_checkout

from diffusers_3d.backends import trellis_glb

pytestmark = pytest.mark.reference_parity

TRELLIS_ROOT = Path(os.environ.get("DIFFUSERS_3D_TRELLIS_REFERENCE_ROOT", "/tmp/TRELLIS"))
UTILS3D_ROOT = Path(os.environ.get("DIFFUSERS_3D_UTILS3D_REFERENCE_ROOT", "/tmp/utils3d"))
UTILS3D_REVISION = "9a4eb15e4021b67b12c460c7057d642626897ec8"


@pytest.fixture(scope="module")
def reference():
    try:
        validate_reference_checkout(
            TRELLIS_ROOT,
            expected_revision="442aa1e1afb9014e80681d3bf604e8d728a86ee7",
            expected_repository="https://github.com/microsoft/TRELLIS.git",
            expected_paths=("trellis/utils/random_utils.py",),
        )
        validate_reference_checkout(
            UTILS3D_ROOT,
            expected_revision=UTILS3D_REVISION,
            expected_repository="https://github.com/EasternJournalist/utils3d.git",
            expected_paths=("utils3d/torch/mesh.py", "utils3d/torch/transforms.py"),
        )
    except ReferenceCheckoutError as error:
        pytest.skip(reference_unavailable(error))
    spec = importlib.util.spec_from_file_location(
        "trellis_random_utils", TRELLIS_ROOT / "trellis/utils/random_utils.py"
    )
    random_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(random_utils)
    # utils3d imports nvdiffrast at import time; only its pure-PyTorch helpers are compared here.
    stubs = {"nvdiffrast": types.ModuleType("nvdiffrast"), "nvdiffrast.torch": types.ModuleType("nvdiffrast.torch")}
    saved = {name: sys.modules.get(name) for name in [*stubs, "utils3d"]}
    sys.modules.update(stubs)
    sys.modules.pop("utils3d", None)
    sys.path.insert(0, str(UTILS3D_ROOT))
    try:
        import utils3d.torch as utils3d

        yield random_utils, utils3d
    finally:
        sys.path.remove(str(UTILS3D_ROOT))
        for name in [name for name in sys.modules if name == "utils3d" or name.startswith("utils3d.")]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def _equal(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_cameras_match_trellis_and_utils3d_bit_for_bit(reference):
    random_utils, utils3d = reference
    for count in (100, 1000):
        for index in range(count):
            assert list(trellis_glb._sphere_hammersley(index, count)) == random_utils.sphere_hammersley_sequence(
                index, count
            )
    up = torch.tensor([0.0, 0.0, 1.0])
    for eye in (torch.tensor([0.3, -1.9, 0.4]), torch.tensor([-1.0, 0.2, 1.7])):
        _equal(trellis_glb._look_at(eye, up, opencv=True), utils3d.extrinsics_look_at(eye, torch.zeros(3), up))
        _equal(trellis_glb._look_at(eye, up, opencv=False), utils3d.view_look_at(eye, torch.zeros(3), up))
    fov = torch.deg2rad(torch.tensor(40.0))
    intrinsics = trellis_glb._intrinsics_from_fov(fov)
    _equal(intrinsics, utils3d.intrinsics_from_fov_xy(fov, fov))
    _equal(
        trellis_glb._gl_perspective(intrinsics, 0.1, 10.0),
        utils3d.intrinsics_to_perspective(intrinsics, near=0.1, far=10.0),
    )
    fov = torch.deg2rad(torch.tensor(40))
    _equal(trellis_glb._gl_perspective(None, 1, 3, fov=fov), utils3d.perspective_from_fov_xy(fov, fov, 1, 3))


def test_mesh_helpers_match_utils3d_exactly(reference):
    _, utils3d = reference
    generator = np.random.default_rng(0)
    # Several strips of quads with shuffled faces: multiple components, boundary edges, and interior edges.
    faces = []
    for offset in (0, 40, 80):
        for i in range(9):
            a, b, c, d = offset + i, offset + i + 1, offset + 20 + i, offset + 21 + i
            faces += [[a, b, c], [b, d, c]]
    faces = torch.tensor(np.asarray(faces)[generator.permutation(len(faces))], dtype=torch.int32)

    edges, face_to_edge, counts = trellis_glb._edges(faces)
    for actual, expected in zip((edges, face_to_edge, counts), utils3d.compute_edges(faces)):
        _equal(actual, expected)
    components = trellis_glb._face_components(faces)
    expected = utils3d.compute_connected_components(faces)
    assert len(components) == len(expected) == 3
    for actual, wanted in zip(components, expected):
        _equal(actual, wanted)
    dual_edges, dual_edge_to_edge = trellis_glb._group(face_to_edge.flatten(), required_group_size=2)
    expected_dual, expected_dual_to_edge = utils3d.compute_dual_graph(face_to_edge)
    _equal(dual_edges // 3, expected_dual)
    _equal(dual_edge_to_edge, expected_dual_to_edge)
    boundary = torch.nonzero(counts == 1).reshape(-1)
    vertices, loop_edges = torch.unique(edges[boundary].flatten(), return_inverse=True)
    loops = trellis_glb._connected_components(len(boundary), loop_edges.view(-1, 2), vertices.shape[0])
    for actual, wanted in zip(loops, utils3d.compute_edge_connected_components(edges[boundary]), strict=True):
        _equal(actual, wanted)
