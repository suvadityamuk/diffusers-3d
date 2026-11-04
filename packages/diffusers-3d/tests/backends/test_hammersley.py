from __future__ import annotations

import torch

from diffusers_3d.backends._hammersley import hammersley_3d, radical_inverse
from diffusers_3d.families.trellis import flexicubes


def test_radical_inverse_and_hammersley_points():
    assert [radical_inverse(2, index) for index in range(5)] == [0.0, 0.5, 0.25, 0.75, 0.125]
    assert radical_inverse(3, 1) == 1 / 3 and radical_inverse(3, 3) == 1 / 9
    points = hammersley_3d(4)
    assert points.dtype == torch.float32 and points.shape == (4, 3)
    assert points[:, 0].tolist() == [0.0, 0.25, 0.5, 0.75]


def test_flexicubes_tables_have_their_shapes_and_are_built_once_per_device():
    dmc, num_vd, check, edges = flexicubes._tables(torch.device("cpu"))
    assert dmc.shape == (256, 4, 7) and num_vd.shape == (256,) and check.shape == (256, 5) and edges.shape == (24,)
    assert flexicubes._tables(torch.device("cpu"))[0] is dmc
    # Every case lists exactly as many dual vertices as it has non-empty edge groups.
    assert torch.equal((dmc[..., 0] >= 0).sum(dim=1), num_vd)
