"""Classic marching cubes in plain PyTorch, matching the CPU kernel of ``torchmcubes`` that TripoSR calls.

``torchmcubes`` (https://github.com/tatsy/torchmcubes, revision 3381600, the HEAD that TripoSR's unpinned requirement
installed at TripoSR's pinned revision) polygonises every cell with Paul Bourke's lookup tables. The triangle table
below is Bourke's public table as PyMCubes (BSD-3-Clause) and torchmcubes ship it; the code is an independent
vectorized implementation of the same rules:

- cells run in C order over ``volume[i, j, k]``; a corner is "inside" when its value is strictly below ``threshold``;
- an edge vertex snaps to its first endpoint when ``|threshold - v1| < 1e-4``, to its second when
  ``|threshold - v2| < 1e-4``, to the first when ``|v1 - v2| < 1e-4``, and otherwise sits at ``p1 + mu * (p2 - p1)``
  with ``mu = (threshold - v1) / (v2 - v1)``, all in float32 (``p2 - p1`` is 0 or ±1, so fused multiply-adds round
  the same way);
- triangles are emitted in table order and vertices with exactly equal positions are merged, numbered in order of
  first appearance. torchmcubes built with OpenMP appends whole ``i`` slabs in thread completion order, so its vertex
  numbering is only reproducible single-threaded; this follows the single-threaded order.

On CUDA tensors this follows torchmcubes' CUDA kernel instead, which TripoSR runs on GPU: it keeps only the first
four of the five triangles that 32 of the 256 cases produce, and does not merge vertices (triangle ``t`` owns vertices
``3t`` to ``3t + 2``). Cells, corners, edges, and vertex positions are as above. Where nothing crosses the threshold,
that kernel returns one all-zero triangle; this returns an empty mesh.
"""

from __future__ import annotations

import torch

# Corners of a cell as (i, j, k) offsets, in Bourke's corner numbering as torchmcubes lays it on ``volume[i, j, k]``.
_CORNERS = ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))
# Bourke's edges as (first, second) corners; vertices interpolate from the first corner.
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
# Edges of each triangle, three per triangle, for every cube index (bit ``c`` set when corner ``c`` is inside).
_TRIANGLES = (
    (),
    (0, 8, 3),
    (0, 1, 9),
    (1, 8, 3, 9, 8, 1),
    (1, 2, 10),
    (0, 8, 3, 1, 2, 10),
    (9, 2, 10, 0, 2, 9),
    (2, 8, 3, 2, 10, 8, 10, 9, 8),
    (3, 11, 2),
    (0, 11, 2, 8, 11, 0),
    (1, 9, 0, 2, 3, 11),
    (1, 11, 2, 1, 9, 11, 9, 8, 11),
    (3, 10, 1, 11, 10, 3),
    (0, 10, 1, 0, 8, 10, 8, 11, 10),
    (3, 9, 0, 3, 11, 9, 11, 10, 9),
    (9, 8, 10, 10, 8, 11),
    (4, 7, 8),
    (4, 3, 0, 7, 3, 4),
    (0, 1, 9, 8, 4, 7),
    (4, 1, 9, 4, 7, 1, 7, 3, 1),
    (1, 2, 10, 8, 4, 7),
    (3, 4, 7, 3, 0, 4, 1, 2, 10),
    (9, 2, 10, 9, 0, 2, 8, 4, 7),
    (2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4),
    (8, 4, 7, 3, 11, 2),
    (11, 4, 7, 11, 2, 4, 2, 0, 4),
    (9, 0, 1, 8, 4, 7, 2, 3, 11),
    (4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1),
    (3, 10, 1, 3, 11, 10, 7, 8, 4),
    (1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4),
    (4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3),
    (4, 7, 11, 4, 11, 9, 9, 11, 10),
    (9, 5, 4),
    (9, 5, 4, 0, 8, 3),
    (0, 5, 4, 1, 5, 0),
    (8, 5, 4, 8, 3, 5, 3, 1, 5),
    (1, 2, 10, 9, 5, 4),
    (3, 0, 8, 1, 2, 10, 4, 9, 5),
    (5, 2, 10, 5, 4, 2, 4, 0, 2),
    (2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8),
    (9, 5, 4, 2, 3, 11),
    (0, 11, 2, 0, 8, 11, 4, 9, 5),
    (0, 5, 4, 0, 1, 5, 2, 3, 11),
    (2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5),
    (10, 3, 11, 10, 1, 3, 9, 5, 4),
    (4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10),
    (5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3),
    (5, 4, 8, 5, 8, 10, 10, 8, 11),
    (9, 7, 8, 5, 7, 9),
    (9, 3, 0, 9, 5, 3, 5, 7, 3),
    (0, 7, 8, 0, 1, 7, 1, 5, 7),
    (1, 5, 3, 3, 5, 7),
    (9, 7, 8, 9, 5, 7, 10, 1, 2),
    (10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3),
    (8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2),
    (2, 10, 5, 2, 5, 3, 3, 5, 7),
    (7, 9, 5, 7, 8, 9, 3, 11, 2),
    (9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11),
    (2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7),
    (11, 2, 1, 11, 1, 7, 7, 1, 5),
    (9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11),
    (5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0),
    (11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0),
    (11, 10, 5, 7, 11, 5),
    (10, 6, 5),
    (0, 8, 3, 5, 10, 6),
    (9, 0, 1, 5, 10, 6),
    (1, 8, 3, 1, 9, 8, 5, 10, 6),
    (1, 6, 5, 2, 6, 1),
    (1, 6, 5, 1, 2, 6, 3, 0, 8),
    (9, 6, 5, 9, 0, 6, 0, 2, 6),
    (5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8),
    (2, 3, 11, 10, 6, 5),
    (11, 0, 8, 11, 2, 0, 10, 6, 5),
    (0, 1, 9, 2, 3, 11, 5, 10, 6),
    (5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11),
    (6, 3, 11, 6, 5, 3, 5, 1, 3),
    (0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6),
    (3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9),
    (6, 5, 9, 6, 9, 11, 11, 9, 8),
    (5, 10, 6, 4, 7, 8),
    (4, 3, 0, 4, 7, 3, 6, 5, 10),
    (1, 9, 0, 5, 10, 6, 8, 4, 7),
    (10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4),
    (6, 1, 2, 6, 5, 1, 4, 7, 8),
    (1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7),
    (8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6),
    (7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9),
    (3, 11, 2, 7, 8, 4, 10, 6, 5),
    (5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11),
    (0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6),
    (9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6),
    (8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6),
    (5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11),
    (0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7),
    (6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9),
    (10, 4, 9, 6, 4, 10),
    (4, 10, 6, 4, 9, 10, 0, 8, 3),
    (10, 0, 1, 10, 6, 0, 6, 4, 0),
    (8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10),
    (1, 4, 9, 1, 2, 4, 2, 6, 4),
    (3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4),
    (0, 2, 4, 4, 2, 6),
    (8, 3, 2, 8, 2, 4, 4, 2, 6),
    (10, 4, 9, 10, 6, 4, 11, 2, 3),
    (0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6),
    (3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10),
    (6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1),
    (9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3),
    (8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1),
    (3, 11, 6, 3, 6, 0, 0, 6, 4),
    (6, 4, 8, 11, 6, 8),
    (7, 10, 6, 7, 8, 10, 8, 9, 10),
    (0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10),
    (10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0),
    (10, 6, 7, 10, 7, 1, 1, 7, 3),
    (1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7),
    (2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9),
    (7, 8, 0, 7, 0, 6, 6, 0, 2),
    (7, 3, 2, 6, 7, 2),
    (2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7),
    (2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7),
    (1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11),
    (11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1),
    (8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6),
    (0, 9, 1, 11, 6, 7),
    (7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0),
    (7, 11, 6),
    (7, 6, 11),
    (3, 0, 8, 11, 7, 6),
    (0, 1, 9, 11, 7, 6),
    (8, 1, 9, 8, 3, 1, 11, 7, 6),
    (10, 1, 2, 6, 11, 7),
    (1, 2, 10, 3, 0, 8, 6, 11, 7),
    (2, 9, 0, 2, 10, 9, 6, 11, 7),
    (6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8),
    (7, 2, 3, 6, 2, 7),
    (7, 0, 8, 7, 6, 0, 6, 2, 0),
    (2, 7, 6, 2, 3, 7, 0, 1, 9),
    (1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6),
    (10, 7, 6, 10, 1, 7, 1, 3, 7),
    (10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8),
    (0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7),
    (7, 6, 10, 7, 10, 8, 8, 10, 9),
    (6, 8, 4, 11, 8, 6),
    (3, 6, 11, 3, 0, 6, 0, 4, 6),
    (8, 6, 11, 8, 4, 6, 9, 0, 1),
    (9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6),
    (6, 8, 4, 6, 11, 8, 2, 10, 1),
    (1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6),
    (4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9),
    (10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3),
    (8, 2, 3, 8, 4, 2, 4, 6, 2),
    (0, 4, 2, 4, 6, 2),
    (1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8),
    (1, 9, 4, 1, 4, 2, 2, 4, 6),
    (8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1),
    (10, 1, 0, 10, 0, 6, 6, 0, 4),
    (4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3),
    (10, 9, 4, 6, 10, 4),
    (4, 9, 5, 7, 6, 11),
    (0, 8, 3, 4, 9, 5, 11, 7, 6),
    (5, 0, 1, 5, 4, 0, 7, 6, 11),
    (11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5),
    (9, 5, 4, 10, 1, 2, 7, 6, 11),
    (6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5),
    (7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2),
    (3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6),
    (7, 2, 3, 7, 6, 2, 5, 4, 9),
    (9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7),
    (3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0),
    (6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8),
    (9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7),
    (1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4),
    (4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10),
    (7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10),
    (6, 9, 5, 6, 11, 9, 11, 8, 9),
    (3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5),
    (0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11),
    (6, 11, 3, 6, 3, 5, 5, 3, 1),
    (1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6),
    (0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10),
    (11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5),
    (6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3),
    (5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2),
    (9, 5, 6, 9, 6, 0, 0, 6, 2),
    (1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8),
    (1, 5, 6, 2, 1, 6),
    (1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6),
    (10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0),
    (0, 3, 8, 5, 6, 10),
    (10, 5, 6),
    (11, 5, 10, 7, 5, 11),
    (11, 5, 10, 11, 7, 5, 8, 3, 0),
    (5, 11, 7, 5, 10, 11, 1, 9, 0),
    (10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1),
    (11, 1, 2, 11, 7, 1, 7, 5, 1),
    (0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11),
    (9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7),
    (7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2),
    (2, 5, 10, 2, 3, 5, 3, 7, 5),
    (8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5),
    (9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2),
    (9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2),
    (1, 3, 5, 3, 7, 5),
    (0, 8, 7, 0, 7, 1, 1, 7, 5),
    (9, 0, 3, 9, 3, 5, 5, 3, 7),
    (9, 8, 7, 5, 9, 7),
    (5, 8, 4, 5, 10, 8, 10, 11, 8),
    (5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0),
    (0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5),
    (10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4),
    (2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8),
    (0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11),
    (0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5),
    (9, 4, 5, 2, 11, 3),
    (2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4),
    (5, 10, 2, 5, 2, 4, 4, 2, 0),
    (3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9),
    (5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2),
    (8, 4, 5, 8, 5, 3, 3, 5, 1),
    (0, 4, 5, 1, 0, 5),
    (8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5),
    (9, 4, 5),
    (4, 11, 7, 4, 9, 11, 9, 10, 11),
    (0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11),
    (1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11),
    (3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4),
    (4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2),
    (9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3),
    (11, 7, 4, 11, 4, 2, 2, 4, 0),
    (11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4),
    (2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9),
    (9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7),
    (3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10),
    (1, 10, 2, 8, 7, 4),
    (4, 9, 1, 4, 1, 7, 7, 1, 3),
    (4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1),
    (4, 0, 3, 7, 4, 3),
    (4, 8, 7),
    (9, 10, 8, 10, 11, 8),
    (3, 0, 9, 3, 9, 11, 11, 9, 10),
    (0, 1, 10, 0, 10, 8, 8, 10, 11),
    (3, 1, 10, 11, 3, 10),
    (1, 2, 11, 1, 11, 9, 9, 11, 8),
    (3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9),
    (0, 2, 11, 8, 0, 11),
    (3, 2, 11),
    (2, 3, 8, 2, 8, 10, 10, 8, 9),
    (9, 10, 2, 0, 9, 2),
    (2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8),
    (1, 10, 2),
    (1, 3, 8, 9, 1, 8),
    (0, 9, 1),
    (0, 3, 8),
    (),
)


def unique_rows_in_order(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly equal rows of ``(N, D)`` ``rows``, numbered in order of first appearance.

    Returns the index of each distinct row's first appearance and, for every row, the number of its distinct row.
    """

    unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
    positions = torch.arange(rows.shape[0], device=rows.device)
    first = positions.new_full((unique.shape[0],), rows.shape[0]).scatter_reduce(0, inverse, positions, "amin")
    order = first.argsort()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.shape[0], device=rows.device)
    return first[order], rank[inverse]


def marching_cubes(
    volume: torch.Tensor, threshold: float = 0.0, *, _cuda_kernel: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mesh where the float32 ``(I, J, K)`` ``volume`` crosses ``threshold``, as ``torchmcubes.marching_cubes``.

    Returns float32 vertices ``(V, 3)`` in index units, ordered ``(i, j, k)`` like the volume axes (torchmcubes
    returns ``(k, j, i)``), and int64 faces ``(F, 3)``, on the volume's device. Triangles face the side below
    ``threshold``; degenerate triangles from snapped vertices are kept. A volume that never crosses ``threshold``
    gives an empty mesh. ``_cuda_kernel`` (default: ``volume.is_cuda``) picks torchmcubes' CUDA-kernel behaviour.
    """

    if volume.ndim != 3:
        raise ValueError(f"volume must be a rank-3 tensor, got shape {tuple(volume.shape)}")
    cuda_kernel = volume.is_cuda if _cuda_kernel is None else _cuda_kernel
    volume = volume.detach().float()
    device = volume.device
    level = torch.tensor(threshold, dtype=torch.float32, device=device)
    snap = torch.tensor(1e-4, dtype=torch.float32, device=device)
    cells = tuple(max(size - 1, 0) for size in volume.shape)
    cube = torch.zeros(cells, dtype=torch.uint8, device=device)
    for bit, (i, j, k) in enumerate(_CORNERS):
        corner = volume[i : i + cells[0], j : j + cells[1], k : k + cells[2]]
        cube |= (corner < level).to(torch.uint8) << bit
    cube = cube.flatten()
    cell = ((cube != 0) & (cube != 255)).nonzero()[:, 0]

    table = torch.tensor([edges + (-1,) * (15 - len(edges)) for edges in _TRIANGLES], device=device)
    if cuda_kernel:
        table = table[:, :12]
    edges = table[cube[cell].long()]
    valid = edges >= 0
    edge = edges[valid]
    cell = cell[:, None].expand_as(edges)[valid]
    origin = torch.stack((cell // (cells[1] * cells[2]), cell // cells[2] % cells[1], cell % cells[2]), dim=1)
    corners = torch.tensor(_CORNERS, dtype=torch.int64, device=device)
    endpoints = torch.tensor(_EDGES, dtype=torch.int64, device=device)[edge]
    p1, p2 = origin + corners[endpoints[:, 0]], origin + corners[endpoints[:, 1]]
    v1, v2 = volume[p1.unbind(1)], volume[p2.unbind(1)]
    p1, p2 = p1.float(), p2.float()
    mu = (level - v1) / (v2 - v1)
    points = p1 + mu[:, None] * (p2 - p1)
    points = torch.where(((v1 - v2).abs() < snap)[:, None], p1, points)
    points = torch.where(((level - v2).abs() < snap)[:, None], p2, points)
    points = torch.where(((level - v1).abs() < snap)[:, None], p1, points)
    if cuda_kernel or points.shape[0] == 0:
        return points, torch.arange(points.shape[0], device=device).reshape(-1, 3)
    first, inverse = unique_rows_in_order(points)
    return points[first], inverse.reshape(-1, 3)


__all__ = ["marching_cubes", "unique_rows_in_order"]
