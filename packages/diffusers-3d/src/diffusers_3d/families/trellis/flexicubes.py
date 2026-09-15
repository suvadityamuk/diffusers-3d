# Portions of this file reproduce FlexiCubes iso-surface extraction as pinned by Microsoft TRELLIS:
# https://github.com/MaxtirError/FlexiCubes (revision 815e075a2a400d06c48d94c347674344ed6ae5c5)
# and the sparse cube-to-mesh glue from https://github.com/microsoft/TRELLIS
# (trellis/representations/mesh/{cube2mesh,utils_cube}.py, revision 442aa1e1afb9014e80681d3bf604e8d728a86ee7).
#
# FlexiCubes: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Apache License, Version 2.0.
# TRELLIS: MIT License. Copyright (c) Microsoft Corporation.
#
# Modified: inference only (no regularizers), the grid is built sparsely around the active voxels instead of
# materializing the dense ``res**3`` grid, and ambiguous-face resolution hashes cube coordinates instead of
# indexing a dense volume. Outputs match the dense reference up to vertex order.

"""Pure-PyTorch FlexiCubes mesh extraction for the TRELLIS SLAT mesh decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .flexicubes_tables import CHECK_TABLE, DMC_TABLE, NUM_VD_TABLE

FLEXICUBES_REFERENCE_REVISION = "815e075a2a400d06c48d94c347674344ed6ae5c5"

# Corner ``c`` of a unit cube sits at ``CUBE_CORNERS[c]``; ``CUBE_EDGES`` lists the 12 edges as corner pairs.
CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
CUBE_EDGES = (0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4)
_QUAD_SPLIT_1 = (0, 1, 2, 0, 2, 3)
_QUAD_SPLIT_2 = (0, 1, 3, 3, 1, 2)
_WEIGHT_SCALE = 0.99


@dataclass(frozen=True)
class FlexiCubesResult:
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_colors: torch.Tensor | None


def _codes(coordinates: torch.Tensor, extent: int) -> torch.Tensor:
    return (coordinates[:, 0] * extent + coordinates[:, 1]) * extent + coordinates[:, 2]


def _linear_interp(edge_weights: torch.Tensor, edge_values: torch.Tensor) -> torch.Tensor:
    """Zero crossing along each edge: weights ``(E, 2, 1)`` -> value at ``w1 * x0 - w0 * x1`` over ``w1 - w0``."""

    flipped = torch.cat([edge_weights[:, 1:2], -edge_weights[:, 0:1]], dim=1)
    return (edge_values * flipped).sum(1) / flipped.sum(1)


def _resolve_case_ids(
    occupancy: torch.Tensor,
    cube_coordinates: torch.Tensor,
    check_table: torch.Tensor,
) -> torch.Tensor:
    """Dual-marching-cubes case per surface cube, with the C16/C19 ambiguous-face inversion (suppl. sec. 1.3)."""

    weights = torch.pow(2, torch.arange(8, device=occupancy.device))
    case_ids = (occupancy.to(torch.int64) * weights).sum(-1)
    config = check_table[case_ids]
    to_check = config[:, 0] == 1
    if not bool(to_check.any()):
        return case_ids
    problem_coordinates = cube_coordinates[to_check]
    problem_config = config[to_check]
    adjacent = problem_coordinates + problem_config[:, 1:4]
    extent = int(max(cube_coordinates.max().item(), adjacent.max().item())) + 2
    problem_codes = _codes(problem_coordinates + 1, extent + 1)
    adjacent_codes = _codes(adjacent + 1, extent + 1)
    # A problematic cube whose neighbour across the ambiguous face is also problematic flips to the inverted case.
    to_invert = torch.isin(adjacent_codes, problem_codes)
    indices = torch.nonzero(to_check, as_tuple=False).reshape(-1)[to_invert]
    case_ids = case_ids.clone()
    case_ids[indices] = problem_config[to_invert][:, 4]
    return case_ids


def flexicubes_extract(
    grid_vertices: torch.Tensor,
    scalar_field: torch.Tensor,
    cube_index: torch.Tensor,
    cube_coordinates: torch.Tensor,
    *,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    gamma_f: torch.Tensor,
    vertex_colors: torch.Tensor | None = None,
) -> FlexiCubesResult:
    """Extract the FlexiCubes iso-surface of ``scalar_field`` sampled at ``grid_vertices``.

    ``cube_index`` is ``(N, 8)`` into the vertices following ``CUBE_CORNERS``; ``cube_coordinates`` gives each cube's
    integer grid position (used only to resolve ambiguous faces). ``beta`` ``(N, 12)``, ``alpha`` ``(N, 8)`` and
    ``gamma_f`` ``(N,)`` are the raw learned weights; ``vertex_colors`` ``(V, C)`` are raw logits interpolated like
    positions. Faces wind towards positive ``scalar_field``.
    """

    device = grid_vertices.device
    if grid_vertices.ndim != 2 or grid_vertices.shape[1] != 3:
        raise ValueError("grid_vertices must have shape (V, 3)")
    if scalar_field.shape != (grid_vertices.shape[0],):
        raise ValueError("scalar_field must have shape (V,)")
    if cube_index.ndim != 2 or cube_index.shape[1] != 8:
        raise ValueError("cube_index must have shape (N, 8)")
    num_cubes = cube_index.shape[0]
    if beta.shape != (num_cubes, 12) or alpha.shape != (num_cubes, 8) or gamma_f.shape != (num_cubes,):
        raise ValueError("beta, alpha and gamma_f must have shapes (N, 12), (N, 8) and (N,)")

    dmc_table = torch.tensor(DMC_TABLE, dtype=torch.int64, device=device).reshape(256, 4, 7)
    num_vd_table = torch.tensor(NUM_VD_TABLE, dtype=torch.int64, device=device)
    check_table = torch.tensor(CHECK_TABLE, dtype=torch.int64, device=device).reshape(256, 5)
    cube_edges = torch.tensor(CUBE_EDGES, dtype=torch.int64, device=device)

    occupied = scalar_field < 0
    occupancy = occupied[cube_index.reshape(-1)].reshape(-1, 8)
    occupied_corners = occupancy.sum(-1)
    surface = (occupied_corners > 0) & (occupied_corners < 8)
    empty_colors = None if vertex_colors is None else grid_vertices.new_zeros(0, vertex_colors.shape[-1])
    if not bool(surface.any()):
        return FlexiCubesResult(
            grid_vertices.new_zeros(0, 3), torch.zeros(0, 3, dtype=torch.int64, device=device), empty_colors
        )

    beta = (torch.tanh(beta) * _WEIGHT_SCALE + 1)[surface]
    alpha = (torch.tanh(alpha) * _WEIGHT_SCALE + 1)[surface]
    gamma_f = (torch.sigmoid(gamma_f) * _WEIGHT_SCALE + (1 - _WEIGHT_SCALE) / 2)[surface]
    if vertex_colors is not None:
        vertex_colors = torch.sigmoid(vertex_colors)

    case_ids = _resolve_case_ids(occupancy[surface], cube_coordinates[surface], check_table)

    # Surface-crossing edges, indexed uniquely; ``edge_map`` maps (surface cube, edge) -> unique surface edge or -1.
    surface_cubes = cube_index[surface]
    all_edges = surface_cubes[:, cube_edges].reshape(-1, 2)
    unique_edges, inverse, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)
    crossing = occupied[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
    crossing_map = torch.full((unique_edges.shape[0],), -1, dtype=torch.int64, device=device)
    crossing_map[crossing] = torch.arange(int(crossing.sum()), device=device)
    edge_map = crossing_map[inverse].reshape(-1, 12)
    surface_edge_mask = crossing[inverse]
    edge_counts = counts[inverse]
    surface_edges = unique_edges[crossing]

    # Dual vertices (sec. 4.2): weighted mean of the alpha-shifted zero crossings of each vertex's edge group.
    alpha_edges = alpha[:, cube_edges].reshape(-1, 12, 2)
    edge_positions = grid_vertices[surface_edges.reshape(-1)].reshape(-1, 2, 3)
    edge_scalars = scalar_field[surface_edges.reshape(-1)].reshape(-1, 2, 1)
    edge_colors = None
    if vertex_colors is not None:
        edge_colors = vertex_colors[surface_edges.reshape(-1)].reshape(-1, 2, vertex_colors.shape[-1])

    num_vd = num_vd_table[case_ids]
    groups, group_to_vd, group_to_cube, vd_gamma = [], [], [], []
    total_vd = 0
    for count in torch.unique(num_vd).tolist():
        cubes = num_vd == count
        emitted = int(cubes.sum()) * count
        edge_group = dmc_table[case_ids[cubes], :count].reshape(-1, count * 7)
        vd_ids = (torch.arange(emitted, device=device).unsqueeze(-1).repeat(1, 7) + total_vd).reshape_as(edge_group)
        total_vd += emitted
        cube_ids = torch.nonzero(cubes, as_tuple=False).reshape(-1).unsqueeze(-1).repeat(1, count * 7)
        valid = edge_group != -1
        groups.append(edge_group[valid])
        group_to_vd.append(vd_ids[valid])
        group_to_cube.append(cube_ids[valid])
        vd_gamma.append(gamma_f[cubes].unsqueeze(-1).repeat(1, count).reshape(-1))
    edge_group = torch.cat(groups)
    group_to_vd = torch.cat(group_to_vd)
    group_to_cube = torch.cat(group_to_cube)
    vd_gamma = torch.cat(vd_gamma)

    flat = group_to_cube * 12 + edge_group
    edge_ids = edge_map.reshape(-1)[flat]
    positions = edge_positions[edge_ids]
    scalars = edge_scalars[edge_ids]
    alpha_group = alpha_edges.reshape(-1, 2)[flat].reshape(-1, 2, 1)
    beta_group = beta.reshape(-1)[flat].reshape(-1, 1)
    crossings = _linear_interp(scalars * alpha_group, positions)
    beta_sum = grid_vertices.new_zeros(total_vd, 1).index_add_(0, group_to_vd, beta_group)
    dual_vertices = grid_vertices.new_zeros(total_vd, 3).index_add_(0, group_to_vd, crossings * beta_group) / beta_sum
    dual_colors = None
    if edge_colors is not None:
        color_crossings = _linear_interp(scalars * alpha_group, edge_colors[edge_ids])
        dual_colors = grid_vertices.new_zeros(total_vd, edge_colors.shape[-1])
        dual_colors = dual_colors.index_add_(0, group_to_vd, color_crossings * beta_group) / beta_sum
    vd_index_map = torch.zeros(case_ids.shape[0] * 12, dtype=torch.int64, device=device)
    vd_index_map = vd_index_map.scatter(0, flat, torch.arange(total_vd, device=device)[group_to_vd])

    # Quads (sec. 4.3): each surface edge shared by four cubes links their dual vertices; split by gamma.
    quad_mask = (edge_counts == 4) & surface_edge_mask
    quad_edges = edge_map.reshape(-1)[quad_mask]
    quad_vd = vd_index_map[quad_mask]
    sorted_edges, order = torch.sort(quad_edges, stable=True)
    quad_vd = quad_vd[order].reshape(-1, 4)
    first_edge = sorted_edges.reshape(-1, 4)[:, 0]
    edge_start_scalar = scalar_field[surface_edges[first_edge][:, 0]]
    flip = edge_start_scalar > 0
    quad_vd = torch.cat([quad_vd[flip][:, [0, 1, 3, 2]], quad_vd[~flip][:, [2, 3, 1, 0]]])
    quad_gamma = vd_gamma[quad_vd.reshape(-1)].reshape(-1, 4)
    split_1 = quad_gamma[:, 0] * quad_gamma[:, 2] > quad_gamma[:, 1] * quad_gamma[:, 3]
    faces = torch.zeros(quad_gamma.shape[0], 6, dtype=torch.int64, device=device)
    faces[split_1] = quad_vd[split_1][:, list(_QUAD_SPLIT_1)]
    faces[~split_1] = quad_vd[~split_1][:, list(_QUAD_SPLIT_2)]
    return FlexiCubesResult(dual_vertices, faces.reshape(-1, 3), dual_colors)


def sparse_cubes_to_mesh(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    resolution: int,
    *,
    use_color: bool,
) -> FlexiCubesResult:
    """TRELLIS ``SparseFeatures2Mesh``: per-voxel cube features -> FlexiCubes mesh in the centred unit cube.

    ``features`` columns follow the released layout ``[sdf (8) | deform (8x3) | weights (21) | color (8x6)]``, one
    value per cube corner for the vertex attributes. Corner values shared by several cubes are averaged; grid
    vertices no active cube touches sit outside (``sdf = 1``) and cubes without features carry zero weights.
    """

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    num_color = 8 * 6 if use_color else 0
    if features.shape != (coordinates.shape[0], 8 + 24 + 21 + num_color):
        raise ValueError(f"features must have shape (N, {8 + 24 + 21 + num_color})")
    device = coordinates.device
    coordinates = coordinates.to(torch.int64)
    corners = torch.tensor(CUBE_CORNERS, dtype=torch.int64, device=device)

    sdf = features[:, :8].reshape(-1, 8, 1) - 1.0 / resolution
    deform = features[:, 8:32].reshape(-1, 8, 3)
    weights = features[:, 32:53]
    attributes = [sdf, deform]
    if use_color:
        attributes.append(features[:, 53:].reshape(-1, 8, 6))
    corner_attributes = torch.cat(attributes, dim=-1)

    # Every cube touching an active corner can cross the surface; dilate the active set by one cube.
    offsets = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=device)] * 3, indexing="ij"), -1).reshape(-1, 3)
    candidates = torch.unique((coordinates.unsqueeze(1) + offsets).reshape(-1, 3), dim=0)
    candidates = candidates[(candidates >= 0).all(1) & (candidates < resolution).all(1)]
    corner_coordinates = (candidates.unsqueeze(1) + corners).reshape(-1, 3)
    grid_vertices, cube_index = torch.unique(corner_coordinates, dim=0, return_inverse=True)
    cube_index = cube_index.reshape(-1, 8)

    extent = resolution + 1
    vertex_codes = _codes(grid_vertices, extent)
    active_corner_codes = _codes((coordinates.unsqueeze(1) + corners).reshape(-1, 3), extent)
    vertex_attributes = features.new_zeros(grid_vertices.shape[0], corner_attributes.shape[-1])
    vertex_attributes[:, 0] = 1.0
    active_rows = torch.searchsorted(vertex_codes, active_corner_codes)
    active_count = torch.bincount(active_rows, minlength=grid_vertices.shape[0]).to(features.dtype)
    touched = active_count > 0
    vertex_attributes[touched] = 0.0
    vertex_attributes.index_add_(0, active_rows, corner_attributes.reshape(-1, corner_attributes.shape[-1]))
    vertex_attributes[touched] /= active_count[touched, None]

    cube_codes = _codes(candidates, extent)
    active_cube_rows = torch.searchsorted(cube_codes, _codes(coordinates, extent))
    cube_weights = features.new_zeros(candidates.shape[0], 21)
    cube_weights[active_cube_rows] = weights

    vertex_positions = grid_vertices.to(features.dtype) / resolution - 0.5
    vertex_positions = vertex_positions + (1 - 1e-8) / (resolution * 2) * torch.tanh(vertex_attributes[:, 1:4])
    return flexicubes_extract(
        vertex_positions,
        vertex_attributes[:, 0],
        cube_index,
        candidates,
        beta=cube_weights[:, :12],
        alpha=cube_weights[:, 12:20],
        gamma_f=cube_weights[:, 20],
        vertex_colors=vertex_attributes[:, 4:] if use_color else None,
    )


__all__ = [
    "CUBE_CORNERS",
    "CUBE_EDGES",
    "FLEXICUBES_REFERENCE_REVISION",
    "FlexiCubesResult",
    "flexicubes_extract",
    "sparse_cubes_to_mesh",
]
