from __future__ import annotations

import torch

from ._validation import normalize_coordinate_system
from .types import CoordinateSystem

# Each convention as a basis written in right-handed Y-up coordinates: row ``i`` holds the Y-up coordinates of the
# convention's ``i``-th axis. Z-up follows TRELLIS (Y-up ``y`` is Z-up ``z``, Y-up ``z`` is Z-up ``-y``); the
# left-handed conventions mirror their right-handed counterpart across the horizontal axis that is not "up".
_BASES: dict[CoordinateSystem, tuple[tuple[float, float, float], ...]] = {
    CoordinateSystem.RIGHT_HANDED_Y_UP: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    CoordinateSystem.RIGHT_HANDED_Z_UP: ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    CoordinateSystem.LEFT_HANDED_Y_UP: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    CoordinateSystem.LEFT_HANDED_Z_UP: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def coordinate_change_matrix(
    source: CoordinateSystem | str,
    target: CoordinateSystem | str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the 3x3 matrix ``M`` with ``target_points = source_points @ M.T``.

    ``M`` is orthonormal: use it for positions and normals alike. Its determinant is ``-1`` exactly when the
    handedness changes, in which case triangle winding must be flipped to keep faces front-facing.
    """

    source_basis = torch.tensor(_BASES[normalize_coordinate_system(source)], device=device, dtype=dtype)
    target_basis = torch.tensor(_BASES[normalize_coordinate_system(target)], device=device, dtype=dtype)
    # points_yup = points_source @ source_basis; points_target = points_yup @ target_basis^T (orthonormal inverse)
    return target_basis @ source_basis.T


def changes_handedness(source: CoordinateSystem | str, target: CoordinateSystem | str) -> bool:
    return bool(torch.det(coordinate_change_matrix(source, target, dtype=torch.float64)) < 0)


__all__ = ["changes_handedness", "coordinate_change_matrix"]
