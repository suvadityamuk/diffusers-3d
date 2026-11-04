"""The Hammersley low-discrepancy sequence that TRELLIS and TripoSplat use to place views and Gaussian offsets."""

from __future__ import annotations

import torch


def radical_inverse(base: int, index: int) -> float:
    """Van der Corput radical inverse of ``index`` in ``base``, accumulated digit by digit as upstream does."""

    value, inverse = 0.0, 1.0 / base
    scale = inverse
    while index > 0:
        value += index % base * scale
        index //= base
        scale *= inverse
    return value


def hammersley_3d(count: int) -> torch.Tensor:
    """``(count, 3)`` float32 points ``(i / count, radical_inverse(2, i), radical_inverse(3, i))``."""

    return torch.tensor(
        [[index / count, radical_inverse(2, index), radical_inverse(3, index)] for index in range(count)],
        dtype=torch.float32,
    )
