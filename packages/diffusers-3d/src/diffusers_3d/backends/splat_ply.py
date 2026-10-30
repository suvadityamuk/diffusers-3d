"""Read and write Gaussian splats in the PLY layout of the 3D Gaussian Splatting reference code.

Each Gaussian is one ``vertex`` of float32 properties: ``x y z``, ``nx ny nz`` (zeros), ``f_dc_0..2``, ``f_rest_*``
(channel-major: all red coefficients, then green, then blue), ``opacity`` (logit), ``scale_0..2`` (log), and
``rot_0..3`` (quaternion w, x, y, z). Viewers such as SuperSplat and the reference SIBR viewer read this layout.
"""

from __future__ import annotations

import math
import os
from typing import BinaryIO

import numpy as np
import torch

from ..objects import CoordinateSystem, GaussianSplatAsset


def write_splat_ply(file: str | os.PathLike[str] | BinaryIO, gaussians: GaussianSplatAsset) -> None:
    gaussians = gaussians.to("cpu", torch.float32)
    if not torch.allclose(gaussians.transform, torch.eye(4)):
        raise ValueError("PLY cannot store an object transform; bake it into the Gaussians first")
    count, coefficients, _ = gaussians.sh_coefficients.shape
    rest = gaussians.sh_coefficients[:, 1:].transpose(1, 2).reshape(count, -1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{index}" for index in range(rest.shape[1])]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    columns = torch.cat(
        [
            gaussians.means,
            torch.zeros(count, 3),
            gaussians.sh_coefficients[:, 0],
            rest,
            gaussians.opacity_logits.reshape(count, 1),
            gaussians.log_scales,
            gaussians.quaternions_wxyz,
        ],
        dim=1,
    )
    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{}end_header\n".format(
        count, "".join(f"property float {name}\n" for name in names)
    )
    payload = header.encode("ascii") + columns.numpy().astype("<f4").tobytes()
    if isinstance(file, (str, os.PathLike)):
        with open(file, "wb") as handle:
            handle.write(payload)
    else:
        file.write(payload)


def read_splat_ply(
    file: str | os.PathLike[str] | BinaryIO,
    *,
    coordinate_system: CoordinateSystem | str = CoordinateSystem.RIGHT_HANDED_Y_UP,
) -> GaussianSplatAsset:
    """Read a binary little-endian Gaussian-splat PLY; quaternions are renormalized."""

    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as handle:
            data = handle.read()
    else:
        data = file.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    lines = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only binary little-endian PLY files are supported")
    count, names, in_vertex = 0, [], False
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            in_vertex = parts[1] == "vertex"
            count = int(parts[2]) if in_vertex else count
        elif parts[:1] == ["property"] and in_vertex:
            if parts[1] not in ("float", "float32"):
                raise ValueError(f"unsupported PLY property type {parts[1]!r} for {parts[-1]!r}")
            names.append(parts[-1])
    table = np.frombuffer(data, dtype=[(name, "<f4") for name in names], count=count, offset=end)

    def stack(prefix: str, size: int) -> torch.Tensor:
        if size == 0:
            return torch.zeros(count, 0)
        return torch.from_numpy(np.stack([table[f"{prefix}{index}"] for index in range(size)], axis=1).copy())

    rest_count = sum(name.startswith("f_rest_") for name in names)
    coefficients = 1 + rest_count // 3
    if (math.isqrt(coefficients)) ** 2 != coefficients:
        raise ValueError(f"{rest_count} f_rest properties do not form a spherical-harmonics basis")
    rest = stack("f_rest_", rest_count).reshape(count, 3, coefficients - 1).transpose(1, 2)
    quaternions = stack("rot_", 4)
    return GaussianSplatAsset(
        means=torch.from_numpy(np.stack([table[axis] for axis in "xyz"], axis=1).copy()),
        log_scales=stack("scale_", 3),
        quaternions_wxyz=quaternions / quaternions.norm(dim=1, keepdim=True),
        opacity_logits=torch.from_numpy(table["opacity"].copy())[:, None],
        sh_coefficients=torch.cat([stack("f_dc_", 3)[:, None], rest], dim=1),
        active_sh_degree=math.isqrt(coefficients) - 1,
        coordinate_system=coordinate_system,
    )


__all__ = ["read_splat_ply", "write_splat_ply"]
