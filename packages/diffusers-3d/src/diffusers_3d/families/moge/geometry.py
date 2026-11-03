# Portions of this file reproduce the MoGe point-map post-processing (``moge/utils/geometry_torch.py``,
# ``moge/utils/geometry_numpy.py``):
# https://github.com/microsoft/MoGe
# Revisions: a8c37341bc0325ca99b9d57981cc3bb2bd3e255b (MoGe) and 74fbce054ebed49800de42d0ad0e83495065719a (MoGe-2)
# MIT License. Copyright (c) Microsoft Corporation.
#
# The shift solver reproduces SciPy's ``least_squares(method="lm")`` (SciPy 1.18.1, BSD-3-Clause, Copyright (c)
# 2001-2002 Enthought, Inc. 2003, SciPy Developers), whose ``lmder``/``lmpar``/``qrsolv`` are a C translation of
# MINPACK (Argonne National Laboratory; Garbow, Hillstrom, More). Specialized to one unknown and run on CPU in
# Python floats and NumPy float64, as upstream runs it, without SciPy.

"""Focal and shift recovery and camera helpers shared by the MoGe and MoGe-2 models."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_EPSILON = 2.220446049250313e-16  # float64 machine epsilon (MINPACK ``dpmpar[0]``)
_DWARF = 2.2250738585072014e-308  # smallest normal float64 (MINPACK ``dpmpar[1]``)


def normalized_view_plane_uv(
    width: int, height: int, aspect_ratio: float | None = None, dtype=None, device=None
) -> torch.Tensor:
    """Pixel-centre view-plane coordinates spanning ``±width / diagonal`` and ``±height / diagonal``, shape (H, W, 2)."""

    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def _residuals(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: float, focal: float | None) -> np.ndarray:
    """``focal * xy / (z + shift) - uv``, with the least-squares focal when ``focal`` is None.

    NumPy, as upstream: the forward-difference Jacobian divides by a 1.5e-8 step, so the reduction order of these
    float64 sums shows up in the solution and has to match NumPy's pairwise summation.
    """

    xy_projected = xy / (z + np.float64(shift))[:, None]
    if focal is None:
        focal = (xy_projected * uv).sum() / np.square(xy_projected).sum()
    return (focal * xy_projected - uv).ravel()


def _sequential_sum(vector: np.ndarray) -> float:
    """Left-to-right sum, as MINPACK's loops accumulate (``np.add.accumulate`` is strictly sequential)."""

    return float(np.add.accumulate(vector)[-1])


def _norm(vector: np.ndarray) -> float:
    """MINPACK ``enorm`` for values between its underflow and overflow guards."""

    return math.sqrt(_sequential_sum(vector * vector))


def solve_shift(
    uv: np.ndarray,
    xy: np.ndarray,
    z: np.ndarray,
    focal: float | None = None,
    *,
    ftol: float = 1e-3,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    factor: float = 100.0,
    max_nfev: int = 100,
) -> tuple[float, int, int]:
    """``least_squares(residuals, x0=0, ftol=1e-3, method="lm")`` over the z shift, as SciPy >= 1.16 runs it.

    The default ``x_scale="jac"`` (MINPACK mode 1) scales by the Jacobian norm, and the Jacobian is a forward
    difference with step ``sqrt(eps) * max(1, |x|)``. Returns the shift, the number of function evaluations
    MINPACK counts, and MINPACK's ``info`` status. ``uv`` (N, 2), ``xy`` (N, 2) and ``z`` (N,) are float32 arrays.
    """

    def residuals(shift: float) -> np.ndarray:
        return _residuals(uv, xy, z, shift, focal)

    x = 0.0
    fvec = residuals(x)
    nfev = 1
    fnorm = _norm(fvec)
    par, iteration, delta, xnorm, diag = 0.0, 1, 0.0, 0.0, 0.0
    while True:
        # Forward-difference Jacobian (``scipy.optimize._numdiff`` "2-point").
        step = _EPSILON**0.5 * (1.0 if x >= 0 else -1.0) * max(1.0, abs(x))
        jacobian = (residuals(x + step) - fvec) / ((x + step) - x)

        # Householder QR of the single column: R = rdiag, ``householder`` the reflector.
        column_norm = _norm(jacobian)
        r = -0.0
        householder = jacobian
        if column_norm != 0.0:
            norm = -column_norm if float(jacobian[0]) < 0.0 else column_norm
            householder = jacobian / norm
            householder[0] += 1.0
            r = -norm
        if iteration == 1:
            diag = column_norm if column_norm != 0.0 else 1.0
            xnorm = abs(diag * x)
            delta = factor * xnorm if factor * xnorm != 0.0 else factor

        # First component of Q^T fvec.
        head = float(householder[0])
        qtf = float(fvec[0])
        if head != 0.0:
            qtf = qtf + head * (-_sequential_sum(householder * fvec) / head)

        gnorm = 0.0
        if fnorm != 0.0 and column_norm != 0.0:
            gnorm = abs(r * (qtf / fnorm) / column_norm)
        if gnorm <= gtol:
            return x, nfev, 4
        diag = max(diag, column_norm)

        while True:
            step, par = _levenberg_marquardt_step(r, diag, qtf, delta, par)
            step = -step
            trial = x + step
            pnorm = abs(diag * step)
            if iteration == 1:
                delta = min(delta, pnorm)
            trial_fvec = residuals(trial)
            nfev += 1
            fnorm1 = _norm(trial_fvec)
            actred = 1.0 - (fnorm1 / fnorm) ** 2 if 0.1 * fnorm1 < fnorm else -1.0
            temp1 = abs(r * step) / fnorm
            temp2 = (math.sqrt(par) * pnorm) / fnorm
            prered = temp1 * temp1 + temp2 * temp2 / 0.5
            dirder = -(temp1 * temp1 + temp2 * temp2)
            ratio = actred / prered if prered != 0.0 else 0.0

            if ratio <= 0.25:
                temp = (0.5 * dirder) / (dirder + 0.5 * actred) if actred < 0.0 else 0.5
                if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                    temp = 0.1
                delta = temp * min(delta, pnorm / 0.1)
                par /= temp
            elif par == 0.0 or ratio >= 0.75:
                delta = pnorm / 0.5
                par *= 0.5

            if ratio >= 1e-4:
                x, fvec, fnorm = trial, trial_fvec, fnorm1
                xnorm = abs(diag * x)
                iteration += 1

            converged = abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1.0
            small_step = delta <= xtol * xnorm
            if converged or small_step:
                return x, nfev, 3 if converged and small_step else (1 if converged else 2)
            # MINPACK assigns these in order, so the last test that holds sets the status.
            status = 0
            if nfev >= max_nfev:
                status = 5
            if abs(actred) <= _EPSILON and prered <= _EPSILON and 0.5 * ratio <= 1.0:
                status = 6
            if delta <= _EPSILON * xnorm:
                status = 7
            if gnorm <= _EPSILON:
                status = 8
            if status:
                return x, nfev, status
            if ratio >= 1e-4:
                break


def _levenberg_marquardt_step(r: float, diag: float, qtf: float, delta: float, par: float) -> tuple[float, float]:
    """MINPACK ``lmpar`` for one unknown: the step and the Levenberg-Marquardt parameter."""

    singular = r == 0.0
    step = 0.0 if singular else qtf / r
    dxnorm = abs(diag * step)
    fp = dxnorm - delta
    if fp <= 0.1 * delta:
        return step, 0.0

    parl = 0.0
    if not singular:
        temp = abs(diag * ((diag * step) / dxnorm) / r)
        parl = ((fp / delta) / temp) / temp
    gnorm = abs((r * qtf) / diag)
    paru = gnorm / delta
    if paru == 0.0:
        paru = _DWARF / min(delta, 0.1)
    par = min(max(par, parl), paru)
    if par == 0.0:
        par = gnorm / dxnorm

    for iteration in range(1, 11):
        if par == 0.0:
            par = max(_DWARF, paru * 0.001)
        damping = math.sqrt(par) * diag
        # ``qrsolv``: one Givens rotation folds the damping row into R.
        sdiag, rhs = r, qtf
        if damping != 0.0:
            if abs(r) < abs(damping):
                cotangent = r / damping
                sine = 0.5 / math.sqrt(0.25 + 0.25 * cotangent**2)
                cosine = sine * cotangent
            else:
                tangent = damping / r
                cosine = 0.5 / math.sqrt(0.25 + 0.25 * tangent**2)
                sine = cosine * tangent
            sdiag = r * cosine
            sdiag += sine * damping
            rhs = cosine * qtf + sine * 0.0
        step = 0.0 if sdiag == 0.0 else rhs / sdiag

        dxnorm = abs(diag * step)
        previous_fp, fp = fp, dxnorm - delta
        if abs(fp) <= 0.1 * delta or (parl == 0.0 and fp <= previous_fp and previous_fp < 0.0) or iteration == 10:
            return step, par
        temp = abs(diag * ((diag * step) / dxnorm) / sdiag)
        parc = ((fp / delta) / temp) / temp
        if fp > 0.0:
            parl = max(parl, par)
        if fp < 0.0:
            paru = min(paru, par)
        par = max(parl, par + parc)
    return step, par


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the focal (relative to the half diagonal) and z shift of an affine point map, as MoGe does.

    ``points`` is (..., H, W, 3) and ``mask`` (..., H, W) bool. The point map, view-plane UVs, and mask are
    nearest-downsampled to ``downsample_size``; the shift is solved on CPU in float64 as upstream does with NumPy and
    SciPy, then cast to float32. With ``focal`` given, only the shift is solved.
    """

    shape = points.shape
    height, width = shape[-3], shape[-2]
    points = points.reshape(-1, *shape[-3:])
    mask = mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest").squeeze(0)
    uv_lr = uv_lr.permute(1, 2, 0)
    mask_lr = F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1) > 0

    points_lr, uv_lr, mask_lr = points_lr.detach().cpu(), uv_lr.cpu(), mask_lr.cpu()
    shifts, focals = [], []
    for index in range(points.shape[0]):
        selected_points, selected_uv = points_lr[index][mask_lr[index]], uv_lr[mask_lr[index]]
        if selected_uv.shape[0] < 2:
            focals.append(1.0)
            shifts.append(0.0)
            continue
        uv_np, xyz_np = selected_uv.to(torch.float32).numpy(), selected_points.to(torch.float32).numpy()
        given_focal = None if focal is None else float(focal[index].to(torch.float32))
        shift, _, _ = solve_shift(uv_np, xyz_np[:, :2], xyz_np[:, 2], given_focal)
        shift = np.float32(shift)
        shifts.append(float(shift))
        if focal is None:
            # Upstream recomputes the focal from the float32 shift, in float32 NumPy arithmetic.
            projected = xyz_np[:, :2] / (xyz_np[:, 2] + shift)[:, None]
            focals.append(float((projected * uv_np).sum() / np.square(projected).sum()))

    shift = torch.tensor(shifts, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal is None:
        focal = torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        focal = focal.reshape(shape[:-3])
    return focal, shift


def focal_to_intrinsics(focal: torch.Tensor, aspect_ratio: float) -> torch.Tensor:
    """Normalized OpenCV intrinsics (principal point at 0.5, 0.5) for a focal relative to the half diagonal."""

    fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
    fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
    # The principal point is a float32 tensor upstream, so half-precision focals come back as float32 intrinsics.
    zeros, ones, half = torch.zeros_like(fx), torch.ones_like(fx), torch.full_like(fx, 0.5, dtype=torch.float32)
    return torch.stack([fx, zeros, half, zeros, fy, half, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, *, homogeneous: bool = False) -> torch.Tensor:
    """Unproject a (..., H, W) depth map through normalized intrinsics into OpenCV camera space (x right, y down).

    MoGe's ``utils3d`` multiplies ``[u, v, 1]`` by the inverse intrinsics and then by depth; MoGe-2's
    ``utils3d_moge`` (``homogeneous=True``) inverts the 4x4 homogeneous intrinsics and applies it to
    ``[u d, v d, d, 1]``. Both are reproduced so the points match bit for bit.
    """

    height, width = depth.shape[-2:]
    u_end, v_end = (
        (1 - 0.5 / width, 1 - 0.5 / height) if homogeneous else ((width - 0.5) / width, (height - 0.5) / height)
    )
    u = torch.linspace(0.5 / width, u_end, width, dtype=depth.dtype, device=depth.device)
    v = torch.linspace(0.5 / height, v_end, height, dtype=depth.dtype, device=depth.device)
    uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)
    ones = torch.ones_like(uv[..., :1])
    if not homogeneous:
        points = torch.cat([uv, ones], dim=-1) @ torch.inverse(intrinsics[..., None, :, :]).transpose(-2, -1)
        return points * depth[..., None]
    transform = torch.zeros(*intrinsics.shape[:-2], 4, 4, dtype=intrinsics.dtype, device=intrinsics.device)
    transform[..., :3, :3] = intrinsics
    transform[..., 3, 3] = 1
    points = torch.cat([uv.expand(*depth.shape, 2), ones.expand(*depth.shape, 1)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return (points @ torch.linalg.inv(transform)[..., None, :, :].mT)[..., :3]
