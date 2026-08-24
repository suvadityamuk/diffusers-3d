# Portions of this file reproduce scheduler semantics from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
# Community License Agreement. Copyright (C) 2025 Tencent. All Rights Reserved.
# This file has been modified for native Diffusers integration.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin  # noqa: F401 - external-component loading


class Hunyuan3DFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """Diffusers Euler flow scheduler with Hunyuan's noise-to-data direction.

    Hunyuan3D-2.1 integrates from sigma 0 (noise) to sigma 1 (shape) and
    appends a terminal sigma of 1. Diffusers' base scheduler appends 0 for its
    usual reverse direction, so using it directly changes the final Euler step.
    """

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: Sequence[float] | None = None,
        mu: float | None = None,
        timesteps: Sequence[float] | None = None,
    ) -> None:
        if timesteps is not None:
            raise ValueError("Hunyuan3D scheduler accepts sigma schedules, not custom timesteps")
        if sigmas is None:
            if (
                not isinstance(num_inference_steps, int)
                or isinstance(num_inference_steps, bool)
                or num_inference_steps <= 0
            ):
                raise ValueError("num_inference_steps must be a positive integer")
            sigmas = np.linspace(0.0, 1.0, num_inference_steps, dtype=np.float32)
        else:
            sigmas = np.asarray(tuple(sigmas), dtype=np.float32)
            if sigmas.ndim != 1 or len(sigmas) == 0:
                raise ValueError("sigmas must be a non-empty rank-one sequence")
            if not np.isfinite(sigmas).all() or (np.diff(sigmas) < 0).any():
                raise ValueError("Hunyuan3D sigmas must be finite and monotonically increasing")
            if num_inference_steps is not None and num_inference_steps != len(sigmas):
                raise ValueError("num_inference_steps must match the custom sigma count")
            num_inference_steps = len(sigmas)

        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=list(sigmas),
            mu=mu,
        )
        self.sigmas[-1] = 1.0


__all__ = ["Hunyuan3DFlowMatchEulerDiscreteScheduler"]
