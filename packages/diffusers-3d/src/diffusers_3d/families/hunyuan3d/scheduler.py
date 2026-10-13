# Portions of this file reproduce scheduler semantics from Tencent Hunyuan3D-2.1:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
# Revision: 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#
# Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement.
# Copyright (C) 2025 Tencent. All Rights Reserved.
# Modified: expressed as the stock Diffusers flow-match Euler scheduler with Hunyuan's sigma schedule.

from __future__ import annotations

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin  # noqa: F401 - required by external-component loading


class Hunyuan3DFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """Stock flow-match Euler stepping that integrates from noise at sigma 0 to the shape at sigma 1.

    Hunyuan3D-2.1 samples ``linspace(0, 1, num_inference_steps)`` and appends a terminal sigma of 1, so the final
    step leaves the latents unchanged. The stock scheduler runs the opposite direction and appends 0.
    """

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        super().set_timesteps(sigmas=np.linspace(0.0, 1.0, num_inference_steps), device=device)
        self.sigmas[-1] = 1.0


__all__ = ["Hunyuan3DFlowMatchEulerDiscreteScheduler"]
