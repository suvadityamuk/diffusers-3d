# Portions of this file reproduce sampler semantics from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for the Diffusers scheduler lifecycle and package-owned sparse tensors.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin  # noqa: F401 - Diffusers looks the base class up here

from ..trellis.scheduler import (
    TrellisFlowEulerScheduler,
    TrellisFlowEulerSchedulerOutput,
    _features,
    _replace,
    _validate_pair,
)
from ..trellis.sparse import TrellisSparseTensor

Trellis2FlowEulerSchedulerOutput = TrellisFlowEulerSchedulerOutput


class Trellis2FlowEulerScheduler(TrellisFlowEulerScheduler):
    """TRELLIS's Euler flow scheduler with TRELLIS.2's guidance: ``w * cond + (1 - w) * neg`` and x0 rescaling."""

    def original_to_velocity(
        self,
        sample: torch.Tensor | TrellisSparseTensor,
        original: torch.Tensor | TrellisSparseTensor,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor | TrellisSparseTensor:
        _validate_pair(sample, original, names="sample and original")
        sample_features = _features(sample)
        original_features = _features(original)
        timestep = torch.as_tensor(timestep, device=sample_features.device, dtype=sample_features.dtype)
        velocity = ((1 - self.config.sigma_min) * sample_features - original_features) / (
            self.config.sigma_min + (1 - self.config.sigma_min) * timestep
        )
        return _replace(sample, velocity)

    def apply_guidance(
        self,
        conditional: torch.Tensor | TrellisSparseTensor,
        negative: torch.Tensor | TrellisSparseTensor,
        guidance_strength: float,
        *,
        sample: torch.Tensor | TrellisSparseTensor | None = None,
        timestep: torch.Tensor | float | None = None,
        guidance_rescale: float = 0.0,
    ) -> torch.Tensor | TrellisSparseTensor:
        """Apply upstream ``w * cond + (1 - w) * neg`` CFG and optional x0 rescale."""

        _validate_pair(conditional, negative, names="conditional and negative predictions")
        if not 0 <= guidance_rescale <= 1:
            raise ValueError("guidance_rescale must lie in [0, 1]")
        strength = float(guidance_strength)
        if not math.isfinite(strength):
            raise ValueError("guidance_strength must be finite")
        if strength == 1:
            return conditional
        if strength == 0:
            return negative
        conditional_features = _features(conditional)
        negative_features = _features(negative)
        guided_features = strength * conditional_features + (1 - strength) * negative_features
        guided = _replace(conditional, guided_features)
        if guidance_rescale == 0:
            return guided
        if sample is None or timestep is None:
            raise ValueError("sample and timestep are required when guidance_rescale is non-zero")
        _validate_pair(sample, guided, names="sample and guided prediction")
        positive_x0 = self.velocity_to_original(sample, conditional, timestep)
        guided_x0 = self.velocity_to_original(sample, guided, timestep)
        positive_features = _features(positive_x0)
        guided_x0_features = _features(guided_x0)
        if isinstance(sample, TrellisSparseTensor):
            rescaled = torch.empty_like(guided_x0_features)
            for batch_index in range(sample.batch_size):
                mask = sample.coordinates[:, 0] == batch_index
                positive_std = positive_features[mask].std()
                guided_std = guided_x0_features[mask].std()
                rescaled[mask] = guided_x0_features[mask] * (positive_std / guided_std)
        else:
            dimensions = tuple(range(1, guided_x0_features.ndim))
            positive_std = positive_features.std(dim=dimensions, keepdim=True)
            guided_std = guided_x0_features.std(dim=dimensions, keepdim=True)
            rescaled = guided_x0_features * (positive_std / guided_std)
        blended_x0 = float(guidance_rescale) * rescaled + (1 - float(guidance_rescale)) * guided_x0_features
        return self.original_to_velocity(sample, _replace(sample, blended_x0), timestep)

    @staticmethod
    def guidance_is_active(timestep: torch.Tensor | float, interval: Sequence[float]) -> bool:
        if len(interval) != 2:
            raise ValueError("guidance_interval must contain two values")
        lower, upper = (float(value) for value in interval)
        if not 0 <= lower <= upper <= 1:
            raise ValueError("guidance_interval must be an increasing pair within [0, 1]")
        value = float(torch.as_tensor(timestep))
        return lower <= value <= upper


__all__ = ["Trellis2FlowEulerScheduler", "Trellis2FlowEulerSchedulerOutput"]
