# Portions of this file reproduce sampler semantics from Microsoft TRELLIS:
# https://github.com/microsoft/TRELLIS
# Revision: 442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# MIT License. Copyright (c) Microsoft Corporation.
# This file has been modified for the Diffusers scheduler lifecycle.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from .sparse import TrellisSparseTensor


@dataclass
class TrellisFlowEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor | TrellisSparseTensor
    pred_original_sample: torch.Tensor | TrellisSparseTensor


def _features(value: torch.Tensor | TrellisSparseTensor) -> torch.Tensor:
    return value.features if isinstance(value, TrellisSparseTensor) else value


def _replace(
    reference: torch.Tensor | TrellisSparseTensor,
    value: torch.Tensor,
) -> torch.Tensor | TrellisSparseTensor:
    return reference.replace(value) if isinstance(reference, TrellisSparseTensor) else value


class TrellisFlowEulerScheduler(SchedulerMixin, ConfigMixin):
    """Euler integration for TRELLIS's ``t=1`` noise to ``t=0`` data flow."""

    order = 1
    init_noise_sigma = 1.0

    @register_to_config
    def __init__(self, sigma_min: float = 1e-5, num_train_timesteps: int = 1000) -> None:
        if not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must lie in [0, 1)")
        if (
            not isinstance(num_train_timesteps, int)
            or isinstance(num_train_timesteps, bool)
            or num_train_timesteps <= 0
        ):
            raise ValueError("num_train_timesteps must be a positive integer")
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self._next_timesteps = torch.tensor([], dtype=torch.float32)
        self.num_inference_steps: int | None = None
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        *,
        timesteps: Sequence[float] | None = None,
        rescale_t: float = 1.0,
    ) -> None:
        if not isinstance(rescale_t, (float, int)) or isinstance(rescale_t, bool) or rescale_t <= 0:
            raise ValueError("rescale_t must be positive")
        if timesteps is None:
            if (
                not isinstance(num_inference_steps, int)
                or isinstance(num_inference_steps, bool)
                or num_inference_steps <= 0
            ):
                raise ValueError("num_inference_steps must be a positive integer")
            schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        else:
            schedule = torch.as_tensor(tuple(timesteps), dtype=torch.float32, device=device)
            if schedule.ndim != 1 or schedule.numel() < 2:
                raise ValueError("timesteps must contain at least current and terminal values")
            if not bool(torch.isfinite(schedule).all()) or bool((schedule[1:] > schedule[:-1]).any()):
                raise ValueError("timesteps must be finite and monotonically decreasing")
            if float(schedule[0]) > 1 or float(schedule[-1]) < 0:
                raise ValueError("timesteps must lie in [0, 1]")
            if num_inference_steps is not None and num_inference_steps != schedule.numel() - 1:
                raise ValueError("num_inference_steps must match the custom timestep count")
            num_inference_steps = schedule.numel() - 1
        schedule = float(rescale_t) * schedule / (1 + (float(rescale_t) - 1) * schedule)
        self.timesteps = schedule[:-1]
        self._next_timesteps = schedule[1:]
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def scale_model_input(
        self,
        sample: torch.Tensor | TrellisSparseTensor,
        timestep: torch.Tensor | float | None = None,
    ) -> torch.Tensor | TrellisSparseTensor:
        del timestep
        return sample

    def model_timestep(
        self,
        timestep: torch.Tensor | float,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        value = torch.as_tensor(timestep, dtype=torch.float32, device=device)
        return (value * self.config.num_train_timesteps).expand(batch_size)

    def velocity_to_original(
        self,
        sample: torch.Tensor | TrellisSparseTensor,
        velocity: torch.Tensor | TrellisSparseTensor,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor | TrellisSparseTensor:
        sample_features = _features(sample)
        velocity_features = _features(velocity)
        timestep = torch.as_tensor(timestep, device=sample_features.device, dtype=sample_features.dtype)
        original = (1 - self.config.sigma_min) * sample_features - (
            self.config.sigma_min + (1 - self.config.sigma_min) * timestep
        ) * velocity_features
        return _replace(sample, original)

    def step(
        self,
        model_output: torch.Tensor | TrellisSparseTensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor | TrellisSparseTensor,
        *,
        return_dict: bool = True,
    ) -> TrellisFlowEulerSchedulerOutput | tuple[torch.Tensor | TrellisSparseTensor]:
        if self.num_inference_steps is None or self._step_index >= len(self.timesteps):
            raise ValueError("set_timesteps must be called before each denoising run")
        sample_features = _features(sample)
        output_features = _features(model_output)
        if sample_features.shape != output_features.shape:
            raise ValueError("model_output and sample must have identical shapes")
        if isinstance(sample, TrellisSparseTensor) != isinstance(model_output, TrellisSparseTensor):
            raise TypeError("sparse scheduler steps require sparse sample and model_output values")
        if isinstance(sample, TrellisSparseTensor) and not torch.equal(sample.coordinates, model_output.coordinates):
            raise ValueError("sparse scheduler steps require identical coordinates")

        current = self.timesteps[self._step_index].to(device=sample_features.device, dtype=sample_features.dtype)
        supplied = torch.as_tensor(timestep, device=sample_features.device, dtype=sample_features.dtype)
        if not torch.allclose(supplied, current):
            raise ValueError("timestep does not match the scheduler's current step")
        previous = self._next_timesteps[self._step_index].to(
            device=sample_features.device,
            dtype=sample_features.dtype,
        )
        previous_sample = sample_features - (current - previous) * output_features
        original_sample = _features(self.velocity_to_original(sample, model_output, current))
        self._step_index += 1
        previous_output = _replace(sample, previous_sample)
        original_output = _replace(sample, original_sample)
        if not return_dict:
            return (previous_output,)
        return TrellisFlowEulerSchedulerOutput(
            prev_sample=previous_output,
            pred_original_sample=original_output,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if original_samples.shape != noise.shape:
            raise ValueError("original_samples and noise must have identical shapes")
        timesteps = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.reshape(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - timesteps) * original_samples + (
            self.config.sigma_min + (1 - self.config.sigma_min) * timesteps
        ) * noise

    @staticmethod
    def apply_guidance(
        conditional: torch.Tensor | TrellisSparseTensor,
        unconditional: torch.Tensor | TrellisSparseTensor,
        guidance_scale: float,
    ) -> torch.Tensor | TrellisSparseTensor:
        conditional_features = _features(conditional)
        unconditional_features = _features(unconditional)
        if conditional_features.shape != unconditional_features.shape:
            raise ValueError("conditional and unconditional predictions must have identical shapes")
        guided = (1 + guidance_scale) * conditional_features - guidance_scale * unconditional_features
        return _replace(conditional, guided)


__all__ = ["TrellisFlowEulerScheduler", "TrellisFlowEulerSchedulerOutput"]
