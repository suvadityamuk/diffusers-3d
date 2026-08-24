# Portions of this file reproduce sampler semantics from Microsoft TRELLIS.2:
# https://github.com/microsoft/TRELLIS.2
# Revision: 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#
# MIT License. Copyright (c) Microsoft Corporation.
# Modified for the Diffusers scheduler lifecycle and package-owned sparse tensors.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from ..trellis.sparse import TrellisSparseTensor


@dataclass
class Trellis2FlowEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor | TrellisSparseTensor
    pred_original_sample: torch.Tensor | TrellisSparseTensor


def _features(value: torch.Tensor | TrellisSparseTensor) -> torch.Tensor:
    return value.features if isinstance(value, TrellisSparseTensor) else value


def _replace(
    reference: torch.Tensor | TrellisSparseTensor,
    value: torch.Tensor,
) -> torch.Tensor | TrellisSparseTensor:
    return reference.replace(value) if isinstance(reference, TrellisSparseTensor) else value


def _validate_pair(
    left: torch.Tensor | TrellisSparseTensor,
    right: torch.Tensor | TrellisSparseTensor,
    *,
    names: str,
) -> None:
    if isinstance(left, TrellisSparseTensor) != isinstance(right, TrellisSparseTensor):
        raise TypeError(f"{names} must both be dense tensors or both be TrellisSparseTensor values")
    if _features(left).shape != _features(right).shape:
        raise ValueError(f"{names} must have identical feature shapes")
    if isinstance(left, TrellisSparseTensor) and not torch.equal(left.coordinates, right.coordinates):
        raise ValueError(f"{names} must have identical sparse coordinates")


class Trellis2FlowEulerScheduler(SchedulerMixin, ConfigMixin):
    """TRELLIS.2 Euler flow scheduler with its distinct CFG and rescale equations."""

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
        _validate_pair(sample, velocity, names="sample and velocity")
        sample_features = _features(sample)
        velocity_features = _features(velocity)
        timestep = torch.as_tensor(timestep, device=sample_features.device, dtype=sample_features.dtype)
        original = (1 - self.config.sigma_min) * sample_features - (
            self.config.sigma_min + (1 - self.config.sigma_min) * timestep
        ) * velocity_features
        return _replace(sample, original)

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
        conditional_features = _features(conditional)
        negative_features = _features(negative)
        guided_features = (
            float(guidance_strength) * conditional_features + (1 - float(guidance_strength)) * negative_features
        )
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

    def step(
        self,
        model_output: torch.Tensor | TrellisSparseTensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor | TrellisSparseTensor,
        *,
        return_dict: bool = True,
    ) -> Trellis2FlowEulerSchedulerOutput | tuple[torch.Tensor | TrellisSparseTensor]:
        if self.num_inference_steps is None or self._step_index >= len(self.timesteps):
            raise ValueError("set_timesteps must be called before each denoising run")
        _validate_pair(sample, model_output, names="sample and model_output")
        sample_features = _features(sample)
        output_features = _features(model_output)
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
        return Trellis2FlowEulerSchedulerOutput(
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


__all__ = ["Trellis2FlowEulerScheduler", "Trellis2FlowEulerSchedulerOutput"]
