from __future__ import annotations

import torch

from diffusers_3d import Hunyuan3DFlowMatchEulerDiscreteScheduler


def test_reference_sigma_and_euler_math(tmp_path):
    scheduler = Hunyuan3DFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(3)
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([0.0, 500.0, 1000.0]))
    torch.testing.assert_close(scheduler.sigmas, torch.tensor([0.0, 0.5, 1.0, 1.0]))

    sample = torch.tensor([1.0])
    velocity = torch.tensor([2.0])
    first = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    second = scheduler.step(velocity, scheduler.timesteps[1], first).prev_sample
    third = scheduler.step(velocity, scheduler.timesteps[2], second).prev_sample
    torch.testing.assert_close(first, sample + 0.5 * velocity)
    torch.testing.assert_close(second, sample + velocity)
    torch.testing.assert_close(third, second)

    scheduler.save_pretrained(tmp_path)
    loaded = Hunyuan3DFlowMatchEulerDiscreteScheduler.from_pretrained(tmp_path)
    loaded.set_timesteps(3)
    torch.testing.assert_close(loaded.sigmas, scheduler.sigmas)
