from __future__ import annotations

import torch

from diffusers_3d import TrellisFlowEulerScheduler, TrellisSparseTensor


def test_released_rescaled_schedule_guidance_and_euler_equations():
    scheduler = TrellisFlowEulerScheduler(sigma_min=1e-5)
    scheduler.set_timesteps(2, rescale_t=3.0)
    torch.testing.assert_close(scheduler.timesteps, torch.tensor([1.0, 0.75]))
    torch.testing.assert_close(scheduler._next_timesteps, torch.tensor([0.75, 0.0]))
    torch.testing.assert_close(
        scheduler.model_timestep(scheduler.timesteps[0], 2, device="cpu"),
        torch.tensor([1000.0, 1000.0]),
    )

    conditional = torch.tensor([2.0])
    unconditional = torch.tensor([0.5])
    velocity = scheduler.apply_guidance(conditional, unconditional, 5.0)
    torch.testing.assert_close(velocity, torch.tensor([9.5]))
    sample = torch.tensor([1.0])
    first = scheduler.step(velocity, scheduler.timesteps[0], sample)
    torch.testing.assert_close(first.prev_sample, sample - 0.25 * velocity)
    torch.testing.assert_close(
        first.pred_original_sample,
        (1 - 1e-5) * sample - (1e-5 + (1 - 1e-5)) * velocity,
    )
    second = scheduler.step(velocity, scheduler.timesteps[1], first.prev_sample)
    torch.testing.assert_close(second.prev_sample, first.prev_sample - 0.75 * velocity)


def test_scheduler_noise_equation_and_sparse_coordinate_preservation():
    scheduler = TrellisFlowEulerScheduler(sigma_min=0.1)
    clean = torch.tensor([[1.0], [2.0]])
    noise = torch.tensor([[3.0], [4.0]])
    timesteps = torch.tensor([0.25, 0.75])
    expected = (1 - timesteps[:, None]) * clean + (0.1 + 0.9 * timesteps[:, None]) * noise
    torch.testing.assert_close(scheduler.add_noise(clean, noise, timesteps), expected)

    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int64)
    sample = TrellisSparseTensor(coordinates, torch.ones(2, 1))
    velocity = sample.replace(torch.full((2, 1), 2.0))
    scheduler.set_timesteps(1)
    result = scheduler.step(velocity, scheduler.timesteps[0], sample)
    assert torch.equal(result.prev_sample.coordinates, coordinates)
    torch.testing.assert_close(result.prev_sample.features, torch.full((2, 1), -1.0))


def test_scheduler_config_roundtrip(tmp_path):
    scheduler = TrellisFlowEulerScheduler(sigma_min=1e-5, num_train_timesteps=1000)
    scheduler.save_pretrained(tmp_path)
    loaded = TrellisFlowEulerScheduler.from_pretrained(tmp_path)
    assert loaded.config.sigma_min == 1e-5
    assert loaded.config.num_train_timesteps == 1000
