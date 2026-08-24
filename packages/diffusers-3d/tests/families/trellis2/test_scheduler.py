from __future__ import annotations

import torch

from diffusers_3d import Trellis2FlowEulerScheduler, TrellisSparseTensor


def test_released_rescaled_schedule_cfg_rescale_interval_timestep_and_euler_equations(tmp_path):
    scheduler = Trellis2FlowEulerScheduler(sigma_min=1e-5)
    scheduler.set_timesteps(2, rescale_t=5.0)
    base = torch.tensor([1.0, 0.5, 0.0])
    expected_schedule = 5.0 * base / (1 + 4.0 * base)
    torch.testing.assert_close(scheduler.timesteps, expected_schedule[:-1])
    assert torch.equal(scheduler.model_timestep(scheduler.timesteps[0], 2, device="cpu"), torch.full((2,), 1000.0))
    assert scheduler.guidance_is_active(torch.tensor(0.75), (0.6, 1.0))
    assert not scheduler.guidance_is_active(torch.tensor(0.5), (0.6, 1.0))

    conditional = torch.tensor([[[[[-1.0, 0.0], [1.0, 2.0]]]]])
    negative = torch.tensor([[[[[0.5, -0.5], [1.5, -1.5]]]]])
    sample = torch.tensor([[[[[0.25, 1.0], [-0.75, 0.5]]]]])
    guided = scheduler.apply_guidance(conditional, negative, 7.5)
    torch.testing.assert_close(guided, 7.5 * conditional + (1 - 7.5) * negative, atol=0.0, rtol=0.0)

    rescaled = scheduler.apply_guidance(
        conditional,
        negative,
        7.5,
        sample=sample,
        timestep=0.75,
        guidance_rescale=0.7,
    )
    positive_x0 = scheduler.velocity_to_original(sample, conditional, 0.75)
    guided_x0 = scheduler.velocity_to_original(sample, guided, 0.75)
    expected_rescaled_x0 = guided_x0 * (
        positive_x0.std(dim=(1, 2, 3, 4), keepdim=True) / guided_x0.std(dim=(1, 2, 3, 4), keepdim=True)
    )
    expected_x0 = 0.7 * expected_rescaled_x0 + 0.3 * guided_x0
    expected_velocity = scheduler.original_to_velocity(sample, expected_x0, 0.75)
    torch.testing.assert_close(rescaled, expected_velocity, atol=0.0, rtol=0.0)

    current = scheduler.timesteps[0]
    output = scheduler.step(guided, current, sample)
    expected_previous = sample - (expected_schedule[0] - expected_schedule[1]) * guided
    torch.testing.assert_close(output.prev_sample, expected_previous, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        output.pred_original_sample,
        scheduler.velocity_to_original(sample, guided, current),
        atol=0.0,
        rtol=0.0,
    )

    scheduler.save_pretrained(tmp_path)
    restored = Trellis2FlowEulerScheduler.from_pretrained(tmp_path)
    assert restored.config.sigma_min == scheduler.config.sigma_min


def test_scheduler_sparse_equations_preserve_coordinates():
    scheduler = Trellis2FlowEulerScheduler()
    coordinates = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int64)
    sample = TrellisSparseTensor(coordinates, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    positive = sample.replace(torch.tensor([[0.5, -0.5], [1.0, -1.0]]))
    negative = sample.replace(torch.tensor([[0.0, 1.0], [-0.5, 0.5]]))
    guided = scheduler.apply_guidance(positive, negative, 2.0)
    assert torch.equal(guided.coordinates, coordinates)
    torch.testing.assert_close(guided.features, 2.0 * positive.features - negative.features)

    scheduler.set_timesteps(1)
    output = scheduler.step(guided, scheduler.timesteps[0], sample)
    assert torch.equal(output.prev_sample.coordinates, coordinates)
    torch.testing.assert_close(output.prev_sample.features, sample.features - guided.features)
