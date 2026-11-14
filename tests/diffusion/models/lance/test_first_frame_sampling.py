# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.bagel import bagel_transformer
from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _StepOutput:
    prev_sample: torch.Tensor
    log_prob: torch.Tensor


class _GaussianScheduler:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(23)
        self.shapes = []

    def step(self, model_output, timestep, sample, dt, *, prev_sample=None):
        self.shapes.append(sample.shape)
        mean = sample.float() - model_output.float() * dt
        if prev_sample is None:
            prev_sample = mean + 0.25 * torch.randn(mean.shape, generator=self.generator)
        log_prob = torch.distributions.Normal(mean, 0.25).log_prob(prev_sample).mean()
        return _StepOutput(prev_sample, log_prob)


def _generation(mocker, mode, dtype, indexes):
    model = object.__new__(Bagel)
    torch.nn.Module.__init__(model)
    model.parallel_config = mocker.Mock(sequence_parallel_size=2 if mode.startswith("sp") else 1)
    model._denoise_schedule_extra_step = True
    world_size = 2 if mode == "parallel_cfg" else 1
    mocker.patch.object(bagel_transformer, "get_classifier_free_guidance_world_size", return_value=world_size)
    mocker.patch.object(bagel_transformer, "get_cfg_group", return_value=mocker.Mock())
    calls = []

    def velocity(x_t, timestep, **kwargs):
        calls.append((x_t.clone(), timestep.clone()))
        return 0.2 * x_t.float() + 0.03 * timestep[:, None]

    mocker.patch.object(model, "forward", side_effect=velocity)
    mocker.patch.object(model, "forward_single_branch", side_effect=velocity)
    mocker.patch.object(
        model,
        "predict_noise_with_multi_branch_cfg",
        side_effect=lambda *, branches_kwargs, **kwargs: velocity(**branches_kwargs[0]),
    )
    initial = torch.arange(24, dtype=torch.float32).reshape(8, 3).to(dtype) / 8
    args = {
        "packed_text_ids": torch.tensor([0, 1]),
        "packed_text_indexes": torch.tensor([0, 9]),
        "packed_init_noises": initial.clone(),
        "packed_vae_position_ids": torch.arange(8),
        "packed_vae_token_indexes": torch.arange(1, 9),
        "packed_seqlens": torch.tensor([10]),
        "packed_position_ids": torch.arange(10),
        "past_key_values": None,
        "num_timesteps": 3,
        "cfg_text_scale": 4.0 if mode.endswith("cfg") else 1.0,
        "cfg_img_scale": 1.0,
        "cfg_text_packed_position_ids": torch.arange(10),
        "cfg_text_past_key_values": None,
        "frame_condition_token_indexes": indexes,
        "return_trajectory_latents": True,
    }
    return model, args, initial, calls


@pytest.mark.parametrize("mode", ["sequential", "sequential_cfg", "sp", "sp_cfg", "parallel_cfg"])
@pytest.mark.parametrize("use_scheduler", [False, True], ids=["euler", "sde"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_reference_rows_stay_fixed_and_recorded_logps_replay(mocker, mode, use_scheduler, dtype, conditioned):
    indexes = torch.tensor([0, 3]) if conditioned else None
    model, args, initial, calls = _generation(mocker, mode, dtype, indexes)
    scheduler = _GaussianScheduler() if use_scheduler else None

    result, trajectory, timesteps, logps = model.generate_image(**args, scheduler=scheduler)

    keep = torch.ones(8, dtype=torch.bool)
    if indexes is not None:
        keep[indexes] = False
        for latents in trajectory:
            torch.testing.assert_close(latents[indexes], initial[indexes].to(latents.dtype), rtol=0, atol=0)
        for latents, timestep in calls:
            torch.testing.assert_close(latents[indexes], initial[indexes].to(latents.dtype), rtol=0, atol=0)
            assert torch.count_nonzero(timestep[indexes]) == 0
    assert len(trajectory) == 4
    assert len(timesteps) == 3
    torch.testing.assert_close(result[0], trajectory[-1], rtol=0, atol=0)
    assert not torch.equal(trajectory[-1][keep], initial[keep])

    if scheduler is None:
        assert logps is None
    else:
        assert scheduler.shapes == [(int(keep.sum()), 3)] * 3
        for step, logp in enumerate(logps):
            current = trajectory[step][keep].float()
            velocity = 0.2 * current + 0.03 * timesteps[step]
            mean = current - velocity / 3
            expected = torch.distributions.Normal(mean, 0.25).log_prob(trajectory[step + 1][keep]).mean()
            torch.testing.assert_close(logp, expected, rtol=1e-5, atol=1e-6)


def test_conditioned_scheduler_requires_generated_rows(mocker):
    model, args, _, calls = _generation(mocker, "sequential", torch.float32, torch.arange(8))
    scheduler = _GaussianScheduler()

    with pytest.raises(ValueError, match="at least one generated token"):
        model.generate_image(**args, scheduler=scheduler)

    assert not calls
    assert not scheduler.shapes


@pytest.mark.parametrize("use_scheduler", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_empty_reference_indexes_match_unconditioned_rollout(mocker, use_scheduler, dtype):
    model, args, _, _ = _generation(mocker, "sequential", dtype, torch.tensor([], dtype=torch.long))
    scheduler = _GaussianScheduler() if use_scheduler else None
    _, empty_trajectory, _, empty_logps = model.generate_image(**args, scheduler=scheduler)
    args["frame_condition_token_indexes"] = None
    scheduler = _GaussianScheduler() if use_scheduler else None

    _, trajectory, _, logps = model.generate_image(**args, scheduler=scheduler)

    for actual, expected in zip(empty_trajectory, trajectory, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if use_scheduler:
        torch.testing.assert_close(torch.stack(empty_logps), torch.stack(logps), rtol=0, atol=0)


def test_scheduler_replay_slices_previous_sample_without_mutating_kwargs():
    sample = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    previous = sample + 2
    indexes = torch.tensor([1, 3])
    kwargs = {"prev_sample": previous}
    scheduler = _GaussianScheduler()

    result, logp = bagel_transformer._step_image_latents(
        torch.ones_like(sample), torch.tensor(0.7), sample, torch.tensor(0.2), scheduler, kwargs, indexes
    )

    torch.testing.assert_close(result[indexes], previous[indexes], rtol=0, atol=0)
    torch.testing.assert_close(result[[0, 2]], sample[[0, 2]], rtol=0, atol=0)
    assert torch.isfinite(logp)
    assert kwargs["prev_sample"] is previous
    assert scheduler.shapes == [(2, 3)]
