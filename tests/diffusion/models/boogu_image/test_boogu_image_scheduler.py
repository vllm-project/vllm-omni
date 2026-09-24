# SPDX-License-Identifier: Apache-2.0
"""L1 unit tests for the Boogu time-shifting flow-match Euler scheduler.

Golden values were generated from the upstream boogu scheduler
(scripts/test_ported_scheduler_parity.py) with the released checkpoint's
scheduler_config.json, so CI does not need the `boogu` package installed.
"""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.boogu_image import FlowMatchEulerDiscreteScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Ground truth from the released checkpoint's scheduler/scheduler_config.json.
CHECKPOINT_CONFIG = dict(
    num_train_timesteps=1000,
    do_shift=True,
    dynamic_time_shift=False,
    time_shift_version="v1",
    seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
)

NUM_TOKENS = 4096 * 4  # 1024x1024 resolution

# Generated from the upstream boogu scheduler with CHECKPOINT_CONFIG.
GOLDEN_TIMESTEPS = {
    4: [0.00000000, 0.09546924, 0.24048907, 0.48715585],
    10: [
        0.00000000,
        0.03398621,
        0.07335263,
        0.11948693,
        0.17429835,
        0.24048907,
        0.32201332,
        0.42489707,
        0.55880034,
        0.74024153,
    ],
}


def make_scheduler(**overrides) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(**{**CHECKPOINT_CONFIG, **overrides})


@pytest.mark.parametrize("n_steps", sorted(GOLDEN_TIMESTEPS))
def test_golden_timesteps_checkpoint_config(n_steps):
    scheduler = make_scheduler()
    scheduler.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)
    np.testing.assert_allclose(
        scheduler.timesteps.numpy(),
        np.asarray(GOLDEN_TIMESTEPS[n_steps], dtype=np.float32),
        atol=1e-7,
    )


@pytest.mark.parametrize("n_steps", [4, 10, 30])
def test_timestep_grid_structure(n_steps):
    scheduler = make_scheduler()
    scheduler.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)

    assert scheduler.timesteps.shape == (n_steps,)
    assert scheduler._timesteps.shape == (n_steps + 1,)
    # Ascending 0 -> 1 convention with the terminal 1.0 appended.
    assert scheduler.timesteps[0].item() == 0.0
    assert torch.all(scheduler._timesteps[1:] > scheduler._timesteps[:-1])
    assert scheduler._timesteps[-1].item() == 1.0
    assert torch.equal(scheduler._timesteps[:-1], scheduler.timesteps)


def test_no_shift_is_uniform_grid():
    n_steps = 8
    scheduler = make_scheduler(do_shift=False)
    scheduler.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)
    expected = np.linspace(0, 1, n_steps + 1, dtype=np.float32)[:-1]
    np.testing.assert_allclose(scheduler.timesteps.numpy(), expected, atol=1e-7)


def test_static_shift_ignores_num_tokens():
    # dynamic_time_shift=False: the shift comes from configured seq_len, so
    # per-call num_tokens must not change the schedule.
    scheduler_a = make_scheduler()
    scheduler_a.set_timesteps(num_inference_steps=10, num_tokens=NUM_TOKENS)
    scheduler_b = make_scheduler()
    scheduler_b.set_timesteps(num_inference_steps=10, num_tokens=320 * 320)
    assert torch.equal(scheduler_a.timesteps, scheduler_b.timesteps)


def test_step_arithmetic_and_index_advancement():
    n_steps = 4
    scheduler = make_scheduler()
    scheduler.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)

    sample = torch.zeros(1, 4, 8, 8)
    model_output = torch.ones(1, 4, 8, 8)

    assert scheduler.step_index is None
    out = scheduler.step(model_output, scheduler.timesteps[0], sample)
    assert scheduler.step_index == 1

    # prev_sample = sample + (t_next - t) * model_output
    expected_dt = scheduler._timesteps[1] - scheduler._timesteps[0]
    assert torch.allclose(out.prev_sample, expected_dt * model_output)

    # Full loop reaches the end of the grid.
    sample = out.prev_sample
    for t in scheduler.timesteps[1:]:
        sample = scheduler.step(model_output, t, sample, return_dict=False)[0]
    assert scheduler.step_index == n_steps
    # Integrating a constant velocity of 1 over t in [0, 1] gives exactly 1.
    assert torch.allclose(sample, torch.ones_like(sample), atol=1e-6)


def test_step_rejects_integer_timestep():
    scheduler = make_scheduler()
    scheduler.set_timesteps(num_inference_steps=4, num_tokens=NUM_TOKENS)
    with pytest.raises(ValueError, match="integer indices"):
        scheduler.step(torch.zeros(1), 0, torch.zeros(1))


def test_set_begin_index():
    scheduler = make_scheduler()
    scheduler.set_timesteps(num_inference_steps=4, num_tokens=NUM_TOKENS)
    scheduler.set_begin_index(2)
    assert scheduler.begin_index == 2

    scheduler.step(torch.zeros(1), scheduler.timesteps[2], torch.zeros(1))
    assert scheduler.step_index == 3


def test_len_and_config_roundtrip():
    scheduler = make_scheduler()
    assert len(scheduler) == CHECKPOINT_CONFIG["num_train_timesteps"]
    # ConfigMixin must retain the checkpoint config (needed for from_pretrained).
    assert scheduler.config.do_shift is True
    assert scheduler.config.dynamic_time_shift is False
    assert scheduler.config.time_shift_version == "v1"
    assert scheduler.config.seq_len == 4096


def test_matches_upstream_boogu_scheduler():
    """Differential check against the upstream boogu package, when installed."""
    boogu_schedulers = pytest.importorskip("boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting")
    UpstreamScheduler = boogu_schedulers.FlowMatchEulerDiscreteScheduler

    for dynamic in (False, True):
        for version in ("v1", "v2"):
            for n_steps in (4, 10, 30):
                config = {**CHECKPOINT_CONFIG, "dynamic_time_shift": dynamic, "time_shift_version": version}
                upstream = UpstreamScheduler(**config)
                ported = make_scheduler(**config)
                upstream.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)
                ported.set_timesteps(num_inference_steps=n_steps, num_tokens=NUM_TOKENS)
                assert torch.equal(upstream.timesteps, ported.timesteps), (dynamic, version, n_steps)
                assert torch.equal(upstream._timesteps, ported._timesteps), (dynamic, version, n_steps)
