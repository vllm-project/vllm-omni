# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 tests for the Lingbot World Fast scheduler."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world_fast.flow_scheduler import LingbotFlowScheduler
from vllm_omni.diffusion.models.lingbot_world_fast.fm_solvers_unipc import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.lingbot_world_fast.pipeline_lingbot_world_fast import (
    CONFIG,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_scheduler() -> FlowUniPCMultistepScheduler:
    # Same construction as ``LingbotWorldFastPipeline.__init__``.
    base_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=CONFIG["num_train_timesteps"],
        shift=1,
        use_dynamic_shifting=False,
    )
    base_scheduler.set_timesteps(CONFIG["num_train_timesteps"], shift=CONFIG["sample_shift"])

    timesteps = base_scheduler.timesteps[CONFIG["timesteps_index"]]
    scheduler = LingbotFlowScheduler(base_scheduler, timesteps)

    return (base_scheduler, scheduler)


def test_timesteps_index_selects_exactly_four_steps() -> None:
    base_scheduler, _ = _make_scheduler()
    selected = base_scheduler.timesteps[CONFIG["timesteps_index"]]

    assert selected.shape == (4,)
    # Monotonically decreasing — flow matching schedulers walk t from high to low.
    diffs = selected[1:] - selected[:-1]
    assert torch.all(diffs < 0), f"timesteps must be strictly decreasing, got {selected.tolist()}"


def test_timesteps_full_schedule_length_matches_num_train_timesteps() -> None:
    base_scheduler, _ = _make_scheduler()
    assert base_scheduler.num_inference_steps == CONFIG["num_train_timesteps"]
    assert base_scheduler.timesteps.shape == (CONFIG["num_train_timesteps"],)


def test_convert_flow_pred_to_x0_passthrough_when_pred_is_zero() -> None:
    base_scheduler, scheduler = _make_scheduler()
    xt = torch.randn(1, 4, 1, 4, 4, dtype=torch.float32)
    timestep = base_scheduler.timesteps[0]
    flow_pred = torch.zeros_like(xt)

    x0 = scheduler._convert_flow_pred_to_x0(
        flow_pred=flow_pred,
        xt=xt,
        timestep=timestep,
    )
    assert torch.allclose(x0, xt, atol=1e-6)


def test_convert_flow_pred_to_x0_recovers_x0_from_synthesized_pair() -> None:
    base_scheduler, scheduler = _make_scheduler()
    timestep = base_scheduler.timesteps[CONFIG["timesteps_index"][1]]

    sigmas = base_scheduler.sigmas
    timesteps = base_scheduler.timesteps
    timestep_id = torch.argmin((timesteps - timestep).abs())
    sigma_t = sigmas[timestep_id].item()

    x0 = torch.randn(1, 4, 1, 4, 4, dtype=torch.float32)
    noise = torch.randn_like(x0)
    flow_pred = noise - x0
    xt = (1.0 - sigma_t) * x0 + sigma_t * noise

    recovered = scheduler._convert_flow_pred_to_x0(
        flow_pred=flow_pred,
        xt=xt,
        timestep=timestep,
    )

    # The function does the math in float64 internally, casts back to the
    # input dtype. float32 inputs ⇒ ~1e-5 absolute tolerance is plenty.
    assert torch.allclose(recovered, x0, atol=1e-4)


def test_timesteps_index_is_within_schedule_bounds() -> None:
    """Defensive guard: an out-of-range index would silently wrap."""
    assert isinstance(CONFIG["timesteps_index"], list)
    assert len(CONFIG["timesteps_index"]) == 4
    for idx in CONFIG["timesteps_index"]:
        assert 0 <= idx < CONFIG["num_train_timesteps"]


def test_sample_shift_constant_is_positive() -> None:
    """``sample_shift`` controls the timestep curve; a non-positive value
    would corrupt the flow-matching trajectory."""
    assert CONFIG["sample_shift"] > 0
    # Reasonable upper bound — Wan models use shift ~5–10.
    assert math.isfinite(CONFIG["sample_shift"])
    assert CONFIG["sample_shift"] <= 100
