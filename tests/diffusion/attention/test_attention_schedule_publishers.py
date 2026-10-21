# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts shared by every denoise progress publisher."""

from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange
from vllm_omni.diffusion.forward_context import (
    bind_attention_schedule,
    get_attention_schedule_profile,
    get_forward_context,
    request_denoise_progress,
    set_forward_context,
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_timestep,
    set_forward_context_denoise_total_steps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _ranges(*entries):
    """A normalized schedule from (start, end, profile) triples."""
    return tuple(AttentionScheduleRange(start=start, end=end, profile=profile) for start, end, profile in entries)


def _progress(ctx):
    return ctx.denoise_step_idx, ctx.denoise_timestep, ctx.total_denoise_steps


class _RecordingPagedRuntime:
    """Stands in for the runner-owned paged runtime and records each ensure_active call."""

    def __init__(self):
        self.activated: list[int] = []

    def ensure_active(self, step_idx):
        self.activated.append(step_idx)


@pytest.mark.parametrize("schedule", [None, ()], ids=["unbound", "disabled"])
def test_request_denoise_progress_publishes_without_activating_an_unscheduled_context(schedule):
    with set_forward_context(), bind_attention_schedule(schedule):
        ctx = get_forward_context()
        with request_denoise_progress(2, 6, timestep=torch.tensor(0.5)):
            assert _progress(ctx) == (2, 0.5, 6)
            assert type(ctx.denoise_timestep) is float
            assert get_attention_schedule_profile() is None
        assert _progress(ctx) == (None, None, None)


@pytest.mark.parametrize(
    ("outer_published", "fail"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["outer-unpublished-exit", "outer-unpublished-exception", "outer-published-exit", "outer-published-exception"],
)
def test_request_denoise_progress_restores_the_previous_progress(outer_published, fail):
    runtime = _RecordingPagedRuntime()
    with (
        set_forward_context(paged_kv_runtime=runtime),
        bind_attention_schedule(_ranges((0, None, "sparse"))),
    ):
        ctx = get_forward_context()
        if outer_published:
            set_forward_context_denoise_step_idx(1)
            set_forward_context_denoise_timestep(0.9)
            set_forward_context_denoise_total_steps(7)
        outer = _progress(ctx)
        activated_before = list(runtime.activated)
        raises = pytest.raises(RuntimeError, match="request forward failed") if fail else nullcontext()

        with raises:
            with request_denoise_progress(4, 5, timestep=0.25):
                assert _progress(ctx) == (4, 0.25, 5)
                assert get_attention_schedule_profile() == "sparse"
                if fail:
                    raise RuntimeError("request forward failed")

        assert _progress(ctx) == outer
    assert runtime.activated == activated_before + [4]
