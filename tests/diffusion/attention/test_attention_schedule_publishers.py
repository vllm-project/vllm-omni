# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts shared by every denoise progress publisher."""

from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    InvalidAttentionScheduleError,
)
from vllm_omni.diffusion.forward_context import (
    begin_scheduled_denoise,
    bind_attention_schedule,
    get_forward_context,
    is_forward_context_available,
    request_denoise_progress,
    set_forward_context,
    set_forward_context_denoise_step_idx,
    set_forward_context_denoise_timestep,
    set_forward_context_denoise_total_steps,
)
from vllm_omni.errors import OmniClientError, client_error_metadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _ranges(*entries):
    """A normalized schedule from (start, end, profile) triples."""
    return tuple(AttentionScheduleRange(start=start, end=end, profile=profile) for start, end, profile in entries)


def _progress(ctx):
    return ctx.denoise_step_idx, ctx.denoise_timestep, ctx.total_denoise_steps, ctx.attention_schedule_denoise_active


def test_begin_scheduled_denoise_returns_the_total_for_a_bound_schedule():
    with set_forward_context(), bind_attention_schedule(_ranges((0, 2, "sparse")), denoise=False):
        assert begin_scheduled_denoise(2) == 2
        assert begin_scheduled_denoise(8) == 8
        assert _progress(get_forward_context()) == (None, None, None, False)


@pytest.mark.parametrize("scope", ["no-context", "unbound", "disabled"])
def test_begin_scheduled_denoise_returns_none_without_checking_when_no_schedule_is_bound(scope):
    if scope == "no-context":
        assert is_forward_context_available() is False
        assert begin_scheduled_denoise(0) is None
    else:
        with set_forward_context(), bind_attention_schedule(None if scope == "unbound" else (), denoise=False):
            assert begin_scheduled_denoise(0) is None


@pytest.mark.parametrize(
    "schedule",
    [_ranges((3, 6, "sparse")), _ranges((5, None, "sparse"))],
    ids=["end-past-total", "open-range-starts-at-total"],
)
def test_begin_scheduled_denoise_rejects_a_range_past_the_actual_total(schedule):
    with set_forward_context(), bind_attention_schedule(schedule, denoise=False):
        with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=5") as excinfo:
            begin_scheduled_denoise(5)

    assert isinstance(excinfo.value, OmniClientError)
    assert client_error_metadata(excinfo.value) == (400, "invalid_attention_schedule")


class _RecordingPagedRuntime:
    """Stands in for the runner-owned paged runtime and records each ensure_active call."""

    def __init__(self):
        self.activated: list[int] = []

    def ensure_active(self, step_idx):
        self.activated.append(step_idx)


@pytest.mark.parametrize("schedule", [None, ()], ids=["unbound", "disabled"])
def test_request_denoise_progress_publishes_without_activating_an_unscheduled_context(schedule):
    with set_forward_context(), bind_attention_schedule(schedule, denoise=False):
        ctx = get_forward_context()
        with request_denoise_progress(2, 6, timestep=torch.tensor(0.5)):
            assert _progress(ctx) == (2, 0.5, 6, False)
            assert type(ctx.denoise_timestep) is float
        assert _progress(ctx) == (None, None, None, False)


@pytest.mark.parametrize(
    ("outer_published", "fail"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["outer-unpublished-exit", "outer-unpublished-exception", "outer-published-exit", "outer-published-exception"],
)
def test_request_denoise_progress_restores_the_previous_progress(outer_published, fail):
    runtime = _RecordingPagedRuntime()
    with (
        set_forward_context(paged_kv_runtime=runtime),
        bind_attention_schedule(_ranges((0, None, "sparse")), denoise=False),
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
                assert _progress(ctx) == (4, 0.25, 5, True)
                if fail:
                    raise RuntimeError("request forward failed")

        assert _progress(ctx) == outer
    assert runtime.activated == activated_before + [4]
