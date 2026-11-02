# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR execution coalescing uses a bounded gate, separate from admission."""

import pytest
from vllm.v1.core.sched.interface import PauseState

from tests.core.sched.test_generation_scheduler_atomic_chunk import FakeAdapter, _make_request
from tests.core.sched.test_generation_scheduler_init import _config
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def scheduler(*, native=False, extras=None, population=4):
    s = OmniARScheduler.__new__(OmniARScheduler)
    s.vllm_config = _config(native=native, extras=extras)
    s.max_num_running_reqs = 128
    s._native_data_plane = native
    s.chunk_transfer_adapter = None if native else FakeAdapter()
    s.input_coordinator = OmniSchedulingCoordinator(stage_id=1, async_chunk=True) if native else None
    s._pause_state = PauseState.UNPAUSED
    s.requests = {str(i): _make_request(str(i), 8) for i in range(population)}
    s._init_prefill_admission_policy()
    return s


def knobs(**overrides):
    return {"generation_min_batch_size": 4, "generation_max_wait_ms": 12, **overrides}


@pytest.mark.parametrize("native", [False, True])
def test_configuration_supports_both_runners_and_additional_config(native):
    s = scheduler(native=native)
    s.vllm_config.additional_config = knobs()
    s._init_prefill_admission_policy()
    assert s._generation_min_batch_size == 4
    assert s._generation_max_wait_s == 0.012
    assert s._generation_always_wait_target_batch is False


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("generation_min_batch_size", 0, "positive"),
        ("generation_min_batch_size", 129, "max_num_seqs"),
        ("generation_max_wait_ms", -1, "finite and nonnegative"),
        ("generation_max_wait_ms", float("nan"), "finite and nonnegative"),
        ("generation_max_wait_ms", float("inf"), "finite and nonnegative"),
        ("generation_always_wait_target_batch", "false", "boolean"),
        ("generation_coalescing_policy", "idle_wait", "fixed"),
    ],
)
def test_rejects_invalid_configuration(key, value, message):
    with pytest.raises(ValueError, match=message):
        scheduler(extras=knobs(**{key: value}))


@pytest.mark.parametrize(
    "old_knobs",
    [
        {"ar_prefill_admission_min_batch": 4, "ar_prefill_admission_max_wait_ms": 12},
        {"ar_prefill_admission_max_wait_ms": 12},
        {"ar_prefill_admission_min_running": 2},
    ],
)
def test_rejects_combining_admission_and_execution_coalescing(old_knobs):
    with pytest.raises(ValueError, match="cannot be enabled together"):
        scheduler(extras=knobs(**old_knobs))


def test_deadline_does_not_restart_on_partial_arrivals(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(extras=knobs())
    assert s._should_defer_ar_generation({"0"})
    clock[0] += 0.006
    assert s._should_defer_ar_generation({"0", "1"})
    assert s._generation_batch_deadline == pytest.approx(10.012)
    clock[0] += 0.006
    assert not s._should_defer_ar_generation({"0", "1"})
    assert s._generation_batch_deadline is None


@pytest.mark.parametrize(
    "always_wait,population,ready,wait",
    [
        (False, 4, 1, True),
        (False, 4, 4, False),
        (False, 3, 1, False),
        (True, 3, 1, True),
        (True, 3, 3, False),
        (True, 4, 0, False),
    ],
)
def test_batch_target_policy(always_wait, population, ready, wait):
    s = scheduler(extras=knobs(generation_always_wait_target_batch=always_wait), population=population)
    assert s._should_defer_ar_generation({str(i) for i in range(ready)}) is wait
