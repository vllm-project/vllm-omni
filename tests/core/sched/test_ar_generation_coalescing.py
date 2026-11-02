# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR execution coalescing uses a bounded gate, separate from admission."""

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from tests.core.sched.test_generation_scheduler_atomic_chunk import FakeAdapter, _make_request
from tests.core.sched.test_generation_scheduler_init import _config
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, VLLMScheduler
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
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    s.kv_holding_waiting = create_request_queue(s.policy)
    s.deferred_waiting = set()
    s.use_v2_model_runner = native
    s.max_num_active_reqs = 128
    s.max_num_scheduled_tokens = 128
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


@pytest.mark.parametrize("native", [False, True])
def test_candidates_combine_input_prefill_and_continuation_without_duplicates(native):
    s = scheduler(native=native, extras=knobs(), population=7)
    receiver = s.input_coordinator if native else s.chunk_transfer_adapter
    receiver.requests_with_ready_chunks.update({"0", "1", "4", "5", "gone"})
    # The same stream appears in the input set and waiting queue.
    s.waiting.add_request(s.requests["0"])
    for i in (1, 2, 3):
        request = s.requests[str(i)]
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = request.num_prompt_tokens
        s.running.append(request)
    # A normal continuation and an async continuation both have token work.
    s.requests["1"].append_output_token_ids(1)
    s.requests["2"].num_output_placeholders = 1
    s.requests["2"].num_in_flight_tokens = 1
    # Running request 3 has no work; 4 is parked; 5 is finished.
    s.requests["4"].status = RequestStatus.WAITING_FOR_CHUNK
    s.requests["5"].status = RequestStatus.FINISHED_ABORTED
    s.requests["6"].status = RequestStatus.PREEMPTED
    s.kv_holding_waiting.add_request(s.requests["6"])
    assert s._get_ar_coalescing_ready_request_ids() == {"0", "1", "2", "6"}
    assert receiver.requests_with_ready_chunks == {"0", "1", "4", "5", "gone"}


def prepare_schedule_boundary(monkeypatch, s, *, fail=False):
    """Keep the real AR gate; isolate upstream allocation and output packaging."""
    for name in (
        "_process_pending_omni_inputs",
        "_resync_streaming_input_counter",
        "_postprocess_omni_schedule_output",
    ):
        monkeypatch.setattr(s, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "get_finished_requests_needing_kv_transfer", lambda: {})
    monkeypatch.setattr(s, "_wrap_omni_scheduler_output", lambda output, **kwargs: output)
    budgets = []

    def upstream_schedule(self, throttle_prefills=False):
        budgets.append(self.max_num_scheduled_tokens)
        if fail:
            raise RuntimeError("allocation failed")
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {"finished"}
        if self.max_num_scheduled_tokens:
            output.num_scheduled_tokens = {"0": 8}
            output.total_num_scheduled_tokens = 8
        return output

    monkeypatch.setattr(VLLMScheduler, "schedule", upstream_schedule)
    return budgets


@pytest.mark.parametrize("native", [False, True])
def test_schedule_defers_before_allocation_and_preserves_control_output(monkeypatch, native):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(native=native, extras=knobs())
    s.waiting.add_request(s.requests["0"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    assert s.schedule().finished_req_ids == {"finished"}
    assert budgets == [0]
    assert s.max_num_scheduled_tokens == 128
    assert s.requests["0"].num_computed_tokens == 0
    assert list(s.waiting) == [s.requests["0"]]
    clock[0] += 0.012
    assert s.schedule().num_scheduled_tokens == {"0": 8}
    assert budgets == [0, 128]


def test_schedule_restores_budget_on_upstream_exception(monkeypatch):
    s = scheduler(extras=knobs())
    s.waiting.add_request(s.requests["0"])
    prepare_schedule_boundary(monkeypatch, s, fail=True)
    with pytest.raises(RuntimeError, match="allocation failed"):
        s.schedule()
    assert s.max_num_scheduled_tokens == 128
