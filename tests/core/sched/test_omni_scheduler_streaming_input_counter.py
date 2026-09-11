"""Tests for the ``num_waiting_for_streaming_input`` resync placed in
``OmniARScheduler.finish_requests`` / ``OmniGenerationScheduler.finish_requests``
and in ``OmniARScheduler.schedule``.

Upstream keeps that counter incrementally (``+1`` when a resumable request
parks in a waiting queue as ``WAITING_FOR_STREAMING_REQ``, ``-1`` when the
next update arrives or the request is finished) and
``Scheduler.get_num_unfinished_requests`` *subtracts* it from the waiting
queues. Omni rewrites ``request.status`` outside both hooks -- the
chunk-transfer adapter's park/restore and
``_realign_request_status_to_queues`` -- so a counted request can leave
``WAITING_FOR_STREAMING_REQ`` without the matching decrement and inflate the
counter for the lifetime of the process.

An inflated counter does not degrade throughput, it hangs the stage:
``EngineCore.has_work()`` reads false while a live request sits in
``waiting``, so the engine blocks in ``input_queue.get()`` and never calls
``schedule()`` again. Regression for the Qwen3-Omni duplex symptom where the
server answered the first conversation after a boot and was silent from the
second on while still reporting healthy.
"""

from __future__ import annotations

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
import vllm_omni.core.sched.omni_generation_scheduler as gen_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubRequest:
    def __init__(self, request_id: str, status: RequestStatus) -> None:
        self.request_id = request_id
        self.status = status

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


_SCHEDULER_PARAMS = [
    pytest.param(OmniARScheduler, ar_sched_mod, id="ar"),
    pytest.param(OmniGenerationScheduler, gen_sched_mod, id="generation"),
]


def _make_scheduler(scheduler_cls, *, requests, running, waiting, counter, skipped=None):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler.requests = requests
    scheduler.running = running
    scheduler.waiting = waiting
    scheduler.skipped_waiting = skipped if skipped is not None else []
    scheduler.num_waiting_for_streaming_input = counter
    return scheduler


@pytest.mark.parametrize(("scheduler_cls", "scheduler_mod"), _SCHEDULER_PARAMS)
def test_finish_requests_clears_counter_left_by_a_stomped_status(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    """The abort case that wedged the talker.

    The request was counted while parked for streaming input, then had its
    status rewritten to ``RUNNING`` by omni's queue surgery, so upstream's
    ``finish_requests`` took the running branch and never decremented. With
    no parked requests left, the counter must land at 0.
    """
    aborted = _StubRequest("req-aborted", RequestStatus.RUNNING)
    scheduler = _make_scheduler(
        scheduler_cls,
        requests={},  # upstream already dropped it
        running=[],
        waiting=[],
        counter=1,
    )

    monkeypatch.setattr(
        scheduler_mod.VLLMScheduler,
        "finish_requests",
        lambda self, request_ids, finished_status: [aborted],
    )

    scheduler_cls.finish_requests(scheduler, [aborted.request_id], RequestStatus.FINISHED_ABORTED)

    assert scheduler.num_waiting_for_streaming_input == 0


def test_resync_counts_parked_requests_in_skipped_waiting() -> None:
    """``get_num_unfinished_requests`` sums ``waiting`` and ``skipped_waiting``,
    so the resync has to look at both or it under-counts and reintroduces the
    spin the counter exists to prevent."""
    parked = _StubRequest("req-skipped", RequestStatus.WAITING_FOR_STREAMING_REQ)
    scheduler = _make_scheduler(
        OmniARScheduler,
        requests={parked.request_id: parked},
        running=[],
        waiting=[],
        counter=0,
        skipped=[parked],
    )

    scheduler._resync_streaming_input_counter()

    assert scheduler.num_waiting_for_streaming_input == 1


def test_ar_schedule_resyncs_before_delegating_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins placement in ``schedule``: the chunk-transfer adapter rewrites
    status during ``process_pending_chunks``, so the resync has to run after
    that and before upstream schedules, while ``self.waiting`` is still the
    real queue."""
    parked = _StubRequest("req-resumed", RequestStatus.RUNNING)
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler.requests = {parked.request_id: parked}
    scheduler.running = [parked]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.num_waiting_for_streaming_input = 1  # leaked by the status rewrite
    scheduler.policy = "fcfs"

    seen: dict[str, int] = {}

    monkeypatch.setattr(
        ar_sched_mod.OmniSchedulerMixin, "_consume_pending_connector_output", lambda self, model_mode: None
    )
    monkeypatch.setattr(ar_sched_mod.OmniSchedulerMixin, "_process_pending_input_timeouts", lambda self: None)
    monkeypatch.setattr(OmniARScheduler, "_should_defer_waiting_admission", lambda self: False)

    def fake_schedule(self, throttle_prefills=False):
        seen["counter"] = self.num_waiting_for_streaming_input
        raise _StopScheduleError

    class _StopScheduleError(Exception):
        pass

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "schedule", fake_schedule)

    with pytest.raises(_StopScheduleError):
        OmniARScheduler.schedule(scheduler)

    assert seen["counter"] == 0


def test_generation_scheduler_sweep_removes_aborted_requests_from_all_queues() -> None:
    aborted = _StubRequest("req-generation-aborted", RequestStatus.FINISHED_ABORTED)
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.waiting.add_request(aborted)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting.add_request(aborted)
    scheduler.running = [aborted]

    scheduler._drop_aborted_queued_requests()

    assert list(scheduler.waiting) == []
    assert list(scheduler.skipped_waiting) == []
    assert scheduler.running == []
