from __future__ import annotations

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Coordinator:
    _async_chunk = True

    def __init__(self) -> None:
        self.metadata_calls = []
        self.chunk_calls = []

    def update_request_metadata(self, requests, metadata, model_mode) -> None:
        self.metadata_calls.append((requests, metadata, model_mode))

    def process_pending_chunks(self, waiting, running, ready, finished) -> None:
        self.chunk_calls.append((waiting, running, ready, finished))


class _Scheduler(OmniSchedulerMixin):
    def __init__(self) -> None:
        self.requests = {"r1": object(), "r2": object()}
        self.waiting = object()
        self.running = []
        self.input_coordinator = _Coordinator()
        self._latest_omni_connector_output = None
        self._init_omni_connector_output_inbox()


def test_direct_ready_inbox_merges_events_before_scheduler_admission() -> None:
    scheduler = _Scheduler()
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={"r1"},
            request_metadata={"r1": {"decode_token_end": 2}},
        )
    )
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={"r2"},
            chunk_finished_req_ids={"r2"},
            request_metadata={"r2": {"left_context_size": 25}},
        )
    )

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.input_coordinator.metadata_calls == [
        (
            scheduler.requests,
            {
                "r1": {"decode_token_end": 2},
                "r2": {"left_context_size": 25},
            },
            "ar",
        )
    ]
    assert scheduler.input_coordinator.chunk_calls == [(scheduler.waiting, scheduler.running, {"r1", "r2"}, {"r2"})]

    scheduler._consume_pending_connector_output(model_mode="ar")
    assert len(scheduler.input_coordinator.metadata_calls) == 1
    assert scheduler.input_coordinator.chunk_calls[-1][2:] == (set(), set())


def test_ready_inbox_merges_output_carried_fallback_in_same_drain() -> None:
    scheduler = _Scheduler()
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"r1"}))
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        chunk_ready_req_ids={"r2"},
        request_metadata={"r2": {"decode_token_end": 3}},
    )

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.input_coordinator.chunk_calls == [(scheduler.waiting, scheduler.running, {"r1", "r2"}, set())]
    assert scheduler._latest_omni_connector_output is None


def test_ready_inbox_coalesces_duplicate_ready_events() -> None:
    scheduler = _Scheduler()
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"r1"}))
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"r1"}))

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.input_coordinator.chunk_calls == [(scheduler.waiting, scheduler.running, {"r1"}, set())]


def test_ready_inbox_drops_late_events_for_aborted_request() -> None:
    scheduler = _Scheduler()
    scheduler.requests = {"r1": object()}
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={"r1", "aborted"},
            chunk_finished_req_ids={"aborted"},
            request_metadata={
                "r1": {"decode_token_end": 2},
                "aborted": {"decode_token_end": 99},
            },
        )
    )

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert scheduler.input_coordinator.metadata_calls == [(scheduler.requests, {"r1": {"decode_token_end": 2}}, "ar")]
    assert scheduler.input_coordinator.chunk_calls == [(scheduler.waiting, scheduler.running, {"r1"}, set())]
