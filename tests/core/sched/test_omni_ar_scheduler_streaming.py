"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.outputs import OmniConnectorOutput

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(
    *,
    stage_id: int = 0,
    async_chunk: bool = False,
    model_stage: str = "thinker",
    final_output: bool = False,
) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched._omni_pending_segment_finished = set()
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=stage_id,
            async_chunk=async_chunk,
            final_output=final_output,
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            model_stage=model_stage,
        )
    )
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    return sched


def test_terminal_only_input_coordinator_request_finishes_without_model_run() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        finished_requests={"req-live", "req-stale", "req-ready"},
        requests_with_ready_chunks={"req-ready"},
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))
        for request_id in request_ids:
            sched.requests.pop(request_id, None)

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == [(["req-live"], RequestStatus.FINISHED_STOPPED)]
    assert sched.input_coordinator.finished_requests == {"req-ready"}


def test_final_output_terminal_only_request_runs_terminal_flush_step() -> None:
    sched = _make_scheduler(stage_id=2, final_output=True)
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        finished_requests={"req-live", "req-stale", "req-ready"},
        requests_with_ready_chunks={"req-ready"},
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == []
    assert sched.input_coordinator.requests_with_ready_chunks == {"req-live", "req-ready"}
    assert sched.input_coordinator.finished_requests == {"req-live", "req-ready"}


def test_full_payload_input_coordinator_request_is_not_terminal_only() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=False,
        finished_requests={"req-live"},
        requests_with_ready_chunks=set(),
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == []
    assert sched.input_coordinator.finished_requests == {"req-live"}


def test_final_output_segment_only_connector_output_wakes_flush_step() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, final_output=True)
    sched.requests = {"req-live": object()}
    sched.waiting = []
    sched.running = []
    sched._omni_pending_upstream_segment_finished = set()
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_segment_finished_req_ids={"req-live"},
    )
    chunk_calls = []
    full_payload_calls = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids):
        chunk_calls.append((set(chunk_ready_req_ids), set(chunk_finished_req_ids)))

    def process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids):
        full_payload_calls.append(set(stage_recv_req_ids))

    sched.input_coordinator = SimpleNamespace(
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=process_pending_full_payload_inputs,
    )

    sched._consume_pending_connector_output(model_mode="ar")

    assert sched._latest_omni_connector_output is None
    assert sched._omni_pending_upstream_segment_finished == {"req-live"}
    assert chunk_calls == [({"req-live"}, set())]
    assert full_payload_calls == [set()]


def test_segment_ready_connector_output_remains_pending_for_flush_step() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, final_output=True)
    sched.requests = {"req-live": object()}
    sched.waiting = []
    sched.running = []
    sched._omni_pending_upstream_segment_finished = set()
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_ready_req_ids={"req-live"},
        chunk_segment_finished_req_ids={"req-live"},
    )
    chunk_calls = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids):
        chunk_calls.append((set(chunk_ready_req_ids), set(chunk_finished_req_ids)))

    sched.input_coordinator = SimpleNamespace(
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=lambda waiting, running, stage_recv_req_ids: None,
    )

    sched._consume_pending_connector_output(model_mode="ar")

    assert sched._omni_pending_upstream_segment_finished == {"req-live"}
    assert chunk_calls == [({"req-live"}, set())]


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_stage0_model_runner_final_commit_emits_segment_terminal() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.streaming_queue = deque()
    sched.requests = {session.request_id: session}
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests
    final_commit = _make_request()

    sched.add_request(final_commit)

    assert sched._omni_pending_segment_finished == {session.request_id}
    assert finished == [(session.request_id, RequestStatus.FINISHED_STOPPED)]
    assert sched.has_requests()


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2
