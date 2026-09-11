# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from argparse import Namespace
from collections import defaultdict
from threading import get_ident
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus, StreamingUpdate

from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniChunkRecvHandle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    pass


class _RequestQueue:
    def __init__(self, *requests):
        self.requests = list(requests)

    def add_request(self, request):
        self.requests.append(request)

    def remove_requests(self, requests):
        self.requests = [request for request in self.requests if request not in requests]

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)

    def __contains__(self, request):
        return request in self.requests


class _NativeAppendBase:
    """Minimal vLLM 0.28 update contract, without retired prompt utilities."""

    def _update_request_as_session(self, request, update):
        assert isinstance(update, StreamingUpdate)
        assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        computed = request.num_computed_tokens
        kept_output_tokens = request._all_token_ids[request.num_prompt_tokens : computed]
        del request._all_token_ids[computed:]
        request._output_token_ids.clear()
        request.prompt_token_ids.extend(kept_output_tokens)
        request.prompt_token_ids.extend(update.prompt_token_ids)
        request._all_token_ids.extend(update.prompt_token_ids)
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request.arrival_time = update.arrival_time
        request.sampling_params = update.sampling_params
        request.status = RequestStatus.WAITING


class _NativeAppendScheduler(OmniSchedulerMixin, _NativeAppendBase):
    pass


class _FailingMetricsNativeAppendScheduler(_NativeAppendScheduler):
    def get_streaming_prompt_metrics(self, request_id):
        del request_id
        raise RuntimeError("metrics unavailable")


class _FailOnceCommitNativeAppendScheduler(_NativeAppendScheduler):
    fail_next_commit: bool

    def _commit_native_append(self, request_id):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise RuntimeError("transient commit failure")
        return super()._commit_native_append(request_id)


class _CommitThenFailOnceNativeAppendScheduler(_NativeAppendScheduler):
    fail_after_commit: bool

    def _commit_native_append(self, request_id):
        self.commit_calls += 1
        super()._commit_native_append(request_id)
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise RuntimeError("reply construction failed after commit")


class _StreamingRequest(Namespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, **kwargs):
        defaults = {
            "request_id": "req",
            "max_tokens": 20,
            "sampling_params": SamplingParams(max_tokens=20),
            "arrival_time": 1.0,
        }
        super().__init__(**(defaults | kwargs))

    def is_finished(self):
        return RequestStatus.is_finished(self.status)


def test_streaming_prompt_metrics_returns_not_found_after_terminal_cleanup():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}

    assert scheduler.get_streaming_prompt_metrics("closed-request") == {
        "status": "NOT_FOUND",
        "num_prompt_tokens": 0,
        "num_computed_tokens": 0,
        "is_finished": True,
        "omni_request_found": False,
    }


def test_native_input_error_survives_request_cleanup_in_bounded_metrics_cache():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}
    for index in range(257):
        scheduler._record_native_model_input_error(f"req-{index}", "native_duplex_prefill_failed: encoder empty")
    assert len(scheduler._omni_native_model_input_errors) == 256
    assert scheduler.get_streaming_prompt_metrics("req-0")["status"] == "NOT_FOUND"
    metrics = scheduler.get_streaming_prompt_metrics("req-256")
    assert metrics["status"] == "FINISHED_ERROR"
    assert metrics["omni_model_input_error"] == "native_duplex_prefill_failed: encoder empty"


def test_commit_preserves_input_error_after_request_cleanup():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}
    scheduler._record_native_model_input_error("failed", "native_duplex_prefill_failed: encoder empty")
    with pytest.raises(RuntimeError, match="native_duplex_prefill_failed: encoder empty"):
        scheduler._commit_native_append("failed")


def test_commit_unknown_request_keeps_not_found_semantics():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}
    with pytest.raises(KeyError, match="unknown"):
        scheduler._commit_native_append("unknown")


def test_streaming_prompt_metrics_report_scheduler_queue_and_stale_state():
    request = _StreamingRequest(
        request_id="req",
        status=RequestStatus.WAITING,
        num_prompt_tokens=113,
        num_computed_tokens=88,
        num_stale_output_tokens=3,
        num_in_flight_tokens=3,
        num_output_placeholders=0,
        drop_stale_output=False,
        _all_token_ids=[0] * 113,
        is_finished=lambda: False,
    )
    other_request = _StreamingRequest(request_id="other")
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.running = []
    scheduler.waiting = _RequestQueue(other_request)
    scheduler.skipped_waiting = _RequestQueue(request)
    scheduler.num_waiting_for_streaming_input = 0

    metrics = scheduler.get_streaming_prompt_metrics("req")

    assert metrics["omni_scheduler_running_requests"] == 0
    assert metrics["omni_scheduler_waiting_requests"] == 1
    assert metrics["omni_scheduler_skipped_waiting_requests"] == 1
    assert metrics["omni_scheduler_waiting_for_streaming_input"] == 0
    assert metrics["omni_request_in_running"] is False
    assert metrics["omni_request_in_waiting"] is False
    assert metrics["omni_request_in_skipped_waiting"] is True
    assert metrics["omni_num_stale_output_tokens"] == 3
    assert metrics["omni_num_in_flight_tokens"] == 3
    assert metrics["omni_num_output_placeholders"] == 0
    assert metrics["omni_drop_stale_output"] is False


def test_native_duplex_append_supports_resumable_streaming_request_contract():
    request = _StreamingRequest(
        request_id="req",
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_computed_tokens=5,
        max_tokens=20,
        sampling_params=SamplingParams(max_tokens=20),
        arrival_time=1.0,
        _all_token_ids=[1, 2, 3, 10, 11, 99],
        _output_token_ids=[10, 11, 99],
        is_finished=lambda: False,
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}

    result = scheduler.append_streaming_prompt_unit(
        "req",
        [20],
        {"duplex": {"seq": 2}},
        operation_id="resumable-append",
        operation_fingerprint=b"complete-request-fingerprint",
    )

    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._all_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._output_token_ids == []
    assert request.num_prompt_tokens == 6
    assert request._omni_native_append_committed is True
    assert request.model_intermediate_buffer == {"duplex": {"seq": 2}}
    assert result["deduplicated"] is False
    assert result["omni_context_tokens"] == 6

    retry = scheduler.append_streaming_prompt_unit(
        "req",
        [20],
        {"duplex": {"seq": 2}},
        operation_id="resumable-append",
        operation_fingerprint=b"complete-request-fingerprint",
    )

    assert retry["deduplicated"] is True
    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]


def test_native_duplex_unit_is_runnable_and_committed_when_utility_returns(monkeypatch):
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_computed_tokens=5,
        _all_token_ids=[1, 2, 3, 10, 11, 99],
        _output_token_ids=[10, 11, 99],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    caller_thread = get_ident()
    transitions = []
    update = scheduler._update_request_as_session
    commit = scheduler._commit_native_append

    def tracked_update(request, streaming_update):
        update(request, streaming_update)
        transitions.append(("update", get_ident(), request.status, request._omni_native_append_committed))

    def tracked_commit(request_id):
        # The pending journal must already own this mutation before commit.
        assert set(request._omni_native_append_pending) == {"atomic-unit"}
        commit(request_id)
        transitions.append(("commit", get_ident(), request.status, request._omni_native_append_committed))

    monkeypatch.setattr(scheduler, "_update_request_as_session", tracked_update)
    monkeypatch.setattr(scheduler, "_commit_native_append", tracked_commit)

    result = scheduler.append_streaming_prompt_unit(
        "req", [20], operation_id="atomic-unit", operation_fingerprint=b"atomic-unit-fingerprint"
    )

    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._all_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._output_token_ids == []
    assert request.num_prompt_tokens == 6
    assert request._omni_native_append_committed is True
    assert request.status == RequestStatus.WAITING
    assert request._omni_native_append_pending == {}
    assert result["deduplicated"] is False
    assert transitions == [
        ("update", caller_thread, RequestStatus.WAITING, False),
        ("commit", caller_thread, RequestStatus.WAITING, True),
    ]


def test_native_duplex_append_fences_async_lookahead_before_prompt_growth():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_computed_tokens=6,
        num_output_placeholders=1,
        num_in_flight_tokens=1,
        num_stale_output_tokens=0,
        spec_token_ids=[-1],
        _all_token_ids=[1, 2, 3, 10, 11, 99],
        _output_token_ids=[10, 11, 99],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    scheduler.append_streaming_prompt_unit(
        "req",
        [20],
        operation_id="append-after-async-lookahead",
        operation_fingerprint=b"async-lookahead-fingerprint",
    )

    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._all_token_ids == [1, 2, 3, 10, 11, 20]
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == 5
    assert request.num_stale_output_tokens == 1
    assert request.drop_stale_output is True
    assert request.spec_token_ids == []


def test_native_duplex_append_rejects_inconsistent_async_placeholder_state():
    request = _StreamingRequest(
        request_id="req",
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=0,
        num_output_placeholders=1,
        num_in_flight_tokens=1,
        num_stale_output_tokens=0,
        drop_stale_output=False,
        spec_token_ids=[-1],
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(RuntimeError, match="async placeholder state is inconsistent"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [20],
            operation_id="append-inconsistent-lookahead",
            operation_fingerprint=b"async-lookahead-fingerprint",
        )

    assert request.prompt_token_ids == [1]
    assert request.num_output_placeholders == 1


def test_native_duplex_unit_append_commits_and_deduplicates_retry():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        num_computed_tokens=5,
        _all_token_ids=[1, 2, 3, 10, 11, 99],
        _output_token_ids=[10, 11, 99],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    result = scheduler.append_streaming_prompt_unit(
        "req",
        [20],
        {},
        operation_id="append-2",
        operation_fingerprint=b"append-2-fingerprint",
    )

    assert result["deduplicated"] is False
    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]
    assert request._omni_native_append_committed is True
    assert request.status == RequestStatus.WAITING
    assert request.model_intermediate_buffer == {}
    assert request._omni_native_append_metadata_pending is True

    retry = scheduler.append_streaming_prompt_unit(
        "req",
        [20],
        {},
        operation_id="append-2",
        operation_fingerprint=b"append-2-fingerprint",
    )

    assert retry["deduplicated"] is True
    assert request.prompt_token_ids == [1, 2, 3, 10, 11, 20]


def test_native_duplex_unit_requires_operation_id_before_request_lookup():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}

    with pytest.raises(ValueError, match="requires a non-empty operation_id"):
        scheduler.append_streaming_prompt_unit("missing", [20])


def test_native_append_applies_sampling_snapshot_and_scheduler_length_limit():
    previous = SamplingParams(temperature=0.0, max_tokens=2)
    replacement = SamplingParams(temperature=0.8, max_tokens=7)
    request = _StreamingRequest(
        request_id="req",
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=True,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
        sampling_params=previous,
        max_tokens=2,
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    result = scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="update",
        operation_fingerprint=b"sampling-v2",
        sampling_params=replacement,
    )
    assert result["deduplicated"] is False
    assert request.sampling_params is replacement
    assert request.max_tokens == 7
    assert request._omni_native_append_sampling_pending is True
    assert previous.temperature == 0.0
    retry = scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="update",
        operation_fingerprint=b"sampling-v2",
        sampling_params=replacement,
    )
    assert retry["deduplicated"] is True
    assert request.prompt_token_ids == [1, 2]


def test_native_duplex_unit_requires_operation_fingerprint_before_request_lookup():
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {}

    with pytest.raises(ValueError, match="requires a full operation_fingerprint"):
        scheduler.append_streaming_prompt_unit("missing", [20], operation_id="append")


def test_native_duplex_unit_rejects_operation_id_reuse_with_different_tokens():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    scheduler.append_streaming_prompt_unit("req", [2], operation_id="same-id", operation_fingerprint=b"same-input")

    with pytest.raises(ValueError, match="reused with different input"):
        scheduler.append_streaming_prompt_unit("req", [3], operation_id="same-id", operation_fingerprint=b"same-input")


def test_native_duplex_unit_fingerprint_rejects_same_tokens_with_different_payload():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="same-id",
        operation_fingerprint=b"payload-a",
    )

    with pytest.raises(ValueError, match="reused with different input"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [2],
            operation_id="same-id",
            operation_fingerprint=b"payload-b",
        )

    assert request.prompt_token_ids == [1, 2]


@pytest.mark.parametrize("status", [RequestStatus.RUNNING, RequestStatus.WAITING, RequestStatus.FINISHED_STOPPED])
def test_native_duplex_unit_rejects_nonparked_request_before_mutating_prompt(status):
    request = _StreamingRequest(
        status=status,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(RuntimeError, match="not ready for append"):
        scheduler.append_streaming_prompt_unit(
            "req", [2], operation_id="append", operation_fingerprint=b"append-fingerprint"
        )

    assert request.prompt_token_ids == [1]
    assert request._all_token_ids == [1]
    assert request.status == status
    assert request._omni_native_append_committed is False


def test_native_duplex_unit_enforces_scheduler_context_limit_before_mutation():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1, 2],
        num_prompt_tokens=2,
        num_computed_tokens=3,
        _all_token_ids=[1, 2, 9],
        _output_token_ids=[9],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4))
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(RuntimeError, match="streaming_prompt_context_limit_exceeded"):
        scheduler.append_streaming_prompt_unit(
            "req", [3, 4], operation_id="too-large", operation_fingerprint=b"too-large-fingerprint"
        )

    assert request.prompt_token_ids == [1, 2]
    assert request._all_token_ids == [1, 2, 9]


def test_native_duplex_unit_metrics_failure_does_not_make_commit_ambiguous():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _FailingMetricsNativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    result = scheduler.append_streaming_prompt_unit(
        "req", [2], operation_id="append", operation_fingerprint=b"append-fingerprint"
    )
    retry = scheduler.append_streaming_prompt_unit(
        "req", [2], operation_id="append", operation_fingerprint=b"append-fingerprint"
    )

    assert result["deduplicated"] is False
    assert retry["deduplicated"] is True
    assert request.prompt_token_ids == [1, 2]
    assert request._omni_native_append_committed is True


def test_native_duplex_unit_recovers_commit_failure_without_appending_twice():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _FailOnceCommitNativeAppendScheduler()
    scheduler.fail_next_commit = True
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(RuntimeError, match="streaming_prompt_uncertain_operation_requires_retry"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [2],
            operation_id="append-recover-commit",
            operation_fingerprint=b"fingerprint",
        )

    assert request.prompt_token_ids == [1, 2]
    # vLLM 0.28 makes the segment runnable in its synchronous update; the
    # subsequent commit marker failure must not repeat that update on retry.
    assert request.status == RequestStatus.WAITING
    assert request._omni_native_append_committed is False
    assert set(request._omni_native_append_pending) == {"append-recover-commit"}
    with pytest.raises(ValueError, match="reused with different input"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [2],
            operation_id="append-recover-commit",
            operation_fingerprint=b"different-fingerprint",
        )
    with pytest.raises(RuntimeError, match="streaming_prompt_uncertain_operation_requires_retry"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [3],
            operation_id="different-operation",
            operation_fingerprint=b"different-fingerprint",
        )

    recovered = scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="append-recover-commit",
        operation_fingerprint=b"fingerprint",
    )
    replay = scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="append-recover-commit",
        operation_fingerprint=b"fingerprint",
    )

    assert recovered["deduplicated"] is True
    assert replay["deduplicated"] is True
    assert request.prompt_token_ids == [1, 2]
    assert request._omni_native_append_committed is True
    assert request._omni_native_append_pending == {}


@pytest.mark.parametrize("retry_status", [RequestStatus.WAITING, RequestStatus.WAITING_FOR_STREAMING_REQ])
def test_native_duplex_unit_records_receipt_when_commit_completed_before_error(retry_status):
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _CommitThenFailOnceNativeAppendScheduler()
    scheduler.commit_calls = 0
    scheduler.fail_after_commit = True
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(RuntimeError, match="streaming_prompt_uncertain_operation_requires_retry"):
        scheduler.append_streaming_prompt_unit(
            "req",
            [2],
            operation_id="append-committed-before-error",
            operation_fingerprint=b"fingerprint",
        )

    assert request._omni_native_append_committed is True
    # A lost utility reply does not pause decoding: the model may already
    # have stopped the segment again before the same operation is retried.
    request.status = retry_status
    recovered = scheduler.append_streaming_prompt_unit(
        "req",
        [2],
        operation_id="append-committed-before-error",
        operation_fingerprint=b"fingerprint",
    )

    assert recovered["deduplicated"] is True
    assert scheduler.commit_calls == 1
    assert request.prompt_token_ids == [1, 2]
    assert request._omni_native_append_committed is True
    assert request.status == retry_status
    assert request._omni_native_append_pending == {}


def test_evicted_native_append_receipt_is_rejected_instead_of_reapplied(monkeypatch):
    monkeypatch.setattr(omni_scheduler_mixin, "_NATIVE_APPEND_RECEIPT_LIMIT", 1)
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=16))
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    scheduler.append_streaming_prompt_unit(
        "req", [2], operation_id="old-operation", operation_fingerprint=b"old-fingerprint"
    )
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.num_computed_tokens = len(request._all_token_ids)
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    scheduler.append_streaming_prompt_unit(
        "req", [3], operation_id="new-operation", operation_fingerprint=b"new-fingerprint"
    )

    with pytest.raises(RuntimeError, match="streaming_prompt_idempotency_window_expired"):
        scheduler.append_streaming_prompt_unit(
            "req", [2], operation_id="old-operation", operation_fingerprint=b"old-fingerprint"
        )

    assert request.prompt_token_ids == [1, 2, 3]


def test_native_append_rejects_new_work_when_idempotency_tombstones_are_full(monkeypatch):
    monkeypatch.setattr(omni_scheduler_mixin, "_NATIVE_APPEND_RECEIPT_LIMIT", 1)
    monkeypatch.setattr(omni_scheduler_mixin, "_NATIVE_APPEND_TOMBSTONE_LIMIT", 1)
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=16))
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    scheduler.append_streaming_prompt_unit(
        "req", [2], operation_id="operation-1", operation_fingerprint=b"fingerprint-1"
    )
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.num_computed_tokens = len(request._all_token_ids)
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    scheduler.append_streaming_prompt_unit(
        "req", [3], operation_id="operation-2", operation_fingerprint=b"fingerprint-2"
    )

    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.num_computed_tokens = len(request._all_token_ids)
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)
    with pytest.raises(RuntimeError, match="streaming_prompt_idempotency_capacity_exhausted"):
        scheduler.append_streaming_prompt_unit(
            "req", [4], operation_id="operation-3", operation_fingerprint=b"fingerprint-3"
        )

    assert request.prompt_token_ids == [1, 2, 3]


def test_native_append_rejects_empty_unit_before_mutation():
    request = _StreamingRequest(
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_prompt_continuous=True,
        _omni_native_append_committed=False,
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        num_computed_tokens=1,
        _all_token_ids=[1],
        _output_token_ids=[],
    )
    scheduler = _NativeAppendScheduler()
    scheduler.requests = {"req": request}
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue(request)

    with pytest.raises(ValueError, match="requires at least one token"):
        scheduler.append_streaming_prompt_unit(
            "req", [], operation_id="empty", operation_fingerprint=b"empty-fingerprint"
        )

    assert request._all_token_ids == [1]


def test_async_chunk_adapter_initializes_for_stage_zero_sender_and_stage_one_receiver(monkeypatch):
    created_adapters = []

    def adapter_factory(config):
        adapter = SimpleNamespace(vllm_config=config)
        created_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(omni_scheduler_mixin, "OmniChunkTransferAdapter", adapter_factory)

    def init_scheduler(**model_config):
        scheduler = _Scheduler()
        scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(**model_config))
        scheduler._init_omni_io_scheduling_state()
        return scheduler

    producer = init_scheduler(
        stage_id=0,
        async_chunk=True,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func="test.pipeline.produce_async_chunk",
    )
    receiver = init_scheduler(
        stage_id=1,
        async_chunk=True,
        requires_full_payload_input=True,
        custom_process_next_stage_input_func=None,
    )
    terminal_stage = init_scheduler(
        stage_id=2,
        async_chunk=True,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
    )

    assert producer.chunk_transfer_adapter is not None
    assert receiver.chunk_transfer_adapter is not None
    assert terminal_stage.chunk_transfer_adapter is not None
    assert receiver.input_coordinator is None
    assert len(created_adapters) == 3


@pytest.mark.parametrize(
    ("stage_id", "async_chunk", "required", "enabled"),
    [
        (0, False, True, False),
        (1, True, True, False),
        (1, False, False, False),
        (1, False, True, True),
    ],
)
def test_full_payload_coordinator_matches_legacy_gate(monkeypatch, stage_id, async_chunk, required, enabled):
    monkeypatch.setattr(
        omni_scheduler_mixin,
        "OmniChunkTransferAdapter",
        lambda _config: SimpleNamespace(),
    )
    scheduler = _Scheduler()
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=stage_id,
            async_chunk=async_chunk,
            requires_full_payload_input=required,
        )
    )

    scheduler._init_omni_io_scheduling_state()

    assert (scheduler.input_coordinator is not None) is enabled


def test_schedule_lifecycle_helpers_process_and_restore_both_input_paths():
    calls: list[tuple[object, ...]] = []
    scheduler = _Scheduler()
    scheduler.waiting = ["waiting"]
    scheduler.running = ["running"]
    scheduler.requests = {"request": object()}
    scheduler._consume_pending_connector_output = lambda mode: calls.append(("consume", mode))
    scheduler._process_pending_input_timeouts = lambda: calls.append(("timeouts",))

    def _collect_timed_out(timeout_s):
        calls.append(("chunk-timeouts", timeout_s))
        return set()

    def _collect_failed_sends():
        calls.append(("failed-sends",))
        return {}

    scheduler.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=True,
        process_pending_chunks=lambda waiting, running, scheduler_requests: calls.append(
            ("process", waiting, running, scheduler_requests)
        ),
        restore_queues=lambda waiting, running, scheduler_requests: calls.append(
            ("restore-chunks", waiting, running, scheduler_requests)
        ),
        collect_timed_out_request_ids=_collect_timed_out,
        collect_failed_send_request_ids=_collect_failed_sends,
    )
    scheduler.input_coordinator = SimpleNamespace(
        restore_queues=lambda waiting: calls.append(("restore-full", waiting))
    )

    scheduler._process_pending_omni_inputs("ar")
    scheduler._restore_omni_wait_queues()

    assert calls == [
        ("consume", "ar"),
        ("timeouts",),
        ("process", scheduler.waiting, scheduler.running, scheduler.requests),
        # The chunk deadline runs after chunks are applied, so a chunk that
        # arrived this cycle resets the clock before it is measured (R1.1).
        ("chunk-timeouts", omni_scheduler_mixin.DEFAULT_INPUT_WAIT_TIMEOUT_S),
        ("failed-sends",),
        ("restore-chunks", scheduler.waiting, scheduler.running, scheduler.requests),
        ("restore-full", scheduler.waiting),
    ]


@pytest.mark.parametrize(
    ("synthesize_abort_outputs", "expected_finish_reason"),
    [(False, None), (True, FinishReason.ABORT)],
)
def test_finished_request_attachment_keeps_ar_abort_policy_explicit(
    synthesize_abort_outputs,
    expected_finish_reason,
):
    scheduler = _Scheduler()
    scheduler.finished_req_ids_dict = defaultdict(set, {2: {"req-finished"}})
    outputs: dict[int, EngineCoreOutputs] = {}

    scheduler._attach_finished_request_sets(
        outputs,
        synthesize_abort_outputs=synthesize_abort_outputs,
    )

    assert outputs[2].finished_requests == {"req-finished"}
    if expected_finish_reason is None:
        assert outputs[2].outputs == []
    else:
        assert outputs[2].outputs[0].finish_reason == expected_finish_reason
    assert scheduler.finished_req_ids_dict == {}


def test_chunk_receive_handle_carries_minimal_registration_fields():
    handle = OmniChunkRecvHandle(request_id="req", external_req_id="external")
    assert handle.request_id == "req"
    assert handle.external_req_id == "external"


def test_output_helper_preserves_required_nan_counter_default():
    scheduler = _Scheduler()
    request = SimpleNamespace(
        request_id="req-output",
        trace_headers=None,
        take_events=lambda: [],
    )

    output = scheduler._make_omni_engine_output(request, new_token_ids=[])

    assert output.num_nans_in_logits == 0


def test_output_helper_snapshots_final_duplex_segment_input_metadata():
    scheduler = _Scheduler()
    request = SimpleNamespace(
        request_id="req-final-append",
        trace_headers=None,
        take_events=lambda: [],
        model_intermediate_buffer={
            "duplex": {
                "data_plane": True,
                "session_id": "session-1",
                "epoch": 4,
                "seq": 9,
                "turn_id": 2,
                "response_seq": 3,
                "turn_seq": 4,
                "mode": "append_audio_chunk",
                "final": True,
                "payload": {"audio": "must-not-cross-the-output-wire"},
            }
        },
    )

    output = scheduler._make_omni_engine_output(
        request,
        new_token_ids=[151718],
        is_segment_finished=True,
    )

    assert output.streaming_segment_input_metadata == {
        "duplex": {
            "data_plane": True,
            "session_id": "session-1",
            "epoch": 4,
            "seq": 9,
            "turn_id": 2,
            "response_seq": 3,
            "turn_seq": 4,
            "mode": "append_audio_chunk",
            "final": True,
        }
    }


def test_output_helper_omits_segment_metadata_before_boundary():
    scheduler = _Scheduler()
    request = SimpleNamespace(
        request_id="req-in-flight",
        trace_headers=None,
        take_events=lambda: [],
        model_intermediate_buffer={"duplex": {"data_plane": True, "final": True}},
    )

    output = scheduler._make_omni_engine_output(
        request,
        new_token_ids=[7],
        is_segment_finished=False,
    )

    assert output.streaming_segment_input_metadata is None
