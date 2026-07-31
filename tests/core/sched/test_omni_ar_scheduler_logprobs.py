"""Tests for the AR sampled-token logprob contract."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import LogprobsLists
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, _slice_sampled_logprobs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _logprobs(
    token_ids: list[list[int]],
    values: list[list[float]],
    *,
    cumulative_rows: list[int] | None = None,
) -> LogprobsLists:
    return LogprobsLists(
        logprob_token_ids=np.asarray(token_ids, dtype=np.int32),
        logprobs=np.asarray(values, dtype=np.float32),
        sampled_token_ranks=np.zeros(len(token_ids), dtype=np.int32),
        cu_num_generated_tokens=cumulative_rows,
    )


def test_slices_requested_rows_and_preserves_sampled_token_values() -> None:
    all_logprobs = _logprobs(
        [[3, 8], [4, 9], [11, 2], [12, 1]],
        [[-0.3, -1.0], [-0.4, -1.1], [-0.11, -2.0], [-0.12, -2.1]],
        cumulative_rows=[0, 2],
    )

    sliced = _slice_sampled_logprobs(all_logprobs, req_index=1, sampled_token_ids=[11, 12])

    np.testing.assert_array_equal(sliced.logprob_token_ids[:, 0], [11, 12])
    np.testing.assert_allclose(sliced.logprobs[:, 0], [-0.11, -0.12])


def test_rejects_missing_logprobs() -> None:
    with pytest.raises(RuntimeError, match="model runner returned none"):
        _slice_sampled_logprobs(None, req_index=0, sampled_token_ids=[7])


def test_rejects_row_count_mismatch() -> None:
    logprobs = _logprobs([[7, 1]], [[-0.7, -1.0]])

    with pytest.raises(RuntimeError, match="row count does not match"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7, 8])


def test_rejects_sampled_token_mismatch() -> None:
    logprobs = _logprobs([[9, 7]], [[-0.9, -1.0]])

    with pytest.raises(RuntimeError, match="generated_token=7 logprob_token=9"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7])


def test_rejects_non_finite_sampled_logprob() -> None:
    logprobs = _logprobs([[7, 1]], [[float("nan"), -1.0]])

    with pytest.raises(RuntimeError, match="non-finite values at rows \\[0\\]"):
        _slice_sampled_logprobs(logprobs, req_index=0, sampled_token_ids=[7])


class _Request:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.client_index = 0
        self.status = RequestStatus.RUNNING
        self.sampling_params = SimpleNamespace(num_logprobs=1)
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.has_encoder_inputs = False
        self.pooling_params = None
        self.resumable = True
        self.stop_reason = None
        self.trace_headers = None
        self.num_nans_in_logits = 0

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    def take_events(self):
        return None

    def take_prefill_stats(self):
        return None


class _RequestQueue(list):
    def remove_requests(self, requests) -> None:
        for request in requests:
            if request in self:
                self.remove(request)


def test_mid_step_stop_trims_logprob_rows_with_token_ids() -> None:
    """Spec decode can sample tokens past a stop; the trailing tokens are
    trimmed by _update_request_with_output, and the emitted logprob rows must
    be trimmed with them (upstream slices logprobs after the trim)."""
    request = _Request("req")
    request.num_computed_tokens = 8
    scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        chunk_transfer_adapter=None,
        requests={request.request_id: request},
        running=[request],
        waiting=_RequestQueue(),
        skipped_waiting=_RequestQueue(),
        structured_output_manager=SimpleNamespace(should_advance=lambda _request: False),
        transfer_triggered_requests=set(),
        active_kv_transfers=set(),
        pending_stop_after_extraction=set(),
        waiting_for_transfer_free=set(),
        _new_prompt_len_snapshot={},
        finished_req_ids=set(),
        finished_req_ids_dict=defaultdict(set),
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        kv_event_publisher=SimpleNamespace(publish=lambda _events: None),
    )

    def update_request_trimming(req, token_ids):
        # Mirror vllm Scheduler._update_request_with_output: the first sampled
        # token hits EOS, so the spec-accepted tokens behind it are trimmed
        # in place and the request stops.
        del token_ids[1:]
        req.status = RequestStatus.FINISHED_STOPPED
        return token_ids, True

    def free_request(req):
        scheduler.requests.pop(req.request_id, None)
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.finished_req_ids_dict[req.client_index].add(req.request_id)
        return None, None

    scheduler._update_request_with_output = update_request_trimming
    scheduler._process_kv_transfer_trigger = lambda _request, _tokens: False
    scheduler._handle_stopped_request = lambda _request: True
    scheduler._free_request = free_request
    scheduler.make_spec_decoding_stats = lambda *args, **kwargs: None
    scheduler.make_stats = lambda *args, **kwargs: None
    scheduler._capture_omni_connector_output = lambda _output: None

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 3},
        scheduled_spec_decode_tokens={"req": [8, 9]},
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[7, 8, 9]],
        logprobs=_logprobs(
            [[7, 1], [8, 2], [9, 3]],
            [[-0.7, -1.0], [-0.8, -1.1], [-0.9, -1.2]],
            cumulative_rows=[0],
        ),
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"req": 0},
        routed_experts=None,
    )

    outputs = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)
    (output,) = outputs[0].outputs

    assert output.new_token_ids == [7]
    assert output.finish_reason is FinishReason.STOP
    token_rows = np.asarray(output.new_logprobs.logprob_token_ids)
    value_rows = np.asarray(output.new_logprobs.logprobs)
    assert token_rows.shape[0] == len(output.new_token_ids)
    assert value_rows.shape[0] == len(output.new_token_ids)
    np.testing.assert_array_equal(token_rows[:, 0], [7])


def test_invalid_logprobs_finish_only_the_affected_scheduler_request() -> None:
    """A bad runner response must not bring down unrelated batch requests."""
    bad = _Request("bad")
    good = _Request("good")
    scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        chunk_transfer_adapter=None,
        requests={bad.request_id: bad, good.request_id: good},
        running=[bad, good],
        waiting=_RequestQueue(),
        skipped_waiting=_RequestQueue(),
        structured_output_manager=SimpleNamespace(should_advance=lambda _request: False),
        transfer_triggered_requests=set(),
        active_kv_transfers=set(),
        pending_stop_after_extraction=set(),
        waiting_for_transfer_free=set(),
        _new_prompt_len_snapshot={},
        finished_req_ids=set(),
        finished_req_ids_dict=defaultdict(set),
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        kv_event_publisher=SimpleNamespace(publish=lambda _events: None),
    )
    update_calls: list[str] = []

    def update_request(request, token_ids):
        update_calls.append(request.request_id)
        return token_ids, False

    def free_request(request):
        scheduler.requests.pop(request.request_id, None)
        scheduler.finished_req_ids.add(request.request_id)
        scheduler.finished_req_ids_dict[request.client_index].add(request.request_id)
        # vLLM 0.26 contract: (kv_xfer_params, ec_xfer_params)
        return None, None

    scheduler._update_request_with_output = update_request
    scheduler._process_kv_transfer_trigger = lambda _request, _tokens: False
    scheduler._handle_stopped_request = lambda _request: True
    scheduler._free_request = free_request
    scheduler.make_spec_decoding_stats = lambda *args, **kwargs: None
    scheduler.make_stats = lambda *args, **kwargs: None
    scheduler._capture_omni_connector_output = lambda _output: None

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"bad": 1, "good": 1},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[7], [8]],
        logprobs=_logprobs(
            [[9, 1], [8, 2]],
            [[-0.9, -1.0], [-0.8, -1.0]],
            cumulative_rows=[0, 1],
        ),
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={"bad": 0, "good": 1},
        routed_experts=None,
    )

    outputs = OmniARScheduler.update_from_output(scheduler, scheduler_output, model_runner_output)
    output_by_id = {output.request_id: output for output in outputs[0].outputs}

    assert update_calls == ["good"]
    assert bad.status is RequestStatus.FINISHED_ERROR
    assert bad.stop_reason is not None and "misaligned" in bad.stop_reason
    assert "bad" not in scheduler.requests
    assert output_by_id["bad"].finish_reason is FinishReason.ERROR
    assert output_by_id["bad"].new_token_ids == []
    assert output_by_id["good"].new_token_ids == [8]
    np.testing.assert_array_equal(output_by_id["good"].new_logprobs.logprob_token_ids[:, 0], [8])
