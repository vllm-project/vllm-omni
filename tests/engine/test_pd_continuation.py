# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression coverage for importing a producer's first AR output."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import FinishReason
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer, IncrementalDetokenizer
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.pd_continuation import (
    PD_PREFILL_KEY,
    PD_RESUME_KEY,
    PDContinuation,
    initial_output_tokens,
    prepend_initial_output,
    validate_pd_sampling,
)
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState
from vllm_omni.request import OmniRequest
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _params(**kwargs):
    return SamplingParams(
        temperature=0,
        max_tokens=kwargs.pop("max_tokens", 5),
        extra_args={PD_RESUME_KEY: True, "kv_transfer_params": {"do_remote_prefill": True}},
        **kwargs,
    )


def _request(length=1236, params=None, token=791, stop_string=None):
    return OmniRequest(
        request_id="req",
        prompt_token_ids=list(range(length)),
        sampling_params=params or _params(),
        pooling_params=None,
        pd_continuation=PDContinuation(length, [token], stop_string),
    )


@pytest.mark.parametrize("length", [1, 127, 128, 129, 1236])
def test_consumer_history_extends_sequence_without_claiming_kv_ready(length):
    req = _request(length)
    assert req.num_prompt_tokens == length
    assert req.num_tokens == length + 1
    assert list(req.all_token_ids) == list(range(length)) + [791]
    assert list(req.output_token_ids) == [791]
    assert req.num_computed_tokens == 0
    assert req.max_tokens == 5
    req.num_computed_tokens = length  # Connector acknowledgement.
    assert req.all_token_ids[req.num_computed_tokens] == 791
    assert req.num_tokens - req.num_computed_tokens == 1
    data = OmniNewRequestData.from_request(req, ([1],))
    assert data.prompt_token_ids == list(range(length))
    assert data.initial_output_token_ids == [791]
    data.initial_output_token_ids.append(42)
    assert list(req.output_token_ids) == [791]


def test_continuation_survives_wire_and_request_clone():
    core = OmniEngineCoreRequest(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=_params(),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        pd_continuation=PDContinuation(3, [791]),
    )
    clone = OmniEngineCoreRequest.from_request(core)
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(clone), type=OmniEngineCoreRequest)
    assert decoded.pd_continuation == PDContinuation(3, [791])
    req = OmniRequest.from_engine_core_request(decoded, block_hasher=None)
    assert list(req.output_token_ids) == [791]
    assert req.num_computed_tokens == 0


def test_producer_preserves_logical_sampling_masks_and_limits():
    sp = _params(max_tokens=19, min_tokens=7, stop_token_ids=[42], stop=["STOP"], repetition_penalty=1.1)
    p = PDDisaggregationMixin._prepare_prefill_sampling_params("req", sp)
    assert (p.max_tokens, p.min_tokens, p.stop_token_ids) == (19, 7, [42])
    assert p.all_stop_token_ids == sp.all_stop_token_ids
    assert p.repetition_penalty == 1.1
    assert p.stop == []
    assert not p.detokenize
    assert p.extra_args[PD_PREFILL_KEY]
    assert p.extra_args["kv_transfer_params"]["do_remote_decode"]
    assert not p.extra_args["kv_transfer_params"]["do_remote_prefill"]
    assert sp.stop == ["STOP"]
    assert PD_PREFILL_KEY not in sp.extra_args


def test_legacy_pd_preparation_unchanged():
    original = SamplingParams(max_tokens=9, stop_token_ids=[42], stop=["STOP"])
    p = PDDisaggregationMixin._prepare_prefill_sampling_params("req", original)
    assert p.max_tokens == p.min_tokens == 1
    assert p.stop == p.stop_token_ids == []
    assert original.max_tokens == 9


@pytest.mark.parametrize("first_token", [42, 791])
def test_producer_stops_after_one_sample_with_transport_length_reason(first_token):
    p = PDDisaggregationMixin._prepare_prefill_sampling_params("req", _params(stop_token_ids=[42]))
    req = OmniRequest("req", [1, 2, 3], p, None)
    req.status = RequestStatus.RUNNING
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._spec_token_placeholders = []
    scheduler.max_model_len = 4096
    tokens, stopped = scheduler._update_request_with_output(req, [first_token])
    assert tokens == [first_token] and stopped
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert list(req.output_token_ids) == [first_token]


def test_initial_token_emitted_once_and_included_in_limit():
    req = _request(params=_params(max_tokens=2))
    assert prepend_initial_output(req, [], False) == []  # Connector progress is not an output.
    req.append_output_token_ids(1217)
    assert check_stop(req, 4096)
    assert prepend_initial_output(req, [1217], True) == [791, 1217]
    assert list(req.output_token_ids) == [791, 1217]
    assert prepend_initial_output(req, [], True) == []
    assert req.num_output_tokens == req.max_tokens == 2


def test_history_is_preserved_after_recompute_preemption():
    req = _request()
    req.append_output_token_ids([1217, 596])
    req.num_computed_tokens = 0
    assert initial_output_tokens(req) == [791, 1217, 596]
    assert OmniNewRequestData.from_request(req, ([2, 3],)).initial_output_token_ids == [791, 1217, 596]


def test_non_pd_history_and_outputs_are_unchanged():
    req = OmniRequest("normal", [1, 2], SamplingParams(), None)
    assert initial_output_tokens(req) == []
    assert prepend_initial_output(req, [791], False) == [791]
    assert req.num_output_tokens == 0


class _Queue(list):
    def remove_requests(self, requests):
        for req in requests:
            if req in self:
                self.remove(req)


def _receiver(req, scheduler_cls=OmniARScheduler):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler._spec_token_placeholders = []
    scheduler.max_model_len = 4096
    scheduler.requests = {req.request_id: req}
    scheduler.waiting = _Queue([req])
    scheduler.skipped_waiting = _Queue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler._pd_completed_outputs = []
    scheduler._kv_wait_start_ts = {}
    scheduler.chunk_transfer_adapter = None
    scheduler._free_request = Mock(return_value=({"kv_ready": True}, None))
    scheduler.connector = object()
    scheduler.kv_cache_manager = SimpleNamespace(cache_blocks=Mock(), estimate_cached_tokens=Mock(return_value=0))
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req.num_computed_tokens = req.pd_continuation.prompt_len
    return scheduler


@pytest.mark.parametrize(
    "params,token,stop_string,reason",
    [
        ({"max_tokens": 1}, 791, None, FinishReason.LENGTH),
        ({"stop_token_ids": [42]}, 42, None, FinishReason.STOP),
        ({"stop": ["STOP"]}, 791, "STOP", FinishReason.LENGTH),
    ],
)
@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
def test_terminal_first_token_waits_for_kv_then_finishes_without_forward(
    params, token, stop_string, reason, scheduler_cls
):
    req = _request(params=_params(**params), token=token, stop_string=stop_string)
    scheduler = _receiver(req, scheduler_cls)
    scheduler._finish_pd_terminal_receives()
    scheduler._free_request.assert_not_called()
    assert scheduler.waiting == [req]
    scheduler.finished_recving_kv_req_ids.add("req")
    scheduler._finish_pd_terminal_receives()
    assert not scheduler.waiting
    scheduler._free_request.assert_called_once_with(req)
    client, output = scheduler._pd_completed_outputs[0]
    assert client == req.client_index
    assert output.new_token_ids == [token]
    assert output.finish_reason == reason
    assert req.num_computed_tokens == 1236  # No y1 forward or extra sampling.
    assert req.num_output_placeholders == 0
    assert not req.pd_output_prefix_pending
    scheduler._finish_pd_terminal_receives()
    assert len(scheduler._pd_completed_outputs) == 1


def test_nonterminal_continuation_uses_upstream_full_kv_receive():
    req = _request()
    scheduler = _receiver(req)
    scheduler.finished_recving_kv_req_ids.add("req")
    scheduler._finish_pd_terminal_receives()
    assert scheduler.waiting == [req]
    scheduler._update_waiting_for_remote_kv(req)
    assert req.num_computed_tokens == 1236  # Upstream must not back off to 1235.
    assert "req" not in scheduler.finished_recving_kv_req_ids


def test_incomplete_receive_rejected(monkeypatch):
    req = _request()
    scheduler = _receiver(req)
    monkeypatch.setattr(Scheduler, "_update_waiting_for_remote_kv", lambda self, req: None)
    req.num_computed_tokens = 1235
    with pytest.raises(RuntimeError, match="complete prompt KV"):
        scheduler._update_waiting_for_remote_kv(req)


@pytest.mark.parametrize("tokens", [[], [1, 2], [-1], [True]])
def test_malformed_handoff_rejected(tokens):
    with pytest.raises(ValueError, match="exactly one"):
        PDContinuation(3, tokens).validate([1, 2, 3])


@pytest.mark.parametrize("kwargs", [{"temperature": 0.5}, {"n": 2}, {"logprobs": 1}, {"prompt_logprobs": 1}])
def test_unsupported_sampling_is_explicit(kwargs):
    params = SamplingParams(temperature=0)
    for key, value in kwargs.items():
        setattr(params, key, value)
    with pytest.raises(ValueError):
        validate_pd_sampling(params)


@pytest.mark.parametrize("cumulative", [None, [791]])
def test_handoff_extracts_actual_completion(cumulative):
    completion = SimpleNamespace(token_ids=[791], cumulative_token_ids=cumulative, finish_reason="length")
    output = SimpleNamespace(outputs=[completion])
    handoff = PDContinuation.from_output(output, [1, 2, 3])
    assert handoff.token_ids == [791]
    assert handoff.prompt_len == 3
    handoff.token_ids.append(42)
    assert completion.token_ids == [791]


@pytest.mark.parametrize("outputs", [[], [SimpleNamespace(finish_reason="abort", token_ids=[791])]])
def test_missing_or_failed_producer_cannot_resume(outputs):
    with pytest.raises(ValueError):
        PDContinuation.from_output(SimpleNamespace(outputs=outputs), [1, 2, 3])


def test_prompt_length_mismatch_rejected():
    with pytest.raises(ValueError, match="complete original prompt"):
        PDContinuation(4, [791]).validate([1, 2, 3])


def test_failed_receive_does_not_allow_recomputation():
    scheduler = _receiver(_request())
    scheduler.failed_recving_kv_req_ids.add("req")
    with pytest.raises(RuntimeError, match="KV load failed"):
        scheduler._update_waiting_for_remote_kv(scheduler.requests["req"])


@pytest.mark.parametrize("unsupported", ["speculation", "pp", "dcp", "pcp", "resumable"])
def test_unsupported_scheduler_modes_rejected_before_admission(monkeypatch, unsupported):
    scheduler = _receiver(_request(), OmniARAsyncScheduler)
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=True)
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=object() if unsupported == "speculation" else None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2 if unsupported == "pp" else 1,
            decode_context_parallel_size=2 if unsupported == "dcp" else 1,
            prefill_context_parallel_size=2 if unsupported == "pcp" else 1,
        ),
    )
    scheduler.requests["req"].resumable = unsupported == "resumable"
    admitted = Mock()
    monkeypatch.setattr(Scheduler, "add_request", admitted)
    with pytest.raises(ValueError, match="PP/CP=1"):
        scheduler.add_request(scheduler.requests["req"])
    admitted.assert_not_called()


@pytest.mark.parametrize("producer", [False, True])
def test_async_admission_allows_consumer_only(monkeypatch, producer):
    req = _request()
    if producer:
        params = PDDisaggregationMixin._prepare_prefill_sampling_params("req", _params())
        req = OmniRequest("req", [1, 2, 3], params, None)
    scheduler = OmniARAsyncScheduler.__new__(OmniARAsyncScheduler)
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=True)
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=None, parallel_config=SimpleNamespace(pipeline_parallel_size=1)
    )
    admitted = Mock()
    monkeypatch.setattr(Scheduler, "add_request", admitted)
    if producer:
        with pytest.raises(ValueError, match="producer requires synchronous scheduling"):
            scheduler.add_request(req)
        admitted.assert_not_called()
    else:
        scheduler.add_request(req)
        admitted.assert_called_once_with(req)


def _async_decode_step(scheduler, req):
    # Exercise the real AsyncScheduler MRO, including upstream computed-token
    # advancement and one placeholder for each scheduled (not imported) output.
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.enable_return_routed_experts = False
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = False
    output = SimpleNamespace(
        num_scheduled_tokens={req.request_id: 1},
        scheduled_spec_decode_tokens={},
        num_spec_tokens_to_schedule=0,
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )
    scheduler._update_after_schedule(output)


@pytest.mark.parametrize("length", [1279, 1280, 1281])
def test_async_two_token_limit_counts_imported_token_without_placeholder(length):
    req = _request(length, params=_params(max_tokens=2))
    scheduler = _receiver(req, OmniARAsyncScheduler)
    scheduler.finished_recving_kv_req_ids.add("req")
    scheduler._update_waiting_for_remote_kv(req)
    req.status = RequestStatus.RUNNING
    assert req.num_output_placeholders == 0
    _async_decode_step(scheduler, req)
    assert req.num_computed_tokens == length + 1
    assert req.num_output_placeholders == 1
    tokens, stopped = scheduler._update_request_with_output(req, [1217])
    assert stopped and req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert req.num_output_placeholders == 0
    assert scheduler._get_confirmed_num_computed_tokens(req) == length + 1
    assert prepend_initial_output(req, tokens, stopped) == [791, 1217]
    assert req.num_output_tokens == 2


def test_async_inflight_decode_keeps_confirmed_kv_and_emits_prefix_once():
    req = _request(params=_params(max_tokens=8, stop_token_ids=[42]))
    scheduler = _receiver(req, OmniARAsyncScheduler)
    req.status = RequestStatus.RUNNING
    _async_decode_step(scheduler, req)
    _async_decode_step(scheduler, req)  # Another step submitted before y2 arrives.
    assert req.num_output_placeholders == 2
    tokens, stopped = scheduler._update_request_with_output(req, [1217])
    assert not stopped
    assert req.num_output_placeholders == 1
    assert scheduler._get_confirmed_num_computed_tokens(req) == 1237
    assert prepend_initial_output(req, tokens, stopped) == [791, 1217]
    _async_decode_step(scheduler, req)
    tokens, stopped = scheduler._update_request_with_output(req, [42])
    assert stopped
    assert prepend_initial_output(req, tokens, stopped) == [42]
    assert list(req.output_token_ids) == [791, 1217, 42]
    # The extra in-flight forward is not part of the KV exported at the stop.
    assert scheduler._get_confirmed_num_computed_tokens(req) == 1238
    scheduler.kv_cache_manager.cache_blocks.assert_called_with(req, 1238)


def test_async_model_sampler_history_preserves_imported_token_across_batch_reorder():
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    event = Mock()
    runner.input_batch = SimpleNamespace(
        req_ids=["other", "req", "new"],
        req_output_token_ids=[[9, -1], [791, -1], [791]],
        prev_req_id_to_index={"req": 0, "other": 1},
        sampled_token_ids_cpu=torch.tensor([[1217], [10]]),
        async_copy_ready_event=event,
    )
    assert runner._build_model_sampler_output_token_ids() == [[9, 10], [791, 1217], [791]]
    event.synchronize.assert_called_once()
    assert runner.input_batch.req_output_token_ids == [[9, -1], [791, -1], [791]]


def test_terminal_prefill_stats_are_finalized_before_kv_release():
    req = _request(params=_params(max_tokens=1))
    scheduler = _receiver(req)
    scheduler.finished_recving_kv_req_ids.add("req")
    stats = Mock()
    req.prefill_stats = stats

    def free(request):
        scheduler.kv_cache_manager.estimate_cached_tokens.assert_called_once_with(request)
        stats.finalize.assert_called_once_with(0)
        return {"kv_ready": True}, None

    scheduler._free_request.side_effect = free
    scheduler._finish_pd_terminal_receives()
    assert scheduler._pd_completed_outputs[0][1].prefill_stats is stats


class _TextDetokenizer(BaseIncrementalDetokenizer):
    def decode_next(self, next_token_id):
        return {791: "beforeSTOPafter", 1217: " next"}[next_token_id]


@pytest.mark.parametrize("include_stop", [False, True])
@pytest.mark.parametrize("min_tokens", [0, 1])
def test_first_token_string_stop_uses_normal_detokenizer_semantics(monkeypatch, include_stop, min_tokens):
    req = _request(params=_params(stop=["STOP"], include_stop_str_in_output=include_stop, min_tokens=min_tokens))
    processor = MultimodalOutputProcessor.__new__(MultimodalOutputProcessor)
    processor.request_states = {}
    processor.external_req_ids = defaultdict(list)
    processor.tokenizer = object()
    processor.log_stats, processor.tracing_enabled, processor.stream_interval = False, False, 1
    state = SimpleNamespace(external_req_id="req")
    monkeypatch.setattr(OmniRequestState, "from_new_request", Mock(return_value=state))
    monkeypatch.setattr(
        IncrementalDetokenizer,
        "from_new_request",
        classmethod(lambda cls, tokenizer, request: _TextDetokenizer(request)),
    )
    processor.add_request(req, prompt=None)
    assert processor.request_states["req"] is state
    assert req.pd_continuation.stop_string == ("STOP" if min_tokens == 0 else None)
    if min_tokens == 0:
        scheduler = _receiver(req)
        scheduler.finished_recving_kv_req_ids.add("req")
        scheduler._finish_pd_terminal_receives()
        output = scheduler._pd_completed_outputs[0][1]
        detector = _TextDetokenizer(req)
        stop = detector.update(output.new_token_ids, stop_terminated=output.finish_reason == FinishReason.STOP)
        assert stop == "STOP"
        assert detector.output_token_ids == [791]
        assert detector.get_next_output_text(True, True) == ("beforeSTOP" if include_stop else "before")
        assert detector.get_next_output_text(True, True) == ""  # No duplicate delta.


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
def test_terminal_output_flushes_on_transfer_only_step_and_releases_kv_once(scheduler_cls):
    req = _request(params=_params(max_tokens=1))
    scheduler = _receiver(req, scheduler_cls)
    scheduler.finished_recving_kv_req_ids.add("req")
    scheduler._finish_pd_terminal_receives()
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.active_kv_transfers = {"req"}
    scheduler.waiting_for_transfer_free = {"req"}
    scheduler.transfer_triggered_requests = {"req"}
    scheduler.pending_stop_after_extraction = set()
    scheduler.finished_req_ids_dict = defaultdict(set, {req.client_index: {"req"}})
    scheduler.kv_cache_manager.free = Mock()
    scheduler.kv_cache_manager.take_events = Mock(return_value=None)
    scheduler.make_stats = Mock(return_value=None)
    scheduler._capture_omni_connector_output = Mock()
    schedule = SimpleNamespace(num_scheduled_tokens={})
    worker = SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=["req"],
    )
    result = scheduler.update_from_output(schedule, worker)
    assert len(result[req.client_index].outputs) == 1
    output = result[req.client_index].outputs[0]
    assert output.new_token_ids == [791]
    assert output.finish_reason == FinishReason.LENGTH
    scheduler.kv_cache_manager.free.assert_called_once_with(req)
    assert "req" not in scheduler.requests
    assert not scheduler.active_kv_transfers
    assert not scheduler.waiting_for_transfer_free
    assert not scheduler._pd_completed_outputs
    # A late result for the completed request must not emit or free it again.
    if scheduler_cls is OmniARAsyncScheduler:
        schedule.num_scheduled_tokens = {"req": 1}
        worker.sampled_token_ids = [[1217]]
        worker.kv_extracted_req_ids = []
        scheduler.finished_req_ids_dict.clear()
        assert not scheduler.update_from_output(schedule, worker)
        scheduler.kv_cache_manager.free.assert_called_once_with(req)
