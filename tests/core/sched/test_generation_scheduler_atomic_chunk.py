# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Atomic scheduling of connector-delivered async chunks (generation stage).

Regression coverage for the ICL first-chunk clipping crash: the waiting fast
path scheduled ``min(required_tokens, token_budget)``, splitting a stateful
codec chunk whose metadata still describes the complete chunk. With 57 first
chunks of 1168 tokens and a 65536 token budget, the 57th chunk was clipped to
the 128 token remainder and the ValueError killed the whole stage engine.

Rules under test (see docs section 9.5 of the incident analysis):

1. the waiting and running loops defer async chunks intact when the remaining
   budget is insufficient -- never split, and never lose the ready marker;
2. ready chunks larger than the full step budget fail their request through
   the standard finish path and queue an explicit ERROR output, which a
   zero-token engine step must still emit exactly once;
3. non-chunk inputs (sender-only adapter / plain prompts) keep their previous
   behavior on both paths.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request
# before Request / RequestStatus are bound in this module.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BUDGET = 65536
FIRST_CHUNK_TOKENS = (72 + 1) * 16  # 73 frames x 16 quantizers = 1168
FOLLOWUP_TOKENS = 25 * 16  # 400
OVERSIZED_TOKENS = BUDGET + 16  # 65552: can never fit a full step


class FakeBlocks:
    def get_block_ids(self, allow_none: bool = False):
        return ()


class FakeKV:
    """Stands in for kv_cache_manager; allocation never fails by default."""

    def __init__(self, fail_allocation: bool = False):
        self.fail_allocation = fail_allocation

    def new_step_starts(self):
        pass

    def allocate_slots(self, request, num_tokens, num_lookahead_tokens=0):
        return None if self.fail_allocation else FakeBlocks()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_events(self):
        return []

    def take_new_block_ids(self):
        return []


class FakeAdapter:
    """Minimal chunk-receiving adapter honoring the ready-chunk contract.

    ``postprocess_scheduler_output`` mirrors the real adapter's
    ``_clear_chunk_ready``: only request ids actually scheduled in this step
    lose their ready marker; deferred chunks stay ready for the next step.
    """

    def __init__(self, receives_chunks: bool = True):
        self.receives_chunks = receives_chunks
        self.requests_with_ready_chunks: set[str] = set()
        self.replaced_streaming_prompt_ids: set[str] = set()
        self.num_running_waiting_for_chunk = 0
        self.restore_called = False
        self.finish_calls: list[tuple] = []
        self.finish_calls_unresolved: list[str] = []

    def process_pending_chunks(self, waiting, running, scheduler_requests=None):
        pass

    def restore_queues(self, waiting, running, scheduler_requests=None):
        self.restore_called = True

    def is_done_receiving_chunks(self, request_id):
        return False

    def collect_failed_receive_request_ids(self):
        return {}

    def collect_timed_out_request_ids(self, timeout_s):
        return set()

    def collect_failed_send_request_ids(self):
        return {}

    def finish_requests(self, request_ids, finished_status, requests):
        # Record the call and lock the real ordering contract: the adapter
        # hook runs while the requests are still registered, so every
        # targeted id must resolve in the passed request dict.
        self.finish_calls.append((set(request_ids), finished_status))
        self.finish_calls_unresolved = [req_id for req_id in request_ids if req_id not in requests]

    def postprocess_scheduler_output(self, scheduler_output):
        for req_id in getattr(scheduler_output, "num_scheduled_tokens", {}):
            self.requests_with_ready_chunks.discard(req_id)


def _make_scheduler(
    *,
    budget: int = BUDGET,
    max_num_seqs: int = 128,
    enable_chunked_prefill: bool = False,
    receives_chunks: bool = True,
    fail_allocation: bool = False,
):
    sched = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    sched.chunk_transfer_adapter = FakeAdapter(receives_chunks=receives_chunks)
    sched.scheduler_config = SimpleNamespace(enable_chunked_prefill=enable_chunked_prefill, async_scheduling=False)
    sched.use_pp = False
    sched.max_num_scheduled_tokens = budget
    sched.max_num_running_reqs = max_num_seqs
    sched.kv_cache_manager = FakeKV(fail_allocation=fail_allocation)
    sched.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    sched.policy = SchedulingPolicy.FCFS
    sched.waiting = create_request_queue(sched.policy)
    sched.skipped_waiting = create_request_queue(sched.policy)
    sched.running = []
    sched.requests = {}
    sched.finished_req_ids = set()
    sched.finished_req_ids_dict = {}
    sched.connector = None
    sched.ec_connector = None
    sched.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    sched.needs_kv_cache_zeroing = False
    sched.log_stats = False
    sched.perf_metrics = None
    sched.num_lookahead_tokens = 0
    sched.num_waiting_for_streaming_input = 0
    sched._retains_state_across_chunks = False
    sched._pause_state = PauseState.UNPAUSED
    sched._pending_finish_reqs = []
    sched._pending_chunk_error_outputs = []
    sched._latest_omni_connector_output = None
    sched.input_coordinator = None
    sched.use_v2_model_runner = False
    sched.async_scheduling = False
    sched.prev_step_scheduled_req_ids = set()
    sched._last_stats_time = time.monotonic()  # make_stats() short-circuits

    def _finish_requests(request_ids, finished_status):
        # Minimal stand-in for the mixin/vLLM finish path, aligned with the
        # real contracts that the scheduler code depends on:
        # 1. the adapter hook runs first, while the requests are still
        #    registered (it can still look them up);
        # 2. the base path then skips already-finished requests and removes
        #    the rest from the queues/registry;
        # 3. finished IDs are recorded under each request's own client index.
        # Resource release (KV/deferred free) is out of scope here; the real
        # cleanup path is covered by test_omni_scheduler_finish_requests_purge.
        sched.chunk_transfer_adapter.finish_requests(request_ids, finished_status, sched.requests)
        finished = []
        for req_id in set(request_ids):
            request = sched.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            request.status = finished_status
            del sched.requests[req_id]
            sched.finished_req_ids.add(req_id)
            sched.finished_req_ids_dict.setdefault(request.client_index, set()).add(req_id)
            if request in sched.running:
                sched.running.remove(request)
            sched.waiting.remove_requests([request])
            sched.chunk_transfer_adapter.requests_with_ready_chunks.discard(req_id)
            finished.append(request)
        return finished

    sched.finish_requests = _finish_requests
    sched._update_after_schedule = lambda scheduler_output: None
    return sched


def _make_request(request_id: str, num_tokens: int) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=100000),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.WAITING
    request.client_index = 0
    return request


def _stage_waiting_chunk(sched, request: Request) -> None:
    """Register a request whose full chunk has been committed by the adapter."""
    sched.requests[request.request_id] = request
    sched.waiting.add_request(request)
    sched.chunk_transfer_adapter.requests_with_ready_chunks.add(request.request_id)


def _stage_running_chunk(sched, request: Request) -> None:
    sched.requests[request.request_id] = request
    sched.running.append(request)
    sched.chunk_transfer_adapter.requests_with_ready_chunks.add(request.request_id)


def _make_runner_output():
    return SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=None,
        num_nans_in_logits=0,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={},
    )


class TestWaitingDefersWholeChunk:
    @pytest.mark.parametrize("enable_chunked_prefill", [False, True])
    def test_57th_first_chunk_deferred_intact(self, enable_chunked_prefill):
        sched = _make_scheduler(enable_chunked_prefill=enable_chunked_prefill)
        requests = [_make_request(f"speech-{i}", FIRST_CHUNK_TOKENS) for i in range(57)]
        for request in requests:
            _stage_waiting_chunk(sched, request)

        output = sched.schedule()

        # 56 whole chunks are scheduled; the 57th is deferred, never sliced.
        assert len(output.num_scheduled_tokens) == 56
        assert set(output.num_scheduled_tokens.values()) == {FIRST_CHUNK_TOKENS}
        assert output.total_num_scheduled_tokens == 56 * FIRST_CHUNK_TOKENS
        assert 128 not in output.num_scheduled_tokens.values()

        deferred = requests[56]
        assert deferred.request_id not in output.num_scheduled_tokens
        # Intact data, still queued, still ready, still tracked.
        assert len(deferred.prompt_token_ids) == FIRST_CHUNK_TOKENS
        assert deferred.request_id in sched.requests
        assert sched.waiting.peek_request() is deferred
        assert deferred.request_id in sched.chunk_transfer_adapter.requests_with_ready_chunks
        assert len(sched.running) == 56
        # Only the scheduled chunks lose their ready marker (clear-on-schedule).
        assert sched.chunk_transfer_adapter.requests_with_ready_chunks == {deferred.request_id}

    def test_exact_fit_chunk_is_scheduled_whole(self):
        sched = _make_scheduler(budget=FIRST_CHUNK_TOKENS)
        request = _make_request("fit-1", FIRST_CHUNK_TOKENS)
        _stage_waiting_chunk(sched, request)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {request.request_id: FIRST_CHUNK_TOKENS}

    def test_allocate_failure_defers_and_does_not_fall_back(self):
        sched = _make_scheduler(fail_allocation=True)
        request = _make_request("alloc-fail", FIRST_CHUNK_TOKENS)
        _stage_waiting_chunk(sched, request)

        output = sched.schedule()

        # No fallback to the base scheduler (it does not know the chunk
        # protocol); an empty adapter-path output is produced instead.
        assert output.num_scheduled_tokens == {}
        assert sched.chunk_transfer_adapter.restore_called
        assert request.request_id in sched.requests
        assert request.request_id in sched.chunk_transfer_adapter.requests_with_ready_chunks

    def test_empty_prompt_request_is_parked_not_rejected(self):
        # A huge stale prompt that is NOT ready must never be rejected by the
        # oversized pre-check; an empty prompt is parked awaiting its chunk.
        sched = _make_scheduler()
        stale = _make_request("stale-huge", OVERSIZED_TOKENS)
        sched.requests[stale.request_id] = stale
        sched.waiting.add_request(stale)  # not in requests_with_ready_chunks

        output = sched.schedule()

        assert sched._pending_chunk_error_outputs == []
        assert stale.request_id in sched.requests
        assert stale.request_id not in output.num_scheduled_tokens


class TestRunningDefersWholeChunk:
    @pytest.mark.parametrize("enable_chunked_prefill", [True, False])
    def test_followup_chunk_deferred_when_budget_insufficient(self, enable_chunked_prefill):
        # Full budget 1000 fits each chunk whole, but after the filler request
        # consumes 800 only 200 remain. With chunked prefill enabled the old
        # guard would split the follow-up (min(400, 200)); the atomic rule
        # must defer it intact instead.
        sched = _make_scheduler(budget=1000, enable_chunked_prefill=enable_chunked_prefill)
        filler = _make_request("filler-1", 800)
        followup = _make_request("followup-1", FOLLOWUP_TOKENS)
        _stage_running_chunk(sched, filler)
        _stage_running_chunk(sched, followup)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {filler.request_id: 800}
        assert 200 not in output.num_scheduled_tokens.values()
        assert len(followup.prompt_token_ids) == FOLLOWUP_TOKENS
        assert followup.num_computed_tokens == 0
        assert followup.request_id in sched.chunk_transfer_adapter.requests_with_ready_chunks

    def test_consumed_running_chunk_not_rescheduled(self):
        sched = _make_scheduler()
        request = _make_request("consumed-1", FOLLOWUP_TOKENS)
        request.num_computed_tokens = FOLLOWUP_TOKENS  # fully consumed
        _stage_running_chunk(sched, request)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {}
        assert request in sched.running  # parked waiting for its next chunk


class TestOversizedRejected:
    def test_oversized_rejected_normal_request_continues(self):
        sched = _make_scheduler()
        oversized = _make_request("oversized-1", OVERSIZED_TOKENS)
        oversized.resumable = True  # B2 must clear this on terminal failure
        normal = _make_request("normal-1", FIRST_CHUNK_TOKENS)
        _stage_waiting_chunk(sched, oversized)
        _stage_waiting_chunk(sched, normal)

        output = sched.schedule()

        # The oversized chunk failed through the standard finish path with an
        # explicit ERROR output queued; the normal chunk ran whole.
        assert output.num_scheduled_tokens == {normal.request_id: FIRST_CHUNK_TOKENS}
        assert oversized.request_id not in sched.requests
        assert oversized.request_id in sched.finished_req_ids
        assert oversized.status == RequestStatus.FINISHED_ERROR
        assert oversized.resumable is False
        # Exactly one batched finish call targeting only the oversized id,
        # issued while the request was still registered.
        assert sched.chunk_transfer_adapter.finish_calls == [({oversized.request_id}, RequestStatus.FINISHED_ERROR)]
        assert sched.chunk_transfer_adapter.finish_calls_unresolved == []
        # The queued terminal output carries the full error contract.
        assert len(sched._pending_chunk_error_outputs) == 1
        client_index, error_output = sched._pending_chunk_error_outputs[0]
        assert client_index == 0
        assert error_output.request_id == oversized.request_id
        assert error_output.finish_reason == FinishReason.ERROR
        assert error_output.new_token_ids == []
        assert error_output.is_segment_finished is False
        assert "65552" in error_output.stop_reason

    def test_two_oversized_rejected_in_single_batched_finish_call(self):
        sched = _make_scheduler()
        oversized_a = _make_request("oversized-a", OVERSIZED_TOKENS)
        oversized_b = _make_request("oversized-b", OVERSIZED_TOKENS + 32)
        normal = _make_request("normal-1", FIRST_CHUNK_TOKENS)
        for request in (oversized_a, oversized_b, normal):
            _stage_waiting_chunk(sched, request)

        output = sched.schedule()

        # One batched finish call covering exactly the two oversized ids.
        assert output.num_scheduled_tokens == {normal.request_id: FIRST_CHUNK_TOKENS}
        assert len(sched.chunk_transfer_adapter.finish_calls) == 1
        assert sched.chunk_transfer_adapter.finish_calls[0] == (
            {oversized_a.request_id, oversized_b.request_id},
            RequestStatus.FINISHED_ERROR,
        )
        assert sched.chunk_transfer_adapter.finish_calls_unresolved == []
        assert len(sched._pending_chunk_error_outputs) == 2
        errored_ids = {error_output.request_id for _, error_output in sched._pending_chunk_error_outputs}
        assert errored_ids == {oversized_a.request_id, oversized_b.request_id}

    def test_nonzero_client_index_routing(self):
        sched = _make_scheduler()
        oversized = _make_request("client-2-oversized", OVERSIZED_TOKENS)
        oversized.client_index = 2
        _stage_waiting_chunk(sched, oversized)

        output = sched.schedule()

        # Finished IDs and the queued output must follow the request's own
        # client index, not a hardcoded one.
        assert sched.finished_req_ids_dict == {2: {oversized.request_id}}
        client_index, error_output = sched._pending_chunk_error_outputs[0]
        assert client_index == 2
        result = sched.update_from_output(output, _make_runner_output())
        assert oversized.request_id in result[2].finished_requests
        assert 0 not in result

    @pytest.mark.parametrize("budget,chunk_tokens", [(1152, FIRST_CHUNK_TOKENS), (1000, FIRST_CHUNK_TOKENS)])
    def test_boundary_budgets_reject_never_slice(self, budget, chunk_tokens):
        sched = _make_scheduler(budget=budget)
        request = _make_request("boundary-1", chunk_tokens)
        _stage_waiting_chunk(sched, request)

        output = sched.schedule()

        # Never sliced to the budget remainder (1152 would look like exactly
        # 72 frames; 1000 is not even frame-aligned).
        assert output.num_scheduled_tokens == {}
        assert request.request_id not in sched.requests
        assert len(sched._pending_chunk_error_outputs) == 1

    def test_zero_token_step_emits_error_output(self):
        # Scope note: update_from_output() is invoked manually here, which
        # proves the scheduler emits the queued error when the step output is
        # processed -- it does not by itself prove the engine loop always
        # reaches this call (that is exercised by engine-level e2e runs).
        sched = _make_scheduler()
        oversized = _make_request("lone-oversized", OVERSIZED_TOKENS)
        _stage_waiting_chunk(sched, oversized)

        output = sched.schedule()
        assert output.num_scheduled_tokens == {}
        assert oversized.request_id in output.finished_req_ids

        result = sched.update_from_output(output, _make_runner_output())

        engine_outputs = result[0]
        assert oversized.request_id in engine_outputs.finished_requests
        assert len(engine_outputs.outputs) == 1
        terminal = engine_outputs.outputs[0]
        assert terminal.request_id == oversized.request_id
        assert terminal.new_token_ids == []
        assert "65552" in terminal.stop_reason
        # Queued errors are drained exactly once.
        assert sched._pending_chunk_error_outputs == []

    def test_error_emitted_exactly_once_across_two_schedules(self):
        # Scope note: both steps here are zero-token; this checks the
        # error-drain discipline of the queued-output list, not real
        # in-flight batch cleanup (the real finish path is covered by
        # test_omni_scheduler_finish_requests_purge).
        sched = _make_scheduler()
        oversized = _make_request("once-1", OVERSIZED_TOKENS)
        _stage_waiting_chunk(sched, oversized)

        first = sched.schedule()
        assert len(sched._pending_chunk_error_outputs) == 1

        # A second scheduling round must not re-emit or duplicate the error.
        second = sched.schedule()
        assert len(sched._pending_chunk_error_outputs) == 1
        assert second.num_scheduled_tokens == {}

        result = sched.update_from_output(second, _make_runner_output())
        assert len(result[0].outputs) == 1
        assert sched._pending_chunk_error_outputs == []
        result_again = sched.update_from_output(first, _make_runner_output())
        # The drained error is not re-emitted; an empty result may not even
        # carry an entry for the client.
        assert not result_again.get(0) or result_again[0].outputs == []


class TestNonChunkBehaviorUnchanged:
    @pytest.mark.parametrize("enable_chunked_prefill", [True, False])
    def test_sender_only_adapter_waiting_still_clips(self, enable_chunked_prefill):
        # Narrow-vs-wide guard: for a plain waiting input with the switch
        # DISABLED, the wide version (atomic OR not enable_chunked_prefill)
        # would defer the request; the narrow version must keep the legacy
        # clip in BOTH switch states. This case is what distinguishes the
        # two designs, so it is asserted for both.
        sched = _make_scheduler(budget=128, enable_chunked_prefill=enable_chunked_prefill, receives_chunks=False)
        request = _make_request("plain-1", 400)
        sched.requests[request.request_id] = request
        sched.waiting.add_request(request)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {request.request_id: 128}

    def test_sender_only_adapter_oversized_never_rejected(self):
        sched = _make_scheduler(budget=BUDGET, enable_chunked_prefill=True, receives_chunks=False)
        request = _make_request("plain-huge", OVERSIZED_TOKENS)
        sched.requests[request.request_id] = request
        sched.waiting.add_request(request)

        output = sched.schedule()

        assert sched._pending_chunk_error_outputs == []
        assert output.num_scheduled_tokens == {request.request_id: BUDGET}

    def test_plain_running_split_kept_when_chunked_prefill_enabled(self):
        sched = _make_scheduler(budget=128, enable_chunked_prefill=True, receives_chunks=False)
        request = _make_request("plain-run-1", FOLLOWUP_TOKENS)
        sched.requests[request.request_id] = request
        sched.running.append(request)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {request.request_id: 128}

    def test_plain_running_deferred_when_chunked_prefill_disabled(self):
        sched = _make_scheduler(budget=128, enable_chunked_prefill=False, receives_chunks=False)
        request = _make_request("plain-run-2", FOLLOWUP_TOKENS)
        sched.requests[request.request_id] = request
        sched.running.append(request)

        output = sched.schedule()

        assert output.num_scheduled_tokens == {}
