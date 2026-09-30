# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Chunk-lifecycle coverage for OmniGenerationScheduler: restore on error,
native terminal state, in-flight no-resubmission, slot release on finish,
and single scheduling of a terminal empty-prompt chunk."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler, _has_async_chunk_payload_to_run
from vllm_omni.engine.serialization import serialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeAdapter:
    """Minimal mock of OmniChunkTransferAdapter tracking restore calls."""

    def __init__(self):
        self.waiting_for_chunk_waiting_requests = deque()
        self.waiting_for_chunk_running_requests = deque()
        self.restore_called = False
        self.done_request_ids = set()

    def process_pending_chunks(self, waiting, running, scheduler_requests=None):
        if running:
            req = running.pop()
            self.waiting_for_chunk_running_requests.append(req)

    def is_done_receiving_chunks(self, request_id):
        return request_id in self.done_request_ids

    def collect_failed_send_request_ids(self):
        return {}

    def restore_queues(self, waiting, running, scheduler_requests=None):
        self.restore_called = True
        running.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(self, output):
        pass


def _make_generation_scheduler(waiting_request, *, use_v2_model_runner=False):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.max_num_scheduled_tokens = 8
    scheduler.max_num_running_reqs = 1
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.waiting.add_request(waiting_request)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.requests = {waiting_request.request_id: waiting_request}
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.chunk_transfer_adapter = FakeAdapter()
    scheduler.input_coordinator = None
    scheduler.log_stats = False
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    scheduler.num_lookahead_tokens = 0
    scheduler.num_spec_tokens = 0
    scheduler.dynamic_sd_lookup = None
    scheduler.reset_preempted_req_ids = set()
    scheduler.kv_cache_manager = SimpleNamespace(
        new_step_starts=lambda: None,
        allocate_slots=lambda *args, **kwargs: SimpleNamespace(get_block_ids=lambda: ([1],)),
        get_num_common_prefix_blocks=lambda request_id: [0],
        take_new_block_ids=lambda: [],
        take_boundary_state_offloads=lambda: {},
    )
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = use_v2_model_runner
    scheduler._retains_state_across_chunks = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: [],
        get_manager_metadata=lambda: None,
    )
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._pending_finish_reqs = []
    scheduler._consume_pending_connector_output = lambda model_mode: None
    scheduler._process_pending_input_timeouts = lambda: None
    scheduler._make_cached_request_data = lambda **kwargs: SimpleNamespace(
        req_ids=[],
        resumed_req_ids=[],
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    scheduler._update_after_schedule = lambda output: None
    scheduler._wrap_omni_scheduler_output = lambda output: output
    return scheduler


def _chunk_request(request_id, **kwargs):
    defaults = dict(
        request_id=request_id,
        prompt_token_ids=[1],
        num_computed_tokens=0,
        num_in_flight_tokens=0,
        status=None,
        sampling_params=None,
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        prompt_is_token_ids=True,
        additional_information=None,
        external_req_id=request_id,
        prefill_stats=None,
        record_event=lambda *args, **kwargs: None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_full_payload_scheduler(requests, *, already_running=False, chunked_prefill=True):
    scheduler = _make_generation_scheduler(requests[0])
    scheduler.max_num_running_reqs = len(requests)
    scheduler.chunk_transfer_adapter = None
    scheduler.scheduler_config.enable_chunked_prefill = chunked_prefill
    scheduler._process_pending_omni_inputs = lambda **kwargs: None
    scheduler._postprocess_omni_schedule_output = lambda output: None
    scheduler._restore_omni_wait_queues = lambda: None
    scheduler._record_prefill_stats = lambda request: None
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.requests = {request.request_id: request for request in requests}
    for request in requests:
        if already_running:
            request.status = RequestStatus.RUNNING
            scheduler.running.append(request)
        else:
            scheduler.waiting.add_request(request)
    # Exercise the real finish_requests/_free_request cleanup on rejection.
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.defer_block_free = False
    scheduler.encoder_cache_manager.free = lambda request: None
    scheduler.kv_cache_manager.free = lambda request: None
    return scheduler


@pytest.mark.parametrize("already_running", [False, True])
@pytest.mark.parametrize("chunked_prefill", [False, True])
def test_full_payload_is_deferred_without_partial_execution(already_running, chunked_prefill):
    requests = [
        Request(name, [0] * 6, SamplingParams(max_tokens=1), pooling_params=None) for name in ("first", "second")
    ]
    scheduler = _make_full_payload_scheduler(requests, already_running=already_running, chunked_prefill=chunked_prefill)
    allocations = []
    allocate = scheduler.kv_cache_manager.allocate_slots

    def record_allocation(request, num_tokens, **kwargs):
        allocations.append((request.request_id, num_tokens))
        return allocate(request, num_tokens, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = record_allocation
    assert scheduler.schedule().num_scheduled_tokens == {"first": 6}
    assert requests[1].num_computed_tokens == 0
    assert allocations == [("first", 6)]
    requests[0].num_computed_tokens = 6
    assert scheduler.schedule().num_scheduled_tokens == {"second": 6}
    assert allocations == [("first", 6), ("second", 6)]
    requests[1].num_computed_tokens = 6
    assert scheduler.schedule().num_scheduled_tokens == {}


@pytest.mark.parametrize("already_running", [False, True])
@pytest.mark.parametrize("chunked_prefill", [False, True])
@pytest.mark.parametrize("with_valid_request", [False, True])
def test_oversized_payload_emits_request_error(already_running, chunked_prefill, with_valid_request):
    oversized = Request("large", [0] * 9, SamplingParams(max_tokens=1), pooling_params=None, client_index=3)
    valid = Request("valid", [0] * 8, SamplingParams(max_tokens=1), pooling_params=None)
    scheduler = _make_full_payload_scheduler(
        [oversized, valid] if with_valid_request else [oversized],
        already_running=already_running,
        chunked_prefill=chunked_prefill,
    )
    freed = []
    scheduler.kv_cache_manager.free = lambda request: freed.append(request.request_id)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == ({"valid": 8} if with_valid_request else {})
    assert oversized.status == RequestStatus.FINISHED_ERROR
    assert "large" not in scheduler.requests
    assert oversized not in scheduler.running
    assert oversized not in list(scheduler.waiting)
    assert freed == ["large"]
    outputs: dict[int, EngineCoreOutputs] = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=False)
    [error] = outputs[3].outputs
    assert error.request_id == "large"
    assert error.finish_reason == FinishReason.ERROR
    assert "9 tokens" in error.stop_reason
    assert "max_num_batched_tokens=8" in error.stop_reason
    outputs = {}
    scheduler._attach_finished_request_sets(outputs, synthesize_abort_outputs=False)
    assert outputs == {}


@pytest.mark.parametrize("already_running", [False, True])
def test_full_payload_allocation_failure_does_not_fall_back(monkeypatch, already_running):
    request = Request("req", [0] * 8, SamplingParams(max_tokens=1), pooling_params=None)
    scheduler = _make_full_payload_scheduler([request], already_running=already_running)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Full payloads must not fall back to the token-splitting scheduler")

    monkeypatch.setattr("vllm.v1.core.sched.scheduler.Scheduler.schedule", unexpected_fallback)
    allocate = scheduler.kv_cache_manager.allocate_slots
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert request.num_computed_tokens == 0
    assert request in scheduler.running or request in list(scheduler.waiting)
    scheduler.kv_cache_manager.allocate_slots = allocate
    assert scheduler.schedule().num_scheduled_tokens == {"req": 8}


@pytest.mark.parametrize("already_running", [False, True])
def test_async_chunk_transport_keeps_existing_budget_behavior(already_running):
    request = Request("chunk", [0] * 9, SamplingParams(max_tokens=1), pooling_params=None)
    scheduler = _make_full_payload_scheduler([request], already_running=already_running)
    scheduler.chunk_transfer_adapter = FakeAdapter()
    assert scheduler.schedule().num_scheduled_tokens == {"chunk": 8}
    assert not request.is_finished()


@pytest.mark.parametrize("already_running", [False, True])
def test_paused_full_payload_scheduler_does_not_execute(already_running):
    request = Request("req", [0] * 6, SamplingParams(max_tokens=1), pooling_params=None)
    scheduler = _make_full_payload_scheduler([request], already_running=already_running)
    scheduler._pause_state = PauseState.PAUSED_ALL
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert request.num_computed_tokens == 0
    scheduler._pause_state = PauseState.UNPAUSED
    assert scheduler.schedule().num_scheduled_tokens == {"req": 6}


def test_chunk_lifecycle_no_resubmit_and_state_survives_requeue(monkeypatch) -> None:
    # Native plane: a completed payload without a new chunk is never executed
    # again; requeued requests keep WAITING_FOR_CHUNK and their token counts.
    scheduler = _make_generation_scheduler(_chunk_request("unused"))
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.create_request_queue", create_request_queue)
    scheduler._native_data_plane = True
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = SimpleNamespace(
        _async_chunk=True, finished_requests=set(), restore_queues=lambda *args, **kwargs: None
    )
    scheduler.running = []
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)

    completed = Request("completed", [1, 2], SamplingParams(max_tokens=4), pooling_params=None)
    completed.status = RequestStatus.RUNNING
    completed.num_computed_tokens = 2
    scheduler.running = [completed]
    scheduler.requests = {"completed": completed}
    scheduler.input_coordinator.finished_requests.add("completed")  # terminal payload, no new chunk

    assert scheduler._is_done_receiving_chunks("completed")
    assert not scheduler._is_done_receiving_chunks("unknown")
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert scheduler._pending_finish_reqs == [completed]

    completed.status = RequestStatus.WAITING_FOR_CHUNK
    scheduler._pending_finish_reqs = []
    scheduler._requeue_completed_native_chunks()
    assert completed.status == RequestStatus.WAITING_FOR_CHUNK
    assert completed.num_computed_tokens == 2
    assert not scheduler.running


def test_generation_scheduler_schedules_terminal_empty_prompt_chunk_once(monkeypatch: pytest.MonkeyPatch) -> None:
    waiting = _chunk_request(
        "terminal",
        prompt_token_ids=[],
        num_prompt_tokens=0,
        additional_information=serialize_additional_information(
            {"codes": {"audio": torch.tensor([[1, 2, 3]], dtype=torch.long)}}
        ),
    )
    scheduler = _make_generation_scheduler(waiting)
    scheduler.chunk_transfer_adapter.done_request_ids.add("terminal")
    scheduler.requests = {"terminal": waiting}
    assert _has_async_chunk_payload_to_run(waiting)
    # A terminal codec request with an empty prompt is still scheduled exactly once.
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.create_request_queue", create_request_queue)

    output = OmniGenerationScheduler.schedule(scheduler)

    assert output.num_scheduled_tokens == {"terminal": 1}
    assert output.scheduled_new_reqs[0].req_id == "terminal"
    assert scheduler._pending_finish_reqs == []


class TestRestoreQueuesOnError:
    """Verify that restore_queues is called even when rewrapping raises."""

    def test_requests_not_lost_on_exception(self):
        """Simulate the error path: process_pending_chunks moves a request
        out, then an exception occurs during rewrapping.
        The finally block must restore the request to the queue."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        # Step 1: process_pending_chunks moves req-B out
        adapter.process_pending_chunks(waiting=[], running=running)
        assert running == ["req-A"]
        assert len(adapter.waiting_for_chunk_running_requests) == 1

        # Step 2: simulate the try/except/finally pattern
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Log error, leave output unchanged
        finally:
            # This is what guarantees restore always runs
            adapter.restore_queues(waiting=[], running=running)

        # Step 3: verify request is restored
        assert adapter.restore_called is True
        assert "req-B" in running
        assert len(adapter.waiting_for_chunk_running_requests) == 0

    def test_requests_lost_without_fix(self):
        """Demonstrate the bug: without restore in except, request is lost."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)
        assert running == ["req-A"]

        # Simulate the BUGGY code: except without restore
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Bug: no restore_queues call

        # Request is lost!
        assert "req-B" not in running
        assert len(adapter.waiting_for_chunk_running_requests) == 1

    def test_happy_path_restores_via_finally(self):
        """When no exception, restore_queues is still called via finally."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)

        # Happy path: no exception, finally still runs
        try:
            pass  # Rewrapping succeeds
        finally:
            adapter.restore_queues(waiting=[], running=running)

        assert adapter.restore_called is True
        assert "req-B" in running
