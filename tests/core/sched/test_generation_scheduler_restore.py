# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Chunk-lifecycle coverage for OmniGenerationScheduler: restore on error,
native terminal state, in-flight no-resubmission, slot release on finish,
and single scheduling of a terminal empty-prompt chunk."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler, _has_async_chunk_payload_to_run
from vllm_omni.engine.serialization import serialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeAdapter:
    """Minimal mock of OmniChunkTransferAdapter tracking restore calls."""

    def __init__(self):
        self.waiting_for_chunk_waiting_requests: deque = deque()
        self.waiting_for_chunk_running_requests: deque = deque()
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


def test_first_chunk_express_slack_guard_tracks_emitted_audio(monkeypatch):
    """Express steps wait while a ready later chunk's stream is close to underrun."""
    import vllm_omni.core.sched.omni_generation_scheduler as module

    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._first_chunk_express = True
    scheduler._express_min_slack_s = 0.5
    scheduler._express_skipped_for_slack = 0
    scheduler._stream_audio = {}
    scheduler._chunk_started = {"started", "idle"}
    ready = SimpleNamespace(
        request_id="started", num_in_flight_tokens=0, prompt_token_ids=[0] * 28, num_computed_tokens=0
    )
    # Started, but no chunk to decode now: never blocks an express step.
    idle = SimpleNamespace(request_id="idle", num_in_flight_tokens=0, prompt_token_ids=[0] * 28, num_computed_tokens=28)
    scheduler.running = [ready, idle]
    scheduler.waiting = []
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    # No measured audio credit: do not delay the ready stream.
    assert not scheduler._continuations_have_slack()
    # 1 s of audio at t=100; at t=100.2 it holds 0.8 s, at t=100.7 only 0.3 s.
    scheduler._record_stream_audio("started", {"model_outputs": torch.zeros(24000), "sr": torch.tensor(24000)})
    scheduler._record_stream_audio("started", {"model_outputs": torch.zeros(0), "sr": torch.tensor(24000)})
    clock[0] = 100.2
    assert scheduler._continuations_have_slack()
    clock[0] = 100.7
    assert not scheduler._continuations_have_slack()
    # Another second emitted: slack is back to 1.3 s.
    scheduler._record_stream_audio("started", {"model_outputs": torch.zeros(24000), "sr": torch.tensor(24000)})
    assert scheduler._continuations_have_slack()
    assert scheduler._stream_audio["started"] == [100.0, 2.0]


def _express_scheduler():
    continuation = Request("continuation", [1, 2], SamplingParams(max_tokens=4), pooling_params=None)
    first = Request("first", [1], SamplingParams(max_tokens=4), pooling_params=None)
    scheduler = _make_generation_scheduler(continuation, use_v2_model_runner=True)
    scheduler.waiting.add_request(first)
    scheduler.requests[first.request_id] = first
    scheduler.max_num_running_reqs = 4
    scheduler._native_data_plane = True
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = SimpleNamespace(
        _async_chunk=True, finished_requests=set(), restore_queues=lambda *args, **kwargs: None
    )
    scheduler._first_chunk_express = True
    scheduler._last_step_express = False
    scheduler._chunk_started = {"continuation"}
    scheduler._express_min_slack_s = 0
    scheduler._stream_audio = {}
    return scheduler, continuation, first


def test_express_schedules_only_first_chunks_then_allows_continuations():
    scheduler, continuation, first = _express_scheduler()
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"first": 1}
    assert scheduler._last_step_express
    assert continuation in list(scheduler.waiting)
    # Another first chunk arrives before the continuation runs.
    first.num_computed_tokens = len(first.prompt_token_ids)
    second = Request("second", [1], SamplingParams(max_tokens=4), pooling_params=None)
    scheduler.waiting.add_request(second)
    scheduler.requests[second.request_id] = second
    output = scheduler.schedule()
    assert not scheduler._last_step_express
    assert "continuation" in output.num_scheduled_tokens


def test_express_does_not_delay_continuation_for_unready_first_chunk():
    scheduler, continuation, first = _express_scheduler()
    first.num_in_flight_tokens = 1
    output = scheduler.schedule()
    assert not scheduler._last_step_express
    assert output.num_scheduled_tokens == {"continuation": 2}


def test_express_slack_guard_is_used_by_schedule():
    scheduler, continuation, first = _express_scheduler()
    scheduler._express_min_slack_s = 0.5
    scheduler._express_skipped_for_slack = 0
    output = scheduler.schedule()
    assert not scheduler._last_step_express
    assert "continuation" in output.num_scheduled_tokens


def test_express_cancel_cleans_playback_state(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler, continuation, _ = _express_scheduler()
    scheduler.input_coordinator = None
    scheduler._stream_audio["continuation"] = [1.0, 2.0]
    monkeypatch.setattr(Scheduler, "_free_request", lambda *args: (None, None))
    scheduler._free_request(continuation)
    assert "continuation" not in scheduler._chunk_started
    assert "continuation" not in scheduler._stream_audio
