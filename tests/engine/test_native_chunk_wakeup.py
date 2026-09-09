# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import queue
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineShutdownState
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.engine.stage_engine_core_proc import _OMNI_CHUNK_READY, StageEngineCoreProc
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _engine():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine._omni_chunk_wakeup = True
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.engines_running = False
    engine.batch_queue = []
    engine.input_queue = queue.Queue()
    engine.aborts_queue = queue.Queue()
    engine.process_input_queue_block = True
    engine.scheduler = SimpleNamespace(requests={"r": object()}, has_runnable_omni_chunks=lambda: False)
    engine.scheduler.has_requests = lambda: engine.scheduler.has_runnable_omni_chunks()
    engine.is_running = lambda: True
    engine._notify_idle_state_callbacks = lambda: None
    return engine


def test_ready_before_wait_is_not_lost():
    engine = _engine()
    engine.scheduler.has_runnable_omni_chunks = lambda: True
    engine.input_queue.put((_OMNI_CHUNK_READY, None))
    engine._process_input_queue()
    assert engine.input_queue.empty()


def test_ready_wakes_blocked_core():
    engine = _engine()
    waiting = threading.Event()
    ready = threading.Event()
    engine.scheduler.has_runnable_omni_chunks = ready.is_set
    original_get = engine.input_queue.get

    def blocking_get(*args, **kwargs):
        waiting.set()
        return original_get(*args, **kwargs)

    engine.input_queue.get = blocking_get
    done = threading.Event()

    def run():
        engine._process_input_queue()
        done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert waiting.wait(2)
        ready.set()
        engine.input_queue.put((_OMNI_CHUNK_READY, None))
        assert done.wait(2)
    finally:
        engine.is_running = lambda: False
        engine.input_queue.put((_OMNI_CHUNK_READY, None))
        thread.join(2)
    assert not thread.is_alive()


def test_parked_streams_get_periodic_deadline_tick():
    engine = _engine()
    engine.input_queue = Mock()
    engine.input_queue.empty.return_value = True
    engine.input_queue.get.side_effect = queue.Empty
    engine._notify_idle_state_callbacks = Mock()
    engine._process_input_queue()
    engine.input_queue.get.assert_called_once_with(block=True, timeout=0.1)
    engine._notify_idle_state_callbacks.assert_not_called()
    assert engine.scheduler._omni_maintenance_due


def test_client_control_messages_are_drained_with_ready_events():
    engine = _engine()
    engine.scheduler.has_runnable_omni_chunks = lambda: True
    handler = Mock()
    engine._handle_client_request = handler
    engine.input_queue.put(("abort", ["r"]))
    engine.input_queue.put((_OMNI_CHUNK_READY, None))
    engine._process_input_queue()
    assert [c.args for c in handler.call_args_list] == [("abort", ["r"]), (_OMNI_CHUNK_READY, None)]


def test_in_flight_batch_prevents_sleep():
    engine = _engine()
    assert not engine.has_work()
    engine.batch_queue.append(object())
    assert engine.has_work()


def test_graceful_drain_keeps_parked_streams_alive():
    engine = _engine()
    scheduler = engine.scheduler
    scheduler._omni_chunk_wakeup_bound = True
    scheduler.has_unfinished_requests = lambda: True
    scheduler.has_requests = lambda: (
        scheduler.has_runnable_omni_chunks()
        if scheduler._omni_chunk_wakeup_bound
        else scheduler.has_unfinished_requests()
    )
    assert not engine.has_work()
    engine.shutdown_state = EngineShutdownState.SHUTTING_DOWN
    assert engine.has_work()
    assert not scheduler._omni_chunk_wakeup_bound


@pytest.mark.parametrize("state", [PauseState.PAUSED_ALL, PauseState.PAUSED_NEW])
def test_pause_keeps_upstream_scheduler_contract(state):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._omni_chunk_wakeup_bound = True
    scheduler._pause_state = state
    with patch.object(Scheduler, "has_requests", return_value=False) as original:
        assert not scheduler.has_requests()
        original.assert_called_once()


@pytest.mark.parametrize("maintenance", [False, True])
def test_async_queue_drains_empty_work_without_refilling_it(maintenance):
    engine = _engine()
    scheduler = engine.scheduler
    scheduler._omni_chunk_wakeup_bound = True
    scheduler._omni_maintenance_due = maintenance
    scheduler.has_requests = lambda: OmniGenerationScheduler.has_requests(scheduler)
    output = SimpleNamespace(total_num_scheduled_tokens=0)
    future: Future[object] = Future()
    future.set_result(object())
    engine.batch_queue = deque([] if maintenance else [(future, output, future)])
    engine.batch_queue_size = 2
    engine.is_ec_consumer = False
    engine.is_pooling_model = True
    engine._should_throttle_prefills = lambda: False
    engine.log_error_detail = lambda _output: nullcontext()
    engine.capture_iteration_details = lambda _output: nullcontext()
    engine._process_aborts_queue = lambda: None
    engine._attach_iteration_details = lambda *_: None
    engine.model_executor = SimpleNamespace(execute_model=Mock(return_value=future))
    scheduler.update_from_output = lambda *_: {}

    def schedule(*_):
        scheduler._omni_maintenance_due = False
        return output

    scheduler.schedule = schedule
    engine.step_with_batch_queue()
    assert not engine.batch_queue
    assert engine.model_executor.execute_model.call_count == int(maintenance)
    assert not engine.has_work()


@pytest.mark.parametrize("status", [RequestStatus.WAITING_FOR_CHUNK, RequestStatus.WAITING_FOR_STREAMING_REQ])
def test_live_parked_request_needs_ready_or_cleanup(status):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._init_omni_connector_output_inbox()
    scheduler._latest_omni_connector_output = None
    scheduler.has_finished_requests = lambda: False
    scheduler._pending_finish_reqs = []
    scheduler._pending_data_plane_terminal_req_ids = set()
    request = SimpleNamespace(status=status, num_in_flight_tokens=0, is_finished=lambda: False)
    scheduler.requests = {"r": request}
    scheduler.waiting = [request]
    scheduler.skipped_waiting = []
    scheduler.running = []
    assert not scheduler.has_runnable_omni_chunks()
    scheduler._latest_omni_connector_output = OmniConnectorOutput()
    assert not scheduler.has_runnable_omni_chunks()
    scheduler.input_coordinator = SimpleNamespace(requests_with_ready_chunks={"r"})
    assert scheduler.has_runnable_omni_chunks()
    scheduler.input_coordinator.requests_with_ready_chunks.clear()
    scheduler._omni_connector_output_inbox.put(object())
    assert scheduler.has_runnable_omni_chunks()
    scheduler._omni_connector_output_inbox.get_nowait()
    scheduler._pending_data_plane_terminal_req_ids.add("r")
    assert scheduler.has_runnable_omni_chunks()
    scheduler._pending_data_plane_terminal_req_ids.clear()
    request.status = RequestStatus.WAITING
    assert scheduler.has_runnable_omni_chunks()
    scheduler.requests.clear()
    scheduler._omni_connector_output_inbox.put(object())
    assert not scheduler.has_runnable_omni_chunks()
