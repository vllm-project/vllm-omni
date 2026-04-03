# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, InvalidStateError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput, DiffusionRequestAbortedError
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, _AbortCmd
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    DiffusionRequestStatus,
    RequestScheduler,
    SchedulerInterface,
    StepScheduler,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, NewRequestData
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(req_id: str, *, request_ids: list[str] | None = None) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=request_ids or [req_id],
    )


def _make_request_output(req_id: str, *, error: str | None = None, finished: bool = True):
    return SimpleNamespace(
        req_id=req_id,
        step_index=None,
        finished=finished,
        result=DiffusionOutput(output=None, error=error),
    )


def _make_step_output(
    req_id: str,
    step_index: int,
    *,
    finished: bool = False,
    error: str | None = None,
):
    return SimpleNamespace(
        req_id=req_id,
        step_index=step_index,
        finished=finished,
        result=DiffusionOutput(output=None, error=error) if error is not None else None,
    )


def _make_step_request(
    req_id: str,
    *,
    num_inference_steps: int = 4,
    step_index: int | None = None,
    sampling_params: OmniDiffusionSamplingParams | None = None,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=sampling_params
        or OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            step_index=step_index,
        ),
        request_ids=[req_id],
    )


def _new_ids(sched_output) -> list[str]:
    return [req.sched_req_id for req in sched_output.scheduled_new_reqs]


def _cached_ids(sched_output) -> list[str]:
    return list(sched_output.scheduled_cached_reqs.sched_req_ids)


def _make_core_loop_engine(
    *,
    scheduler: SchedulerInterface | None = None,
    execute_fn: Mock | None = None,
    executor: Mock | None = None,
) -> DiffusionEngine:
    """Create a lightweight DiffusionEngine backed by the queue core loop.

    Args:
        scheduler: Optional scheduler instance to install on the engine.
        execute_fn: Optional execution callback used by the owner thread.
        executor: Optional executor mock used for RPC and shutdown assertions.

    Returns:
        A minimally initialized ``DiffusionEngine`` whose core loop is already
        running and ready to consume commands.
    """
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = Mock(model_class_name="mock_model", enable_cpu_offload=False)
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.step_execution = isinstance(scheduler, StepScheduler)
    engine.scheduler = scheduler or RequestScheduler()
    if scheduler is None:
        engine.scheduler.initialize(Mock())
    engine.executor = executor or Mock(collective_rpc=Mock(), shutdown=Mock())
    engine.execute_fn = execute_fn or Mock(return_value=_make_request_output("default"))
    engine._start_core_thread()
    engine._wait_for_core_ready()
    return engine


class _StubScheduler(SchedulerInterface):
    def __init__(self, request: OmniDiffusionRequest, output) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._sched_req_id = request.request_ids[0]
        self._state = None
        self._scheduled = False

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = Mock(
            sched_req_id=self._sched_req_id,
            req=request,
            status=DiffusionRequestStatus.RUNNING,
            is_finished=Mock(return_value=False),
        )
        return self._sched_req_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                scheduled_req_ids=[],
                finished_req_ids=set(),
                is_empty=True,
            )
        self._scheduled = True
        return SimpleNamespace(
            scheduled_new_reqs=[NewRequestData.from_state(self._state)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            scheduled_req_ids=[self._state.sched_req_id],
            finished_req_ids=set(),
            is_empty=False,
        )

    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output
        assert output is self._output
        self._state.status = DiffusionRequestStatus.FINISHED_COMPLETED
        self._state.is_finished.return_value = True
        return {self._sched_req_id}

    def has_requests(self) -> bool:
        return not self._scheduled

    def get_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def get_sched_req_id(self, request_id: str) -> str | None:
        if request_id in self._request.request_ids:
            return self._sched_req_id
        return None

    def pop_request_state(self, sched_req_id: str):
        del sched_req_id
        state = self._state
        self._state = None
        return state

    def preempt_request(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def finish_requests(self, sched_req_ids, status) -> None:
        del sched_req_ids
        if self._state is not None:
            self._state.status = status
            self._state.is_finished.return_value = DiffusionRequestStatus.is_finished(status)

    def close(self) -> None:
        return None


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(Mock())

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        assert _cached_ids(sched_output) == []
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output(req_id, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, _make_request_output(req_id_a))

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        first = self.scheduler.schedule()
        assert first.finished_req_ids == {req_id_b}
        assert _new_ids(first) == [req_id_a]

        second = self.scheduler.schedule()
        assert second.finished_req_ids == set()

        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().scheduled_req_ids == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_request_id_mapping_lifecycle(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_map_a", "prompt_map_b"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["map-a", "map-b"],
        )

        sched_req_id = self.scheduler.add_request(request)

        assert self.scheduler.get_sched_req_id("map-a") == sched_req_id
        assert self.scheduler.get_sched_req_id("map-b") == sched_req_id

        self.scheduler.pop_request_state(sched_req_id)

        assert self.scheduler.get_sched_req_id("map-a") is None
        assert self.scheduler.get_sched_req_id("map-b") is None


class TestDiffusionEngine:
    def test_add_req_and_wait_for_response_single_path(self) -> None:
        request = _make_request("engine")
        runner_output = _make_request_output("engine")
        engine = _make_core_loop_engine(execute_fn=Mock(return_value=runner_output))
        try:
            output = engine.add_req_and_wait_for_response(request)
            assert output is runner_output.result
            engine.execute_fn.assert_called_once()
        finally:
            engine.close()

    def test_supports_scheduler_interface_injection(self) -> None:
        request = _make_request("engine_iface")
        runner_output = _make_request_output("engine_iface")
        scheduler = _StubScheduler(request, runner_output)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(model_class_name="mock_model", enable_cpu_offload=False)
        engine.pre_process_func = None
        engine.post_process_func = None
        engine.step_execution = False
        engine.scheduler = scheduler
        engine.executor = Mock(collective_rpc=Mock(), shutdown=Mock())
        engine.execute_fn = Mock(return_value=runner_output)
        engine._start_core_thread()
        engine._wait_for_core_ready()

        try:
            output = engine.add_req_and_wait_for_response(request)
            assert output is runner_output.result
            engine.execute_fn.assert_called_once()
        finally:
            engine.close()

    def test_initializes_injected_scheduler(self) -> None:
        request = _make_request("init")
        scheduler = _StubScheduler(request, DiffusionOutput(output=None))
        od_config = Mock(model_class_name="mock_model")
        fake_executor_cls = Mock(return_value=Mock())

        with (
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class", return_value=fake_executor_cls),
            patch.object(DiffusionEngine, "_dummy_run", return_value=None),
        ):
            engine = DiffusionEngine(od_config, scheduler=scheduler)

        assert scheduler.initialized_with is od_config
        fake_executor_cls.assert_called_once_with(od_config)
        engine.close()

    def test_step_raises_aborted_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = Mock(
            return_value=DiffusionOutput(aborted=True, abort_message="Request req-abort aborted.")
        )

        with pytest.raises(DiffusionRequestAbortedError, match="Request req-abort aborted"):
            engine.step(_make_request("req-abort"))

    def test_abort_waiting_request_completes_pending_future(self) -> None:
        """Abort a waiting request and resolve its future immediately.

        Waiting requests never reached the executor, so the core loop should be
        able to finalize scheduler state and complete the reply future directly
        from the abort command handler.
        """
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine._pending_futures = {}

        req_id = engine.scheduler.add_request(_make_request("req-abort"))
        future: Future[DiffusionOutput] = Future()
        engine._pending_futures[req_id] = future

        engine._handle_command(_AbortCmd(request_ids=["req-abort"]))

        assert engine.scheduler.get_request_state(req_id) is None
        output = future.result(timeout=1)
        assert output.aborted is True
        assert output.abort_message == "Request req-abort aborted."

    def test_batch_child_abort_aborts_entire_scheduler_request(self) -> None:
        """Aborting any child request id in a batch should abort the whole sched req."""
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine._pending_futures = {}

        req = _make_request("batch", request_ids=["batch-0", "batch-1"])
        sched_req_id = engine.scheduler.add_request(req)
        future: Future[DiffusionOutput] = Future()
        engine._pending_futures[sched_req_id] = future

        engine._handle_command(_AbortCmd(request_ids=["batch-1"]))

        output = future.result(timeout=1)
        assert output.aborted is True
        assert output.abort_message == "Request batch-0 aborted."
        assert engine.scheduler.get_request_state(sched_req_id) is None

    def test_double_abort_is_idempotent(self) -> None:
        """Submitting the same abort twice should not raise or leak futures."""
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine._pending_futures = {}

        req_id = engine.scheduler.add_request(_make_request("dup-abort"))
        future: Future[DiffusionOutput] = Future()
        engine._pending_futures[req_id] = future

        engine._handle_command(_AbortCmd(request_ids=["dup-abort"]))
        engine._handle_command(_AbortCmd(request_ids=["dup-abort"]))

        output = future.result(timeout=1)
        assert output.aborted is True
        assert engine._pending_futures == {}

    def test_finalize_finished_request_returns_aborted_output(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())

        req_id = engine.scheduler.add_request(_make_request("req-finalize"))
        engine.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        output = engine._finalize_finished_request(req_id)

        assert output.aborted is True
        assert output.abort_message == "Request req-finalize aborted."

    def test_resolve_finished_request_fails_future_if_finalize_raises(self) -> None:
        """Fail the caller future before re-raising a finalize-time crash.

        A custom scheduler implementation could theoretically lose request
        state between ``update_from_output()`` and ``_finalize_finished_request``.
        The core loop should still wake the blocked caller future before the
        exception escapes and crashes the owner thread.
        """
        engine = DiffusionEngine.__new__(DiffusionEngine)
        future: Future[DiffusionOutput] = Future()
        engine._pending_futures = {"req-finalize-error": future}
        engine._finalize_finished_request = Mock(side_effect=RuntimeError("state missing"))

        with pytest.raises(RuntimeError, match="state missing"):
            engine._resolve_finished_request("req-finalize-error", runner_output=None)

        with pytest.raises(RuntimeError, match="Failed to finalize diffusion request req-finalize-error"):
            future.result(timeout=1)
        assert engine._pending_futures == {}

    def test_fail_pending_futures_ignores_invalid_state_race(self) -> None:
        """Ignore InvalidStateError if another thread wins the completion race."""
        engine = DiffusionEngine.__new__(DiffusionEngine)
        racing_future = Mock(spec=Future)
        racing_future.done.return_value = False
        racing_future.set_exception.side_effect = InvalidStateError("already finished")
        engine._pending_futures = {"req-race": racing_future}

        engine._fail_pending_futures(RuntimeError("closed"))

        racing_future.set_exception.assert_called_once()
        assert engine._pending_futures == {}

    def test_initializes_step_scheduler_when_step_execution_enabled(self) -> None:
        od_config = Mock(model_class_name="mock_model")
        od_config.step_execution = True
        fake_executor = Mock()
        fake_executor_cls = Mock(return_value=fake_executor)

        with (
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class", return_value=fake_executor_cls),
            patch.object(DiffusionEngine, "_dummy_run", return_value=None),
        ):
            engine = DiffusionEngine(od_config)

        assert isinstance(engine.scheduler, StepScheduler)
        assert engine.execute_fn is fake_executor.execute_step
        fake_executor_cls.assert_called_once_with(od_config)
        engine.close()

    def test_dummy_run_raises_on_output_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(model_class_name="mock_model")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()

    def test_wait_for_core_ready_raises_on_timeout(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._core_ready = threading.Event()
        engine.CORE_READY_TIMEOUT_S = 0.01

        with pytest.raises(RuntimeError, match="did not become ready"):
            engine._wait_for_core_ready()

    def test_submit_request_fails_if_core_loop_is_already_gone(self) -> None:
        """Reject a request queued after the owner thread has already exited."""
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._cmd_queue = queue.Queue()
        engine._shutdown_requested = threading.Event()
        engine._core_loop_error = None
        engine._core_thread = threading.Thread()

        future = engine.submit_request(_make_request("late"))

        with pytest.raises(RuntimeError, match="core loop is not running"):
            future.result(timeout=1)
        assert engine._cmd_queue.empty()

    def test_submit_request_fails_if_shutdown_starts_after_precheck(self) -> None:
        """Reject a request if shutdown wins the race immediately after enqueue.

        ``submit_request()`` first checks the shutdown flag, then enqueues the
        command, then performs a post-enqueue owner-health check. This test
        simulates the narrow race where the first shutdown check still sees the
        engine as open, but shutdown begins before the helper re-check runs.
        The just-enqueued request must still fail promptly instead of waiting
        forever in an unowned queue.
        """
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._cmd_queue = queue.Queue()
        engine._core_loop_error = None
        engine._core_thread = Mock()
        engine._core_thread.is_alive.return_value = True

        shutdown_check_count = 0

        def is_shutdown_requested() -> bool:
            nonlocal shutdown_check_count
            shutdown_check_count += 1
            return shutdown_check_count > 1

        engine._shutdown_requested = Mock()
        engine._shutdown_requested.is_set.side_effect = is_shutdown_requested

        future = engine.submit_request(_make_request("late-shutdown"))

        with pytest.raises(RuntimeError, match="DiffusionEngine is closed"):
            future.result(timeout=1)
        assert engine._cmd_queue.empty()

    def test_core_loop_crash_fails_pending_futures_and_latches_error(self) -> None:
        """Crash the core loop and verify all callers are woken with the latched error.

        The first submitted request triggers a scheduler failure on the owner
        thread. Its future must fail promptly, and later submissions should
        observe the same latched crash reason instead of hanging forever.
        """
        scheduler = RequestScheduler()
        scheduler.initialize(Mock())
        scheduler.schedule = Mock(side_effect=RuntimeError("scheduler boom"))

        engine = _make_core_loop_engine(scheduler=scheduler, execute_fn=Mock())
        try:
            future = engine.submit_request(_make_request("crash"))
            with pytest.raises(RuntimeError, match="core loop exited unexpectedly: scheduler boom"):
                future.result(timeout=1)

            follow_up = engine.submit_request(_make_request("after-crash"))
            with pytest.raises(RuntimeError, match="core loop exited unexpectedly: scheduler boom"):
                follow_up.result(timeout=1)
        finally:
            engine.close()

    def test_close_wakes_pending_request_future_when_owner_thread_is_stuck(self) -> None:
        """``close()`` should wake blocked callers even if the core thread cannot unwind.

        ``execute_fn()`` blocks longer than the shortened join timeout, so
        ``close()`` must use its last-resort cleanup path to fail the pending
        request future from the caller thread.
        """
        started = threading.Event()
        release = threading.Event()

        def execute_fn(sched_output):
            del sched_output
            started.set()
            release.wait(timeout=5)
            return _make_request_output("close-me")

        engine = _make_core_loop_engine(execute_fn=execute_fn)
        engine.CORE_THREAD_JOIN_TIMEOUT_S = 0.05
        try:
            future = engine.submit_request(_make_request("close-me"))
            assert started.wait(timeout=1)

            engine.close()

            with pytest.raises(RuntimeError, match="DiffusionEngine is closed"):
                future.result(timeout=1)
        finally:
            release.set()


class TestStepScheduler:
    def setup_method(self) -> None:
        self.scheduler: StepScheduler = StepScheduler()
        self.scheduler.initialize(Mock())

    def test_single_request_step_lifecycle(self) -> None:
        request = _make_step_request("step", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1))
        assert finished == set()
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.RUNNING
        assert request.sampling_params.step_index == 1
        assert self.scheduler.has_requests() is True

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(second, _make_step_output(req_id, step_index=2))
        assert finished == set()
        assert request.sampling_params.step_index == 2

        third = self.scheduler.schedule()
        assert _new_ids(third) == []
        assert _cached_ids(third) == [req_id]

        finished = self.scheduler.update_from_output(
            third,
            _make_step_output(req_id, step_index=3, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert request.sampling_params.step_index == 3
        assert self.scheduler.has_requests() is False

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(first, _make_step_output(req_id_a, step_index=1))
        assert finished == set()

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        finished = self.scheduler.update_from_output(
            second,
            _make_step_output(req_id_a, step_index=2, finished=True),
        )
        assert finished == {req_id_a}

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("err", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1, finished=True, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"
        assert self.scheduler.has_requests() is False

    def test_missing_step_index_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("missing", num_inference_steps=3))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            SimpleNamespace(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=None,
            ),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "Missing step_index in RunnerOutput"

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_step_request("a", num_inference_steps=2))
        req_id_b = self.scheduler.add_request(_make_step_request("b", num_inference_steps=2))

        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_b).status == DiffusionRequestStatus.FINISHED_ABORTED

        running = self.scheduler.schedule()
        assert _new_ids(running) == [req_id_a]

        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        assert self.scheduler.get_request_state(req_id_a).status == DiffusionRequestStatus.FINISHED_ABORTED
        assert self.scheduler.has_requests() is False

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_step_request("has", num_inference_steps=2))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=2, finished=True),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_scheduled_request_aborted_before_update_is_returned_finished(self) -> None:
        req_id = self.scheduler.add_request(_make_step_request("abort-late", num_inference_steps=2))

        sched_output = self.scheduler.schedule()
        self.scheduler.finish_requests(req_id, DiffusionRequestStatus.FINISHED_ABORTED)

        finished = self.scheduler.update_from_output(
            sched_output,
            _make_step_output(req_id, step_index=1),
        )
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_preempt_request_preserves_step_index(self) -> None:
        request = _make_step_request("preempt", num_inference_steps=3)
        req_id = self.scheduler.add_request(request)

        first = self.scheduler.schedule()
        assert self.scheduler.update_from_output(first, _make_step_output(req_id, step_index=1)) == set()
        assert request.sampling_params.step_index == 1

        second = self.scheduler.schedule()
        assert _cached_ids(second) == [req_id]
        assert self.scheduler.preempt_request(req_id) is True
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.PREEMPTED
        assert request.sampling_params.step_index == 1

        third = self.scheduler.schedule()
        assert _cached_ids(third) == [req_id]
        assert request.sampling_params.step_index == 1

    @pytest.mark.parametrize(
        ("sampling_params", "expected_steps"),
        [
            (
                OmniDiffusionSamplingParams(
                    timesteps=torch.tensor([1.0, 0.5, 0.0]),
                    sigmas=[1.0, 0.5, 0.25, 0.0],
                    num_inference_steps=5,
                ),
                3,
            ),
            (
                OmniDiffusionSamplingParams(
                    sigmas=[1.0, 0.5],
                    num_inference_steps=5,
                ),
                2,
            ),
            (
                OmniDiffusionSamplingParams(
                    num_inference_steps=4,
                ),
                4,
            ),
        ],
    )
    def test_total_steps_priority(self, sampling_params: OmniDiffusionSamplingParams, expected_steps: int) -> None:
        request = _make_step_request("priority", sampling_params=sampling_params)
        req_id = self.scheduler.add_request(request)

        for _ in range(expected_steps - 1):
            sched_output = self.scheduler.schedule()
            assert sched_output.scheduled_req_ids == [req_id]
            next_step = request.sampling_params.step_index + 1
            assert (
                self.scheduler.update_from_output(
                    sched_output,
                    _make_step_output(req_id, step_index=next_step),
                )
                == set()
            )

        final_output = self.scheduler.schedule()
        assert final_output.scheduled_req_ids == [req_id]
        assert self.scheduler.update_from_output(
            final_output,
            _make_step_output(req_id, step_index=expected_steps, finished=True),
        ) == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    @pytest.mark.parametrize(
        "sampling_params",
        [
            OmniDiffusionSamplingParams(num_inference_steps=0),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=3),
            OmniDiffusionSamplingParams(num_inference_steps=3, step_index=-1),
        ],
    )
    def test_rejects_invalid_initial_step_state(self, sampling_params: OmniDiffusionSamplingParams) -> None:
        request = _make_step_request("invalid", sampling_params=sampling_params)

        with pytest.raises(ValueError):
            self.scheduler.add_request(request)
