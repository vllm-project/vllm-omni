# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _make_request(req_id: str, *, num_inference_steps: int = 1) -> OmniDiffusionRequest:
    """Create a small diffusion request used by concurrency-focused tests.

    Args:
        req_id: Public request id that also makes assertions easier to read.
        num_inference_steps: Number of denoising steps for step-scheduler
            scenarios.
    """
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_inference_steps),
        request_ids=[req_id],
    )


def _make_engine(
    *,
    scheduler=None,
    execute_fn=None,
    executor=None,
) -> DiffusionEngine:
    """Create a minimal DiffusionEngine instance with the core loop running.

    Args:
        scheduler: Optional scheduler implementation to install.
        execute_fn: Optional execution callback invoked by the core thread.
        executor: Optional executor mock for RPC assertions.

    Returns:
        A test-only engine whose owner thread is already started.
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
    engine.execute_fn = execute_fn or Mock(
        return_value=RunnerOutput(
            req_id="default",
            step_index=None,
            finished=True,
            result=DiffusionOutput(output=None),
        )
    )
    engine._start_core_thread()
    engine._wait_for_core_ready()
    return engine


class TestCoreLoopRequestExecution:
    """Concurrency coverage for queue-based request execution."""

    def test_request_submission_is_non_blocking_while_another_request_executes(self) -> None:
        """Verify new requests can enqueue while another request is executing.

        Request A blocks inside ``execute_fn()`` to simulate a long-running
        worker step. While A is still running, request B is submitted directly
        through ``submit_request()``. B should stay pending until A completes,
        proving that submission itself no longer requires the caller to own the
        schedule/execute loop.
        """
        started = threading.Event()
        release = threading.Event()
        results: dict[str, DiffusionOutput] = {}
        execute_threads: list[str] = []

        def execute_fn(sched_output):
            req_id = sched_output.scheduled_req_ids[0]
            execute_threads.append(threading.current_thread().name)
            if req_id == "A":
                started.set()
                release.wait(timeout=5)
            return RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=DiffusionOutput(error=f"result_for_{req_id}"),
            )

        engine = _make_engine(execute_fn=execute_fn)
        try:
            thread_a = threading.Thread(
                target=lambda: results.setdefault("A", engine.add_req_and_wait_for_response(_make_request("A"))),
                daemon=True,
            )
            thread_a.start()
            assert started.wait(timeout=1)

            future_b = engine.submit_request(_make_request("B"))
            time.sleep(0.05)
            assert future_b.done() is False

            release.set()
            results["B"] = future_b.result(timeout=1)
            thread_a.join(timeout=1)

            assert results["A"].error == "result_for_A"
            assert results["B"].error == "result_for_B"
            assert execute_threads == ["DiffusionEngineCore", "DiffusionEngineCore"]
        finally:
            release.set()
            engine.close()

    def test_running_request_abort_is_processed_before_update_from_output(self) -> None:
        """Abort a running request and verify the second drain observes it."""
        started = threading.Event()
        release = threading.Event()
        result_holder: dict[str, DiffusionOutput] = {}

        def execute_fn(sched_output):
            req_id = sched_output.scheduled_req_ids[0]
            started.set()
            release.wait(timeout=5)
            return RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=DiffusionOutput(output=None),
            )

        engine = _make_engine(execute_fn=execute_fn)
        try:
            worker = threading.Thread(
                target=lambda: result_holder.setdefault(
                    "output",
                    engine.add_req_and_wait_for_response(_make_request("run-abort")),
                ),
                daemon=True,
            )
            worker.start()
            assert started.wait(timeout=1)

            engine.abort("run-abort")
            release.set()
            worker.join(timeout=1)

            assert result_holder["output"].aborted is True
            assert result_holder["output"].abort_message == "Request run-abort aborted."
        finally:
            release.set()
            engine.close()


class TestCoreLoopStepExecution:
    """Coverage for step-scheduler behavior under the core loop."""

    def test_step_scheduler_does_not_resolve_future_on_intermediate_step(self) -> None:
        """Ensure intermediate step outputs do not resolve the caller future."""
        scheduler = StepScheduler()
        scheduler.initialize(Mock())

        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()
        call_count = 0

        def execute_fn(sched_output):
            nonlocal call_count
            call_count += 1
            req_id = sched_output.scheduled_req_ids[0]
            if call_count == 1:
                first_started.set()
                release_first.wait(timeout=5)
                return RunnerOutput(
                    req_id=req_id,
                    step_index=1,
                    finished=False,
                    result=DiffusionOutput(output=None),
                )

            second_started.set()
            release_second.wait(timeout=5)
            return RunnerOutput(
                req_id=req_id,
                step_index=2,
                finished=True,
                result=DiffusionOutput(output=None),
            )

        engine = _make_engine(scheduler=scheduler, execute_fn=execute_fn)
        try:
            future = engine.submit_request(_make_request("step", num_inference_steps=2))
            assert first_started.wait(timeout=1)
            release_first.set()
            assert second_started.wait(timeout=1)
            assert future.done() is False

            release_second.set()
            output = future.result(timeout=1)
            assert isinstance(output, DiffusionOutput)
            assert output.error is None
            assert call_count == 2
        finally:
            release_first.set()
            release_second.set()
            engine.close()


class TestCoreLoopRpcCoordination:
    """Coverage for queue-based RPC timeout and cancellation semantics."""

    def test_collective_rpc_timeout_cancels_queued_rpc_before_execution(self) -> None:
        """Verify a timed-out queued RPC is skipped by the core loop.

        A long-running request keeps the core thread busy. A concurrent
        ``collective_rpc(timeout=...)`` call times out while still waiting in
        the queue, cancels its future, and must therefore never reach
        ``executor.collective_rpc()``.
        """
        started = threading.Event()
        release = threading.Event()
        executor = Mock(collective_rpc=Mock(return_value="rpc-result"), shutdown=Mock())

        def execute_fn(sched_output):
            req_id = sched_output.scheduled_req_ids[0]
            started.set()
            release.wait(timeout=5)
            return RunnerOutput(
                req_id=req_id,
                step_index=None,
                finished=True,
                result=DiffusionOutput(output=None),
            )

        engine = _make_engine(execute_fn=execute_fn, executor=executor)
        try:
            worker = threading.Thread(
                target=lambda: engine.add_req_and_wait_for_response(_make_request("stall")),
                daemon=True,
            )
            worker.start()
            assert started.wait(timeout=1)

            with pytest.raises(TimeoutError, match="RPC call to health timed out"):
                engine.collective_rpc("health", timeout=0.1)

            release.set()
            worker.join(timeout=1)
            time.sleep(0.05)

            executor.collective_rpc.assert_not_called()
        finally:
            release.set()
            engine.close()
