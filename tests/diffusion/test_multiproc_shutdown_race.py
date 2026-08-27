# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MultiprocDiffusionExecutor shutdown / delayed-pump race.

Reproduces hsliuustc0106's #6439 review finding: `shutdown()` joins each result
pump thread with a 2s timeout and only warns if it survives. A pump blocked
inside `unpack_diffusion_output_shm` can resume AFTER `_completed_outputs` was
cleared and repopulate the dictionary, reintroducing the leak this PR was meant
to close.

Both single-output and batch-split code paths must discard deliveries once
`_closed` is set; `drop_output()` must be a no-op after shutdown.
"""

import queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.data import (
    AsyncDiffusionOutput,
    AsyncOutputKind,
    DiffusionOutput,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_executor():
    """Minimal MultiprocDiffusionExecutor-like object; mirrors test_result_pump helper."""
    from vllm_omni.diffusion.executor.multiproc_executor import (
        MultiprocDiffusionExecutor,
    )

    od_config = MagicMock()
    od_config.step_execution = False

    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = od_config
    executor._rpc_id_counter = 0
    executor._rpc_id_lock = threading.Lock()
    executor._rpc_futures = {}
    executor._output_futures = {}
    executor._completed_outputs = {}
    executor._batch_split_map = {}
    executor._futures_lock = threading.RLock()
    executor._pump_running = False
    executor._pump_stop = threading.Event()
    executor._sync_result_buffer = queue.Queue()
    executor._result_mq = MagicMock()
    executor._result_mqs = []
    executor._broadcast_mq = MagicMock()
    executor._closed = False
    executor._is_failed = False
    executor._finalizer = MagicMock()
    executor._shutdown_cleaner = None
    executor._processes = []
    return executor


class _FakeBatchOutput:
    """Batch-level output exposing per-request results (mirrors test_result_pump helper)."""

    def __init__(self, results):
        self._results = results

    def get_request_output(self, req_id):
        result = self._results.get(req_id)
        if result is None:
            return None
        return MagicMock(result=result)


def _run_pump_blocked_until_shutdown(executor, msg, unpack_gate):
    """Drive the pump thread with one *msg*.

    The pump enters `unpack_diffusion_output_shm` and blocks on *unpack_gate*.
    Caller is expected to close the executor, then release the gate. Returns
    the pump thread so the test can join it.
    """
    call_count = [0]

    def mock_dequeue(timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return msg
        # After the single msg, block until pump_stop is set so the pump
        # loops out cleanly (mirrors _feed_one_msg_to_pump behaviour).
        while not executor._pump_stop.is_set():
            time.sleep(0.02)
        raise TimeoutError

    executor._result_mq.dequeue = mock_dequeue
    t = threading.Thread(target=executor._result_pump, daemon=True)
    t.start()
    # Give the pump time to dequeue and reach unpack_gate.wait().
    time.sleep(0.1)
    return t


class TestResultPumpDelayedAfterShutdown:
    """Reproduce hsliuustc0106's finding: a pump blocked inside SHM unpack can
    resume after shutdown clears `_completed_outputs` and repopulate it.

    Without gating, the assertions here fail (leak reintroduced through the
    late-write path). With gating on `_closed`, deliveries after shutdown are
    discarded.
    """

    def test_result_pump_delayed_after_shutdown_single(self, mocker):
        """Single-output pump path: late unpack must not repopulate the dict."""
        executor = _make_executor()

        unpack_gate = threading.Event()

        def _blocked_unpack(*_args, **_kwargs):
            unpack_gate.wait(timeout=10.0)

        mocker.patch(
            "vllm_omni.diffusion.executor.multiproc_executor.unpack_diffusion_output_shm",
            side_effect=_blocked_unpack,
        )

        async_output_id = "abc-single"
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=async_output_id,
            output=DiffusionOutput(output="late"),
        )

        pump_thread = _run_pump_blocked_until_shutdown(executor, msg, unpack_gate)

        # Simulate shutdown() clearing state while the pump is still stuck
        # inside unpack. `_closed = True` mirrors shutdown()'s first line.
        executor._closed = True
        with executor._futures_lock:
            executor._completed_outputs.clear()

        # Release the pump: it resumes past unpack and tries to write into
        # _completed_outputs. Without the gate, this reintroduces the leak.
        unpack_gate.set()

        # Give the pump time to finish delivering and hit the gate.
        time.sleep(0.2)
        executor._pump_stop.set()
        pump_thread.join(timeout=2.0)
        assert not pump_thread.is_alive(), "pump thread did not exit"

        # The core assertion: the delayed pump must not have repopulated
        # `_completed_outputs` after shutdown cleared it.
        assert executor._completed_outputs == {}, (
            f"delayed pump repopulated dict after shutdown: "
            f"{list(executor._completed_outputs)}"
        )

    def test_result_pump_delayed_after_shutdown_batch_split(self, mocker):
        """Batch-split pump path: late unpack must not repopulate the dict."""
        executor = _make_executor()

        unpack_gate = threading.Event()

        def _blocked_unpack(*_args, **_kwargs):
            unpack_gate.wait(timeout=10.0)

        mocker.patch(
            "vllm_omni.diffusion.executor.multiproc_executor.unpack_diffusion_output_shm",
            side_effect=_blocked_unpack,
        )

        batch_id = "batch-delayed"
        # Pre-populate batch split map so pump goes down the batch path,
        # WITHOUT registering per-request waiters (they are the ones aborted).
        with executor._futures_lock:
            executor._batch_split_map[batch_id] = {
                f"{batch_id}/r-a": "r-a",
                f"{batch_id}/r-b": "r-b",
            }

        outputs = {
            "r-a": DiffusionOutput(output="late-a"),
            "r-b": DiffusionOutput(output="late-b"),
        }
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=batch_id,
            output=_FakeBatchOutput(outputs),
        )

        pump_thread = _run_pump_blocked_until_shutdown(executor, msg, unpack_gate)

        # Simulate shutdown during unpack.
        executor._closed = True
        with executor._futures_lock:
            executor._completed_outputs.clear()

        # Release the pump so it enters _deliver_batch_split with no waiters
        # registered; without gating, both per-request results land in
        # _completed_outputs.
        unpack_gate.set()

        time.sleep(0.2)
        executor._pump_stop.set()
        pump_thread.join(timeout=2.0)
        assert not pump_thread.is_alive(), "pump thread did not exit"

        assert executor._completed_outputs == {}, (
            f"delayed batch-split pump repopulated dict after shutdown: "
            f"{list(executor._completed_outputs)}"
        )


class TestDropOutputAfterShutdown:
    """`drop_output()` runs from the abort path and must be a no-op once the
    executor has entered shutdown, so it does not race the pump into an
    already-torn-down `_output_futures` / `_completed_outputs`.
    """

    def test_drop_output_after_shutdown_is_noop(self):
        executor = _make_executor()
        executor._closed = True

        # Before the gate, `drop_output` would register a placeholder waiter
        # in `_output_futures`. After shutdown, that would leak past teardown.
        executor.drop_output("abc-abort")

        assert executor._output_futures == {}, (
            f"drop_output registered waiter after shutdown: "
            f"{list(executor._output_futures)}"
        )
        assert executor._completed_outputs == {}
