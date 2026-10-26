# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``stage_init_timeout`` must bound stage readiness and stop further launches."""

from __future__ import annotations

import multiprocessing
import threading
import time
import types

import pytest
import zmq
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.v1.engine.utils import CoreEngine, CoreEngineLaunch

from vllm_omni.engine import stage_init_deadline
from vllm_omni.engine.stage_init_deadline import wait_for_engine_startup_with_deadline
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _NeverReadyManager:
    """Stands in for a proc manager whose single engine process never reports READY."""

    def __init__(self, proc: multiprocessing.process.BaseProcess) -> None:
        self.processes = [proc]
        self.kill_calls = 0

    def sentinels(self) -> list[int]:
        return [self.processes[0].sentinel]

    def kill(self) -> None:
        self.kill_calls += 1
        for proc in self.processes:
            proc.kill()
            proc.join(timeout=5)


_PARALLEL_CONFIG = types.SimpleNamespace(
    data_parallel_size_local=1,
    data_parallel_hybrid_lb=False,
    data_parallel_external_lb=False,
)


def test_deadline_terminates_stage_and_raises_timeout():
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=time.sleep, args=(120,), name="FakeStageEngine")
    proc.start()
    manager = _NeverReadyManager(proc)
    try:
        handshake_address = get_open_zmq_ipc_path()
        with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
            # Upstream registers exit sentinels only for CoreEngineProcManager
            # instances; other managers (like StageDiffusionProcManager) watch
            # the process explicitly, which is what this fake does too.
            launch = CoreEngineLaunch(engine_manager=manager, coordinator=None, addresses=None, tensor_queue=None)
            launch.watched_frontend_processes = [proc]
            started = time.monotonic()
            with pytest.raises(TimeoutError, match="stage_init_timeout"):
                wait_for_engine_startup_with_deadline(
                    handshake_socket,
                    [CoreEngine(index=0, local=True)],
                    _PARALLEL_CONFIG,
                    False,
                    None,
                    launch,
                    timeout=1,
                    kill_processes=manager.kill,
                    stage_label="fake stage",
                )
            elapsed = time.monotonic() - started
        assert elapsed < 15, f"deadline did not bound the wait: {elapsed:.1f}s"
        assert manager.kill_calls == 1
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)


def test_no_timeout_when_ready_before_deadline(monkeypatch):
    kill_calls: list[int] = []
    monkeypatch.setattr(stage_init_deadline, "wait_for_engine_startup", lambda *args: None)
    wait_for_engine_startup_with_deadline(
        object(),
        [],
        _PARALLEL_CONFIG,
        False,
        None,
        None,
        timeout=60,
        kill_processes=lambda: kill_calls.append(1),
        stage_label="fake stage",
    )
    assert kill_calls == []


def test_disabled_deadline_passes_through(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(stage_init_deadline, "wait_for_engine_startup", lambda *args: seen.append(args))
    wait_for_engine_startup_with_deadline(
        "sock",
        "engines",
        _PARALLEL_CONFIG,
        False,
        None,
        "launch",
        timeout=0,
        kill_processes=lambda: None,
        stage_label="x",
    )
    assert seen == [("sock", "engines", _PARALLEL_CONFIG, False, None, "launch")]


def test_cancel_initialization_skips_remaining_replicas(monkeypatch):
    stage_cfg = types.SimpleNamespace(
        stage_id=0, stage_type="llm", runtime=types.SimpleNamespace(devices="0"), engine_args={}
    )
    runtime = StageRuntime(
        stage_configs=[stage_cfg, stage_cfg],
        model="fake-model",
        config_path="/fake/stages.yaml",
        stage_init_timeout=60,
        async_chunk=False,
    )
    plans = [
        types.SimpleNamespace(stage_idx=0, replicas=[types.SimpleNamespace(replica_id=0)]),
        types.SimpleNamespace(stage_idx=1, replicas=[types.SimpleNamespace(replica_id=0)]),
    ]
    group = [(0, plans[0].replicas[0]), (1, plans[1].replicas[0])]
    monkeypatch.setattr(runtime, "_build_init_groups", lambda _plans: {"inline:0": list(group)})
    launched: list[int] = []

    def _fake_initialize_replica(replica, stage_init_timeout):
        launched.append(id(replica))
        # Simulate an engine-level shutdown racing the stage initialization.
        runtime.cancel_initialization()
        return "client"

    monkeypatch.setattr(runtime, "_initialize_replica", _fake_initialize_replica)

    with pytest.raises(RuntimeError, match="cancelled") as exc_info:
        runtime._initialize_stage_replicas(plans, 60)

    assert len(launched) == 1, "the second stage must not be launched after cancellation"
    # Already-launched replicas are reported so initialize() can shut them down.
    clients = exc_info.value._initialized_clients_by_stage
    assert clients[0] == ["client"]
    assert clients[1] == [None]


def test_ready_racing_deadline_waits_for_the_callback(monkeypatch):
    """READY and the deadline fire together.

    ``Timer.cancel()`` cannot stop a callback that already passed the timer's
    cancellation check, so that callback may still terminate the stage after
    the readiness wait returned. The function must wait for it and then refuse
    to hand back the (now killed) stage instead of reporting success.
    """
    kill_calls: list[int] = []

    class _RacingTimer:
        # Stands in for a Timer whose callback has already passed the
        # cancellation check when cancel() runs: it fires a moment later.
        def __init__(self, interval, function):
            self._function = function
            self._thread: threading.Thread | None = None
            self.daemon = False

        def start(self) -> None:
            pass

        def cancel(self) -> None:
            def _late_callback() -> None:
                time.sleep(0.05)
                self._function()

            self._thread = threading.Thread(target=_late_callback)
            self._thread.start()

        def join(self, timeout=None) -> None:
            if self._thread is not None:
                self._thread.join(timeout)

    monkeypatch.setattr(stage_init_deadline.threading, "Timer", _RacingTimer)
    # READY arrives immediately.
    monkeypatch.setattr(stage_init_deadline, "wait_for_engine_startup", lambda *args: None)

    with pytest.raises(TimeoutError, match="stage_init_timeout"):
        wait_for_engine_startup_with_deadline(
            object(),
            [],
            _PARALLEL_CONFIG,
            False,
            None,
            None,
            timeout=60,
            kill_processes=lambda: kill_calls.append(1),
            stage_label="fake stage",
        )
    # The termination had completed before the function returned.
    assert kill_calls == [1]


def test_shutdown_cancels_initialization_before_joining_orchestrator():
    """``OmniEngineBase.shutdown`` must cancel stage initialization before it
    waits for the orchestrator thread, otherwise a stage that becomes READY
    during that wait still launches the next one."""
    from vllm_omni.engine.omni_engine_base import OmniEngineBase

    class _Runtime:
        def __init__(self) -> None:
            self.cancelled = False
            self.shutdown_calls = 0

        def cancel_initialization(self) -> None:
            self.cancelled = True

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    runtime = _Runtime()
    cancelled_at_join: list[bool] = []

    class _OrchestratorThread:
        # Alive until joined, like an orchestrator that is still initializing.
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            cancelled_at_join.append(runtime.cancelled)
            self._alive = False

    engine = types.SimpleNamespace(
        _shutdown_called=False,
        _weak_finalizer=None,
        request_queue=None,
        output_queue=None,
        rpc_output_queue=None,
        _correlated_rpc_client=None,
        _runtime=runtime,
        orchestrator_thread=_OrchestratorThread(),
    )
    engine.is_alive = lambda: engine.orchestrator_thread.is_alive()

    OmniEngineBase.shutdown(engine)

    assert cancelled_at_join == [True], "initialization must be cancelled before the orchestrator join"
    assert runtime.shutdown_calls == 1
