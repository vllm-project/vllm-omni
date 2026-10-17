# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise idle waiting without constructing a model or CUDA executor."""

import multiprocessing
import queue
import signal
import threading
import time

import pytest

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _idle_engine():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.input_queue = queue.Queue()
    engine.aborts_queue = queue.Queue()
    engine.process_input_queue_block = True
    engine.has_work = lambda: False
    engine.is_running = lambda: True
    engine._notify_idle_state_callbacks = lambda: None
    engine._handle_client_request = lambda *_request: None
    return engine


def _process_input_and_signal(engine: StageEngineCoreProc, finished: threading.Event) -> None:
    engine._process_input_queue()
    finished.set()


def _directed_term_worker(connection):
    engine = _idle_engine()
    waiting = threading.Event()
    engine._notify_idle_state_callbacks = waiting.set

    def on_term(_signum, _frame):
        connection.send("term_handled")
        raise SystemExit(0)

    def send_term():
        waiting.wait()
        # Deliver TERM to a different thread while the main thread is waiting
        # in Queue.get. CPython executes its Python handler in the main thread.
        # Confirm the real Condition wait rather than relying on a sleep before
        # delivery: a slow CI worker could otherwise handle TERM before get.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with engine.input_queue.not_empty:
                if engine.input_queue.not_empty._waiters:
                    break
            time.sleep(0.001)
        else:
            connection.send("queue_wait_not_observed")
            return
        signal.pthread_kill(threading.get_ident(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    threading.Thread(target=send_term, daemon=True).start()
    connection.send("ready")
    engine._process_input_queue()
    connection.send("unexpected_idle_return")


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="requires POSIX thread signals")
def test_idle_input_wait_handles_term_delivered_to_another_thread():
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_directed_term_worker, args=(send,))
    process.start()
    send.close()
    try:
        # Imports in a fresh spawned process are outside the shutdown deadline.
        assert receive.poll(30), "worker did not start"
        assert receive.recv() == "ready"
        assert receive.poll(2.5), "main thread did not process pending TERM"
        assert receive.recv() == "term_handled"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        receive.close()


def test_idle_polling_stays_inside_input_wait_until_a_request_arrives():
    engine = _idle_engine()
    original_queue = engine.input_queue
    waiting = threading.Event()
    finished = threading.Event()
    handled: list[tuple[str, object]] = []
    idle_callbacks: list[None] = []

    def notify_idle():
        idle_callbacks.append(None)
        waiting.set()

    engine._notify_idle_state_callbacks = notify_idle
    engine.has_work = lambda: bool(handled)
    engine._handle_client_request = lambda *request: handled.append(request)

    def wait_for_input():
        try:
            engine._process_input_queue()
        finally:
            finished.set()

    thread = threading.Thread(target=wait_for_input, daemon=True)
    thread.start()
    try:
        assert waiting.wait(2)
        # Returning from this method lets vLLM's busy loop run an engine step.
        # A timeout must not return here or create periodic empty-work steps.
        assert not finished.wait(1.25)
        assert len(idle_callbacks) == 1
        original_queue.put(("request", "payload"))
        assert finished.wait(2)
        assert handled == [("request", "payload")]
        assert engine.input_queue is original_queue
    finally:
        original_queue.put(("stop", None))
        thread.join(2)


def test_nonblocking_input_wait_returns_on_empty_queue():
    engine = _idle_engine()
    engine.process_input_queue_block = False
    finished = threading.Event()
    thread = threading.Thread(target=_process_input_and_signal, args=(engine, finished), daemon=True)
    thread.start()
    try:
        assert finished.wait(0.5)
    finally:
        engine.is_running = lambda: False
        engine.input_queue.put(("stop", None))
        thread.join(1)


def test_input_wait_preserves_request_order_and_drains_pending_requests():
    engine = _idle_engine()
    handled: list[tuple[str, object]] = []
    engine.has_work = lambda: bool(handled)
    engine._handle_client_request = lambda *request: handled.append(request)
    engine.input_queue.put(("first", 1))
    engine.input_queue.put(("second", 2))
    engine._process_input_queue()
    assert handled == [("first", 1), ("second", 2)]


def test_active_engine_does_not_wait_for_an_input_request():
    engine = _idle_engine()
    engine.has_work = lambda: True
    finished = threading.Event()
    thread = threading.Thread(target=_process_input_and_signal, args=(engine, finished), daemon=True)
    thread.start()
    try:
        assert finished.wait(0.5), "active engine was delayed waiting for input"
    finally:
        engine.is_running = lambda: False
        engine.input_queue.put(("stop", None))
        thread.join(2)


def test_idle_wait_stops_when_shutdown_state_changes():
    engine = _idle_engine()
    stopping = threading.Event()
    waiting = threading.Event()
    finished = threading.Event()
    engine.is_running = lambda: not stopping.is_set()
    engine._notify_idle_state_callbacks = waiting.set
    thread = threading.Thread(target=_process_input_and_signal, args=(engine, finished), daemon=True)
    thread.start()
    try:
        assert waiting.wait(2)
        stopping.set()
        assert finished.wait(2), "idle wait ignored the shutdown state"
    finally:
        engine.input_queue.put(("stop", None))
        thread.join(2)
