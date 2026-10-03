# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise worker signal handling with real processes and blocked cleanup."""

import multiprocessing as mp
import os
import signal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run_worker(connection, cleanup_started, block_cleanup):
    from vllm_omni.diffusion.worker import diffusion_worker as module

    class FakeWorker:
        result_mq_handle = None
        context = SimpleNamespace(term=lambda: None)

        def __init__(self, *args, **kwargs):
            pass

        def _worker_busy_loop(self):
            connection.recv()

        def shutdown(self):
            cleanup_started.set()
            if block_cleanup:
                # Model a distributed teardown that cannot finish while peers
                # are still executing a collective.
                mp.Event().wait()

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False, use_hsdp=False, hsdp_replicate_size=1)
    )
    worker_main = module.WorkerProc.worker_main
    with (
        patch.object(module, "WorkerProc", FakeWorker),
        patch.object(module, "_setup_diffusion_worker_proc_title_and_log_prefix"),
        patch("vllm_omni.plugins.load_omni_general_plugins"),
    ):
        worker_main(0, config, connection, None, None)


@pytest.mark.parametrize("termination", ["sigterm", "graceful", "sigint"])
def test_worker_exit_does_not_wait_for_distributed_cleanup(termination):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    cleanup_started = ctx.Event()
    proc = ctx.Process(target=_run_worker, args=(child, cleanup_started, termination == "sigterm"))
    proc.start()
    child.close()
    try:
        assert parent.poll(240), "worker did not initialize"
        assert parent.recv()["status"] == "ready"
        if termination == "graceful":
            parent.send("finish current batch and shut down")
        else:
            os.kill(proc.pid, signal.SIGTERM if termination == "sigterm" else signal.SIGINT)
        proc.join(5)
        assert not proc.is_alive(), "worker is stuck in distributed cleanup after SIGTERM"
        if termination == "sigterm":
            assert proc.exitcode in (-signal.SIGTERM, 128 + signal.SIGTERM)
            assert not cleanup_started.is_set()
        else:
            assert cleanup_started.is_set()
            assert proc.exitcode == (0 if termination == "graceful" else 128 + signal.SIGINT)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
        parent.close()
        proc.close()


def _run_stuck_peer(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    mp.Event().wait()


def test_worker_sigterm_triggers_monitor_and_reaps_stuck_peers(monkeypatch):
    from vllm_omni.diffusion.executor import multiproc_executor as module

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    cleanup_started, peer_ready, failed = ctx.Event(), ctx.Event(), ctx.Event()
    worker = ctx.Process(target=_run_worker, args=(child, cleanup_started, True))
    peer = ctx.Process(target=_run_stuck_peer, args=(peer_ready,))
    processes = [worker, peer]
    monkeypatch.setattr(module, "_WORKER_SHUTDOWN_GRACE_S", 0.2)
    monkeypatch.setattr(module, "_WORKER_TERMINATE_GRACE_S", 0.2)
    cleaner = module._ExecutorShutdownCleaner(processes=processes)
    executor = object.__new__(module.MultiprocDiffusionExecutor)
    executor._processes = processes
    executor._closed = False
    executor._is_failed = False
    executor._failure_callbacks = [failed.set]

    def shutdown():
        executor._closed = True
        cleaner()

    executor.shutdown = shutdown
    worker.start()
    peer.start()
    child.close()
    try:
        assert parent.poll(240), "worker did not initialize"
        assert parent.recv()["status"] == "ready"
        assert peer_ready.wait(10)
        executor._start_worker_monitor()
        os.kill(worker.pid, signal.SIGTERM)
        assert failed.wait(10), "worker monitor did not report failure after group cleanup"
        assert executor._is_failed
        assert cleaner.processes == []
        assert worker.exitcode == -signal.SIGTERM
        assert peer.exitcode == -signal.SIGKILL
        assert not cleanup_started.is_set()
    finally:
        executor._closed = True
        for proc in processes:
            if proc.is_alive():
                proc.kill()
            proc.join(5)
        parent.close()
