# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Startup failure handling for the diffusion multiproc executor."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_omni.diffusion.executor.multiproc_executor as executor_module
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _startup_worker(
    rank: int,
    _config: object,
    writer: Connection,
    _broadcast_handle: object,
    wake_event: Event,
    _worker_extension_cls: object,
    custom_pipeline_args: Any,
) -> None:
    sibling_started, worker_pids = custom_pipeline_args
    if rank == 0:
        sibling_started.wait()
        writer.close()
        os._exit(1)

    worker_pids.put(os.getpid())
    sibling_started.set()
    writer.send({"status": "ready", "result_handle": object()})
    writer.close()
    wake_event.wait()


def _run_startup_scenario(parent: Connection, sibling_started: Any, worker_pids: Any) -> None:
    executor_module.WorkerProc = SimpleNamespace(worker_main=_startup_worker)
    executor_module.get_mp_context = lambda: mp.get_context("fork")
    executor_module.set_multiprocessing_worker_envs = lambda: None

    executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        num_gpus=2,
        worker_extension_cls=None,
        custom_pipeline_args=(sibling_started, worker_pids),
    )
    ctx = mp.get_context("fork")

    try:
        executor._launch_workers(None, [ctx.Event(), ctx.Event()])
        parent.send(("returned", None))
    except BaseException as exc:
        parent.send(("failed", f"{type(exc).__name__}: {exc}"))
    finally:
        parent.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-specific startup regression")
def test_early_worker_failure_under_fork_fails_and_cleans_sibling() -> None:
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    sibling_started = ctx.Event()
    worker_pids = ctx.Queue()
    scenario = ctx.Process(
        target=_run_startup_scenario,
        args=(child, sibling_started, worker_pids),
        name="DiffusionStartupScenario",
    )
    scenario.start()
    child.close()
    sibling_pid: int | None = None

    try:
        assert sibling_started.wait(timeout=5.0), "sibling worker never started"
        sibling_pid = worker_pids.get(timeout=5.0)
        assert sibling_pid is not None

        scenario.join(timeout=5.0)
        if scenario.is_alive():
            os.kill(sibling_pid, signal.SIGKILL)
            scenario.join(timeout=5.0)
            pytest.fail("worker readiness hung after an early fork worker died")

        outcome, detail = parent.recv()
        assert outcome == "failed"
        assert detail.startswith("EOFError:")

        with pytest.raises(ProcessLookupError):
            os.kill(sibling_pid, 0)
    finally:
        if scenario.is_alive():
            scenario.terminate()
            scenario.join(timeout=5.0)
        if sibling_pid is not None:
            try:
                os.kill(sibling_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        worker_pids.close()
        worker_pids.cancel_join_thread()
        parent.close()
