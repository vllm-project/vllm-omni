# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cancellation transport, ownership and collective agreement regressions."""

import multiprocessing as mp
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import pytest

from vllm_omni.diffusion.cancellation import (
    RequestCancellationRegistry,
    check_request_cancellation,
    request_cancellation_scope,
)
from vllm_omni.diffusion.data import DiffusionRequestAbortedError, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def registry():
    signals = RequestCancellationRegistry()
    yield signals
    signals.close()


def test_cancel_is_request_scoped_and_id_reuse_is_fresh(registry):
    first = registry.create("first")
    other = registry.create("other")
    registry.cancel(["missing", "first"])
    with pytest.raises(ValueError, match="Duplicate cancellation request"):
        registry.create("first")
    with request_cancellation_scope([first]):
        with pytest.raises(DiffusionRequestAbortedError):
            check_request_cancellation()
    with request_cancellation_scope([other]):
        check_request_cancellation()
    registry.finish("first")
    registry.finish("first")
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=first)
    replacement = registry.create("first")
    with request_cancellation_scope([replacement]):
        check_request_cancellation()


def test_scope_restores_outer_state_and_does_not_leak_to_threads(registry):
    signal = registry.create("request")
    registry.cancel_all()
    with request_cancellation_scope([signal]):
        with request_cancellation_scope([None]):
            check_request_cancellation()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(check_request_cancellation).result(timeout=10)
        with pytest.raises(DiffusionRequestAbortedError):
            check_request_cancellation()
    check_request_cancellation()


def test_mixed_batch_keeps_uncancelled_request_running(registry):
    first = registry.create("first")
    other = registry.create("other")
    with request_cancellation_scope([first, other]):
        registry.cancel(["first"])
        check_request_cancellation()
        registry.cancel(["other"])
        with pytest.raises(DiffusionRequestAbortedError):
            check_request_cancellation()


def test_checkpoint_observes_cancel_during_device_wait(registry, monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    signal = registry.create("request")
    # DELETE can arrive while the current GPU step completes. It must be
    # observed before another step is queued, not read before the device wait.
    monkeypatch.setattr(current_omni_platform, "synchronize", lambda: registry.cancel(["request"]))
    with request_cancellation_scope([signal]):
        with pytest.raises(DiffusionRequestAbortedError):
            check_request_cancellation(synchronize=True)


def test_close_removes_all_owned_signals(registry):
    names = [registry.create(str(i)) for i in range(3)]
    registry.cancel_all()
    registry.close()
    registry.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def _admission_engine(registry):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler

    engine = object.__new__(DiffusionEngine)
    engine.scheduler = RequestScheduler()
    engine.scheduler.initialize(OmniDiffusionConfig())
    engine._request_cancellations = registry
    engine._cv = threading.Condition(threading.RLock())
    engine._closed = False
    engine._out_streams = {}
    engine.abort_queue = queue.Queue()
    return engine


def _request(request_id):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        prompt="A fox in the snow.", sampling_params=OmniDiffusionSamplingParams(), request_id=request_id
    )


def test_engine_signals_worker_before_busy_loop_drains_abort(registry):
    from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus

    engine = _admission_engine(registry)
    request = _request("running")
    engine._add_prepared_request(request)
    engine.scheduler.schedule()
    signal = request.cancellation_signal
    assert signal is not None
    with request_cancellation_scope([signal]):
        engine.abort("running")
        # The busy loop is still occupied by execution. Only the signal has
        # changed; scheduler mutation must stay on that loop's thread.
        assert engine.scheduler.get_request_state("running").status == DiffusionRequestStatus.RUNNING
        with pytest.raises(DiffusionRequestAbortedError):
            check_request_cancellation()
    engine._process_aborts_queue()
    assert engine._finalize_finished_request("running").aborted
    assert engine.scheduler.get_request_state("running") is None
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=signal)


def test_rejected_admission_releases_signal(registry, monkeypatch):
    engine = _admission_engine(registry)
    allocated_names = []

    def reject(request):
        allocated_names.append(request.cancellation_signal)
        raise ValueError("admission rejected")

    monkeypatch.setattr(engine.scheduler, "add_request", reject)
    request = _request("rejected")
    with pytest.raises(ValueError, match="admission rejected"):
        engine._add_prepared_request(request)
    assert request.cancellation_signal is None
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=allocated_names[0])
    registry.create("rejected")


def _collective_worker(rank, name, rendezvous, ready, first_check, second_check, results):
    import torch
    import torch.distributed as dist

    from vllm_omni.diffusion.distributed import parallel_state

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    # Exercise a real CPU collective without constructing GPU model groups.
    group = object.__new__(parallel_state.GroupCoordinator)
    group.world_size = 2
    group.cpu_group = dist.group.WORLD
    parallel_state._WORLD = group
    try:
        with torch.inference_mode(), request_cancellation_scope([name]):
            dist.barrier()
            if rank == 0:
                ready.set()
            assert first_check.wait(120)
            check_request_cancellation()
            results.put((rank, "partial_wave_kept_running"))
            assert second_check.wait(120)
            try:
                check_request_cancellation()
            except DiffusionRequestAbortedError:
                results.put((rank, "fully_cancelled_wave_stopped"))
            else:
                raise AssertionError("All cancelled ranks must stop")
    finally:
        dist.destroy_process_group()


def test_spawned_ranks_agree_before_leaving_collective_wave(registry, tmp_path):
    ctx = mp.get_context("spawn")
    names = [registry.create(str(rank)) for rank in range(2)]
    ready, first_check, second_check = ctx.Event(), ctx.Event(), ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_collective_worker,
            args=(rank, names[rank], str(tmp_path / "rendezvous"), ready, first_check, second_check, results),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        assert ready.wait(120)
        registry.cancel(["0"])
        first_check.set()
        assert {results.get(timeout=120) for _ in workers} == {
            (0, "partial_wave_kept_running"),
            (1, "partial_wave_kept_running"),
        }
        registry.cancel(["1"])
        second_check.set()
        assert {results.get(timeout=120) for _ in workers} == {
            (0, "fully_cancelled_wave_stopped"),
            (1, "fully_cancelled_wave_stopped"),
        }
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=30)
        results.close()
        results.join_thread()
