# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The copy-engine ring transport (``VLLM_OMNI_RING_CE_TRANSPORT=1``) must be bit-identical to ``RingComm``."""

import gc
import os
import signal
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.attention.backends import ring_flash_attn
from vllm_omni.diffusion.attention.backends.ring import ce_transport
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.distributed import a2a_permute
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# (local sequence, heads, head dim, dtype): the workspace grows, is reused smaller, changes dtype.
_SHAPES = [
    (19, 4, 64, torch.bfloat16),
    (64, 4, 64, torch.bfloat16),
    (33, 4, 64, torch.bfloat16),
    (64, 8, 128, torch.float16),
    (7, 4, 64, torch.bfloat16),
]


def _ring(q, k, v, causal: bool, ce: bool):
    os.environ["VLLM_OMNI_RING_CE_TRANSPORT"] = "1" if ce else "0"
    return ring_flash_attn.ring_flash_attn_forward(
        dist.group.WORLD,
        q,
        k,
        v,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=causal,
        attn_type=AttnType.VLLM_FA,
    )


def _local_qkv(world_size, rank, seqlen, heads, head_dim, dtype, seed):
    torch.manual_seed(seed)
    return tuple(
        torch.randn(1, seqlen * world_size, heads, head_dim, device="cuda", dtype=dtype)
        .chunk(world_size, dim=1)[rank]
        .contiguous()
        for _ in range(3)
    )


def _run_ce_matches_ring_comm(local_rank: int, world_size: int, master_port: int) -> None:
    device = torch.device("cuda", local_rank)
    current_omni_platform.set_device(device)
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    finished = []
    finish = ce_transport.CeRingComm.finish

    def counting_finish(self):
        finished.append(1)
        finish(self)

    ce_transport.CeRingComm.finish = counting_finish
    try:
        with torch.inference_mode():
            for i, (seqlen, heads, head_dim, dtype) in enumerate(_SHAPES):
                q, k, v = _local_qkv(world_size, local_rank, seqlen, heads, head_dim, dtype, seed=100 + i)
                refs = {causal: _ring(q, k, v, causal, ce=False) for causal in (False, True)}
                # Back-to-back copy-engine calls, as consecutive layers issue them.
                for causal in (False, True, False):
                    before = len(finished)
                    out, lse = _ring(q, k, v, causal, ce=True)
                    assert len(finished) == before + 1, "the copy-engine transport was not used"
                    ref_out, ref_lse = refs[causal]
                    assert torch.equal(out, ref_out), (seqlen, heads, head_dim, dtype, causal)
                    assert torch.equal(lse, ref_lse), (seqlen, heads, head_dim, dtype, causal)
    finally:
        ce_transport.CeRingComm.finish = finish
        os.environ.pop("VLLM_OMNI_RING_CE_TRANSPORT", None)
        dist.destroy_process_group()


def _run_ce_falls_back(local_rank: int, world_size: int, master_port: int) -> None:
    device = torch.device("cuda", local_rank)
    current_omni_platform.set_device(device)
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    empty = ce_transport.symm_mem.empty

    def failing_get_buffer(*args, **kwargs):
        raise RuntimeError("no P2P mapping")

    try:
        with torch.inference_mode():
            q, k, v = _local_qkv(world_size, local_rank, 19, 4, 64, torch.bfloat16, seed=7)
            ref_out, ref_lse = _ring(q, k, v, False, ce=False)
            # A failed rendezvous or peer mapping falls back to RingComm.
            ce_transport.symm_mem.empty = failing_get_buffer
            out, lse = _ring(q, k, v, False, ce=True)
            assert torch.equal(out, ref_out) and torch.equal(lse, ref_lse)
            assert not any(workspace.ring_credited for workspace in ce_transport._SYMM_WORKSPACES.values())
    finally:
        ce_transport.symm_mem.empty = empty
        os.environ.pop("VLLM_OMNI_RING_CE_TRANSPORT", None)
        dist.destroy_process_group()


def _require_devices(world_size: int, needs_peer_access: bool = False) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    if needs_peer_access:
        # The transport maps every peer's buffer, so it needs CUDA peer access between all GPU
        # pairs. PCIe hosts without P2P (such as the L4 lane) fall back to RingComm, which
        # test_ce_transport_falls_back_when_peer_mapping_fails covers.
        missing = [
            (i, j)
            for i in range(world_size)
            for j in range(world_size)
            if i != j and not torch.cuda.can_device_access_peer(i, j)
        ]
        if missing:
            pytest.skip(f"the copy-engine transport needs CUDA peer access; GPU pairs without it: {missing[:4]}")


def _spawn(fn, world_size: int, needs_peer_access: bool = False) -> None:
    _require_devices(world_size, needs_peer_access)
    torch.multiprocessing.spawn(fn, args=(world_size, get_open_port()), nprocs=world_size, join=True)


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_matches_ring_comm_2_ranks():
    _spawn(_run_ce_matches_ring_comm, 2, needs_peer_access=True)


@hardware_test(res={"cuda": ["H100"]}, num_cards=8)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_matches_ring_comm_8_ranks():
    _spawn(_run_ce_matches_ring_comm, 8, needs_peer_access=True)


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
def test_ce_transport_falls_back_when_peer_mapping_fails():
    _spawn(_run_ce_falls_back, 2)


@pytest.mark.cpu
def test_ce_transport_falls_back_to_ring_comm_off_cuda(monkeypatch):
    """CPU tensors cannot use the copy-engine transport, so the ring keeps RingComm."""

    class SingleRankComm:
        rank = 0
        world_size = 1
        constructed = 0

        def __init__(self, _group):
            SingleRankComm.constructed += 1

    def fake_attention(q, _k, _v, **_kwargs):
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    monkeypatch.setenv("VLLM_OMNI_RING_CE_TRANSPORT", "1")
    monkeypatch.setattr(ring_flash_attn, "RingComm", SingleRankComm)
    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: fake_attention)
    qkv = torch.randn(1, 4, 2, 8)

    output, _ = ring_flash_attn.ring_flash_attn_forward(
        object(), qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
    )

    assert output.shape == qkv.shape
    assert SingleRankComm.constructed == 1


@dataclass
class _HostStream:
    cuda_stream: int = 1

    def wait_stream(self, _stream):
        pass

    def wait_event(self, _event):
        pass


class _HostEvent:
    def record(self, _stream):
        pass


@dataclass(frozen=True)
class _HostGroup:
    group_name: str = "ring_ce_execution_error"


@dataclass
class _HostHandle:
    fail_peer: bool = False
    credits: list[tuple[int, int]] = field(default_factory=list)

    def get_buffer(self, _rank, shape, dtype, _offset):
        if self.fail_peer:
            raise RuntimeError("injected peer mapping failure")
        return torch.empty(shape, dtype=dtype)

    def put_signal(self, rank, channel):
        self.credits.append((rank, channel))

    def wait_signal(self, _rank, _channel):
        pass


def _run_rank_local_setup_failure(rank: int, rendezvous_path: str) -> None:
    """Real Gloo/store coordination; only CUDA allocation/handles are replaced."""
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    handles: list[_HostHandle] = []
    failure = [""]
    stream = _HostStream()

    def current_stream(_device):
        if rank == 1 and failure[0] == "stream":
            raise RuntimeError("injected stream failure")
        return stream

    def empty(size, *, dtype, device):
        if rank == 1 and failure[0] == "allocation":
            raise RuntimeError("injected allocation failure")
        return torch.empty(size, dtype=dtype, device=device)

    def rendezvous(_allocation, _group):
        if rank == 1 and failure[0] == "rendezvous":
            raise RuntimeError("injected rendezvous failure")
        handle = _HostHandle(fail_peer=rank == 1 and failure[0] == "peer")
        handles.append(handle)
        return handle

    def event():
        if rank == 1 and failure[0] == "event":
            raise RuntimeError("injected event creation failure")
        return object()

    try:
        with pytest.MonkeyPatch.context() as patch:
            # The production workspace and agreement code run unchanged on CPU.
            patch.setattr(ce_transport, "_unavailable_reason", lambda *_args: None)
            patch.setattr(torch.cuda, "current_stream", current_stream)
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
            patch.setattr(torch.cuda, "Event", event)
            patch.setattr(torch.accelerator, "synchronize", lambda _device: None)
            patch.setattr(ce_transport, "_copy_stream", lambda _device: stream)
            patch.setattr(ce_transport.symm_mem, "empty", empty)
            patch.setattr(ce_transport.symm_mem, "rendezvous", rendezvous)
            patch.setattr(a2a_permute, "_SYMM_WORKSPACES", {})
            patch.setattr(ce_transport, "_SYMM_WORKSPACES", a2a_permute._SYMM_WORKSPACES)
            k = torch.empty(1, 4, 2, 8)
            group = dist.group.WORLD
            assert group is not None
            for stage in ("peer", "stream", "allocation", "rendezvous", "event"):
                failure[0] = stage
                handles.clear()
                a2a_permute.clear_a2a_permute_workspaces()
                assert ce_transport.maybe_ce_ring_comm(group, k, k) is None
                # No rank may grant credits when another rank's setup failed.
                assert all(not handle.credits for handle in handles)
                assert not any(workspace.ring_credited for workspace in a2a_permute._SYMM_WORKSPACES.values())

            failure[0] = ""
            handles.clear()
            a2a_permute.clear_a2a_permute_workspaces()
            comm = ce_transport.maybe_ce_ring_comm(group, k, k)
            assert comm is not None
            handle_ref = weakref.ref(comm._handle)
            assert len(handles[-1].credits) == 2

            # Failed growth on one rank preserves the previous usable workspace.
            failure[0] = "allocation"
            larger = torch.empty(1, 8, 2, 8)
            assert ce_transport.maybe_ce_ring_comm(group, larger, larger) is None
            assert a2a_permute._SYMM_WORKSPACES[(k.device, group.group_name)].handle is comm._handle
            failure[0] = ""

            # Reused workspaces must not perform host agreement or grant credits again.
            with pytest.MonkeyPatch.context() as reuse_patch:

                def unexpected_agreement(_group):
                    raise AssertionError("steady-state CE must not coordinate setup")

                reuse_patch.setattr(ce_transport, "_SetupAgreement", unexpected_agreement)
                reused = ce_transport.maybe_ce_ring_comm(group, k, k)
                assert reused is not None
                assert reused._handle is comm._handle
                assert len(handles[-1].credits) == 2
            del reused, comm
            handles.clear()
            a2a_permute.clear_a2a_permute_workspaces()
            gc.collect()  # Failed growth can leave transient exception/traceback cycles.
            assert handle_ref() is None, "credit tracking must not retain the handle after workspace cleanup"

            # A fresh workspace gets fresh credits after cleanup.
            fresh = ce_transport.maybe_ce_ring_comm(group, k, k)
            assert fresh is not None
            assert len(handles[-1].credits) == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu
def test_ce_rank_local_setup_failures_and_cleanup(tmp_path):
    torch.multiprocessing.spawn(
        _run_rank_local_setup_failure, args=(str(tmp_path / "gloo_store"),), nprocs=2, join=True
    )


def _install_cpu_ring(monkeypatch, world_size: int, rank: int):
    """Exercise the real CE loop using host buffers and nonblocking stream doubles."""
    group, stream, handle = _HostGroup(), _HostStream(), _HostHandle()
    qkv = torch.zeros(1, 4, 2, 8)
    nbytes = 4 * qkv.numel() * qkv.element_size()
    workspaces = {
        (qkv.device, group.group_name): a2a_permute._SymmWorkspace(
            torch.empty(nbytes, dtype=torch.uint8), handle, nbytes, stream.cuda_stream, True
        )
    }
    monkeypatch.setenv("VLLM_OMNI_RING_CE_TRANSPORT", "1")
    monkeypatch.setattr(ce_transport, "_unavailable_reason", lambda *_args: None)
    monkeypatch.setattr(dist, "get_rank", lambda _group=None: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda _group=None: world_size)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _HostEvent)
    monkeypatch.setattr(ce_transport, "_copy_stream", lambda _device: stream)
    monkeypatch.setattr(a2a_permute, "_SYMM_WORKSPACES", workspaces)
    monkeypatch.setattr(ce_transport, "_SYMM_WORKSPACES", workspaces)

    def attention(q, _k, _v, **_kwargs):
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: attention)
    return group, qkv


@pytest.mark.cpu
@pytest.mark.parametrize(
    "world_size,rank,step,stage",
    [(2, 0, 0, "attention"), (3, 1, 1, "attention"), (4, 2, 2, "merge"), (3, 0, 0, "commit"), (2, 1, 1, "finish")],
)
def test_ce_execution_errors_exit_instead_of_reusing_credits(monkeypatch, world_size, rank, step, stage):
    group, qkv = _install_cpu_ring(monkeypatch, world_size, rank)
    calls = 0

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected CE execution failure")

    def attention(q, _k, _v, **_kwargs):
        nonlocal calls
        current = calls
        calls += 1
        if stage == "attention" and current == step:
            fail()
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    original_merge = ring_flash_attn.update_out_and_lse

    def merge(*args, **kwargs):
        if stage == "merge" and calls - 1 == step:
            fail()
        return original_merge(*args, **kwargs)

    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: attention)
    monkeypatch.setattr(ring_flash_attn, "update_out_and_lse", merge)
    monkeypatch.setattr(ce_transport.os, "_exit", exit_process)
    if stage in ("commit", "finish"):
        monkeypatch.setattr(ce_transport.CeRingComm, stage, fail)
    with pytest.raises(SystemExit) as exc:
        ring_flash_attn.ring_flash_attn_forward(
            group, qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
        )
    assert exc.value.code == 1


@pytest.mark.cpu
@pytest.mark.parametrize("ce", [False, True])
def test_ce_success_and_nccl_error_behavior_are_preserved(monkeypatch, ce):
    group, qkv = _install_cpu_ring(monkeypatch, 3, 1)

    def unexpected_exit(_code):
        pytest.fail("successful CE and normal NCCL errors must not terminate the process")

    monkeypatch.setattr(ce_transport.os, "_exit", unexpected_exit)
    if ce:
        output, _ = ring_flash_attn.ring_flash_attn_forward(
            group, qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
        )
        assert output.shape == qkv.shape
    else:

        class SingleRankComm:
            rank = 0
            world_size = 1

        def fail(*_args, **_kwargs):
            raise RuntimeError("normal NCCL-path attention error")

        monkeypatch.setenv("VLLM_OMNI_RING_CE_TRANSPORT", "0")
        monkeypatch.setattr(ring_flash_attn, "RingComm", lambda _group: SingleRankComm())
        monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: fail)
        with pytest.raises(RuntimeError, match="normal NCCL-path"):
            ring_flash_attn.ring_flash_attn_forward(
                group, qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
            )


def _run_ce_fatal_worker(marker_path: str) -> None:
    from pathlib import Path

    with pytest.MonkeyPatch.context() as patch:
        group, qkv = _install_cpu_ring(patch, 3, 1)
        calls = 0

        def attention(q, _k, _v, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("rank-local attention failure after the second commit")
            return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

        patch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: attention)
        try:
            ring_flash_attn.ring_flash_attn_forward(
                group, qkv, qkv, qkv, softmax_scale=0.25, causal=False, attn_type=AttnType.FA
            )
        except Exception:
            # Model the worker's recoverable-error path. CE must never reach it.
            Path(marker_path).write_text("worker caught the error and could accept another request")


def _assert_ce_worker_group_exits(
    processes,
    failing_rank: int,
    marker: Path,
    monkeypatch,
    *,
    timeout: float = 30,
    expected_exit_codes: tuple[int, ...] = (1,),
    on_started: Callable[[threading.Event], None] | None = None,
) -> None:
    """Run the real executor monitor with bounded waits and unconditional cleanup."""
    from vllm_omni.diffusion.executor import multiproc_executor

    executor = object.__new__(multiproc_executor.MultiprocDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._failure_callbacks = []
    executor._broadcast_mq = None
    executor._result_mq = None
    executor._shutdown_cleaner = multiproc_executor._ExecutorShutdownCleaner(processes=processes)
    executor._finalizer = weakref.finalize(executor, executor._shutdown_cleaner)
    executor._pump_stop = threading.Event()
    executor._futures_lock = threading.RLock()
    executor._rpc_futures = {}
    executor._output_futures = {}
    executor._batch_split_map = {}
    executor._completed_outputs = {}
    executor._dropped_output_ids = OrderedDict()
    executor._processes = processes
    # Avoid the production graceful-shutdown wait for intentionally blocked peers.
    monkeypatch.setattr(multiproc_executor, "_WORKER_SHUTDOWN_GRACE_S", 0)
    completed = threading.Event()
    executor.register_failure_callback(completed.set)
    try:
        for process in processes:
            process.start()
        executor._start_worker_monitor()
        if on_started is not None:
            on_started(completed)
        assert completed.wait(timeout), "worker death did not trigger group teardown within the timeout"
        assert processes[failing_rank].exitcode in expected_exit_codes, [process.exitcode for process in processes]
        assert not marker.exists(), "CE error reached the worker's recoverable-error path"
        assert executor.is_dead
        assert all(not process.is_alive() for process in processes)
    finally:
        executor.shutdown()


@pytest.mark.cpu
def test_ce_rank_local_error_terminates_worker_and_executor_group(tmp_path, monkeypatch):
    ctx = torch.multiprocessing.get_context("spawn")
    blocked = ctx.Event()
    marker = tmp_path / "reused_worker"
    failing = ctx.Process(target=_run_ce_fatal_worker, args=(str(marker),))
    peers = [ctx.Process(target=blocked.wait, args=(60,)) for _ in range(2)]
    _assert_ce_worker_group_exits([failing, *peers], 0, marker, monkeypatch)


class _InjectedCeExecutionError(RuntimeError):
    """Distinguish the intended GPU fault from unrelated startup/kernel failures."""


def _run_gpu_ce_execution_error(
    local_rank: int,
    world_size: int,
    master_port: int,
    directory: str,
    fault_rank: int,
    fault_step: int,
    stage: str,
    start_fault,
    hold_peer,
) -> None:
    """Real CUDA/NCCL/CE execution; only the selected rank's error is injected."""
    directory_path = Path(directory)
    device = torch.device("cuda", local_rank)
    try:
        current_omni_platform.set_device(device)
        os.environ.update(
            RANK=str(local_rank),
            LOCAL_RANK=str(local_rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(master_port),
        )
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size, timeout=timedelta(seconds=120))
        commits, finishes = 0, 0
        original_commit = ce_transport.CeRingComm.commit
        original_finish = ce_transport.CeRingComm.finish
        original_abort = ce_transport.CeRingComm.abort

        def commit(self):
            nonlocal commits
            original_commit(self)
            commits += 1

        def finish(self):
            nonlocal finishes
            original_finish(self)
            finishes += 1

        def abort(self):
            # A worker exit alone is insufficient: a setup/kernel failure could
            # otherwise pass this test without exercising the intended CE fault.
            if local_rank == fault_rank and isinstance(sys.exc_info()[1], _InjectedCeExecutionError):
                assert commits == min(fault_step + 1, world_size - 1)
                (directory_path / "abort_observed").write_text(f"{local_rank}:{stage}:{commits}")
            # Peers may also encounter CUDA errors once the failing rank exits.
            original_abort(self)

        ce_transport.CeRingComm.commit = commit
        ce_transport.CeRingComm.finish = finish
        ce_transport.CeRingComm.abort = abort
        with torch.inference_mode():
            q, k, v = _local_qkv(world_size, local_rank, 19, 4, 64, torch.bfloat16, seed=29)
            expected_out, expected_lse = _ring(q, k, v, False, ce=False)
            actual_out, actual_lse = _ring(q, k, v, False, ce=True)
            assert finishes == 1, "CE transport was not exercised during warmup"
            assert torch.equal(actual_out, expected_out)
            assert torch.equal(actual_lse, expected_lse)
            torch.accelerator.synchronize()
            dist.barrier()
            (directory_path / f"ready-{local_rank}").touch()
            assert start_fault.wait(300), "parent did not release the GPU fault phase"
            commits = 0
            attention_calls = 0
            select = ring_flash_attn.select_flash_attn_impl
            merge = ring_flash_attn.update_out_and_lse

            def select_attention(*args, **kwargs):
                function = select(*args, **kwargs)

                def attention(*fn_args, **fn_kwargs):
                    nonlocal attention_calls
                    step = attention_calls
                    attention_calls += 1
                    if local_rank == fault_rank and stage == "attention" and step == fault_step:
                        raise _InjectedCeExecutionError("injected rank-local GPU attention error")
                    return function(*fn_args, **fn_kwargs)

                return attention

            def merge_output(*args, **kwargs):
                if local_rank == fault_rank and stage == "merge" and attention_calls - 1 == fault_step:
                    raise _InjectedCeExecutionError("injected rank-local GPU merge error")
                return merge(*args, **kwargs)

            ring_flash_attn.select_flash_attn_impl = select_attention
            ring_flash_attn.update_out_and_lse = merge_output
            try:
                _ring(q, k, v, False, ce=True)
                torch.accelerator.synchronize()
            except Exception:
                (directory_path / "reused_worker").write_text(f"rank {local_rank} reached recoverable error handling")
                raise
            # A successful peer may finish before the failing rank. Keep it alive
            # so only the injected abort can trigger the executor's death monitor.
            hold_peer.wait(300)
            raise AssertionError("executor failed to terminate a surviving peer")
    except BaseException:
        import traceback

        traceback.print_exc()
        # Avoid normal CUDA teardown if unexpected failure left signal kernels
        # pending. The expected abort bypasses this handler via os._exit(1).
        os._exit(2)


def _check_gpu_ce_execution_error(tmp_path, monkeypatch, world_size: int, fault_step: int, stage: str) -> None:
    _require_devices(world_size, needs_peer_access=True)
    ctx = torch.multiprocessing.get_context("spawn")
    start_fault, hold_peer = ctx.Event(), ctx.Event()
    port, fault_rank = get_open_port(), 1
    processes = [
        ctx.Process(
            target=_run_gpu_ce_execution_error,
            args=(rank, world_size, port, str(tmp_path), fault_rank, fault_step, stage, start_fault, hold_peer),
        )
        for rank in range(world_size)
    ]

    def begin_fault(completed: threading.Event) -> None:
        # Bound startup/warmup separately from failure teardown. FA4's initial
        # kernel compilation can be substantially slower than the ring call.
        deadline = time.monotonic() + 240
        while not all((tmp_path / f"ready-{rank}").exists() for rank in range(world_size)):
            assert not completed.is_set(), "a GPU worker exited before CE warmup completed"
            assert time.monotonic() < deadline, "GPU warmup did not complete within 240 seconds"
            time.sleep(0.05)
        start_fault.set()

    _assert_ce_worker_group_exits(
        processes,
        fault_rank,
        tmp_path / "reused_worker",
        monkeypatch,
        timeout=60,
        # A peer can notice the failing GPU's disappearance before its OS exit
        # completes. The executor may then kill that rank during group teardown.
        # The marker below independently proves it reached the intended abort.
        expected_exit_codes=(1, -signal.SIGTERM, -signal.SIGKILL),
        on_started=begin_fault,
    )
    expected_commits = min(fault_step + 1, world_size - 1)
    assert (tmp_path / "abort_observed").read_text() == f"{fault_rank}:{stage}:{expected_commits}"


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
@pytest.mark.parametrize("stage", ["attention", "merge"])
def test_ce_rank_local_execution_error_2_ranks(tmp_path, monkeypatch, stage):
    _check_gpu_ce_execution_error(tmp_path, monkeypatch, 2, fault_step=0, stage=stage)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="the copy-engine transport requires CUDA")
@pytest.mark.parametrize("stage", ["attention", "merge"])
def test_ce_rank_local_execution_error_4_ranks(tmp_path, monkeypatch, stage):
    _check_gpu_ce_execution_error(tmp_path, monkeypatch, 4, fault_step=1, stage=stage)
