# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Perf smoke tests for Ulysses advanced_uaa communication overhead.

This test is intended for CI monitoring only:
- Print per-iteration timings and ratio vs strict Ulysses all-to-all.
- Use a loose sanity bound to catch gross regressions without flakiness.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from tests.utils import hardware_test
from vllm_omni.diffusion.attention.parallel.ulysses import (
    _all_gather_int,
    _ulysses_all_to_all_any_o,
    _ulysses_all_to_all_any_qkv,
)
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _max_all_reduce(pg: dist.ProcessGroup, value: float, *, device: torch.device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pg)
    return float(t.item())


@dataclass(frozen=True, slots=True)
class _PerfCase:
    ulysses_degree: int
    ring_degree: int

    @property
    def world_size(self) -> int:
        return int(self.ulysses_degree * self.ring_degree)


PERF_CASES: list[_PerfCase] = [
    _PerfCase(ulysses_degree=4, ring_degree=1),
    _PerfCase(ulysses_degree=2, ring_degree=2),
]


@pytest.mark.parametrize("case", PERF_CASES)
@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_ulysses_advanced_uaa_comm_overhead(case: _PerfCase) -> None:
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _perf_worker,
        args=(case.world_size, master_port, case.ulysses_degree, case.ring_degree),
        nprocs=case.world_size,
    )


def _perf_worker(local_rank: int, world_size: int, master_port: int, ulysses_degree: int, ring_degree: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=world_size,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

        sp_group = get_sp_group()
        ulysses_pg = sp_group.ulysses_group
        ring_pg = sp_group.ring_group

        ulysses_world_size = dist.get_world_size(ulysses_pg)
        ring_world_size = dist.get_world_size(ring_pg)

        # A moderate tensor size to reduce timing noise while staying fast in CI.
        bsz = 1
        s_local = 256
        head_cnt = 32  # divisible by ulysses_degree in both cases above
        head_dim = 128
        dtype = torch.float16
        use_sync = False

        torch.manual_seed(1234 + local_rank)
        q = torch.randn(bsz, s_local, head_cnt, head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        warmup_iters = 10
        iters = 200

        def strict_comm_step(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
            q_out = SeqAllToAll4D.apply(ulysses_pg, q_in, 2, 1, use_sync)
            SeqAllToAll4D.apply(ulysses_pg, k_in, 2, 1, use_sync)
            SeqAllToAll4D.apply(ulysses_pg, v_in, 2, 1, use_sync)
            o_in = q_out  # dummy
            return SeqAllToAll4D.apply(ulysses_pg, o_in, 1, 2, use_sync)

        def uaa_comm_step(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
            seq_lens = _all_gather_int(ulysses_pg, int(q_in.shape[1]), device=q_in.device)
            s_global = int(sum(seq_lens))
            if ring_world_size > 1:
                ring_s_globals = _all_gather_int(ring_pg, s_global, device=q_in.device)
                if len(set(ring_s_globals)) != 1:
                    raise RuntimeError(f"Unexpected hybrid ring post-Ulysses seq_len mismatch: {ring_s_globals}.")

            q_out, orig_head_cnt = _ulysses_all_to_all_any_qkv(ulysses_pg, q_in, seq_lens=seq_lens, use_sync=use_sync)
            _ulysses_all_to_all_any_qkv(ulysses_pg, k_in, seq_lens=seq_lens, use_sync=use_sync)
            _ulysses_all_to_all_any_qkv(ulysses_pg, v_in, seq_lens=seq_lens, use_sync=use_sync)
            o_in = q_out  # dummy
            return _ulysses_all_to_all_any_o(
                ulysses_pg,
                o_in,
                seq_lens=seq_lens,
                local_seq_len=int(s_local),
                orig_head_cnt=int(orig_head_cnt),
                use_sync=use_sync,
            )

        with torch.no_grad():
            # Warmup (strict)
            for _ in range(warmup_iters):
                _ = strict_comm_step(q, k, v)
            current_omni_platform.synchronize()

            # Timed (strict)
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                _ = strict_comm_step(q, k, v)
            t1.record()
            current_omni_platform.synchronize()
            strict_ms = float(t0.elapsed_time(t1)) / float(iters)

            # Warmup (UAA)
            for _ in range(warmup_iters):
                _ = uaa_comm_step(q, k, v)
            current_omni_platform.synchronize()

            # Timed (UAA)
            u0 = torch.cuda.Event(enable_timing=True)
            u1 = torch.cuda.Event(enable_timing=True)
            u0.record()
            for _ in range(iters):
                _ = uaa_comm_step(q, k, v)
            u1.record()
            current_omni_platform.synchronize()
            uaa_ms = float(u0.elapsed_time(u1)) / float(iters)

        # Reduce across ranks (use worst-rank to be conservative).
        strict_ms_max = _max_all_reduce(dist.group.WORLD, strict_ms, device=device)
        uaa_ms_max = _max_all_reduce(dist.group.WORLD, uaa_ms, device=device)
        ratio = (uaa_ms_max / strict_ms_max) if strict_ms_max > 0 else float("inf")

        # Approx bytes moved per iteration per-rank (send+recv) for 4x all-to-all.
        elem_size = torch.tensor([], dtype=dtype).element_size()
        per_a2a_bytes = int(q.numel()) * elem_size * 2
        comm_bytes = int(4 * per_a2a_bytes)
        strict_gbps = (comm_bytes / (strict_ms_max / 1000.0)) / 1e9 if strict_ms_max > 0 else 0.0
        uaa_gbps = (comm_bytes / (uaa_ms_max / 1000.0)) / 1e9 if uaa_ms_max > 0 else 0.0

        if dist.get_rank() == 0:
            payload = {
                "name": "ulysses_uaa_comm_perf",
                "world_size": int(world_size),
                "ulysses_degree": int(ulysses_degree),
                "ring_degree": int(ring_degree),
                "ulysses_world_size": int(ulysses_world_size),
                "ring_world_size": int(ring_world_size),
                "shape": {
                    "bsz": int(bsz),
                    "s_local": int(s_local),
                    "head_cnt": int(head_cnt),
                    "head_dim": int(head_dim),
                },
                "dtype": str(dtype),
                "iters": int(iters),
                "strict_ms_per_iter_max": float(strict_ms_max),
                "uaa_ms_per_iter_max": float(uaa_ms_max),
                "uaa_over_strict_ratio": float(ratio),
                "comm_bytes_per_iter_per_rank": int(comm_bytes),
                "strict_effective_gbps_per_rank": float(strict_gbps),
                "uaa_effective_gbps_per_rank": float(uaa_gbps),
            }
            print(f"UAA_COMM_PERF_JSON={payload}")

        # Loose bound: we only want to catch severe regressions. The printed JSON
        # payload is used for monitoring smaller changes, while this threshold is
        # intentionally generous to avoid flakiness across GPU types/drivers.
        max_uaa_ms_per_iter = 10.0
        assert uaa_ms_max < max_uaa_ms_per_iter, (
            f"UAA comm too slow: uaa={uaa_ms_max:.3f}ms/iter, strict={strict_ms_max:.3f}ms/iter "
            f"(cap={max_uaa_ms_per_iter:.1f}ms/iter)."
        )
        assert ratio < 3.0, f"UAA comm overhead too high: ratio={ratio:.3f}x (strict={strict_ms_max:.3f}ms)."
    finally:
        destroy_distributed_env()
