# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses attention with copy-engine head-chunk overlap and SM-free cross-GPU signalling.

Plain strict Ulysses exchanges Q/K/V, runs attention and exchanges the output back to back. Here the rank-local heads
are split into ``chunks`` groups and every transfer runs on copy engines (cudaMemcpy2DAsync into peers' symmetric
memory), ordered across GPUs with CUDA stream memory operations (cuStreamWriteValue32 / cuStreamWaitValue32) instead
of kernels. Nothing in the exchange occupies an SM, so it hides behind attention, which runs on the compute stream:
chunk c+1's inputs arrive and chunk c-1's outputs leave while chunk c is computing.

Copies that stay on one GPU (a rank's own rows) are SM work: the driver runs device-to-device memcpy as kernels, which
cannot start while attention holds every SM, so they run on the compute stream (inputs once up front as one strided
copy per tensor, each output chunk right after its attention). Only peer transfers go to copy engines, one stream per
peer so the engines' per-operation latency overlaps across peers.

Each rank keeps the standard Ulysses head assignment (rank r owns heads [r*H/N, (r+1)*H/N)) and computes every head
with the same attention kernel over the same K/V, so the result is bit-identical to plain Ulysses whenever the kernel's
per-head result does not depend on how many heads one call covers. Attention reads its inputs from, and the caller
reads the output from, the symmetric workspace (no copy-out): the output view stays valid until the next call, whose
first action publishes READY only after earlier work on the compute stream has finished.

Flags (uint32, monotonic epochs, one slot per writer rank) live after the data in each rank's workspace:
READY[N]: writer's regions are free for this epoch; IN[chunks][N]: writer's chunk inputs landed here;
OUT[N]: writer's outputs for every chunk landed here.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from functools import cache

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm_omni.diffusion.distributed.a2a_permute import _ce_runtime

_WAIT_GEQ = 0x0
_COPY_DEFAULT = 4  # cudaMemcpyDefault: peer pointers come from the symmetric-memory mappings
_ALIGN = 4096


@cache
def _driver() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    for name in ("cuStreamWriteValue32_v2", "cuStreamWaitValue32_v2"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    return lib


def _check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"{what} failed with error {result}")


class CEOverlapUlysses:
    """One workspace per (group, local shape, chunks); call ``run`` once per attention layer, from one stream."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        seq_local: int,
        heads: int,
        head_dim: int,
        chunks: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        n, rank = dist.get_world_size(group), dist.get_rank(group)
        if heads % (n * chunks):
            raise ValueError(f"heads {heads} not divisible by world size {n} x chunks {chunks}")
        self.n, self.rank, self.s, self.h, self.d, self.k = n, rank, seq_local, heads, head_dim, chunks
        self.hc = heads // (n * chunks)
        self.dtype, self.device = dtype, device
        es = torch.empty((), dtype=dtype).element_size()
        self.es = es
        self.tensor_bytes = n * seq_local * self.hc * head_dim * es  # one of q/k/v for one chunk: [N*s, hc, D]
        self.chunk_bytes = 3 * self.tensor_bytes
        self.out_off = chunks * self.chunk_bytes
        self.flag_off = -(-(self.out_off + seq_local * heads * head_dim * es) // _ALIGN) * _ALIGN
        self.slots = n * (chunks + 2)
        total = self.flag_off + 4 * self.slots
        self.alloc = symm_mem._SymmetricMemory.empty_strided_p2p((total,), (1,), torch.uint8, device)
        self.alloc.zero_()
        self.handle = symm_mem.rendezvous(self.alloc, group.group_name)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)
        self.peer_base = [int(p) for p in self.handle.buffer_ptrs]
        self.local_base = self.alloc.data_ptr()
        self.inputs = [
            self.alloc[c * self.chunk_bytes : (c + 1) * self.chunk_bytes]
            .view(dtype)
            .view(3, 1, n * seq_local, self.hc, head_dim)
            .unbind(0)
            for c in range(chunks)
        ]
        self.output = (
            self.alloc[self.out_off : self.out_off + seq_local * heads * head_dim * es]
            .view(dtype)
            .view(1, seq_local, heads, head_dim)
        )
        # This rank's own input rows of every chunk: [chunks, s, hc, D] strided workspace views, one per tensor.
        elems = self.alloc[: self.out_off].view(dtype)
        self.self_inputs = [
            elems.as_strided(
                (chunks, seq_local, self.hc, head_dim),
                (self.chunk_bytes // es, self.hc * head_dim, head_dim, 1),
                (t * self.tensor_bytes + rank * seq_local * self.hc * head_dim * es) // es,
            )
            for t in range(3)
        ]
        self.peers = [(rank + i) % n for i in range(1, n)]
        self.t_in = {p: torch.cuda.Stream(device=device) for p in self.peers}
        self.t_out = {p: torch.cuda.Stream(device=device) for p in self.peers}
        self.ev_inputs = torch.cuda.Event()
        self.ev_attn = [torch.cuda.Event() for _ in range(chunks)]
        self.ev_in_done = {p: torch.cuda.Event() for p in self.peers}
        self.ev_out_done = {p: torch.cuda.Event() for p in self.peers}
        self.epoch = 0
        _driver()
        _ce_runtime()

    def _slot(self, kind: str, writer: int, chunk: int = 0) -> int:
        n = self.n
        index = {"ready": writer, "in": n + chunk * n + writer, "out": n + self.k * n + writer}[kind]
        return self.flag_off + 4 * index

    def _signal(self, stream: torch.cuda.Stream, peer: int, kind: str, chunk: int = 0) -> None:
        address = self.peer_base[peer] + self._slot(kind, self.rank, chunk)
        _check(_driver().cuStreamWriteValue32_v2(stream.cuda_stream, address, self.epoch, 0), "cuStreamWriteValue32")

    def _wait(self, stream: torch.cuda.Stream, writer: int, kind: str, chunk: int = 0) -> None:
        address = self.local_base + self._slot(kind, writer, chunk)
        _check(
            _driver().cuStreamWaitValue32_v2(stream.cuda_stream, address, self.epoch, _WAIT_GEQ),
            "cuStreamWaitValue32",
        )

    def _copy(self, stream, dst, dpitch, src, spitch, width, height) -> None:
        runtime = _ce_runtime()
        runtime.CUDART_CHECK(
            runtime.lib.cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, _COPY_DEFAULT, stream.cuda_stream)
        )

    @torch.compiler.disable
    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """(1, s, H, D) local Q/K/V -> (1, s, H, D) attention output (a view into the workspace)."""
        n, rank, s, h, d, k, hc, es = self.n, self.rank, self.s, self.h, self.d, self.k, self.hc, self.es
        expected = (1, s, h, d)
        for x in (query, key, value):
            if tuple(x.shape) != expected or x.dtype != self.dtype or x.stride(-1) != 1 or x.stride(-2) != d:
                raise ValueError(
                    f"expected {expected} {self.dtype} with dense heads, got {tuple(x.shape)} {x.stride()}"
                )
        compute = torch.cuda.current_stream(self.device)
        self.epoch += 1
        width, hpr = hc * d * es, h // n

        # Publish READY: everything earlier on this stream (previous attention and output consumers) has finished.
        for p in self.peers:
            self._signal(compute, p, "ready")
        self.ev_inputs.record(compute)

        # Peer inputs: each peer's stream pushes this rank's rows of every chunk, then raises IN for that chunk.
        sources = [(x.data_ptr(), x.stride(1) * es) for x in (query, key, value)]
        for p in self.peers:
            stream = self.t_in[p]
            stream.wait_event(self.ev_inputs)
            self._wait(stream, p, "ready")
            for c in range(k):
                head0 = p * hpr + c * hc
                for t, (ptr, pitch) in enumerate(sources):
                    dst = self.peer_base[p] + c * self.chunk_bytes + t * self.tensor_bytes + rank * s * width
                    self._copy(stream, dst, width, ptr + head0 * d * es, pitch, width, s)
                self._signal(stream, p, "in", c)
            self.ev_in_done[p].record(stream)

        # Own input rows: one strided SM copy per tensor while the first peer chunk is in flight.
        for dst, x in zip(self.self_inputs, (query, key, value)):
            dst.copy_(x[0, :, rank * hpr : (rank + 1) * hpr].view(s, k, hc, d).permute(1, 0, 2, 3))

        for p in self.peers:
            self.t_out[p].wait_event(self.ev_inputs)
            self._wait(self.t_out[p], p, "ready")
        pending = []
        for c in range(k):
            for p in self.peers:
                self._wait(compute, p, "in", c)
            q, kk, v = self.inputs[c]
            out = attn_fn(q, kk, v)  # (1, N*s, hc, D)
            if out.stride(-1) != 1 or out.stride(-2) != d or out.stride(1) != hc * d:
                out = out.contiguous()
            pending.append(out)
            self.ev_attn[c].record(compute)
            col0 = rank * hpr + c * hc
            for p in self.peers:
                stream = self.t_out[p]
                stream.wait_event(self.ev_attn[c])
                dst = self.peer_base[p] + self.out_off + col0 * d * es
                self._copy(stream, dst, h * d * es, out.data_ptr() + p * s * width, width, width, s)
            self.output[0, :, col0 : col0 + hc].copy_(out[0, rank * s : (rank + 1) * s])
        for p in self.peers:
            self._signal(self.t_out[p], p, "out")
            self.ev_out_done[p].record(self.t_out[p])

        # Every peer's output rows have landed; sources and chunk outputs may be released afterwards.
        for p in self.peers:
            compute.wait_event(self.ev_in_done[p])
            compute.wait_event(self.ev_out_done[p])
            self._wait(compute, p, "out")
        del pending
        return self.output


_RUNNERS: dict[tuple, CEOverlapUlysses] = {}


def run_ulysses_ce_overlap(pg, query, key, value, attn_fn, *, chunks: int) -> torch.Tensor:
    """Cached runner per (group, local shape, chunks); creation is collective and happens in lockstep on all ranks."""
    _, s, h, d = query.shape
    cache_key = (pg.group_name, s, h, d, chunks, query.dtype, query.device)
    runner = _RUNNERS.get(cache_key)
    if runner is None:
        runner = _RUNNERS[cache_key] = CEOverlapUlysses(
            pg, seq_local=s, heads=h, head_dim=d, chunks=chunks, dtype=query.dtype, device=query.device
        )
    return runner.run(query, key, value, attn_fn)
