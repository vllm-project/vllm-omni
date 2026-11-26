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

Flags (uint32, one producer and one consumer per slot) live at the start of each rank's workspace:
``READY[N]``: writer's regions are free for this epoch; ``IN[chunks][N]``: writer's chunk inputs landed here;
``OUT[N]``: writer's outputs for every chunk landed here.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from functools import cache
from itertools import accumulate

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


class PreparationError(RuntimeError):
    """A peer could not prepare its local resources before starting the exchange."""


def prepare_local(fn, control):
    # Complete all purely local work before any rank can enter rendezvous.
    # An unexpected local exception must also reach the other ranks. Only
    # resource exhaustion and unavailable driver libraries are safe to retry
    # as native attention. A poisoned device must leave the worker instead.
    failure = None
    try:
        result = fn()
    except Exception as exc:
        recoverable = isinstance(exc, (PreparationError, torch.OutOfMemoryError, MemoryError, OSError))
        failure = (recoverable, f"{type(exc).__name__}: {exc}")
    peers = [None] * dist.get_world_size(control)
    dist.all_gather_object(peers, failure, group=control)
    errors = [(rank, item[1]) for rank, item in enumerate(peers) if item is not None]
    if errors:
        error_type = PreparationError if all(item[0] for item in peers if item) else RuntimeError
        raise error_type(f"Ulysses local preparation failed: {errors}")
    return result


class CEOverlapUlysses:
    """One grow-only workspace per (group, device); call ``run`` once per attention layer, from one stream.

    The workspace is sized for the largest local shape seen and smaller shapes reuse it, so mixed resolutions and
    durations do not add allocations. Growth is collective (every rank sees the same shapes in lockstep).
    """

    def __init__(self, group: dist.ProcessGroup, *, device: torch.device) -> None:
        self.group = group
        self.n, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.device = device
        self.alloc: torch.Tensor | None = None
        self.capacity = 0
        self.max_chunks = 0
        self.layout: tuple | None = None
        self.peers = [(self.rank + i) % self.n for i in range(1, self.n)]
        self.t_in = {p: torch.cuda.Stream(device=device) for p in self.peers}
        self.t_out = {p: torch.cuda.Stream(device=device) for p in self.peers}
        self.ev_inputs = torch.cuda.Event()
        self.ev_attn: list[torch.cuda.Event] = []
        self.ev_in_done = {p: torch.cuda.Event() for p in self.peers}
        self.ev_out_done = {p: torch.cuda.Event() for p in self.peers}
        _driver()
        _ce_runtime()

    def _flag_bytes(self, chunks: int) -> int:
        # READY[N], OUT[N], IN[chunks][N] at the start of the workspace, so their addresses never hold data.
        return -(-4 * self.n * (chunks + 2) // _ALIGN) * _ALIGN

    def _configure(
        self, seq_local, heads, head_dim, chunks, dtype, *, prefix=0, front=True, tail=0, defer_rendezvous=False
    ) -> None:
        n, rank = self.n, self.rank
        if isinstance(chunks, int):
            if chunks <= 0 or heads % (n * chunks):
                raise ValueError(f"heads {heads} must divide evenly across {n} ranks and {chunks} chunks")
            widths = (heads // (n * chunks),) * chunks
        else:
            widths = tuple(chunks)
        if not widths or min(widths) <= 0 or sum(widths) * n != heads or not 0 <= tail <= seq_local:
            raise ValueError("Head widths must cover each rank's heads and the tail must fit its local query")
        layout = (seq_local, heads, head_dim, widths, dtype, prefix, front, tail)
        if layout == self.layout:
            return
        es = dtype.itemsize
        rows = n * seq_local
        input_rows = (rows, rows + prefix, rows + prefix)
        input_bytes = sum(input_rows) * sum(widths) * head_dim * es
        max_chunks = max(len(widths), self.max_chunks)
        required = self._flag_bytes(max_chunks) + input_bytes + seq_local * heads * head_dim * es
        if self.alloc is None or required > self.capacity or max_chunks > self.max_chunks:
            self._grow(max(required, self.capacity), max_chunks, defer_rendezvous)
        self.s, self.h, self.d, self.k, self.es, self.dtype = seq_local, heads, head_dim, len(widths), es, dtype
        self.widths, self.offsets = widths, tuple(accumulate((0, *widths[:-1])))
        self.tail, self.prefix = tail, prefix
        self.data_off = offset = self._flag_bytes(self.max_chunks)
        self.inputs, self.peer_inputs, self.local_inputs, self.prefix_inputs = [], [], [], []
        # Store offsets relative to a peer's mapped base. Q may place the owner's
        # rows last for the final FA tail. K/V retain native sequence order.
        for c, hc in enumerate(widths):
            parts, local, replicated = [], [], []
            destinations = {peer: [] for peer in self.peers}
            pitch = hc * head_dim * es
            for t, length in enumerate(input_rows):
                end = offset + length * pitch
                part = self.alloc[offset:end].view(dtype).view(1, length, hc, head_dim)
                parts.append(part)
                start = prefix if t and front else 0
                for peer in (*self.peers, rank):
                    row = self._query_row(rank, peer) if tail and c == self.k - 1 and t == 0 else rank
                    first = start + row * seq_local
                    if peer == rank:
                        local.append(part[:, first : first + seq_local])
                    else:
                        destinations[peer].append(offset + first * pitch)
                if t and prefix:
                    replicated.append(part[:, :prefix] if front else part[:, rows:])
                offset = end
            self.inputs.append(parts)
            self.local_inputs.append(local)
            self.peer_inputs.append(destinations)
            self.prefix_inputs.append(replicated)
        self.out_off = offset
        self.output = (
            self.alloc[offset : offset + seq_local * heads * head_dim * es]
            .view(dtype)
            .view(1, seq_local, heads, head_dim)
        )
        self.staging = ()
        self.self_inputs = []
        if not prefix and not tail and len(set(widths)) == 1:
            # Uniform groups can stage all chunks in one strided copy per field.
            hc = widths[0]
            self.self_inputs = [
                view.as_strided((self.k, seq_local, hc, head_dim), (3 * rows * hc * head_dim, *view.stride()[1:]))
                for view in self.local_inputs[0]
            ]
        while len(self.ev_attn) < self.k:
            self.ev_attn.append(torch.cuda.Event())
        self.layout = layout

    def _query_row(self, source, owner):
        if source == owner:
            return self.n - 1
        return owner if source == self.n - 1 else source

    def _grow(self, capacity: int, max_chunks: int, defer_rendezvous=False) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "ulysses_ce_overlap: cannot grow the symmetric-memory workspace during CUDA graph capture; "
                "warm up the maximum request shape before capture"
            )
        # Like a2a_permute's grow-only workspace: nothing may still use the previous allocation when it is released.
        torch.accelerator.synchronize(self.device)
        alloc = symm_mem._SymmetricMemory.empty_strided_p2p((capacity,), (1,), torch.uint8, self.device)
        alloc[: self._flag_bytes(max_chunks)].zero_()
        torch.accelerator.synchronize(self.device)
        self.alloc, self.capacity, self.max_chunks = alloc, capacity, max_chunks
        self.local_base = alloc.data_ptr()
        self.layout = None
        if not defer_rendezvous:
            self.connect()

    def connect(self):
        self.handle = symm_mem.rendezvous(self.alloc, self.group.group_name)
        self.peer_base = tuple(self.handle.buffer_ptrs)

    def _slot(self, kind: str, writer: int, chunk: int = 0) -> int:
        n = self.n
        return 4 * {"ready": writer, "out": n + writer, "in": 2 * n + chunk * n + writer}[kind]

    def _signal(self, stream: torch.cuda.Stream, peer: int, kind: str, chunk: int = 0) -> None:
        address = self.peer_base[peer] + self._slot(kind, self.rank, chunk)
        _check(_driver().cuStreamWriteValue32_v2(stream.cuda_stream, address, 1, 0), "cuStreamWriteValue32")

    def _wait(self, stream: torch.cuda.Stream, writer: int, kind: str, chunk: int = 0) -> None:
        address = self.local_base + self._slot(kind, writer, chunk)
        _check(
            _driver().cuStreamWaitValue32_v2(stream.cuda_stream, address, 1, _WAIT_GEQ),
            "cuStreamWaitValue32",
        )
        # Consume the signal. Replaying a captured graph uses the same immediate
        # value each time. The next READY/IN/OUT round cannot overtake this reset.
        _check(_driver().cuStreamWriteValue32_v2(stream.cuda_stream, address, 0, 0), "cuStreamWriteValue32")

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
        *,
        joint_kv=(),
    ) -> torch.Tensor:
        """(1, s, H, D) local Q/K/V -> (1, s, H, D) attention output (a view into the workspace)."""
        rank, s, h, d, es = self.rank, self.s, self.h, self.d, self.es
        if len(joint_kv) != (2 if self.prefix else 0):
            raise ValueError("Replicated K/V must match the configured prefix")
        expected = (1, s, h, d)
        for x in (query, key, value):
            if tuple(x.shape) != expected or x.dtype != self.dtype or x.stride(-1) != 1 or x.stride(-2) != d:
                raise ValueError(
                    f"expected {expected} {self.dtype} with dense heads, got {tuple(x.shape)} {x.stride()}"
                )
        compute = torch.cuda.current_stream(self.device)
        hpr = h // self.n

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
            for c, hc in enumerate(self.widths):
                width = hc * d * es
                head0 = p * hpr + self.offsets[c]
                for offset, (ptr, pitch) in zip(self.peer_inputs[c][p], sources):
                    self._copy(stream, self.peer_base[p] + offset, width, ptr + head0 * d * es, pitch, width, s)
                self._signal(stream, p, "in", c)
            self.ev_in_done[p].record(stream)

        local_head = rank * hpr
        if self.self_inputs:
            hc = self.widths[0]
            for dst, x in zip(self.self_inputs, (query, key, value)):
                dst.copy_(x[0, :, local_head : local_head + hpr].view(s, self.k, hc, d).permute(1, 0, 2, 3))
        if self.staging:
            pointers = tuple(ptr for ptr, _ in sources) + tuple(x.data_ptr() for x in joint_kv)
            for stage in self.staging:
                stage(pointers)
        pending = []
        for c, hc in enumerate(self.widths):
            head = local_head + self.offsets[c]
            columns = slice(head, head + hc)
            if not self.self_inputs and not self.staging:
                for dst, x in zip(self.local_inputs[c], (query, key, value)):
                    dst.copy_(x[:, :, columns])
                for dst, x in zip(self.prefix_inputs[c], joint_kv, strict=True):
                    dst.copy_(x[:, :, columns])
            for p in self.peers:
                self._wait(compute, p, "in", c)
            q, kk, v = self.inputs[c]
            tail = self.tail if c == self.k - 1 else 0
            out = attn_fn(q[:, :-tail] if tail else q, kk, v)
            if out.stride(-1) != 1 or out.stride(-2) != d:
                out = out.contiguous()
            pending.append(out)
            self.ev_attn[c].record(compute)
            # Receiving IN implies the peer has published READY. OUT needs no
            # second wait on READY, which has a single consumer on the input stream.
            for p in self.peers:
                stream = self.t_out[p]
                stream.wait_event(self.ev_attn[c])
                row = self._query_row(p, rank) if tail else p
                pitch = out.stride(1) * es
                self._copy(
                    stream,
                    self.peer_base[p] + self.out_off + head * d * es,
                    h * d * es,
                    out.data_ptr() + row * s * pitch,
                    pitch,
                    hc * d * es,
                    s,
                )
            own = self.output[:, :, columns]
            row = (self.n - 1 if tail else rank) * s
            if s > tail:
                own[:, : s - tail].copy_(out[:, row : row + s - tail])
            if tail:
                own[:, -tail:].copy_(attn_fn(q[:, -tail:], kk, v))
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
# Whether the overlap is bit-identical to the standard exchange, per (group, attention kernel, shape, dtype, chunks).
# Shared by every layer; all ranks run the layers in the same order, so they fill it identically.
VERDICTS: dict[tuple, bool] = {}


def run_ulysses_ce_overlap(pg, query, key, value, attn_fn, *, chunks: int) -> torch.Tensor:
    """One runner per (group, device), grown to the largest shape seen; growth is collective and happens in lockstep
    on all ranks."""
    _, s, h, d = query.shape
    cache_key = (pg.group_name, query.device)
    runner = _RUNNERS.get(cache_key)
    if runner is None:
        runner = _RUNNERS[cache_key] = CEOverlapUlysses(pg, device=query.device)
    runner._configure(s, h, d, chunks, query.dtype)
    return runner.run(query, key, value, attn_fn)


def clear_ulysses_ce_overlap_runners() -> None:
    """Release the runners' symmetric-memory workspaces during worker shutdown (called by
    ``a2a_permute.clear_a2a_permute_workspaces``)."""
    VERDICTS.clear()
    if not _RUNNERS:
        return
    for device in {runner.device for runner in _RUNNERS.values()}:
        torch.accelerator.synchronize(device)
    _RUNNERS.clear()
