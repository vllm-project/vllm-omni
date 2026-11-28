# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-chunked strict Ulysses with HCCL communication/compute overlap on Ascend NPU.

``ulysses_ce_overlap`` hides the exchange behind attention with NVIDIA copy engines and stream memory
operations. Ascend has neither, so the same pipeline is built from what it does provide: HCCL
collectives that run asynchronously while the compute stream keeps working, plus NPU stream events for
the kernel-to-collective direction.

The rank-local heads are cut into ``chunks`` groups, each exchanged with one ``all_to_all_single`` per
tensor. Chunk ``c``'s attention waits for chunk ``c``'s input exchange, and chunk ``c``'s output
exchange is issued right after chunk ``c``'s attention, so the output leaves while chunk ``c+1``
computes. Every head is still computed by the same kernel over the same K/V, so the result is
bit-identical to plain Ulysses whenever the kernel's per-head result does not depend on how many heads
one call covers; the caller verifies that per shape and kernel.

Ordering is asymmetric because the two directions have different primitives. An event recorded on the
compute stream after a kernel is faithfully observed by the side stream, so attention completion is
propagated with ``Event``. HCCL work submitted with ``async_op=True`` is *not* observable that way -- a
stream event recorded after the submission does not wait for the transfer -- so the communication
completion is taken from the returned ``Work`` objects with ``wait()``. That blocks the host, not the
device: the next chunk's collective is already submitted, so the device stays busy while the host
waits, and attention is enqueued immediately afterwards.

All staging buffers and the output are owned by the runner and reused across layers and calls. Reuse is
safe because each call starts by ordering the side stream behind the compute stream, which already
contains every earlier read of them.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist


class NPUOverlapUlysses:
    """One runner per (group, device); ``run`` once per attention layer from one compute stream."""

    def __init__(self, group: dist.ProcessGroup, *, device: torch.device) -> None:
        self.group = group
        self.n = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = torch.device(device)
        self.stream = torch.npu.Stream(device=self.device)
        self.ev_attn: list[torch.npu.Event] = []
        self.layout: tuple | None = None
        self.fwd_in: list[torch.Tensor] = []
        self.fwd_out: list[torch.Tensor] = []
        self.rev_staging: list[torch.Tensor] = []
        self.output: torch.Tensor | None = None

    def _configure(self, seq_local: int, heads: int, head_dim: int, chunks: int, dtype: torch.dtype) -> None:
        layout = (seq_local, heads, head_dim, chunks, dtype)
        if layout == self.layout:
            return
        n = self.n
        if heads % (n * chunks):
            raise ValueError(f"heads {heads} not divisible by world size {n} x chunks {chunks}")
        # Preserve the previous buffers until no queued work can still touch them.
        torch.npu.synchronize(self.device)
        self.s, self.h, self.d, self.k = seq_local, heads, head_dim, chunks
        self.hpr, self.hc = heads // n, heads // (n * chunks)
        self.dtype = dtype
        # Forward wire layout: [chunk] x {q,k,v}, each [dest rank, local seq, batch, chunk heads, D].
        self.fwd_in = [
            torch.empty((3, n, seq_local, 1, self.hc, head_dim), dtype=dtype, device=self.device) for _ in range(chunks)
        ]
        self.fwd_out = [
            torch.empty((3, n, seq_local, 1, self.hc, head_dim), dtype=dtype, device=self.device) for _ in range(chunks)
        ]
        # Reverse wire layout: [chunk] [source rank, local seq, batch, chunk heads, D].
        self.rev_staging = [
            torch.empty((n, seq_local, 1, self.hc, head_dim), dtype=dtype, device=self.device) for _ in range(chunks)
        ]
        self.output = torch.empty((1, seq_local, heads, head_dim), dtype=dtype, device=self.device)
        while len(self.ev_attn) < chunks:
            self.ev_attn.append(torch.npu.Event())
        self.layout = layout

    @torch.compiler.disable
    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """(1, s, H, D) local Q/K/V -> (1, s, H, D) attention output (a view into the workspace)."""
        s, h, d, k = self.s, self.h, self.d, self.k
        n, hpr, hc = self.n, self.hpr, self.hc
        expected = (1, s, h, d)
        for x in (query, key, value):
            if tuple(x.shape) != expected or x.dtype != self.dtype or x.stride(-1) != 1 or x.stride(-2) != d:
                raise ValueError(
                    f"expected {expected} {self.dtype} with dense heads, got {tuple(x.shape)} {x.stride()}"
                )
            # The head block only has to be contiguous (stride(-2)=d, stride(-1)=1) for
            # ``.view(..., n, hpr, d)``; a slice of a fused QKV projection keeps a larger
            # stride(1) and is still copied correctly.

        compute = torch.npu.current_stream(self.device)
        side = self.stream
        # The inputs were produced on the compute stream; everything queued there also covers the
        # previous call's reads of our staging buffers.
        side.wait_stream(compute)

        # Forward: every chunk's head slice is permuted and submitted; the next chunk's collective is
        # already on the side stream while the host waits for this one.
        fwd_works: list[list] = []
        with torch.npu.stream(side):
            for c in range(k):
                src = (query, key, value)
                works = []
                for t in range(3):
                    # [1, s, H, D] -> this chunk's [dest rank, s, 1, hc, D] block.
                    self.fwd_in[c][t].copy_(
                        src[t].view(1, s, n, hpr, d)[:, :, :, c * hc : (c + 1) * hc, :].permute(2, 1, 0, 3, 4)
                    )
                    works.append(
                        dist.all_to_all_single(self.fwd_out[c][t], self.fwd_in[c][t], group=self.group, async_op=True)
                    )
                fwd_works.append(works)

        # Attention chunk c waits only for its own exchange; its output exchange is submitted right after.
        rev_works: list = []
        head_blocks = self.output.view(1, s, n, hpr, d)
        for c in range(k):
            for work in fwd_works[c]:
                work.wait()
            out = attn_fn(
                self.fwd_out[c][0].view(1, n * s, hc, d),
                self.fwd_out[c][1].view(1, n * s, hc, d),
                self.fwd_out[c][2].view(1, n * s, hc, d),
            )
            if out.stride(-1) != 1 or out.stride(-2) != d or out.stride(1) != hc * d:
                out = out.contiguous()
            self.ev_attn[c].record(compute)
            side.wait_event(self.ev_attn[c])
            with torch.npu.stream(side):
                rev_works.append(
                    dist.all_to_all_single(
                        self.rev_staging[c], out.view(n, s, 1, hc, d), group=self.group, async_op=True
                    )
                )

        # Chunk c gathers the same head slice from every rank into this rank's rows.
        for c in range(k):
            rev_works[c].wait()
            head_blocks[0, :, :, c * hc : (c + 1) * hc, :].copy_(
                self.rev_staging[c].view(n, s, hc, d).permute(1, 0, 2, 3)
            )
        return self.output


_RUNNERS: dict[tuple, NPUOverlapUlysses] = {}


def run_ulysses_overlap_npu(
    pg: dist.ProcessGroup,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    chunks: int,
) -> torch.Tensor:
    _, s, h, d = query.shape
    cache_key = (pg.group_name, query.device)
    runner = _RUNNERS.get(cache_key)
    if runner is None:
        runner = _RUNNERS[cache_key] = NPUOverlapUlysses(pg, device=query.device)
    runner._configure(s, h, d, chunks, query.dtype)
    return runner.run(query, key, value, attn_fn)


def clear_ulysses_overlap_npu_runners() -> None:
    """Release the runners' staging buffers during worker shutdown."""
    if not _RUNNERS:
        return
    for device in {runner.device for runner in _RUNNERS.values()}:
        torch.npu.synchronize(device)
    _RUNNERS.clear()
