# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses attention with the all-to-all overlapped against attention compute.

Plain strict Ulysses runs the Q/K/V all-to-all, attention and the output all-to-all back to back on the
compute stream, so without a fast interconnect the communication is fully exposed. With
``ulysses_overlap_chunks=k`` this path:

1. splits the rank-local heads into ``k`` groups. Chunk ``c`` of rank ``r`` owns the global heads
   ``c*P*hc + r*hc + [0, hc)`` (P = Ulysses degree, hc = heads per rank per chunk). Every rank exchanges
   chunk ``c`` on a side communication stream while the compute stream runs attention for the chunks
   already received, and returns each chunk's output while later chunks compute. Heads are independent,
   so the head-to-rank assignment differs from plain Ulysses without changing the result.
2. with ``ulysses_overlap_int8`` sends Q/K/V as int8 with one fp32 scale per (token, head) row, packed into
   one uint8 buffer per chunk: 132 instead of 256 bytes per bf16 row. The receiver dequantizes to the
   input dtype before attention.
3. packs and unpacks with Triton kernels that read the source layout through strides and write the
   receive side directly in the attention (NHD) layout, so no permute copies sit on the critical path.
   The output scatter into the final (B, S_local, H, D) layout runs on the side stream.

Ordering: every buffer is allocated on the compute stream before the side stream starts, the side stream
waits for an event recorded after the inputs were produced, and the compute stream waits for the side
stream's final event before returning, so no buffer is reused while the other stream still reads it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

ROW_BYTES_EXTRA = 4  # one fp32 scale per (token, head) row
_BLOCK_ROWS = 32  # (token, head) rows per Triton program


@triton.jit
def _quant_pack_kernel(
    x_ptr,
    x_stride_s,
    x_stride_h,  # source (seq_len, heads, head_dim), last dim contiguous
    out_i8_ptr,
    out_f32_ptr,  # packed buffer viewed as int8 / fp32
    n_rows,
    seq_len,
    hc,
    world_size,
    chunk,
    t,
    n_t,  # rows: (r, s, j) for tensor t of n_t
    head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    mask = rows < n_rows
    j = rows % hc
    s = (rows // hc) % seq_len
    r = rows // (hc * seq_len)
    head = chunk * world_size * hc + r * hc + j
    offs = tl.arange(0, head_dim)
    src = x_ptr + (s * x_stride_s + head * x_stride_h)[:, None] + offs[None, :]
    x = tl.load(src, mask=mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax, 1e-8) / 127.0
    y = x / scale[:, None]
    q = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    row = ((r * seq_len + s) * n_t + t) * hc + j  # buffer layout (P, seq_len, n_t, hc, row_bytes)
    tl.store(out_i8_ptr + (row * row_bytes)[:, None] + offs[None, :], q, mask=mask[:, None])
    tl.store(out_f32_ptr + row * (row_bytes // 4) + head_dim // 4, scale, mask=mask)


@triton.jit
def _dequant_unpack_kernel(
    in_i8_ptr,
    in_f32_ptr,  # received buffer (P, seq_len, n_t, hc, row_bytes)
    out_ptr,
    out_stride_s,
    out_stride_h,  # destination (P * seq_len, hc, head_dim)
    n_rows,
    seq_len,
    hc,
    t,
    n_t,
    head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    mask = rows < n_rows
    j = rows % hc
    s = (rows // hc) % seq_len
    p = rows // (hc * seq_len)
    row = ((p * seq_len + s) * n_t + t) * hc + j
    offs = tl.arange(0, head_dim)
    q = tl.load(in_i8_ptr + (row * row_bytes)[:, None] + offs[None, :], mask=mask[:, None], other=0)
    scale = tl.load(in_f32_ptr + row * (row_bytes // 4) + head_dim // 4, mask=mask, other=0.0)
    dst = out_ptr + ((p * seq_len + s) * out_stride_s + j * out_stride_h)[:, None] + offs[None, :]
    tl.store(dst, (q.to(tl.float32) * scale[:, None]).to(out_ptr.dtype.element_ty), mask=mask[:, None])


def pack_int8(x: torch.Tensor, buf: torch.Tensor, *, chunk: int, t: int, n_t: int, world_size: int, hc: int) -> None:
    """Quantize tensor ``t`` of ``n_t`` from ``x`` (S, H, D) into ``buf`` (P, S, n_t, hc, D + 4) for ``chunk``."""
    seq_len, _, head_dim = x.shape
    rows = world_size * seq_len * hc
    _quant_pack_kernel[(triton.cdiv(rows, _BLOCK_ROWS),)](
        x,
        x.stride(0),
        x.stride(1),
        buf.view(torch.int8),
        buf.view(torch.float32),
        rows,
        seq_len,
        hc,
        world_size,
        chunk,
        t,
        n_t,
        head_dim=head_dim,
        row_bytes=head_dim + ROW_BYTES_EXTRA,
        block_rows=_BLOCK_ROWS,
    )


def unpack_int8(buf: torch.Tensor, out: torch.Tensor, *, t: int, n_t: int) -> None:
    """Dequantize tensor ``t`` of ``buf`` (P, S, n_t, hc, D + 4) into ``out`` (P * S, hc, D)."""
    world_size, seq_len, _, hc, row_bytes = buf.shape
    rows = world_size * seq_len * hc
    _dequant_unpack_kernel[(triton.cdiv(rows, _BLOCK_ROWS),)](
        buf.view(torch.int8),
        buf.view(torch.float32),
        out,
        out.stride(0),
        out.stride(1),
        rows,
        seq_len,
        hc,
        t,
        n_t,
        head_dim=row_bytes - ROW_BYTES_EXTRA,
        row_bytes=row_bytes,
        block_rows=_BLOCK_ROWS,
    )


class UlyssesOverlapRunner:
    """Per-device side stream; the attention callable runs on the current (compute) stream."""

    def __init__(self) -> None:
        self._streams: dict[torch.device, torch.cuda.Stream] = {}

    def _comm_stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self._streams.get(device)
        if s is None:
            s = self._streams[device] = torch.cuda.Stream(device=device)
        return s

    @torch.compiler.disable
    def run(
        self,
        pg: dist.ProcessGroup,
        query: torch.Tensor,  # (1, S_local, H, D)
        key: torch.Tensor,
        value: torch.Tensor,
        attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        chunks: int,
        int8: bool,
    ) -> torch.Tensor:
        P = dist.get_world_size(pg)
        _, S, H, D = query.shape
        assert query.shape[0] == 1 and key.shape == query.shape and value.shape == query.shape
        assert H % (P * chunks) == 0, f"heads {H} not divisible by P*chunks {P * chunks}"
        hc = H // (P * chunks)
        dev, dtype = query.device, query.dtype
        compute = torch.cuda.current_stream(dev)
        comm = self._comm_stream(dev)
        srcs = [x[0] for x in (query, key, value)]  # (S, H, D) views
        for x in srcs:
            assert x.stride(-1) == 1, "last dim must be contiguous"

        # Allocate everything on the compute stream, before any side-stream work (see ordering note).
        row = D + ROW_BYTES_EXTRA
        if int8:
            send = [torch.empty((P, S, 3, hc, row), dtype=torch.uint8, device=dev) for _ in range(chunks)]
            recv = [torch.empty_like(b) for b in send]
        else:
            send = [[torch.empty((P, S, hc, D), dtype=dtype, device=dev) for _ in range(3)] for _ in range(chunks)]
            recv = [[torch.empty_like(b) for b in grp] for grp in send]
        if int8:
            qkv_in = [
                [torch.empty((1, P * S, hc, D), dtype=dtype, device=dev) for _ in range(3)] for _ in range(chunks)
            ]
        else:  # the receive buffer already is the contiguous (1, P*S, hc, D) NHD layout
            qkv_in = [[b.view(1, P * S, hc, D) for b in grp] for grp in recv]
        out_send: list[torch.Tensor] = []  # keeps each chunk's output alive until the side stream is done
        out_recv = [torch.empty((P, S, hc, D), dtype=dtype, device=dev) for _ in range(chunks)]
        final = torch.empty((1, S, H, D), dtype=dtype, device=dev)
        final_v = final[0].view(S, chunks, P, hc, D)

        start = torch.cuda.Event()
        start.record(compute)
        in_ready = [torch.cuda.Event() for _ in range(chunks)]
        attn_done = [torch.cuda.Event() for _ in range(chunks)]
        all_done = torch.cuda.Event()

        # ---- input exchange: all chunks queued back to back on the side stream
        with torch.cuda.stream(comm):
            comm.wait_event(start)
            for c in range(chunks):
                if int8:
                    for t, x in enumerate(srcs):
                        pack_int8(x, send[c], chunk=c, t=t, n_t=3, world_size=P, hc=hc)
                    dist.all_to_all_single(recv[c], send[c], group=pg)
                    for t in range(3):
                        unpack_int8(recv[c], qkv_in[c][t][0], t=t, n_t=3)
                else:
                    for t, x in enumerate(srcs):
                        send[c][t].copy_(x.view(S, chunks, P, hc, D)[:, c].permute(1, 0, 2, 3))
                        dist.all_to_all_single(recv[c][t], send[c][t], group=pg)
                in_ready[c].record(comm)

        # ---- attention on the compute stream, output exchange behind it on the side stream
        for c in range(chunks):
            compute.wait_event(in_ready[c])
            out = attn_fn(*qkv_in[c]).contiguous()  # (1, P*S, hc, D)
            out_send.append(out)
            attn_done[c].record(compute)
            with torch.cuda.stream(comm):
                comm.wait_event(attn_done[c])
                dist.all_to_all_single(out_recv[c], out.view(P, S, hc, D), group=pg)
                # out_recv[c][r] holds heads c*P*hc + r*hc + [0, hc) of this rank's sequence shard
                final_v[:, c].copy_(out_recv[c].permute(1, 0, 2, 3))
        with torch.cuda.stream(comm):
            all_done.record(comm)
        compute.wait_event(all_done)
        # out_send buffers were allocated on the compute stream and read on the side stream;
        # the wait above orders their reuse after those reads.
        return final


_RUNNER = UlyssesOverlapRunner()


def run_ulysses_overlap(pg, query, key, value, attn_fn, *, chunks: int, int8: bool) -> torch.Tensor:
    return _RUNNER.run(pg, query, key, value, attn_fn, chunks=chunks, int8=int8)
