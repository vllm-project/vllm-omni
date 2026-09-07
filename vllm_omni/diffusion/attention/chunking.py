# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backend-neutral scheduling for chunked attention calls.

Splitting one wide attention call into several narrower ones along the query
sequence and/or head axes is a *scheduling* decision — which ranges to call,
in what order, and how to reassemble the outputs — that does not depend on
the attention operator underneath. This module owns that decision so every
backend can share it:

* :class:`AttnChunkingOptions` — static user-facing knobs (parsed once per
  attention layer from the diffusion config).
* :class:`ChunkCall` — one scheduled dispatch: a query-row range x head range.
* :func:`build_chunk_plan` — pure function turning shapes + options into the
  call list. No tensors, no side effects: trivially unit-testable.
* :func:`run_chunked` — generic executor for backends without per-call
  bookkeeping (a ``make_call`` callable is invoked per scheduled call and the
  outputs are reassembled).

Backend adapters consume the plan however fits their operator. The NPU FIA
fp8 path (``vllm_omni/platforms/npu/quant/kv_quant_npu.py``) quantizes once
and drives ``fused_infer_attention_score_v2`` per call from the plan, keeping
its own loop because per-call dequant-scale slices are FIA-specific; it takes
the plan as a duck-typed sequence so this module stays importable both inside
and outside the ``vllm_omni`` package.

Chunking is a power/latency mitigation for specific machine types (each
compute burst stays inside the NPU power envelope; per-call K/V slices
become L2-resident), not a general win: it is opt-in via CLI flags and inert
by default.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "DEFAULT_HEAD_CHUNK_MIN_KV",
    "AttnChunkingOptions",
    "ChunkCall",
    "build_chunk_plan",
    "run_chunked",
    "validate_options",
]

# Default L2-residency gate for head chunking: below this valid kv length the
# K/V slice is L2-resident already and splitting heads only adds per-call
# overhead. Tuned on Ascend 950PR long-video workloads (kv >= 50k tokens).
DEFAULT_HEAD_CHUNK_MIN_KV = 50000


@dataclass(frozen=True)
class AttnChunkingOptions:
    """Static chunking knobs shared by every attention backend.

    Args:
        q_chunk: Split the query sequence into up to N row chunks per call
            list (K/V kept whole). 1 = off.
        head_chunk: Call the operator on H heads at a time. 0 = off.
            Requires MHA (``num_kv_heads == num_heads``): grouped KV heads
            would need re-grouping per slice. GQA layers run with full heads.
        head_chunk_min_kv: Head chunking only engages when the valid kv
            length reaches this threshold — the L2-residency gate. 0 =
            always engage (when ``head_chunk`` is set).
    """

    q_chunk: int = 1
    head_chunk: int = 0
    head_chunk_min_kv: int = DEFAULT_HEAD_CHUNK_MIN_KV

    @property
    def active(self) -> bool:
        """True when any chunking is requested (non-default knobs)."""
        return self.q_chunk > 1 or self.head_chunk > 0


def validate_options(options: AttnChunkingOptions) -> None:
    """Range-check the knobs; raise ValueError on impossible values.

    Config surfaces call this at post-init so a typo (e.g. ``q_chunk=0``)
    fails fast instead of silently disabling chunking at plan time.
    """
    if options.q_chunk < 1:
        raise ValueError(f"q_chunk must be >= 1, got {options.q_chunk}")
    if options.head_chunk < 0:
        raise ValueError(f"head_chunk must be >= 0, got {options.head_chunk}")
    if options.head_chunk_min_kv < 0:
        raise ValueError(f"head_chunk_min_kv must be >= 0, got {options.head_chunk_min_kv}")


@dataclass(frozen=True)
class ChunkCall:
    """One scheduled attention dispatch.

    Query rows ``[row0, row1)`` x heads ``[h0, h1)``. Plans are emitted
    q-chunk-major / head-minor: consecutive calls sharing a row range belong
    to the same q chunk, and their outputs concatenate on the head axis.
    """

    row0: int
    row1: int
    h0: int
    h1: int


def _row_chunk_bounds(seq_len: int, n_chunks: int, align: int) -> list[tuple[int, int]]:
    """``[start, end)`` query-row chunk boundaries covering ``[0, seq_len)``.

    Boundaries start on ``align``-multiples so callers whose kernels attach
    per-row-block metadata (e.g. block-quant dequant scales) get exact block
    ranges per chunk; only the last chunk may be ragged. At most ``n_chunks``
    chunks are emitted — fewer when there are not enough whole blocks.
    """
    if n_chunks <= 1 or seq_len <= align:
        return [(0, seq_len)]
    chunk = -(-seq_len // (n_chunks * align)) * align
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(seq_len, start + chunk)
        bounds.append((start, end))
        start = end
    return bounds


def _head_bounds(num_heads: int, head_chunk: int) -> list[tuple[int, int]]:
    """``[start, end)`` head slice boundaries covering ``[0, num_heads)``."""
    if head_chunk <= 0 or head_chunk >= num_heads:
        return [(0, num_heads)]
    return [(h0, min(h0 + head_chunk, num_heads)) for h0 in range(0, num_heads, head_chunk)]


def build_chunk_plan(
    *,
    seq_len: int,
    num_heads: int,
    options: AttnChunkingOptions | None = None,
    num_kv_heads: int | None = None,
    kv_len: int | None = None,
    row_align: int = 1,
) -> list[ChunkCall]:
    """Schedule the attention calls for one operator invocation.

    Pure function over shapes and knobs: returns the q-chunk-major /
    head-minor call list (cartesian product of row chunks and head slices).
    ``options=None`` (or all-default) yields the single-call plan, so the
    unchunked path is the degenerate case of the same schedule.

    Args:
        seq_len: Real query sequence length. Chunks cover exactly
            ``[0, seq_len)``; operator-side padding rows beyond it are never
            scheduled (their outputs are dropped anyway).
        num_heads: Query head count.
        options: Chunking knobs; ``None`` = single call.
        num_kv_heads: KV head count. Head chunking collapses to full heads
            for GQA (``num_kv_heads != num_heads``); the caller detects the
            collapse (first call spans all heads) and warns.
        kv_len: Valid kv length for this invocation, used by the
            ``head_chunk_min_kv`` L2-residency gate.
        row_align: Row-block alignment for q-chunk boundaries. 1 for plain
            backends; backends with per-row-block metadata (FIA block-quant
            scales) pass their block size so per-chunk metadata is an exact
            slice of the full-length metadata.

    Returns:
        Non-empty call list covering all rows and heads exactly once.
    """
    if seq_len < 1:
        raise ValueError(f"build_chunk_plan: seq_len must be >= 1, got {seq_len}")
    if num_heads < 1:
        raise ValueError(f"build_chunk_plan: num_heads must be >= 1, got {num_heads}")
    if row_align < 1:
        raise ValueError(f"build_chunk_plan: row_align must be >= 1, got {row_align}")
    if options is not None:
        validate_options(options)

    row_bounds = _row_chunk_bounds(seq_len, options.q_chunk if options is not None else 1, row_align)

    head_chunk = 0
    if options is not None and options.head_chunk > 0:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            head_chunk = 0  # GQA: grouped KV heads cannot be split per slice; run full heads.
        elif kv_len is not None and kv_len < options.head_chunk_min_kv:
            head_chunk = 0  # Short KV is L2-resident; splitting only adds per-call overhead.
        else:
            head_chunk = options.head_chunk
    head_bounds = _head_bounds(num_heads, head_chunk)

    return [ChunkCall(r0, r1, h0, h1) for (r0, r1) in row_bounds for (h0, h1) in head_bounds]


def run_chunked(
    plan: Sequence[ChunkCall],
    *,
    seq_dim: int,
    head_dim: int,
    make_call: Callable[[ChunkCall], torch.Tensor],
    chunk_callback: Callable[[torch.Tensor, ChunkCall], None] | None = None,
) -> torch.Tensor | None:
    """Execute a chunk plan for backends without per-call bookkeeping.

    Groups consecutive calls sharing a row range (the plan's q-chunk-major
    ordering), concatenates each group's outputs on the head axis, then
    concatenates the groups on the sequence axis. ``make_call`` must return
    one call's output already in caller layout, with the operator's own
    padding trimmed.

    With ``chunk_callback`` set, each q chunk's head-merged output is handed
    to ``callback(out_chunk, first_call)`` instead of being reassembled and
    None is returned — the caller consumes chunks itself (e.g. interleaving
    low-power communication between compute bursts).

    Args:
        plan: Call list from :func:`build_chunk_plan` (q-chunk-major order).
        seq_dim: Sequence axis of the output in caller layout.
        head_dim: Head axis of the output in caller layout.
        make_call: ``fn(call) -> out`` invoking the backend operator once.
        chunk_callback: Optional per-q-chunk consumer; see above.

    Returns:
        Reassembled output covering all scheduled rows/heads; None when
        ``chunk_callback`` consumed the chunks.
    """
    if not plan:
        raise ValueError("run_chunked: plan must not be empty")

    out_parts: list[torch.Tensor] = []
    head_parts: list[torch.Tensor] = []
    cur: ChunkCall | None = None

    def _flush_rows() -> None:
        if not head_parts:
            return
        merged = head_parts[0] if len(head_parts) == 1 else torch.cat(head_parts, dim=head_dim)
        head_parts.clear()
        if chunk_callback is not None:
            assert cur is not None  # head_parts non-empty implies a group started
            chunk_callback(merged, cur)
        else:
            out_parts.append(merged)

    for call in plan:
        if cur is not None and (call.row0, call.row1) != (cur.row0, cur.row1):
            _flush_rows()
        cur = call
        head_parts.append(make_call(call))
    _flush_rows()

    if chunk_callback is not None:
        return None
    if len(out_parts) == 1:
        return out_parts[0]
    return torch.cat(out_parts, dim=seq_dim)
