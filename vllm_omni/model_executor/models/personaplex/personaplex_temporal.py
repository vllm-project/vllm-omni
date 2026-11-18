# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ring-KV and RoPE primitives for PersonaPlex streaming transformers.

The Mimi codec's encoder/decoder transformers (``personaplex_mimi.py``) run as
stateful per-frame steppers over a sliding context window. This module holds
the pieces they share, mirroring the Moshi streaming transformer (MIT; the
reference for the ``nvidia/personaplex-7b-v1`` checkpoint) op for op:

- interleaved (non-neox) RoPE applied at the ABSOLUTE offset in fp32
- a fixed-capacity ring KV whose position table drives the relative-position
  causal mask with ``context``-window truncation

All shapes are static so a whole step is CUDA-graphable, and every row carries
its own offset so batched duplex serving can recycle one slot (``reset_row``)
while the others keep streaming.
"""

from __future__ import annotations

import math

import torch


def _rope_tables(offset: torch.Tensor, seq_len: int, head_dim: int, max_period: float = 10_000.0):
    """Per-row RoPE ``cos``/``sin`` tables for positions ``offset + [0, seq_len)``.

    ``offset`` is the per-row ``[B]`` absolute position. The tables depend only
    on ``(offset, seq_len, head_dim)``, so a streaming stack builds them once per
    ``step()`` and hands the same pair to every layer. Returns ``(rotr, roti)``,
    each ``[B, 1, seq_len, head_dim // 2]`` fp32.
    """
    D = head_dim
    ds = torch.arange(D // 2, device=offset.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(seq_len, device=offset.device, dtype=torch.float32)
    ts = ts.view(-1, 1, seq_len, 1)
    return torch.cos(freqs * ts), torch.sin(freqs * ts)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor):
    """Interleaved RoPE with precomputed ``_rope_tables``, fp32 rotation (moshi ``apply_rope``)."""
    D = q.shape[-1]
    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


def _ringkv_positions(offset: torch.Tensor, seq_len: int, capacity: int, active: torch.Tensor):
    """Ring write indexes and pre-mask absolute positions for one ``complete()``.

    ``offset`` is the per-row ``[B]`` end offset *before* this write; inactive
    rows do not advance. Every ``_RingKV`` of a stack ticks in lockstep with the
    stack's ``_offset``, so a stack builds these once per ``step()`` and shares
    them across layers. Returns ``indexes [B, seq_len]`` and ``positions [B, capacity]``
    (``-1`` for never-written cells). The per-row ``start_offset`` mask is NOT
    applied here: it is per-ring elastic-recycle state and stays in ``complete()``.
    """
    indexes = (
        torch.arange(seq_len, device=offset.device, dtype=offset.dtype).view(1, -1) + offset.view(-1, 1)
    ) % capacity
    end_offset = (offset + seq_len * active.to(offset.dtype)).view(-1, 1)
    idx = torch.arange(capacity, device=offset.device, dtype=torch.long)
    invalid = idx.view(1, -1) >= end_offset
    end_index = end_offset % capacity
    delta = idx.view(1, -1) - end_index
    # `delta <= 0` (not `< 0`) is moshi's exact convention (transformer.py
    # RingKVCache.complete). It labels the just-past-newest slot as the future
    # write position, so once the ring has wrapped the single oldest in-window
    # cell is excluded and the effective window is capacity-1. This is inherited
    # verbatim from the reference and only shows after the window fills (Helium
    # ~3000 frames / 240 s); it costs one frame out of thousands. Do NOT change
    # this to `< 0`: it would diverge from moshi and break greedy bit-parity.
    positions = torch.where(delta <= 0, end_offset + delta, end_offset + delta - capacity)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)
    return indexes, positions


class _RingKV:
    """Fixed-capacity KV ring with per-row valid-window start (elastic recycle).

    Uniform writes (all rows tick together) + a per-row mask via absolute
    positions; with ``start_offset == 0`` everywhere this is the plain streaming
    cache. All shapes static -> CUDA-graph safe.
    """

    def __init__(self, batch_size: int, num_heads: int, dim_per_head: int, capacity: int, device, dtype):
        self.capacity = capacity
        self.cache = torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=dtype)
        self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.start_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset(self) -> None:
        self.end_offset.zero_()
        self.start_offset.zero_()

    def reset_row(self, b: int) -> None:
        # Restart row b at position 0. Every cached entry of the row sits at or
        # past the new end offset, so all of them are masked until overwritten.
        self.end_offset[b] = 0
        self.start_offset[b] = 0

    def complete(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        active: torch.Tensor,
        indexes: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ):
        """Write ``k``/``v`` and return ``(keys, values, positions [B, capacity])``.

        ``indexes``/``positions`` are this write's ``_ringkv_positions`` tables;
        streaming stacks pass the copy they hoisted once per ``step()``. When
        omitted (standalone use) they are built from this ring's ``end_offset``.
        """
        B, H, T, D = k.shape
        if indexes is None or positions is None:
            indexes, positions = _ringkv_positions(self.end_offset, T, self.capacity, active)
        idx4 = indexes.view(B, 1, T, 1).expand(-1, H, -1, D)
        # Keep inactive rows completely inert. Once the ring is full, the
        # physical future-write slot is also addressable by the position mask;
        # writing it for an inactive row would therefore leak padded data into
        # its next attention step.
        active_view = active.view(B, 1, 1, 1)
        old_k = self.cache[0].gather(2, idx4)
        old_v = self.cache[1].gather(2, idx4)
        k = torch.where(active_view, k, old_k)
        v = torch.where(active_view, v, old_v)
        self.cache[0].scatter_(2, idx4, k)
        self.cache[1].scatter_(2, idx4, v)
        self.end_offset.add_(T * active.to(self.end_offset.dtype))

        below = positions < self.start_offset.view(-1, 1)  # [B, capacity]
        positions = torch.where(below, torch.full_like(positions, -1), positions)
        return self.cache[0], self.cache[1], positions
