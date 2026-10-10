# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Block-sparse attention for H3's VSA routing on b12x (SM120/SM121).

Upstream's H3 VSA routing computes a per-(batch, head, q block) selection in PyTorch
(``build_prefix_dense_block_map`` over mean-pooled q/k scores) and hands it to the
FastVideo provider. b12x's block-list attention consumes the same 64-token block
granularity, so the selection needs no re-derivation -- only a layout change, because
b12x's list is one CSR for a packed batch while the selection is per head.

The heads are therefore passed as the varlen batch: q/k/v are laid out head-major
``[H*S, 1, D]`` with ``cu_seqlens = H`` segments of length ``S``, and the CSR carries one
row per (head, q tile) using ``per_segment_tiles=True``. Blocks are ``tile_m=64`` so one
b12x q tile is exactly one VSA q block; ``S = logical_blocks * 64`` is tile-aligned, which
is what per-segment indexing requires.
"""

from __future__ import annotations

import os

import torch

TILE = 64
_CAPACITY_STEP = 8192  # plan capacity bucket: the live list length is a launch scalar
_PLAN_CACHE: dict[tuple, object] = {}
_SCRATCH_CACHE: dict[tuple, torch.Tensor] = {}
_MAX_CACHED = 16


def enabled() -> bool:
    return os.environ.get("VLLM_OMNI_BLOCK_SPARSE_B12X", "0") == "1"


def available() -> bool:
    try:
        from b12x.attention import varlen  # noqa: F401
    except Exception:
        return False
    return True


def _capacity(n: int) -> int:
    return max(_CAPACITY_STEP, ((int(n) + _CAPACITY_STEP - 1) // _CAPACITY_STEP) * _CAPACITY_STEP)


def _csr_from_block_map(block_map: torch.Tensor, logical_blocks: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(head, q tile) CSR from a [B, H, L, L] boolean block map.

    The kernel walks ``mBlockOffsets`` by the GLOBAL q tile of the segment, so rows are
    laid out segment-major: head 0's L tiles, then head 1's, and so on.
    """
    bm = block_map[0, :, :logical_blocks, :logical_blocks]  # [H, L, L]
    nz = bm.nonzero()  # row-major (tile, block)
    counts = bm.sum(dim=-1)  # [H, L]
    indices = nz[:, 2].to(torch.int32)  # (head, q tile, K block) -> K block id
    offsets = counts.reshape(-1).cumsum(0, dtype=torch.int64)
    offsets = torch.cat([offsets.new_zeros(1), offsets]).to(torch.int32)
    return indices.contiguous(), offsets.contiguous()


def _plan_for(heads, seq_len, head_dim, dtype, device, num_tiles, cap):
    from b12x.attention import varlen
    from b12x.attention.varlen import VarlenAttentionConfig

    key = (heads, seq_len, head_dim, dtype, int(device.index or 0), num_tiles, cap)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        placeholder = torch.empty((heads * seq_len, 1, head_dim), dtype=dtype, device=device)
        cu = torch.arange(0, (heads + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        plan = varlen.plan(
            placeholder,
            placeholder,
            placeholder,
            cu,
            cu,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=False,
            block_sparse=True,
            num_q_tiles=num_tiles,
            total_blocks_cap=cap,
            per_segment_tiles=True,
            override=VarlenAttentionConfig(tile_m=TILE, tile_n=TILE),
        )
        _PLAN_CACHE[key] = plan
        while len(_PLAN_CACHE) > _MAX_CACHED:
            _PLAN_CACHE.popitem(next(iter(_PLAN_CACHE)))
    return key, plan


def b12x_block_sparse_attn_bshd(query, key, value, block_map, variable_block_sizes, logical_blocks):
    """Drop-in for the provider call, on BHSD tensors (the custom op transposed already)."""
    from b12x.attention import varlen
    from b12x.preparation import require_prepared

    del variable_block_sizes  # b12x carries block ids, not per-block sizes
    batch, heads, seq_len, head_dim = query.shape
    if batch != 1:
        raise ValueError(f"H3 VSA routing expects batch 1, got {batch}")
    live = logical_blocks * TILE
    device = query.device
    dtype = query.dtype

    q = query[0, :, :live].contiguous().reshape(heads * live, 1, head_dim)
    k = key[0, :, :live].contiguous().reshape(heads * live, 1, head_dim)
    v = value[0, :, :live].contiguous().reshape(heads * live, 1, head_dim)

    indices, offsets = _csr_from_block_map(block_map, logical_blocks, device)
    num_tiles = int(offsets.numel() - 1)
    total_listed = max(1, int(offsets[-1].item()))
    cap = _capacity(total_listed)

    cache_key, plan = _plan_for(heads, live, head_dim, dtype, device, num_tiles, cap)
    state = require_prepared(plan, "attention.varlen", device)
    scratch = _SCRATCH_CACHE.get(cache_key)
    if scratch is None:
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        _SCRATCH_CACHE[cache_key] = scratch

    cu = torch.arange(0, (heads + 1) * live, live, dtype=torch.int32, device=device)
    binding = varlen.bind(
        plan,
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=live,
        max_seqlen_k=live,
        block_indices=indices,
        block_offsets=offsets,
    )
    result = state.run(binding)
    out = result[0] if isinstance(result, tuple) else result
    out = out.reshape(heads, live, head_dim)

    restored = query.new_empty(query.shape)
    restored[0, :, :live] = out
    if live < seq_len:
        restored[0, :, live:] = 0
    return restored
