# SPDX-License-Identifier: Apache-2.0
"""Generic paging mechanics + chunk-window eviction spec for the BDE engine.

Engine-generic, model-agnostic primitives — the layer a second model (e.g. the
Cosmos port) reuses unchanged. Three concerns live here:

* **Slot mapping** — absolute token positions → physical KV-cache slots, the
  standard PagedAttention layout ``slot(pos) = block_id(pos) * block_size +
  (pos % block_size)``. Phase 1 is single-request and gathers the resident
  window into the model's existing attention rather than calling vLLM's paged
  kernel, so a thin slot-mapping helper is used instead of vLLM's ``BlockTables``.
* **Pool I/O** — allocate per-layer flat K/V pools, write a committed chunk into
  slots, and gather the resident window back as a contiguous tensor. The
  per-forward gather copy is the deliberate Phase-1 simplification the fused
  paged backend retires in Phase 2.
* **Chunk-window eviction** — a ``SlidingWindowSpec`` subclass whose unit is a
  *chunk* (``sliding_window = window_chunks * chunk_size``) plus a manager that
  evicts at chunk boundaries. Memory policy / refcounting / ``null_block``
  replacement stay in vLLM's ``BlockPool``; only the token-skip math is here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec


# ── Slot mapping ────────────────────────────────────────────────────────────


def compute_slot_mapping(
    block_ids: Sequence[int],
    positions: torch.Tensor | Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Map absolute token positions to physical KV-cache slots.

    Args:
        block_ids: physical block id per block index (the request's block table).
        positions: absolute token positions to map (1-D).
        block_size: tokens per block.

    Returns:
        ``LongTensor`` of physical slots, one per position.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    table = torch.as_tensor(block_ids, dtype=torch.long)
    pos = torch.as_tensor(positions, dtype=torch.long)
    block_index = torch.div(pos, block_size, rounding_mode="floor")
    offset = pos % block_size
    return table[block_index] * block_size + offset


def chunk_slot_mapping(
    block_ids: Sequence[int],
    num_computed_tokens: int,
    chunk_size: int,
    block_size: int,
) -> torch.Tensor:
    """Slot mapping for the in-flight chunk's tokens (the commit write target).

    The chunk occupies absolute positions
    ``[num_computed_tokens, num_computed_tokens + chunk_size)``.
    """
    positions = torch.arange(
        num_computed_tokens,
        num_computed_tokens + chunk_size,
        dtype=torch.long,
    )
    return compute_slot_mapping(block_ids, positions, block_size)


def resident_block_ids(block_ids: Sequence[int], null_block_id: int) -> list[int]:
    """Real (non-null) blocks currently resident, in table order.

    These are the blocks the read path gathers the attention window from;
    out-of-window positions are the shared ``null_block`` and are excluded.
    """
    return [int(b) for b in block_ids if int(b) != null_block_id]


# ── Pool I/O (allocate / write / gather) ────────────────────────────────────


def allocate_kv_pool(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Allocate per-layer paged K and V pools on *device*.

    Each pool is a contiguous ``(num_blocks * block_size, num_kv_heads, head_dim)``
    tensor — a flat address space indexed by ``slot = block_id * block_size + offset``.
    The slot mapping (above) tells writes and gathers where each token lives.
    """
    k_pools: list[torch.Tensor] = []
    v_pools: list[torch.Tensor] = []
    for _ in range(num_layers):
        k_pools.append(torch.empty(num_blocks * block_size, num_kv_heads, head_dim, dtype=dtype, device=device))
        v_pools.append(torch.empty(num_blocks * block_size, num_kv_heads, head_dim, dtype=dtype, device=device))
    return k_pools, v_pools


def pool_write_chunk(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write the committed chunk's K/V into pool slots (in place).

    ``new_k`` / ``new_v`` shape: ``(batch, chunk_size, num_kv_heads, head_dim)``.
    ``slot_mapping`` shape: ``(chunk_size,)`` — one physical slot per token position.
    The batch dim is written to the same slots (multi-batch writes the same
    positions; the caller is responsible for per-sequence bookkeeping).

    This is the *commit* write — called once per chunk, after the last denoise step.
    """
    k_pool[slot_mapping] = new_k[0]  # batch index 0
    v_pool[slot_mapping] = new_v[0]


def build_window_slots(window_block_ids: list[int], block_size: int, device: torch.device) -> torch.Tensor:
    """Flat slot index covering all resident blocks, in table order.

    Vectorized (no Python loop): ``slot = block_id * block_size + offset`` for every
    ``offset in [0, block_size)``. The window's blocks are identical across all layers
    within a forward, so this is built once per forward and shared.
    """
    if not window_block_ids:
        # No resident blocks (e.g. prefill start, before the first chunk is
        # committed): an empty slot index gathers a zero-length window, which is
        # the concat-identity the model's attention expects as its initial KV.
        return torch.empty(0, dtype=torch.long, device=device)
    starts = torch.tensor(window_block_ids, dtype=torch.long, device=device) * block_size
    offsets = torch.arange(block_size, device=device)
    return (starts[:, None] + offsets[None, :]).reshape(-1)


def pool_gather_window(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    window_block_ids: list[int],
    block_size: int,
    max_attention_size: int,
    *,
    slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the resident window K/V as a contiguous tensor.

    Gathers all slots belonging to ``window_block_ids`` (in table order), trims to
    the last ``max_attention_size`` tokens (matching the existing ``cat`` + slice),
    and returns a ``(2, batch, window_tokens, n_heads, head_dim)`` tensor — exactly
    the format DreamZero's attention expects as ``kv_cache``. Pass a precomputed
    ``slots`` (from :func:`build_window_slots`) to share the index across layers.
    """
    if slots is None:
        slots = build_window_slots(window_block_ids, block_size, k_pool.device)
    # Gather then trim to the attention window.
    wk = k_pool[slots]  # (window_slots, n_heads, head_dim)
    wv = v_pool[slots]
    wk = wk[-max_attention_size:]  # (window, n_heads, head_dim)
    wv = wv[-max_attention_size:]
    # Add the batch dim then stack as (2, batch, window, n_heads, head_dim).
    wk = wk.unsqueeze(0)  # (1, window, n_heads, head_dim)
    wv = wv.unsqueeze(0)
    window = torch.stack([wk, wv], dim=0)  # (2, 1, window, n_heads, head_dim)
    return window


def _drop_batch(t: torch.Tensor) -> torch.Tensor:
    """``(1, L, n_heads, head_dim)`` -> ``(L, n_heads, head_dim)``; pass 3-D through."""
    if t.dim() == 4 and t.shape[0] == 1:
        return t[0]
    return t


# ── Chunk-window eviction (spec + manager) ──────────────────────────────────


def chunk_window_skipped_tokens(
    num_computed_tokens: int,
    *,
    chunk_size: int,
    sliding_window: int,
    sink_chunks: int,
    reset_at_boundary: bool,
) -> int:
    """Tokens outside the resident chunk window, snapped to a chunk boundary.

    Pure function so the eviction policy is unit-testable without constructing a
    manager. Two strategies:

    - ``reset_at_boundary`` (DreamZero): at each chunk boundary everything past
      the sink is dropped.
    - otherwise (VGGT-style sliding replace): keep the last ``window`` tokens
      (plus the sink); the skip count snaps down to a chunk boundary so a chunk
      is never half-evicted.
    """
    sink = sink_chunks * chunk_size
    if reset_at_boundary:
        completed = (num_computed_tokens // chunk_size) * chunk_size
        return max(0, completed - sink)
    skipped = max(0, num_computed_tokens - sliding_window - sink)
    return (skipped // chunk_size) * chunk_size


class ChunkWindowManager(SlidingWindowManager):
    """``SlidingWindowManager`` that evicts at chunk boundaries.

    ``self.sliding_window`` is set by the base ``__init__``; the chunk fields are
    read from ``self.kv_cache_spec`` (a :class:`ChunkWindowSpec`).
    """

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        spec = self.kv_cache_spec
        return chunk_window_skipped_tokens(
            num_computed_tokens,
            chunk_size=spec.chunk_size,
            sliding_window=self.sliding_window,
            sink_chunks=spec.sink_chunks,
            reset_at_boundary=spec.reset_at_boundary,
        )


# Register so KVCacheManager resolves ChunkWindowSpec to ChunkWindowManager.
# Dispatch walks the spec's MRO, so without explicit registration the subclass
# would silently fall back to the parent SlidingWindowManager (override ignored).
# uniform_type_base_spec=None => its own KV cache group.
@register_kv_cache_spec(manager_class=ChunkWindowManager, uniform_type_base_spec=None)
@dataclass(frozen=True, kw_only=True)
class ChunkWindowSpec(SlidingWindowSpec):
    # sliding_window (inherited) MUST equal window_chunks * chunk_size.
    chunk_size: int
    window_chunks: int
    sink_chunks: int = 0
    reset_at_boundary: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window != self.window_chunks * self.chunk_size:
            raise ValueError(
                "ChunkWindowSpec.sliding_window must equal "
                f"window_chunks * chunk_size ({self.window_chunks} * "
                f"{self.chunk_size}), got {self.sliding_window}"
            )
