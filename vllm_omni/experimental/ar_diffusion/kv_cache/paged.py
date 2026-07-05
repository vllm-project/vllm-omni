# SPDX-License-Identifier: Apache-2.0
"""Generic paging mechanics + chunk-window eviction spec for the AR-Diffusion engine.

Engine-generic, model-agnostic primitives — the layer a second model (e.g. the
Cosmos port) reuses unchanged. Three concerns live here:

* **Slot mapping** — absolute token positions → physical KV-cache slots, the
  standard PagedAttention layout ``slot(pos) = block_id(pos) * block_size +
  (pos % block_size)``.
* **Pool I/O** — allocate per-layer FlashAttention-compatible paged K/V pools
  plus flat compatibility views for direct slot writes.
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

try:
    from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec
except ModuleNotFoundError:

    def register_kv_cache_spec(*, manager_class, uniform_type_base_spec=None):
        """Compatibility shim for vLLM versions without kv_cache_spec_registry."""

        def decorator(spec_class):
            from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

            spec_manager_map[spec_class] = manager_class
            return spec_class

        return decorator


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


# ── Pool I/O (allocate / write) ─────────────────────────────────────────────


def allocate_kv_pool_with_views(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Allocate vLLM-style paged KV pools plus flat slot-write views.

    The owning tensor follows FlashAttention's block-table cache layout:
    ``(2, num_blocks, block_size, num_kv_heads, head_dim)``, where dim-0 is
    ``[K, V]``.  The flat K/V views keep ``slot = block_id * block_size +
    offset`` writes simple without changing the kernel-facing cache layout.
    """
    kv_pools: list[torch.Tensor] = []
    k_pools: list[torch.Tensor] = []
    v_pools: list[torch.Tensor] = []
    for _ in range(num_layers):
        kv = torch.empty(2, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        kv_pools.append(kv)
        k_pools.append(kv[0].reshape(num_blocks * block_size, num_kv_heads, head_dim))
        v_pools.append(kv[1].reshape(num_blocks * block_size, num_kv_heads, head_dim))
    return kv_pools, k_pools, v_pools


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
