# SPDX-License-Identifier: Apache-2.0
"""Pool write + window gather for the BDE engine (Phase 1).

Phase 1 keeps the existing attention kernel unchanged — it only changes *where*
the past-window KV comes from (the pool) and *where* the committed chunk's KV is
stored (the pool). The model's ``self_attn`` still receives a contiguous
``(batch, window, n_heads, head_dim)`` K/V pair; the gather materializes that
tensor from the resident pool blocks. The per-forward gather copy is the
deliberate Phase-1 simplification the fused paged backend retires in Phase 2.
"""

from __future__ import annotations

import torch


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
    The slot mapping (``bde.kv_cache.slot_mapping``) tells writes and gathers where
    each token lives.
    """
    k_pools: list[torch.Tensor] = []
    v_pools: list[torch.Tensor] = []
    for _ in range(num_layers):
        k_pools.append(
            torch.empty(
                num_blocks * block_size, num_kv_heads, head_dim, dtype=dtype, device=device
            )
        )
        v_pools.append(
            torch.empty(
                num_blocks * block_size, num_kv_heads, head_dim, dtype=dtype, device=device
            )
        )
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


def pool_gather_window(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    window_block_ids: list[int],
    block_size: int,
    max_attention_size: int,
) -> torch.Tensor:
    """Materialize the resident window K/V as a contiguous tensor.

    Gathers all slots belonging to ``window_block_ids`` (in table order), trims to
    the last ``max_attention_size`` tokens (matching the existing ``cat`` + slice),
    and returns a ``(2, batch, window_tokens, n_heads, head_dim)`` tensor — exactly
    the format DreamZero's attention expects as ``kv_cache``.
    """
    if not window_block_ids:
        raise ValueError("window_block_ids is empty — no blocks are resident")
    # Build flat slot range covering all resident blocks in table order.
    starts = torch.tensor(
        [b * block_size for b in window_block_ids],
        dtype=torch.long,
        device=k_pool.device,
    )
    slots = torch.cat([torch.arange(s, s + block_size, device=k_pool.device) for s in starts])
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


class BDEPipelineMixin:
    """Mixin that replaces model-local KV state with pool-backed storage.

    A DreamZero pipeline inheriting this mixin overrides the KV access pattern
    (get / update / reset) to route through the ``BDEKVCache`` owned by the
    ``BDEModelRunner``, without changing the attention kernel.

    Usage from the model runner (per request):
        pipeline.set_kv_cache(runner.kv_cache)   # once, before the rollout
        pipeline.forward(req)                    # pool gather/write happens inside
    """

    _bde_kv_cache = None

    def set_kv_cache(self, kv_cache) -> None:
        self._bde_kv_cache = kv_cache

    def kv_get_window(
        self, layer_index: int, is_negative: bool
    ) -> torch.Tensor | None:
        """Return the gathered window K/V for a layer, or None if KV is off."""
        raise NotImplementedError("subclass provides this")
