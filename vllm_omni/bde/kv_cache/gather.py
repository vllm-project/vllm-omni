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
from vllm.logger import init_logger

_log = init_logger(__name__)


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


class BDEKVState:
    """Per-request bridge between a DreamZero rollout and the BDE KV pool.

    Replaces the model-local ``DreamZeroState`` KV methods::
      state.get_kv_caches(neg)    →  bde_state.get_kv_caches(neg)
      state.update_kv_cache(i,kv) →  bde_state.update_kv_cache(i,kv,neg)
      state.create_kv_caches(...) →  no-op (pool is pre-allocated)

    One ``BDEKVState`` per request holds the ``BDEKVCache`` and both CFG adapters
    (positive / negative). The runner sets ``pipeline._bde_kv_state`` before
    ``pipeline.forward(req)``; the pipeline delegates every KV call to it when
    the attribute is present and not None.

    Call ``commit_chunk()`` after all layers have been written for a chunk
    (once per chunk, not per denoise step).
    """

    def __init__(self, kv_cache, pos_adapter, neg_adapter, num_layers: int) -> None:
        self.kv_cache = kv_cache
        self.pos = pos_adapter
        self.neg = neg_adapter
        self.num_layers = num_layers
        # --- Approach 1: faithful window store (shape-agnostic) ----------------
        # DreamZero's update_kv_cache stores the whole current (trimmed) window per
        # step; its length and rank vary. We mirror it exactly by holding the
        # model's ``updated_kv`` tensors per (branch, layer) — so the output is
        # bit-identical to the model-local path. On the first get (nothing stored
        # yet) we defer to the model-local empty cache via ``fallback``. The
        # pool-backed, chunk-paged version is Approach 2.
        self._store: dict[bool, list] = {
            False: [None] * num_layers,
            True: [None] * num_layers,
        }

    def get_kv_caches(self, is_negative: bool, fallback) -> list:
        branch = "neg" if is_negative else "pos"
        store = self._store[is_negative]
        if store[0] is None:  # nothing written this rollout yet
            out = fallback()
            _log.info("BDE GET   [%s] source=model-local-empty layers=%d", branch, len(out))
            return out
        _log.info(
            "BDE GET   [%s] source=bde-store layers=%d window0=%s",
            branch, self.num_layers, tuple(store[0].shape),
        )
        for i in range(self.num_layers):
            _log.debug("BDE GET   [%s] layer=%d shape=%s", branch, i, tuple(store[i].shape))
        return store

    def update_kv_cache(self, layer_idx: int, updated_kv: torch.Tensor, is_negative: bool) -> None:
        branch = "neg" if is_negative else "pos"
        self._store[is_negative][layer_idx] = updated_kv
        if layer_idx == 0:
            _log.info(
                "BDE WRITE [%s] window=%s storing %d layers into BDE pool state",
                branch, tuple(updated_kv.shape), self.num_layers,
            )
        _log.debug("BDE WRITE [%s] layer=%d shape=%s", branch, layer_idx, tuple(updated_kv.shape))

    def commit_chunk(self) -> None:
        # Window-store mode keeps no per-chunk bookkeeping (Approach 2 will).
        _log.info("BDE COMMIT (pos/neg window stores retained across forwards)")


class BDEPipelineMixin:
    """Mixin that replaces model-local KV state with pool-backed storage.

    A DreamZero pipeline inheriting this mixin delegates KV access through
    proxy methods (``_bde_kv_get`` / ``_bde_kv_create`` / ``_bde_kv_update``)
    that check for ``self._bde_kv_state`` and route to :class:`BDEKVState`
    when present, falling back to the ``DreamZeroState`` methods otherwise.
    """

    _bde_kv_state: BDEKVState | None = None

    # -- proxy methods (call these instead of state.*) -----------------------

    def _bde_kv_get(self, state, is_negative):
        if self._bde_kv_state is not None:
            return self._bde_kv_state.get_kv_caches(is_negative)
        return state.get_kv_caches(is_negative)

    def _bde_kv_create(self, state, batch_size, dtype, device, num_layers, num_heads, head_dim):
        if self._bde_kv_state is not None:
            return  # pool already allocated
        state.create_kv_caches(batch_size, dtype, device, num_layers, num_heads, head_dim)

    def _bde_kv_update(self, state, layer_idx, updated_kv, is_negative):
        if self._bde_kv_state is not None:
            self._bde_kv_state.update_kv_cache(layer_idx, updated_kv, is_negative)
            return
        state.update_kv_cache(layer_idx, updated_kv, is_negative=is_negative)

    def _bde_kv_commit(self):
        if self._bde_kv_state is not None:
            self._bde_kv_state.commit_chunk()
