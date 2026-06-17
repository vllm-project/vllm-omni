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


def _drop_batch(t: torch.Tensor) -> torch.Tensor:
    """``(1, L, n_heads, head_dim)`` -> ``(L, n_heads, head_dim)``; pass 3-D through."""
    if t.dim() == 4 and t.shape[0] == 1:
        return t[0]
    return t


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
        # --- Approach 2: true chunk-paged write / gather -----------------------
        # Each get gathers the resident window from the paged pool blocks; each
        # update writes only the genuinely-new tokens (seq_len growth) into newly
        # allocated frame-blocks, evicting out-of-window blocks via the
        # ChunkWindowManager. ``_committed`` is the absolute token count written to
        # the pool per branch; ``_pending`` holds this forward's new-chunk slot maps.
        self._committed: dict[bool, int] = {False: 0, True: 0}
        self._pending: dict[bool, list] = {False: [], True: []}
        # Cross-attn KV is populated once after text encoding (eager), then
        # read every denoising step.  Reset clears these flags so the next
        # forward repopulates from the new text encoding.
        self._cross_populated: dict[bool, bool] = {False: False, True: False}

    def _adapter(self, is_negative: bool):
        return self.neg if is_negative else self.pos

    def get_kv_caches(self, is_negative: bool, fallback) -> list:
        branch = "neg" if is_negative else "pos"
        if self._committed[is_negative] == 0:  # nothing committed yet (prefill start)
            out = fallback()
            _log.info("BDE GET   [%s] source=model-local-empty layers=%d", branch, len(out))
            return out
        adapter = self._adapter(is_negative)
        windows = [self.kv_cache.gather_window(i, adapter) for i in range(self.num_layers)]
        _log.info(
            "BDE GET   [%s] source=paged-gather layers=%d window0=%s resident_blocks=%d/%d",
            branch, self.num_layers, tuple(windows[0].shape),
            len(self.kv_cache.window_block_ids(adapter)), self.kv_cache.spec.window_chunks,
        )
        return windows

    def get_cross_kv_caches(self, is_negative: bool, fallback) -> list[dict]:
        """Return pool-backed cross-attn cache dicts, or fallback if not populated."""
        if self._cross_populated[is_negative] and self.kv_cache.cross_attn_length > 0:
            return [self.kv_cache.read_cross_kv(i, is_negative) for i in range(self.num_layers)]
        return fallback()

    def update_kv_cache(self, layer_idx: int, updated_kv: torch.Tensor, is_negative: bool, seq_len) -> None:
        branch = "neg" if is_negative else "pos"
        adapter = self._adapter(is_negative)
        cs = self.kv_cache.spec.chunk_size
        if layer_idx == 0:
            # The K appended this forward is the current observation's length
            # (seq_len), not the cumulative-minus-committed delta. Allocate those
            # frame-blocks once per forward (shared across layers).
            new_count = int(seq_len)
            n_chunks = new_count // cs
            slots = []
            for _ in range(n_chunks):
                self.kv_cache.allocate_chunk(adapter)        # allocate + evict old
                slots.append(self.kv_cache.chunk_write_slots(adapter))
                adapter.on_chunk_committed()
            self._pending[is_negative] = slots
            self._committed[is_negative] += new_count
            _log.info(
                "BDE WRITE [%s] new_tokens=%d -> %d frame-chunks  resident=%d/%d  storing %d layers",
                branch, new_count, n_chunks,
                len(self.kv_cache.window_block_ids(adapter)), self.kv_cache.spec.window_chunks,
                self.num_layers,
            )
        slots = self._pending[is_negative]
        if not slots:
            return
        n = len(slots) * cs
        k_all = _drop_batch(updated_kv[0])[-n:]  # (n, n_heads, head_dim) — the new tokens
        v_all = _drop_batch(updated_kv[1])[-n:]
        kpool = self.kv_cache._k_pools[layer_idx]
        vpool = self.kv_cache._v_pools[layer_idx]
        for c, sm in enumerate(slots):
            k = k_all[c * cs : (c + 1) * cs].unsqueeze(0).to(kpool.dtype)
            v = v_all[c * cs : (c + 1) * cs].unsqueeze(0).to(vpool.dtype)
            pool_write_chunk(kpool, vpool, k, v, sm)

    def commit_chunk(self) -> None:
        _log.info("BDE COMMIT (paged; resident window retained across forwards)")

    def reset(self) -> None:
        """Drop this session's KV — mirrors the model-local ``state.reset()``.

        DreamZero resets at the attention-window boundary (``should_reset``); the
        model starts a fresh sliding window, so BDE must free the resident pool
        blocks and start a fresh adapter for each branch. The ``BDEKVState``
        object (and thus ``pipeline._bde_kv_state``) is preserved across the reset
        so the runner's session mapping stays valid — only the pool-backed state
        is recycled.
        """
        pos_id, neg_id = self.pos.request_id, self.neg.request_id
        self.kv_cache.end_request(self.pos)        # free resident blocks
        self.kv_cache.end_request(self.neg)
        self.pos = self.kv_cache.begin_request(pos_id)   # fresh, empty window
        self.neg = self.kv_cache.begin_request(neg_id)
        self._committed = {False: 0, True: 0}
        self._pending = {False: [], True: []}
        self._cross_populated = {False: False, True: False}
        _log.info("BDE RESET [%s/%s] session KV cleared (window boundary)", pos_id, neg_id)


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
