# SPDX-License-Identifier: Apache-2.0
"""BDEKVState — per-request bridge between a model rollout and the BDE KV pool.

This is the one model-facing piece of the KV stack (everything in ``paged.py`` /
``manager.py`` is engine-generic). A second model (e.g. the Cosmos port) adds a
sibling here that reuses the same pool primitives.

Phase 1 keeps the existing attention kernel unchanged — it only changes *where*
the past-window KV comes from (the pool) and *where* the committed chunk's KV is
stored (the pool). The model's ``self_attn`` still receives a contiguous
``(batch, window, n_heads, head_dim)`` K/V pair; the gather materializes that
tensor from the resident pool blocks.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

from vllm_omni.experimental.bde.kv_cache.paged import _drop_batch, pool_write_chunk

_log = init_logger(__name__)


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
        # The resident window is frozen across a forward's denoise steps (KV is
        # only written back after the last step), so the per-step gather is
        # re-materializing the same tensors ~16x. Memoize the gathered window per
        # branch; invalidate on write-back (window grows) and on reset.
        self._gather_cache: dict[bool, list | None] = {False: None, True: None}
        # Profiling escape hatch: BDE_KV_NO_MEMO=1 disables the per-forward gather
        # memoization so every denoise step re-gathers the window (the pre-memo
        # behavior), for A/B measurement of the memoization win. Default: enabled.
        self._memo_enabled = os.environ.get("BDE_KV_NO_MEMO") != "1"
        # Cross-attn KV is populated once after text encoding (eager), then
        # read every denoising step.  Reset clears these flags so the next
        # forward repopulates from the new text encoding.
        self._cross_populated: dict[bool, bool] = {False: False, True: False}

    def _adapter(self, is_negative: bool):
        return self.neg if is_negative else self.pos

    def get_kv_caches(self, is_negative: bool) -> list:
        # Engine owns the read end-to-end: at prefill start the adapter has no
        # resident blocks, so gather returns zero-length windows (the empty seed
        # the model concatenates against) — no model-local fallback needed.
        branch = "neg" if is_negative else "pos"
        cached = self._gather_cache[is_negative] if self._memo_enabled else None
        if cached is not None:  # window unchanged this forward -> reuse (no re-gather)
            return cached
        adapter = self._adapter(is_negative)
        windows = self.kv_cache.gather_window_all_layers(adapter)
        if self._memo_enabled:
            self._gather_cache[is_negative] = windows
        _log.info(
            "BDE GET   [%s] source=paged-gather layers=%d window0=%s resident_blocks=%d/%d",
            branch,
            self.num_layers,
            tuple(windows[0].shape),
            len(self.kv_cache.window_block_ids(adapter)),
            self.kv_cache.spec.window_chunks,
        )
        return windows

    def get_cross_kv_caches(self, is_negative: bool) -> list[dict]:
        """Return pool-backed cross-attn cache dicts.

        Under BDE the cross-attn pool is always populated by ``_kv_populate_cross``
        before the first read, so this must never be reached unpopulated — if it
        were, the model would lazily project and write cross KV itself, violating
        engine ownership. Fail loud rather than fall back to the model.
        """
        if not (self._cross_populated[is_negative] and self.kv_cache.cross_attn_length > 0):
            raise RuntimeError(
                "BDE cross-attn read before _kv_populate_cross (neg=%s, cross_attn_length=%d) "
                "— the engine must own all cross KV" % (is_negative, self.kv_cache.cross_attn_length)
            )
        return [self.kv_cache.read_cross_kv(i, is_negative) for i in range(self.num_layers)]

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
                self.kv_cache.allocate_chunk(adapter)  # allocate + evict old
                slots.append(self.kv_cache.chunk_write_slots(adapter))
                adapter.on_chunk_committed()
            self._pending[is_negative] = slots
            self._committed[is_negative] += new_count
            # The window just grew; the memoized gather is stale -> force re-gather
            # on the next forward's first read of this branch.
            self._gather_cache[is_negative] = None
            _log.info(
                "BDE WRITE [%s] new_tokens=%d -> %d frame-chunks  resident=%d/%d  storing %d layers",
                branch,
                new_count,
                n_chunks,
                len(self.kv_cache.window_block_ids(adapter)),
                self.kv_cache.spec.window_chunks,
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
        """No-op for DreamZero (logs only).

        The per-chunk advance already happened in :meth:`update_kv_cache`, which
        calls ``adapter.on_chunk_committed()`` at layer 0 while allocating each
        frame-chunk. This method exists so the pipeline's ``_kv_commit`` bridge has
        a symmetric call; the resident window is retained across forwards.
        """
        _log.info("BDE COMMIT (paged; resident window retained across forwards)")

    def close(self) -> None:
        """Final teardown — free both branches' resident pool blocks.

        Unlike :meth:`reset`, this does **not** restart the adapters: the caller is
        dropping this ``BDEKVState`` for good (session eviction / shutdown), so the
        blocks return to the ``BlockPool`` and the dead adapters are discarded with
        the state. Frees pool ownership that would otherwise leak when a session is
        evicted from the runner's session map.
        """
        self.kv_cache.end_request(self.pos)
        self.kv_cache.end_request(self.neg)

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
        self.close()  # free resident blocks for both branches
        self.pos = self.kv_cache.begin_request(pos_id)  # fresh, empty window
        self.neg = self.kv_cache.begin_request(neg_id)
        self._committed = {False: 0, True: 0}
        self._pending = {False: [], True: []}
        self._gather_cache = {False: None, True: None}
        self._cross_populated = {False: False, True: False}
        _log.info("BDE RESET [%s/%s] session KV cleared (window boundary)", pos_id, neg_id)
