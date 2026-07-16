# SPDX-License-Identifier: Apache-2.0
"""DreamZero-facing AR-Diffusion KV bridge backed by paged self-attention."""

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedForwardContext,
    ARDiffusionPagedLayerContext,
)

_log = init_logger(__name__)


class ARDiffusionKVState:
    """Per-request bridge between a DreamZero rollout and the AR-Diffusion KV pool.

    The self-attention path is paged-only: ``get_kv_caches`` returns lightweight
    per-layer contexts, and each attention layer writes current K/V into the
    pool before calling the FlashAttention block-table kernel.  No resident
    window is gathered into a contiguous tensor.
    """

    def __init__(self, kv_cache, pos_adapter, neg_adapter, num_layers: int) -> None:
        """One state per DreamZero session, holding BOTH CFG branches' adapters.

        ``pos``/``neg`` are the classifier-free-guidance branches: ``pos`` is
        the conditional pass (real prompt), ``neg`` the unconditional pass
        (negative prompt). Each owns an independent request adapter — and
        therefore independent pool blocks — because the two branches produce
        different K/V for the same frames. Every ``is_negative`` flag in this
        module selects between them (``False`` -> ``pos``, ``True`` -> ``neg``).
        """
        self.kv_cache = kv_cache
        self.pos = pos_adapter
        self.neg = neg_adapter
        self.num_layers = num_layers
        self._committed: dict[bool, int] = {False: 0, True: 0}
        self._paged_pending: dict[bool, ARDiffusionPagedForwardContext | None] = {False: None, True: None}
        # Cross-attn KV is populated once after text encoding (eager), then
        # read every denoising step. Tracked per half: the text K/V depend only
        # on the session prompt (survive window-boundary resets), while the I2V
        # image-token K/V come from the current observation's CLIP features and
        # must be re-projected on every window restart.
        self._cross_text_populated: dict[bool, bool] = {False: False, True: False}
        self._cross_img_populated: dict[bool, bool] = {False: False, True: False}

    def _adapter(self, is_negative: bool):
        return self.neg if is_negative else self.pos

    def get_kv_caches(
        self,
        is_negative: bool,
        seq_len: int | None = None,
        commit_current: bool = False,
    ) -> list[ARDiffusionPagedLayerContext]:
        if seq_len is None:
            raise ValueError("AR-Diffusion paged self-attention requires seq_len in get_kv_caches()")
        return self.prepare_paged_context(is_negative, seq_len, commit_current)

    def prepare_paged_context(
        self,
        is_negative: bool,
        seq_len: int,
        commit_current: bool,
    ) -> list[ARDiffusionPagedLayerContext]:
        """Return per-layer paged attention contexts for one branch forward.

        Allocation is lazy: constructing both CFG branches' kwargs must not
        allocate manager blocks for a branch that a CFG-parallel rank will not run.
        The first self-attn layer that consumes the context allocates the current
        video slots, then all layers share that branch-level state.
        """
        cs = self.kv_cache.spec.chunk_size
        if int(seq_len) % cs != 0:
            raise AssertionError(
                f"AR-Diffusion expects frame-aligned seq_len (multiple of chunk_size={cs}), got {seq_len}"
            )

        pending = self._paged_pending.get(is_negative)
        if pending is not None and pending.commit_current and pending._allocated_video and not pending._committed:
            raise RuntimeError("AR-Diffusion paged context replaced before its managed current chunk was committed")

        adapter = self._adapter(is_negative)
        branch = "neg" if is_negative else "pos"
        forward_ctx = ARDiffusionPagedForwardContext(
            kv_cache=self.kv_cache,
            adapter=adapter,
            is_negative=is_negative,
            history_block_ids=self.kv_cache.window_block_ids(adapter),
            seq_len=int(seq_len),
            commit_current=bool(commit_current),
            max_video_tokens=int(self.kv_cache.spec.sliding_window),
        )
        self._paged_pending[is_negative] = forward_ctx
        _log.debug(
            "AR-Diffusion GET   [%s] source=paged-attn layers=%d history_blocks=%d seq_len=%d commit_current=%s",
            branch,
            self.num_layers,
            len(forward_ctx.history_block_ids),
            int(seq_len),
            bool(commit_current),
        )
        return [ARDiffusionPagedLayerContext(layer_idx=i, forward_ctx=forward_ctx) for i in range(self.num_layers)]

    def commit_paged_context(self, is_negative: bool) -> None:
        """Commit the managed current video blocks after a successful forward."""
        ctx = self._paged_pending.get(is_negative)
        if ctx is None:
            return
        branch = "neg" if is_negative else "pos"
        if ctx.commit_current and ctx._allocated_video:
            n_chunks = ctx.seq_len // self.kv_cache.spec.chunk_size
            for _ in range(n_chunks):
                ctx.adapter.on_chunk_committed()
            self._committed[is_negative] += ctx.seq_len
            _log.debug(
                "AR-Diffusion COMMIT [%s] paged-attn new_tokens=%d chunks=%d resident=%d/%d",
                branch,
                ctx.seq_len,
                n_chunks,
                len(self.kv_cache.window_block_ids(ctx.adapter)),
                self.kv_cache.spec.window_chunks,
            )
        ctx.mark_committed()
        self._paged_pending[is_negative] = None

    def get_cross_kv_caches(self, is_negative: bool) -> list[dict]:
        """Return pool-backed cross-attn cache dicts.

        Under AR-Diffusion the cross-attn pool is always populated by ``_kv_populate_cross``
        before the first read, so this must never be reached unpopulated — if it
        were, the model would lazily project and write cross KV itself, violating
        engine ownership. Fail loud rather than fall back to the model.
        """
        img_ok = not self.kv_cache._cross_k_img or self._cross_img_populated[is_negative]
        if not (self._cross_text_populated[is_negative] and img_ok and self.kv_cache.cross_attn_length > 0):
            raise RuntimeError(
                f"AR-Diffusion cross-attn read before _kv_populate_cross (neg={is_negative}, "
                f"cross_attn_length={self.kv_cache.cross_attn_length}, "
                f"text={self._cross_text_populated[is_negative]} "
                f"img={self._cross_img_populated[is_negative]}) — the engine must own all cross KV"
            )
        return [self.kv_cache.read_cross_kv(i, is_negative) for i in range(self.num_layers)]

    def close(self) -> None:
        """Final teardown: free both branches' resident pool blocks."""
        self.kv_cache.end_request(self.pos)
        self.kv_cache.end_request(self.neg)
        self._paged_pending = {False: None, True: None}

    def reset(self, *, keep_cross_text: bool = False) -> None:
        """Drop this session's KV — mirrors the model-local ``state.reset()``.

        DreamZero resets at the attention-window boundary (``should_reset``); the
        model starts a fresh sliding window, so AR-Diffusion must free the resident pool
        blocks and start a fresh adapter for each branch. The ``ARDiffusionKVState``
        object (and thus ``pipeline._ar_diffusion_kv_state``) is preserved across the reset
        so the runner's session mapping stays valid — only the pool-backed state
        is recycled.

        Args:
            keep_cross_text: On window ("inference") resets the session prompt is
                unchanged, so the text cross-attn K/V in the static pool buffers are
                still valid — keep them and only force the image half to repopulate.
                Session resets (prompt change / explicit reset) pass ``False``.
        """
        pos_id, neg_id = self.pos.request_id, self.neg.request_id
        self.close()
        self.pos = self.kv_cache.begin_request(pos_id)
        self.neg = self.kv_cache.begin_request(neg_id)
        self._committed = {False: 0, True: 0}
        self._paged_pending = {False: None, True: None}
        if not keep_cross_text:
            self._cross_text_populated = {False: False, True: False}
        self._cross_img_populated = {False: False, True: False}
        _log.info(
            "AR-Diffusion RESET [%s/%s] session KV cleared (window boundary; cross text %s)",
            pos_id,
            neg_id,
            "kept" if keep_cross_text else "cleared",
        )
