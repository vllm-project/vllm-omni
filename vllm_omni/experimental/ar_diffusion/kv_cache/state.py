# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-facing session state backed by paged AR-Diffusion KV storage."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedForwardContext,
    ARDiffusionPagedLayerContext,
)

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.kv_cache.manager import (
        ARDiffusionKVCache,
        ARDiffusionRequestAdapter,
    )

_log = init_logger(__name__)

# One cached forward context: (cache key, forward context, per-layer contexts).
_FctxCacheEntry = tuple[tuple, ARDiffusionPagedForwardContext, list[ARDiffusionPagedLayerContext]]


class ARDiffusionKVState:
    """Runner-owned KV state for one session and any number of KV branches.

    KV branches are addressed by the names in ``ARDiffusionKVCacheSpec``. Each
    branch owns an independent request adapter and therefore independent
    resident self-attention blocks. Named cross-attention allocations are also
    session-scoped and are released by :meth:`close`.
    """

    def __init__(
        self,
        kv_cache: ARDiffusionKVCache,
        session_id: str,
        adapters: Mapping[str, ARDiffusionRequestAdapter],
        *,
        num_layers: int,
    ) -> None:
        expected = tuple(kv_branch.name for kv_branch in kv_cache.kv_branches)
        if set(adapters) != set(expected):
            raise ValueError(
                "AR-Diffusion session adapters must match the configured KV branches; "
                f"expected {expected}, got {tuple(adapters)}"
            )
        self.kv_cache = kv_cache
        self.session_id = session_id
        self.adapters = dict(adapters)
        self.num_layers = num_layers
        self._committed: dict[str, int] = dict.fromkeys(expected, 0)
        self._paged_pending: dict[str, ARDiffusionPagedForwardContext | None] = dict.fromkeys(expected)
        self._closed = False

        # Forward-context reuse across denoise steps. A fresh
        # ARDiffusionPagedForwardContext used to be built on every DiT forward, so
        # prepare() re-did the whole host-side setup 17x per request: CPU arange,
        # compute_slot_mapping, the padded block table, three H2D copies and one
        # layer-context object per layer.
        #
        # Inside the denoise loop the KV mapping does not move -- the pipeline
        # calls in with commit_current=False there, so nothing is allocated or
        # committed. Only the prefill/commit forwards change the mapping. So one
        # context per (branch, geometry, commit epoch) serves every step in between.
        #
        # One entry per KV branch, never more: a new key evicts the old one, so the
        # cache cannot grow and cannot keep a finished request's tensors alive.
        self._paged_cache: dict[str, _FctxCacheEntry] = {}
        self._fctx_commit_seq = 0
        self.fctx_counters: dict[str, int] = {
            "fctx_cache_hit": 0,
            "fctx_cache_miss": 0,
            "fctx_cache_invalidate": 0,
        }
        self._fctx_last_miss_reason: str | None = None

    @property
    def kv_branch_names(self) -> tuple[str, ...]:
        return tuple(self.adapters)

    def adapter(self, kv_branch: str) -> ARDiffusionRequestAdapter:
        """Return the request adapter for one logical KV branch."""
        if self._closed:
            raise RuntimeError(f"AR-Diffusion session {self.session_id!r} is closed")
        try:
            return self.adapters[kv_branch]
        except KeyError as exc:
            raise KeyError(f"Unknown AR-Diffusion KV branch {kv_branch!r}; expected {self.kv_branch_names}") from exc

    def get_kv_caches(
        self,
        kv_branch: str,
        seq_len: int | None = None,
        commit_current: bool = False,
    ) -> list[ARDiffusionPagedLayerContext]:
        if seq_len is None:
            raise ValueError("AR-Diffusion paged self-attention requires seq_len in get_kv_caches()")
        return self.prepare_paged_context(kv_branch, seq_len, commit_current)

    def prepare_paged_context(
        self,
        kv_branch: str,
        seq_len: int,
        commit_current: bool,
    ) -> list[ARDiffusionPagedLayerContext]:
        """Return per-layer paged attention contexts for one KV branch forward.

        Allocation is lazy so distributed workers allocate only for KV branches
        they actually execute.
        """
        cs = self.kv_cache.spec.chunk_size
        if int(seq_len) % cs != 0:
            raise AssertionError(
                f"AR-Diffusion expects frame-aligned seq_len (multiple of chunk_size={cs}), got {seq_len}"
            )

        pending = self._paged_pending.get(kv_branch)
        if pending is not None and pending.commit_current and pending._allocated_video and not pending._committed:
            raise RuntimeError("AR-Diffusion paged context replaced before its managed current chunk was committed")

        adapter = self.adapter(kv_branch)

        # Reuse the forward context when the KV mapping cannot have moved.
        cache_key = self._fctx_cache_key(kv_branch, adapter, seq_len, commit_current)
        cached = self._paged_cache.get(kv_branch)
        if cached is not None:
            prev_key, prev_ctx, prev_layers = cached
            if prev_key == cache_key:
                self.fctx_counters["fctx_cache_hit"] += 1
                # Re-publish as the pending context so commit_paged_context() and
                # the "replaced before commit" guard above still see it.
                self._paged_pending[kv_branch] = prev_ctx
                return prev_layers
            self.fctx_counters["fctx_cache_invalidate"] += 1
            self._fctx_last_miss_reason = self._fctx_key_diff(prev_key, cache_key)
            _log.debug(
                "AR-Diffusion fctx cache INVALIDATE [%s] %s",
                kv_branch,
                self._fctx_last_miss_reason,
            )
            # Drop the stale entry before building the replacement so its device
            # tensors are not held alongside the new ones.
            self._paged_cache.pop(kv_branch, None)
        else:
            self._fctx_last_miss_reason = "no entry for branch"
        self.fctx_counters["fctx_cache_miss"] += 1

        forward_ctx = ARDiffusionPagedForwardContext(
            kv_cache=self.kv_cache,
            adapter=adapter,
            kv_branch=kv_branch,
            history_block_ids=self.kv_cache.window_block_ids(adapter),
            seq_len=int(seq_len),
            commit_current=bool(commit_current),
            max_video_tokens=int(
                self.kv_cache.spec.sliding_window + self.kv_cache.spec.sink_chunks * self.kv_cache.spec.chunk_size
            ),
        )
        self._paged_pending[kv_branch] = forward_ctx
        _log.debug(
            "AR-Diffusion GET [%s] source=paged-attn layers=%d history_blocks=%d seq_len=%d commit_current=%s",
            kv_branch,
            self.num_layers,
            len(forward_ctx.history_block_ids),
            int(seq_len),
            bool(commit_current),
        )
        layers = [ARDiffusionPagedLayerContext(layer_idx=i, forward_ctx=forward_ctx) for i in range(self.num_layers)]
        # Cache the layer-context list too: they are immutable objects wrapping
        # only (layer_idx, forward_ctx), so rebuilding them per forward is pure
        # allocation churn.
        self._paged_cache[kv_branch] = (cache_key, forward_ctx, layers)
        return layers

    def _fctx_cache_key(
        self,
        kv_branch: str,
        adapter: ARDiffusionRequestAdapter,
        seq_len: int,
        commit_current: bool,
    ) -> tuple:
        """Every value that can change the forward context's addressing.

        Built from host-side ints only -- no tensor reads, so validating a hit
        costs no device synchronization. A sync here would defeat the purpose.

        Deliberately conservative:
          * ``commit_current=True`` makes the key unique per call via the epoch
            token, so a committing forward never reuses a cached entry and never
            has its own entry reused. Only the read-only denoise-loop forwards
            are shared.
          * ``num_computed_tokens`` and the committed-token counter both advance on
            commit, so a post-commit forward cannot match a pre-commit key.
          * ``window_block_ids`` changes length when the sliding window evicts,
            which changes the block table.
        """
        return (
            self.session_id,
            kv_branch,
            id(adapter),
            int(seq_len),
            bool(commit_current),
            # KV "version": both advance on commit, so they act as the epoch.
            int(self._committed.get(kv_branch, 0)),
            int(adapter.num_computed_tokens),
            # Window geometry: affects history_block_ids and therefore the table.
            len(self.kv_cache.window_block_ids(adapter)),
            int(self.kv_cache.block_size),
            # A committing forward must be unique: never share, never be shared.
            self._fctx_epoch_token(commit_current),
        )

    def _fctx_epoch_token(self, commit_current: bool) -> int:
        """Monotonic token that makes committing forwards non-cacheable."""
        if not commit_current:
            return 0
        self._fctx_commit_seq += 1
        return self._fctx_commit_seq

    @staticmethod
    def _fctx_key_diff(prev: tuple, cur: tuple) -> str:
        names = (
            "session_id",
            "kv_branch",
            "adapter_id",
            "seq_len",
            "commit_current",
            "committed_tokens",
            "num_computed_tokens",
            "window_blocks",
            "block_size",
            "commit_epoch",
        )
        parts = [f"{n}: {a!r} -> {b!r}" for n, a, b in zip(names, prev, cur, strict=False) if a != b]
        return "; ".join(parts) if parts else "key equal but entry dropped"

    def fctx_cache_stats(self) -> dict[str, int | str | None]:
        """Cache counters, for benchmarks and for asserting the reuse pattern."""
        out: dict[str, int | str | None] = dict(self.fctx_counters)
        out["fctx_cache_entries"] = len(self._paged_cache)
        out["fctx_last_miss_reason"] = self._fctx_last_miss_reason
        return out

    def _fctx_invalidate(self, kv_branch: str | None = None, *, reason: str = "explicit") -> None:
        """Drop cached forward contexts so the next forward rebuilds."""
        if kv_branch is None:
            n = len(self._paged_cache)
            self._paged_cache.clear()
        else:
            n = 1 if kv_branch in self._paged_cache else 0
            self._paged_cache.pop(kv_branch, None)
        if n:
            self.fctx_counters["fctx_cache_invalidate"] += n
            self._fctx_last_miss_reason = reason
            _log.debug("AR-Diffusion fctx cache DROP [%s] reason=%s", kv_branch or "*", reason)

    def commit_paged_context(self, kv_branch: str) -> None:
        """Commit managed current blocks after one successful KV branch forward."""
        self.adapter(kv_branch)
        ctx = self._paged_pending.get(kv_branch)
        if ctx is None:
            return
        if ctx.commit_current and ctx._allocated_video:
            n_chunks = ctx.seq_len // self.kv_cache.spec.chunk_size
            for _ in range(n_chunks):
                self.kv_cache.commit_chunk(ctx.adapter)
            self._committed[kv_branch] += ctx.seq_len
            _log.debug(
                "AR-Diffusion COMMIT [%s] new_tokens=%d chunks=%d resident=%d/%d",
                kv_branch,
                ctx.seq_len,
                n_chunks,
                len(self.kv_cache.window_block_ids(ctx.adapter)),
                self.kv_cache.spec.window_chunks,
            )
        ctx.mark_committed()
        self._paged_pending[kv_branch] = None
        # A commit advances the KV mapping (new resident blocks, possibly an
        # eviction), so anything cached for this branch is stale. The key would
        # already miss on committed_tokens/num_computed_tokens; dropping it here is
        # the belt-and-braces guarantee against stale block-table or slot-mapping
        # reuse, and it releases the device tensors immediately.
        self._fctx_invalidate(kv_branch, reason="kv commit")

    def is_cross_attention_populated(self, kv_branch: str, cache_name: str) -> bool:
        self.adapter(kv_branch)
        return self.kv_cache.is_cross_attention_populated(self.session_id, cache_name, kv_branch)

    def populate_cross_attention(
        self,
        kv_branch: str,
        cache_name: str,
        layer_kv: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Publish one named logical-branch cache after all layers are written.

        The iterable is consumed once in layer order. If projection, validation,
        or copying fails, the previous complete cache (if any) remains visible.
        """
        self.adapter(kv_branch)
        self.kv_cache.populate_cross_attention(self.session_id, cache_name, kv_branch, layer_kv)

    def get_cross_attention_kv(self, kv_branch: str, cache_name: str) -> list[dict[str, torch.Tensor | bool]]:
        """Return all layers for one populated named cross-attention cache."""
        if not self.is_cross_attention_populated(kv_branch, cache_name):
            raise RuntimeError(
                f"AR-Diffusion cross-attention cache {cache_name!r} for KV branch {kv_branch!r} "
                "was read before it was populated"
            )
        return [
            self.kv_cache.read_cross_attention_kv(self.session_id, cache_name, i, kv_branch)
            for i in range(self.num_layers)
        ]

    def clear_cross_attention(self) -> None:
        """Invalidate all named cross-attention caches without dropping self-KV.

        Prompt changes may preserve the autoregressive world history while
        requiring fresh text projections. This operation deliberately leaves
        branch adapters and their resident self-attention blocks untouched.
        """
        if self._closed:
            raise RuntimeError(f"AR-Diffusion session {self.session_id!r} is closed")
        self.kv_cache.release_cross_attention(self.session_id)

    def close(self) -> None:
        """Release all self- and cross-attention storage owned by this session."""
        if self._closed:
            return
        for adapter in self.adapters.values():
            self.kv_cache.end_request(adapter)
        self.kv_cache.release_cross_attention(self.session_id)
        self._paged_pending = dict.fromkeys(self.adapters)
        # Never let a cached context (and its device tensors) outlive the session:
        # that would both leak and risk reuse against freed blocks.
        self._fctx_invalidate(reason="session close")
        self._closed = True

    def reset(self, *, keep_cross_attention: Collection[str] = ()) -> None:
        """Release resident blocks and reopen this session with fresh adapters.

        ``keep_cross_attention`` names allocations whose conditioning remains
        valid across a model-internal window reset. Ordinary session reset
        should leave it empty.
        """
        unknown = set(keep_cross_attention) - set(self.kv_cache.cross_attention_lengths)
        if unknown:
            raise KeyError(f"Unknown AR-Diffusion cross-attention caches to keep: {sorted(unknown)}")
        request_ids = {kv_branch: adapter.request_id for kv_branch, adapter in self.adapters.items()}
        if keep_cross_attention:
            for adapter in self.adapters.values():
                self.kv_cache.end_request(adapter)
            self.kv_cache.retain_cross_attention(self.session_id, keep_cross_attention)
        else:
            self.close()
        self.adapters = {
            kv_branch: self.kv_cache.begin_request(request_id) for kv_branch, request_id in request_ids.items()
        }
        self._committed = dict.fromkeys(self.adapters, 0)
        self._paged_pending = dict.fromkeys(self.adapters)
        # reset() installs brand new adapters and zeroes the committed counters, so
        # every cached block table and slot mapping now addresses blocks this
        # session no longer owns. Drop unconditionally -- the keep_cross_attention
        # branch above skips close(), so relying on close()'s invalidation alone
        # would leave stale entries behind on exactly that path.
        self._fctx_invalidate(reason="session reset")
        self._closed = False
        _log.info(
            "AR-Diffusion RESET session=%s KV branches=%s kept_cross=%s",
            self.session_id,
            self.kv_branch_names,
            sorted(keep_cross_attention),
        )
