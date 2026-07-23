# SPDX-License-Identifier: Apache-2.0
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
        return [ARDiffusionPagedLayerContext(layer_idx=i, forward_ctx=forward_ctx) for i in range(self.num_layers)]

    def commit_paged_context(self, kv_branch: str) -> None:
        """Commit managed current blocks after one successful KV branch forward."""
        self.adapter(kv_branch)
        ctx = self._paged_pending.get(kv_branch)
        if ctx is None:
            return
        if ctx.commit_current and ctx._allocated_video:
            n_chunks = ctx.seq_len // self.kv_cache.spec.chunk_size
            for _ in range(n_chunks):
                ctx.adapter.on_chunk_committed()
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

    def close(self) -> None:
        """Release all self- and cross-attention storage owned by this session."""
        if self._closed:
            return
        for adapter in self.adapters.values():
            self.kv_cache.end_request(adapter)
        self.kv_cache.release_cross_attention(self.session_id)
        self._paged_pending = dict.fromkeys(self.adapters)
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
        self._closed = False
        _log.info(
            "AR-Diffusion RESET session=%s KV branches=%s kept_cross=%s",
            self.session_id,
            self.kv_branch_names,
            sorted(keep_cross_attention),
        )
