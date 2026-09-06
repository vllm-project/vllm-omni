# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned CPU block mirror (mirrors vLLM BlockPool)."""

import logging

import torch

from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig

logger = logging.getLogger(__name__)


class PrefixBlockPool:
    """Durable CPU mirror of vLLM KV slots, one slab per tensor name.

    Storage is ``(num_blocks, block_size, feat)`` per key, viewed flat as
    ``(num_slots, feat)`` so vLLM slot ids index rows directly. Write
    access is single-writer (the controller committer thread); readers
    take row views.

    ``_caches`` is not an unbounded store and has no LRU. Dict keys are
    tensor names (``__hidden_states__`` plus token-major mm fields),
    opened once by ``ensure_key`` and never dropped — a model emits a
    handful of those names, not a per-request set. Each value is a
    fixed slab sized to the same ``num_blocks`` as the upstream KV
    pool. Rows *are* those kv slots: when vLLM recycles a block, the
    next scatter overwrites that row. Slot reuse is the eviction.
    """

    def __init__(self, config: PrefixCacheConfig):
        self._config = config
        self._caches: dict[str, torch.Tensor] = {}

    def _alloc(self, dtype: torch.dtype, feat: int) -> torch.Tensor:
        return torch.zeros(
            (self._config.num_blocks, self._config.block_size, feat),
            dtype=dtype,
            device="cpu",
            # Pinning enables true async D2H and fast scatter; unsupported on
            # CPU-only builds where the async pipeline is off anyway.
            pin_memory=torch.cuda.is_available(),
        )

    def ensure_key(self, key: str, dtype: torch.dtype, feat: int) -> None:
        if key in self._caches:
            return
        self._caches[key] = self._alloc(dtype, feat)
        logger.info("prefix_cache: initialized mirror %s for key %s", list(self._caches[key].shape), key)

    def has_key(self, key: str) -> bool:
        return key in self._caches

    def keys(self) -> set[str]:
        return set(self._caches.keys())

    def _flat(self, key: str) -> torch.Tensor:
        cache = self._caches[key]
        return cache.view(-1, cache.shape[-1])

    def rows(self, key: str, slots: torch.Tensor) -> torch.Tensor:
        """Gather rows for non-contiguous slots (returns a copy)."""
        return self._flat(key).index_select(0, slots)

    def write(self, key: str, slots: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Row scatter; caller (committer thread) is the single writer."""
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        # index_copy_ dispatches to a faster single-dim CPU path than
        # advanced-indexing assignment (aten::index_put_).
        self._flat(key).index_copy_(0, slots, src_cpu)
