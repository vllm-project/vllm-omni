# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned CPU block mirror (mirrors vLLM BlockPool)."""

import logging

import torch

from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig, TensorName

logger = logging.getLogger(__name__)


class PrefixBlockPool:
    """Durable CPU mirror of vLLM KV slots, one tensor per name.

    Storage is ``(num_blocks, block_size, feat)`` per key, viewed flat as
    ``(num_slots, feat)`` so vLLM slot ids index rows directly. Write
    access is single-writer (the controller committer thread); readers
    take row views.

    ``_caches`` is not an unbounded store and has no LRU. Dict keys are
    tensor names (``__hidden_states__`` plus mm whose first dim is tokens),
    opened once by ``ensure_key`` and never dropped — a model emits a
    handful of those names, not a per-request set. Each value is a
    fixed tensor sized to the same ``num_blocks`` as the upstream KV
    pool. Rows *are* those kv slots: when vLLM recycles a block, the
    next pool write overwrites that row. Slot reuse is the eviction.
    """

    def __init__(self, config: PrefixCacheConfig):
        self._config = config
        self._caches: dict[TensorName, torch.Tensor] = {}

    def _alloc(self, dtype: torch.dtype, feat: int) -> torch.Tensor:
        return torch.zeros(
            (self._config.num_blocks, self._config.block_size, feat),
            dtype=dtype,
            device="cpu",
            # Pinning lets device→host overlap compute; unsupported on
            # CPU-only builds where that overlap is off anyway.
            pin_memory=torch.cuda.is_available(),
        )

    def alloc_key(self, key: str, dtype: torch.dtype, feat: int) -> torch.Tensor | None:
        """Allocate storage for a new ``key`` without publishing it (None if
        already open). Pinned allocation is slow; run this unlocked and
        hand the tensor to ``install_key`` under the manager's state lock."""
        if key in self._caches:
            return None
        return self._alloc(dtype, feat)

    def install_key(self, key: str, storage: torch.Tensor) -> None:
        if key in self._caches:
            return
        self._caches[key] = storage
        logger.info("prefix_cache: initialized mirror %s for key %s", list(storage.shape), key)

    def ensure_key(self, key: str, dtype: torch.dtype, feat: int) -> None:
        storage = self.alloc_key(key, dtype, feat)
        if storage is not None:
            self.install_key(key, storage)

    def has_key(self, key: str) -> bool:
        return key in self._caches

    def keys(self) -> set[TensorName]:
        return set(self._caches.keys())

    def memory_stats(self) -> tuple[dict[TensorName, int], int]:
        """Return per-key storage bytes and the pinned subset."""
        cache_bytes = {key: tensor.numel() * tensor.element_size() for key, tensor in self._caches.items()}
        pinned_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self._caches.values() if tensor.is_pinned()
        )
        return cache_bytes, pinned_bytes

    def key_metadata(self, key: TensorName) -> tuple[int, str] | None:
        """Return feature width and dtype for an allocated key."""
        tensor = self._caches.get(key)
        if tensor is None:
            return None
        return int(tensor.shape[-1]), str(tensor.dtype)

    def _flat(self, key: str) -> torch.Tensor:
        cache = self._caches[key]
        return cache.view(-1, cache.shape[-1])

    def rows(self, key: str, slots: torch.Tensor) -> torch.Tensor:
        """Gather rows for non-contiguous slots (returns a copy)."""
        return self._flat(key).index_select(0, slots)

    def write(self, key: str, slots: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Write rows into the pool; caller (committer thread) is the single writer."""
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        # index_copy_ dispatches to a faster single-dim CPU path than
        # advanced-indexing assignment (aten::index_put_).
        self._flat(key).index_copy_(0, slots, src_cpu)
