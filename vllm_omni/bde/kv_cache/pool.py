# SPDX-License-Identifier: Apache-2.0
"""Construct a vLLM ``KVCacheManager`` for the BDE engine.

Phase 1 uses static pool sizing (no profiling run): the worker passes the free
device memory and a budget fraction. Construction is metadata-only — the backing
KV tensors are allocated by the worker; here we only build the block bookkeeping
so allocation / eviction / free flow through vLLM's ``KVCacheManager``.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)


def compute_num_blocks(
    available_bytes: int,
    gpu_memory_fraction: float,
    page_size_bytes: int,
) -> int:
    """Number of KV blocks that fit in ``fraction`` of the memory budget."""
    if page_size_bytes <= 0:
        raise ValueError(f"page_size_bytes must be positive, got {page_size_bytes}")
    if not 0.0 < gpu_memory_fraction <= 1.0:
        raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
    budget = int(available_bytes * gpu_memory_fraction)
    return max(0, budget // page_size_bytes)


def build_kv_manager(
    spec: KVCacheSpec,
    layer_names: Sequence[str],
    num_blocks: int,
    max_model_len: int,
    *,
    enable_caching: bool = False,
) -> KVCacheManager:
    """Build a ``KVCacheManager`` with a single KV cache group for ``spec``.

    Args:
        spec: The KV cache spec for the group (e.g. a ``ChunkWindowSpec``).
        layer_names: Attention layers sharing this group's block table.
        num_blocks: Total physical blocks in the pool.
        max_model_len: Upper bound on a request's sequence length.
        enable_caching: Cross-request prefix caching (Phase 3); off in Phase 1.
    """
    layer_names = list(layer_names)
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
    tensors = [KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=layer_names)]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[group],
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        enable_caching=enable_caching,
    )
