# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for omni schedulers."""

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


def free_kv_blocks_in_physical_order(manager: KVCacheManager, request: Request) -> None:
    """Return non-cached request blocks in increasing physical order.

    vLLM's block pool deliberately honors the eviction-priority order passed
    to ``free_blocks``.  For a non-caching P/D pool there is no LRU priority to
    preserve, so physical order lets native connectors coalesce adjacent
    source/destination pages on the next allocation.
    """

    if manager.enable_caching:
        manager.free(request)
        return
    blocks = manager.pop_blocks_for_free(request)
    manager.block_pool.free_blocks(sorted(blocks, key=lambda block: block.block_id))
