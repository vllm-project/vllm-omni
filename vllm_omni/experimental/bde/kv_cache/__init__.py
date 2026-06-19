# SPDX-License-Identifier: Apache-2.0
"""BDE engine-level KV cache helpers.

Thin glue over vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` /
``SlidingWindowManager``), used by the Block Diffusion Engine to manage KV for
AR-diffusion models. See ``BDE_doc/dreamzero_kv_phase1_plan.md``.
"""

from vllm_omni.experimental.bde.kv_cache.adapter import BDERequestAdapter
from vllm_omni.experimental.bde.kv_cache.chunk_window import ChunkWindowManager, ChunkWindowSpec
from vllm_omni.experimental.bde.kv_cache.config import BDEKVConfig
from vllm_omni.experimental.bde.kv_cache.gather import (
    allocate_kv_pool,
    pool_gather_window,
    pool_write_chunk,
)
from vllm_omni.experimental.bde.kv_cache.manager import BDEKVCache
from vllm_omni.experimental.bde.kv_cache.pool import build_kv_manager, compute_num_blocks
from vllm_omni.experimental.bde.kv_cache.slot_mapping import (
    chunk_slot_mapping,
    compute_slot_mapping,
    resident_block_ids,
)

__all__ = [
    "BDEKVCache",
    "BDEKVConfig",
    "BDERequestAdapter",
    "ChunkWindowManager",
    "ChunkWindowSpec",
    "allocate_kv_pool",
    "build_kv_manager",
    "chunk_slot_mapping",
    "compute_num_blocks",
    "compute_slot_mapping",
    "pool_gather_window",
    "pool_write_chunk",
    "resident_block_ids",
]
