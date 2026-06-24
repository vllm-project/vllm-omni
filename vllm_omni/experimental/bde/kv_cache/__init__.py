# SPDX-License-Identifier: Apache-2.0
"""BDE engine-level KV cache helpers.

Thin glue over vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` /
``SlidingWindowManager``), used by the Block Diffusion Engine to manage KV for
AR-diffusion models. See ``BDE_doc/dreamzero_kv_phase1_plan.md``.

Layout: ``config`` (public knob) · ``paged`` (engine-generic paging mechanics +
chunk-window eviction spec) · ``manager`` (the BDEKVCache orchestrator + its
request adapter / pool builders) · ``state`` (the model-facing BDEKVState bridge).
"""

from vllm_omni.experimental.bde.kv_cache.config import BDEKVConfig
from vllm_omni.experimental.bde.kv_cache.manager import (
    BDEKVCache,
    BDERequestAdapter,
    build_kv_manager,
    compute_num_blocks,
)
from vllm_omni.experimental.bde.kv_cache.paged import (
    ChunkWindowManager,
    ChunkWindowSpec,
    allocate_kv_pool,
    chunk_slot_mapping,
    compute_slot_mapping,
    pool_gather_window,
    pool_write_chunk,
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
