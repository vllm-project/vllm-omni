# SPDX-License-Identifier: Apache-2.0
"""AR-Diffusion engine-level KV cache helpers.

Thin glue over vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` /
``SlidingWindowManager``), used by the AR-Diffusion Engine to manage KV for
AR-diffusion models.

Layout: ``config`` (public knob) · ``paged`` (engine-generic paging mechanics +
chunk-window eviction spec) · ``manager`` (the ARDiffusionKVCache orchestrator + its
request adapter / pool builders) · ``state`` (the model-facing ARDiffusionKVState bridge).
"""

from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import (
    ARDiffusionKVCache,
    ARDiffusionRequestAdapter,
    build_kv_manager,
    compute_num_blocks,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    ChunkWindowManager,
    ChunkWindowSpec,
    allocate_kv_pool_with_views,
    chunk_slot_mapping,
    compute_slot_mapping,
    pool_write_chunk,
    resident_block_ids,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedForwardContext,
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
    set_current_paged_kv_cache,
)

__all__ = [
    "ARDiffusionKVCache",
    "ARDiffusionKVConfig",
    "ARDiffusionPagedForwardContext",
    "ARDiffusionPagedLayerContext",
    "ARDiffusionPagedLayerInputs",
    "ARDiffusionRequestAdapter",
    "ChunkWindowManager",
    "ChunkWindowSpec",
    "allocate_kv_pool_with_views",
    "ar_diffusion_paged_attention",
    "build_kv_manager",
    "chunk_slot_mapping",
    "compute_num_blocks",
    "compute_slot_mapping",
    "paged_write_attn",
    "pool_write_chunk",
    "resident_block_ids",
    "set_current_paged_kv_cache",
]
