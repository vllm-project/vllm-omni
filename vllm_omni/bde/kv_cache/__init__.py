# SPDX-License-Identifier: Apache-2.0
"""BDE engine-level KV cache helpers.

Thin glue over vLLM's paged KV stack (``KVCacheManager`` / ``BlockPool`` /
``SlidingWindowManager``), used by the Block Diffusion Engine to manage KV for
AR-diffusion models. See ``BDE_doc/dreamzero_kv_phase1_plan.md``.
"""

from vllm_omni.bde.kv_cache.adapter import BDERequestAdapter
from vllm_omni.bde.kv_cache.chunk_window import ChunkWindowManager, ChunkWindowSpec
from vllm_omni.bde.kv_cache.config import BDEKVConfig
from vllm_omni.bde.kv_cache.pool import build_kv_manager, compute_num_blocks

__all__ = [
    "BDEKVConfig",
    "BDERequestAdapter",
    "ChunkWindowManager",
    "ChunkWindowSpec",
    "build_kv_manager",
    "compute_num_blocks",
]
