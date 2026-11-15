# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduler test helpers shared by AR / chunk-transfer unit tests."""

from __future__ import annotations

from types import MethodType

import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


def make_kv_cache_manager(block_size: int) -> KVCacheManager:
    """Build a small real manager for scheduler-side CPU connector tests."""
    # A foreign spec can populate the registry before native specs are loaded.
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=32,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=False,
    )


def bind_omits_transfer_helpers(sched) -> None:
    """Bind the real omit helpers so a MagicMock scheduler does not skip save_async.

    MagicMock is truthy, so an unbound ``_request_omits_chunk_transfer_to_next_stage``
    silently skips the chunk put.
    """
    sched._omits_kv_transfer_cache = {}
    sched._omni_final_stage_flags = MethodType(OmniARScheduler._omni_final_stage_flags, sched)
    sched._request_omits_kv_transfer_to_next_stage = MethodType(
        OmniARScheduler._request_omits_kv_transfer_to_next_stage,
        sched,
    )
    sched._request_omits_chunk_transfer_to_next_stage = MethodType(
        OmniARScheduler._request_omits_chunk_transfer_to_next_stage,
        sched,
    )
