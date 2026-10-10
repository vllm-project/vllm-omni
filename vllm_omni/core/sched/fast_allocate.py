# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Skip ``allocate_slots`` for running decode rows whose blocks already fit.

The scheduler asks the KV cache manager for slots for every running request
on every step, but a decoding request needs a new block only once every
``block_size`` tokens. At high concurrency the full ``allocate_slots`` path
(watermark, skipped-block removal, allocation counting, caching) is a large
share of the scheduler's host time. For the configuration where it is a pure
function of the request's block count (one full-attention group, prefix
caching off) the common case is answered directly with the manager's empty
result; everything else takes the original path.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


def install_decode_allocate_fast_path(kv_cache_manager: Any) -> bool:
    """Wrap ``kv_cache_manager.allocate_slots`` on this instance when the fast path is exact."""
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    coordinator = getattr(kv_cache_manager, "coordinator", None)
    managers = getattr(coordinator, "single_type_managers", None)
    if (
        getattr(kv_cache_manager, "enable_caching", True)
        or type(coordinator) is not KVCacheCoordinatorNoPrefixCache
        or managers is None
        or len(managers) != 1
        or type(managers[0]) is not FullAttentionManager
        or getattr(managers[0], "_partial_hit_reqs", None)
    ):
        return False
    manager = managers[0]
    req_to_blocks = manager.req_to_blocks
    block_size = int(manager.block_size)
    max_model_len = int(kv_cache_manager.max_model_len)
    block_pool = kv_cache_manager.block_pool
    empty = kv_cache_manager.empty_kv_cache_blocks
    original = kv_cache_manager.allocate_slots
    running = RequestStatus.RUNNING

    def allocate_slots(
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        if (
            num_new_tokens > 0
            and request.status == running
            and new_computed_blocks is None
            and not (num_new_computed_tokens or num_external_computed_tokens or num_encoder_tokens)
            and not (full_sequence_must_fit or delay_cache_blocks)
        ):
            blocks = req_to_blocks.get(request.request_id)
            need = min(request.num_computed_tokens + num_new_tokens + num_lookahead_tokens, max_model_len)
            # The original returns None when the free pool is below the reservation, even for 0 blocks.
            if blocks and len(blocks) * block_size >= need and block_pool.get_num_free_blocks() >= reserved_blocks:
                return empty
        return original(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )

    kv_cache_manager.allocate_slots = allocate_slots
    logger.info("Decode allocate_slots fast path enabled (block_size=%d)", block_size)
    return True
