# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared utilities for omni schedulers."""

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


def streaming_update_replaces_prompt(update_infos: tuple[object, ...]) -> bool:
    """Whether this streaming update is a new utterance, not a prompt extension.

    MiniCPM sets a nested dict ``meta.replace_streaming_prompt``. Sentence TTS
    serializes the same flag through ``serialize_payload``, so it arrives as
    an ``AdditionalInformationPayload`` entry ``meta.replace_streaming_prompt``.
    The dict-only check missed that wire shape and appended the next sentence
    onto leftover Talker codec KV.
    """
    for info in update_infos:
        if isinstance(info, dict):
            meta = info.get("meta")
            if isinstance(meta, dict) and meta.get("replace_streaming_prompt") is True:
                return True
            if info.get("meta.replace_streaming_prompt") is True:
                return True
            continue
        entries = getattr(info, "entries", None)
        if not isinstance(entries, dict):
            continue
        entry = entries.get("meta.replace_streaming_prompt")
        if entry is not None and getattr(entry, "scalar_data", None) is True:
            return True
    return False


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
