# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen Talker generated-block owner identity on the native prefix hash chain.

Generated blocks are private to a scheduler admission. Codec production and
replay remain on the runner; this module only extends native block hashing.
"""

from collections.abc import Callable
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys, hash_block_tokens
from vllm.v1.request import Request

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner


def get_talker_request_block_hasher(
    hash_block_size: int, caching_hash_fn: Callable[[Any], bytes]
) -> Callable[[Request], list[BlockHash]]:
    """Extend native full-block hashing with request-local generated identity.

    Pass the resolved *hash* block size, not a guessed KV page size. Pure
    prefill retains native hashes. Generated-containing blocks augment native
    extras with the scheduler admission owner, isolating generated history to
    the request/admission that produced the immutable local codec decisions.
    Only new full blocks are inspected, with no prompt-wide hash or tensor read
    during decode.
    """
    if type(hash_block_size) is not int or hash_block_size <= 0:
        raise ValueError("Talker hash block size must be a positive integer")

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start = len(request.block_hashes) * hash_block_size
        mm_index = -1 if start else 0
        parent = request.block_hashes[-1] if request.block_hashes else None
        hashes: list[BlockHash] = []
        owner = getattr(request, "_omni_prefix_cache_owner", None)
        frontier = request.num_tokens
        for end in range(start + hash_block_size, frontier + 1, hash_block_size):
            extra_keys, mm_index = generate_block_hash_extra_keys(request, start, end, mm_index)
            if end > request.num_prompt_tokens:
                if not isinstance(owner, PrefixCacheRequestOwner):
                    raise ValueError("Talker generated block hashing requires a scheduler-owned admission")
                owner_key = ("qwen3-omni.generated-owner.v1", (owner.admission_id, owner.generation))
                extra_keys = (extra_keys or ()) + (owner_key,)
            parent = hash_block_tokens(caching_hash_fn, parent, request.all_token_ids[start:end], extra_keys)
            hashes.append(parent)
            start = end
        return hashes

    return request_block_hasher
