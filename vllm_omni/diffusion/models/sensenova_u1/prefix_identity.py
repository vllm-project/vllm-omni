# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical block-hash computation for diffusion prefix caching (B3).

`DiffusionKVRequest.block_hashes` documents the rule this module implements:
a cached prefix may only be published once model preprocessing supplies **one
canonical hash for every cacheable full block**. Today nothing populates that
field, so `skip_reading_prefix_cache` stays True and prefix reuse never happens
(`vllm_omni/diffusion/diffusion_kv/request.py`).

This module is the hash-production step only. It is intentionally pure:

* it does not import `vllm` at module scope, so it is importable and testable on
  a CPU-only stack;
* it does not decide *where* hashes are computed, *whether* caching is enabled,
  or how the cache is owned — those remain the scheduler/managers' decisions;
* it changes no runtime behaviour by itself.

Identity model. A block's identity must cover everything its computation
depends on, otherwise two different computations collide on one cache entry:

``branch``
    `cond` / `uncond` / `img_cond` are computed from different conditioning, so
    a first version must not share across them (plan 4.1).
``model_epoch``
    weight revision / adapter revision. KV computed under other weights must
    never be reused.
``tp_rank`` / ``tp_size``
    KV is rank-local; a handle from another rank or another sharding degree is a
    different object.
``token_block``
    the block's own token ids, which the parent chain does not cover.
``parent``
    chained ancestor hash, so a shared suffix under a different prefix cannot
    collide.

This mirrors what vLLM already does natively: `vllm.v1.core.kv_cache_utils.
hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids,
extra_keys)` — the branch/epoch/rank layers belong in `extra_keys`, and the
ancestor chaining in `parent_block_hash`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PrefixCacheIdentity",
    "REUSABLE_BRANCHES",
    "canonical_block_hash",
    "prefix_block_hashes",
    "reusable_block_count",
    "stable_hash_function",
    "truncate_to_reusable_blocks",
    "hashes_fit_blocks",
]

# Branches whose prefixes are computed differently and must not be shared in the
# first version (plan 4.1). Kept as a closed set so an unknown branch fails loudly
# rather than silently sharing a namespace.
REUSABLE_BRANCHES = ("cond", "uncond", "img_cond")


def stable_hash_function(data: Any) -> bytes:
    """Deterministic bytes hash over a nested structure of ints/strs/tuples/bytes.

    Used as vLLM's `hash_function` argument. Python's built-in `hash()` is salted
    per process and must not be used for a cross-process cache identity, so this
    serialises explicitly and hashes with sha256.
    """
    digest = hashlib.sha256()

    def feed(value: Any) -> None:
        if value is None:
            digest.update(b"n")
        elif isinstance(value, bool):
            digest.update(b"b1" if value else b"b0")
        elif isinstance(value, int):
            digest.update(b"i" + str(value).encode())
        elif isinstance(value, str):
            payload = value.encode()
            digest.update(b"s" + str(len(payload)).encode() + b":" + payload)
        elif isinstance(value, (bytes, bytearray)):
            digest.update(b"y" + str(len(value)).encode() + b":" + bytes(value))
        elif isinstance(value, (tuple, list)):
            digest.update(b"t" + str(len(value)).encode())
            for item in value:
                feed(item)
        else:
            raise TypeError(f"unsupported value for a stable block hash: {type(value).__name__}")

    feed(data)
    return digest.digest()


@dataclass(frozen=True)
class PrefixCacheIdentity:
    """The model-execution part of a block's cache identity.

    Everything here must be constant for the whole lifetime of a cache entry: a
    change to any field makes previously published KV invalid.
    """

    branch: str
    model_epoch: int
    tp_rank: int = 0
    tp_size: int = 1
    # Optional extra execution-identity keys (e.g. a genuinely participating
    # adapter id). Only things that change the prefix computation belong here:
    # seed / output size / sampling parameters generally do not.
    extra: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.branch not in REUSABLE_BRANCHES:
            raise ValueError(f"unknown branch {self.branch!r}; expected one of {REUSABLE_BRANCHES}")
        if self.model_epoch < 0:
            raise ValueError(f"model_epoch must be non-negative, got {self.model_epoch}")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"tp_rank must be in [0, {self.tp_size}), got {self.tp_rank}")

    def extra_keys(self) -> tuple[Any, ...]:
        """The `extra_keys` payload for vLLM's `hash_block_tokens`.

        Includes the whole rank topology, not only this rank: a handle is only
        meaningful within one TP degree.
        """
        return (
            f"branch={self.branch}",
            f"epoch={self.model_epoch}",
            f"tp={self.tp_rank}/{self.tp_size}",
            *self.extra,
        )


def canonical_block_hash(
    identity: PrefixCacheIdentity,
    token_ids: Sequence[int],
    block_index: int,
    block_size: int,
    *,
    parent_hash: bytes | None = None,
    hash_function: Callable[[Any], bytes] = stable_hash_function,
) -> bytes:
    """Hash exactly one full block of `token_ids`.

    `token_ids` must be the complete current prefix input, so that block
    boundaries are derived from the real token sequence rather than from
    separately-tokenised fragments (plan 4.1). Raises if the block is not full:
    a partially-filled block has no canonical identity and must not be published.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_index < 0:
        raise ValueError(f"block_index must be non-negative, got {block_index}")
    start = block_index * block_size
    end = start + block_size
    if end > len(token_ids):
        raise ValueError(
            f"block {block_index} is not a full block of {len(token_ids)} tokens "
            f"(needs {end})"
        )
    block = tuple(int(t) for t in token_ids[start:end])
    if parent_hash is None:
        parent_hash = b""
    return hash_function((parent_hash, block, identity.extra_keys()))


def prefix_block_hashes(
    identity: PrefixCacheIdentity,
    token_ids: Sequence[int],
    block_size: int,
    num_reusable_tokens: int,
    *,
    hash_function: Callable[[Any], bytes] = stable_hash_function,
) -> list[bytes]:
    """Ancestor-chained hashes for the reusable prefix, one per full block.

    `num_reusable_tokens` must come from the semantic cut (the largest cut that
    is both block-aligned and dependency-closed for the AR mask), not from a
    rounding of the matched length — see the B3 contract note. It must be a
    multiple of `block_size`.
    """
    if num_reusable_tokens < 0:
        raise ValueError(f"num_reusable_tokens must be non-negative, got {num_reusable_tokens}")
    if num_reusable_tokens % block_size:
        raise ValueError(
            f"num_reusable_tokens={num_reusable_tokens} is not a multiple of block_size={block_size}"
        )
    hashes: list[bytes] = []
    parent: bytes | None = None
    for block_index in range(num_reusable_tokens // block_size):
        parent = canonical_block_hash(
            identity,
            token_ids,
            block_index,
            block_size,
            parent_hash=parent,
            hash_function=hash_function,
        )
        hashes.append(parent)
    return hashes


def reusable_block_count(num_tokens: int, block_size: int) -> int:
    """How many whole blocks of a request can ever be reused.

    Native vLLM caps the reported prefix hit at `num_tokens - 1` (the request's
    own final block must be recomputed so its last token is available for
    sampling), rounded down to whole blocks. Verified against vLLM 0.26's
    `KVCacheManager` over 18 published/extra combinations; see
    `artifacts/b3_native_manager.py`.

    This matters for B3 reporting: a full "hash hit" over N published tokens does
    **not** mean N tokens are reused.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_tokens <= 0:
        return 0
    return (num_tokens - 1) // block_size


def truncate_to_reusable_blocks(num_tokens: int, block_size: int) -> int:
    """The largest token count that may be reused for a `num_tokens` request."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_tokens <= 0:
        return 0
    return ((num_tokens - 1) // block_size) * block_size


def hashes_fit_blocks(hashes: Sequence[bytes], num_tokens: int, block_size: int) -> bool:
    """Whether a hash list is within what the request can actually reuse."""
    return len(hashes) * block_size <= truncate_to_reusable_blocks(num_tokens, block_size)
