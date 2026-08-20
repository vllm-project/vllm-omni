# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import RequestStatus


@dataclass(frozen=True)
class DiffusionKVContext:
    """An independently managed K/V context outside the primary sequence.

    ``context_id`` identifies the logical context within one execution
    sequence. ``cache_role`` binds it to a logical attention cache role exposed
    by the Worker. Physical cache geometry remains native ``KVCacheSpec`` /
    ``KVCacheConfig`` state and is deliberately absent here.
    """

    context_id: str
    cache_role: str
    num_tokens: int
    block_hashes: tuple[BlockHash, ...] = ()

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id must be non-empty")
        if not self.cache_role:
            raise ValueError("cache_role must be non-empty")
        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {self.num_tokens}")
        object.__setattr__(self, "block_hashes", tuple(self.block_hashes))


class DiffusionKVRequest:
    """Scheduler-owned KV state for one diffusion execution sequence.

    The primary sequence follows an ordered ``[prefix | target]`` policy, such
    as one Hunyuan CFG row. ``prefix_len`` is the contiguous reusable prefix;
    ``target_len`` is overwritten by every denoise step; ``num_tokens`` is the
    complete first-step allocation boundary.

    Independent cross/joint-attention K/V does not belong to that token axis
    and is described by ``kv_contexts``. This object also exposes the minimal
    mutable Request surface consumed by native ``KVCacheManager``.

    An empty ``block_hashes`` sequence means the prefix has no canonical cache
    identity yet. Such a request may use native request-local page allocation,
    but consumers must not publish it through ``KVCacheManager.cache_blocks``.
    Prefix publication becomes valid only after model preprocessing supplies
    one canonical hash for every cacheable full block.
    """

    def __init__(
        self,
        request_id: str,
        *,
        sequence_id: int,
        prefix_len: int,
        target_len: int,
        seq_len: int,
        block_hashes: Sequence[BlockHash] = (),
        kv_contexts: Sequence[DiffusionKVContext] = (),
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if sequence_id < 0:
            raise ValueError(f"sequence_id must be non-negative, got {sequence_id}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if prefix_len + target_len > seq_len:
            raise ValueError(
                "prefix_len + target_len must not exceed seq_len: "
                f"prefix_len={prefix_len}, target_len={target_len}, seq_len={seq_len}"
            )

        contexts = tuple(kv_contexts)
        if any(not isinstance(context, DiffusionKVContext) for context in contexts):
            raise TypeError("kv_contexts must contain only DiffusionKVContext values")
        context_ids = [context.context_id for context in contexts]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError(f"kv_contexts must have unique context_id values, got {context_ids}")

        # Diffusion semantics consumed by the Scheduler-side facade.
        self.sequence_id = sequence_id
        self.prefix_len = prefix_len
        self.target_len = target_len
        self.kv_contexts = contexts

        # Native vLLM Request surface. Keep this list intentionally small and
        # cover it with real-KVCacheManager conformance tests.
        self.request_id = request_id
        self.num_tokens = seq_len
        self.num_prompt_tokens = prefix_len
        self.num_computed_tokens = 0
        self.block_hashes = list(block_hashes)
        self.skip_reading_prefix_cache = not self.block_hashes
        self.status = RequestStatus.WAITING
        self.num_preemptions = 0
        self.num_in_flight_tokens = 0
        # vLLM 0.26+ uses this optional boundary when publishing cached blocks;
        # zero means that no sparse-retention boundary is active.
        self.shared_prefix_boundary = 0

    @property
    def seq_len(self) -> int:
        """Complete first-step sequence length and allocation boundary."""

        return self.num_tokens
