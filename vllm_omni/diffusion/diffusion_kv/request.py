# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys, hash_block_tokens
from vllm.v1.request import RequestStatus


@dataclass(frozen=True)
class DiffusionKVContext:
    """An independently managed K/V context outside the primary sequence.

    ``context_id`` identifies the logical context within one public request.
    Multiple execution sequences may reference the same context, but must give
    it the same definition. ``cache_role`` binds it to a logical attention
    cache role exposed by the Worker. Physical cache geometry remains native
    ``KVCacheSpec`` / ``KVCacheConfig`` state and is deliberately absent here.
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
    With prefix caching enabled, Engine invokes model input preparation to
    attach token IDs and native multimodal identities / positions. The Manager
    builds one canonical hash for every cacheable full block before lookup;
    publication requires those hashes and successfully materialized KV.
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
        cache_token_ids: Sequence[int] = (),
        mm_features: Sequence[MultiModalFeatureSpec] = (),
        cache_namespace: str = "diffusion-primary-v1",
        kv_contexts: Sequence[DiffusionKVContext] = (),
        kv_transfer_params: dict[str, Any] | None = None,
        prompt_token_ids: list[int] | None = None,
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
        token_ids = tuple(cache_token_ids)
        if token_ids and len(token_ids) < prefix_len:
            raise ValueError(
                f"cache_token_ids must cover the reusable prefix: tokens={len(token_ids)}, prefix_len={prefix_len}"
            )
        if any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("cache_token_ids must contain only integers")
        if not isinstance(cache_namespace, str) or not cache_namespace:
            raise ValueError("cache_namespace must be a non-empty string")

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
        self.cache_token_ids = token_ids
        self.mm_features = list(mm_features)
        self.cache_namespace = cache_namespace
        # Native generate_block_hash_extra_keys consumes this Request surface.
        # Diffusion adapter ID + scale are already isolated in cache_namespace;
        # there are no separate LoRA-name, salt, or prompt-embedding inputs.
        self.lora_request = None
        self.cache_salt = None
        self.prompt_embeds = None

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
        # Opaque request-scoped parameters consumed by an upstream KVConnector.
        self.kv_transfer_params = kv_transfer_params
        self.prompt_token_ids = prompt_token_ids

    def _validate_mm_features(self) -> None:
        # Native block traversal expects sorted, non-overlapping placeholders.
        # Do not silently accept nested/overlapping ranges: the traversal can
        # skip a feature and incorrectly reuse KV. Combine such dependencies
        # in the feature identifier instead.
        previous_end = 0
        for feature in self.mm_features:
            if not isinstance(feature, MultiModalFeatureSpec):
                raise TypeError("mm_features must contain native MultiModalFeatureSpec values")
            position = feature.mm_position
            end = position.offset + position.length
            if position.offset < previous_end or position.length <= 0 or end > self.seq_len:
                raise ValueError("mm_features must have valid sorted, non-overlapping positions within seq_len")
            if not isinstance(feature.identifier, str) or not feature.identifier:
                raise ValueError("mm_features must have non-empty content identifiers")
            previous_end = end

    def build_block_hashes(
        self,
        hash_block_size: int,
        hash_function: Callable[[object], bytes],
    ) -> None:
        """Build native chained hashes for complete reusable-prefix blocks."""

        if self.block_hashes:
            self.skip_reading_prefix_cache = False
            return
        if hash_block_size <= 0:
            raise ValueError(f"hash_block_size must be positive, got {hash_block_size}")
        num_cache_tokens = self.prefix_len // hash_block_size * hash_block_size
        if num_cache_tokens == 0 or not self.cache_token_ids:
            return
        if len(self.cache_token_ids) < num_cache_tokens:
            raise ValueError(
                "cache_token_ids do not cover the block-aligned reusable prefix: "
                f"tokens={len(self.cache_token_ids)}, required={num_cache_tokens}"
            )
        self._validate_mm_features()

        # Keep the first hash deterministic without mutating vLLM's process-wide
        # ``NONE_HASH``.  A diffusion manager can be created more than once in
        # one process (for example in tests or worker reinitialization); using
        # the global random root would make requests built by different manager
        # instances impossible to match.
        parent_hash: BlockHash | None = BlockHash(hash_function(("diffusion-prefix-root-v1", self.cache_namespace)))
        hashes: list[BlockHash] = []
        mm_index = 0
        for start in range(0, num_cache_tokens, hash_block_size):
            end = start + hash_block_size
            # Reuse upstream MM intersection/offset logic directly. Keep our
            # stable-prefix bound: native get_request_block_hasher hashes the
            # entire token axis, including diffusion's mutable target region.
            extra_keys, mm_index = generate_block_hash_extra_keys(self, start, end, mm_index)
            parent_hash = hash_block_tokens(
                hash_function=hash_function,
                parent_block_hash=parent_hash,
                curr_block_token_ids=self.cache_token_ids[start:end],
                extra_keys=extra_keys,
            )
            hashes.append(parent_hash)

        self.block_hashes = hashes
        self.skip_reading_prefix_cache = not hashes

    @property
    def seq_len(self) -> int:
        """Complete first-step sequence length and allocation boundary."""

        return self.num_tokens
