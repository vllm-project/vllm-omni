# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_omni.core.sched.utils import free_kv_blocks_in_physical_order
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest


class DiffusionKVAdmissionError(ValueError):
    """A request-specific KV admission failure safe to return to the caller."""


class DiffusionKVCacheManager:
    """Atomic public-request facade over native vLLM KVCacheManager."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        *,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_hash_algo: str = "sha256",
    ) -> None:
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        if kv_cache_config.num_blocks <= 0 or not kv_cache_config.kv_cache_groups:
            raise ValueError("Diffusion KVCacheConfig must contain a positive block pool and at least one group")
        if scheduler_block_size <= 0 or hash_block_size <= 0:
            raise ValueError("scheduler_block_size and hash_block_size must be positive")
        if max_in_flight_tokens is not None and max_in_flight_tokens <= 0:
            raise ValueError(f"max_in_flight_tokens must be positive when provided, got {max_in_flight_tokens}")
        self._hash_function = get_hash_fn_by_name(prefix_caching_hash_algo)
        fine_grained_hits = enable_prefix_caching and hash_block_size < scheduler_block_size
        replace_unitary_coordinator = fine_grained_hits and len(kv_cache_config.kv_cache_groups) == 1
        self.native_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            scheduler_block_size=scheduler_block_size,
            # The unitary coordinator rejects fine-grained hashes. It is
            # replaced below before any request can observe this temporary
            # full-block coordinator.
            hash_block_size=scheduler_block_size if replace_unitary_coordinator else hash_block_size,
            max_in_flight_tokens=max_in_flight_tokens,
            enable_caching=enable_prefix_caching,
        )
        if fine_grained_hits:
            coordinator = (
                HybridKVCacheCoordinator(
                    kv_cache_config=kv_cache_config,
                    max_model_len=max_model_len,
                    max_in_flight_tokens=max_in_flight_tokens or max_model_len,
                    use_eagle=False,
                    enable_caching=True,
                    enable_kv_cache_events=False,
                    dcp_world_size=1,
                    pcp_world_size=1,
                    scheduler_block_size=scheduler_block_size,
                    hash_block_size=hash_block_size,
                    allow_partial_hash_hits=False,
                )
                if replace_unitary_coordinator
                else self.native_manager.coordinator
            )
            unsupported_managers = [
                type(manager).__name__
                for manager in coordinator.single_type_managers
                if not manager.supports_fine_grained_hash_lookup or manager.block_size % hash_block_size
            ]
            if unsupported_managers:
                raise ValueError(
                    f"Fine-grained diffusion prefix hits require compatible cache managers: {unsupported_managers}"
                )
            # Upstream enables this mode automatically only for hybrid
            # Mamba/full-attention layouts. Diffusion image boundaries need the
            # same native fine-grained lookup and COW path for full attention.
            coordinator.enable_partial_hash_hits = True
            for manager in coordinator.single_type_managers:
                manager.cache_hit_alignment_tokens = hash_block_size
            if replace_unitary_coordinator:
                self.native_manager.coordinator = coordinator
                self.native_manager.block_pool = coordinator.block_pool
                self.native_manager.retained_hit_group_ids = tuple(
                    manager.kv_cache_group_id
                    for manager in coordinator.single_type_managers
                    if manager.retains_longer_hit
                )
        zeroing_managers = [
            type(manager).__name__
            for manager in self.native_manager.coordinator.single_type_managers
            if manager.records_new_block_ids
        ]
        if zeroing_managers:
            raise NotImplementedError(
                f"Diffusion Worker KV block zeroing is not implemented for cache managers: {zeroing_managers}"
            )
        self.max_model_len = max_model_len
        self.hash_block_size = hash_block_size
        self.enable_prefix_caching = enable_prefix_caching
        # Native vLLM may reserve a null block, so an idle BlockPool does not
        # necessarily report ``kv_cache_config.num_blocks`` free blocks.
        self._empty_pool_num_free_blocks = self.native_manager.block_pool.get_num_free_blocks()
        self._requests: dict[str, tuple[DiffusionKVRequest, ...]] = {}
        self._metadata: dict[str, DiffusionKVMetadata] = {}
        self._cow_retained_blocks: dict[str, list[KVCacheBlock]] = {}
        self._internal_request_ids: set[str] = set()
        self._next_allocation_generation = 1

    def _prepare_block_hashes(self, requests: Sequence[DiffusionKVRequest]) -> None:
        if not self.enable_prefix_caching:
            return
        for request in requests:
            request.build_block_hashes(self.hash_block_size, self._hash_function)

    def _clip_computed_blocks(
        self,
        blocks: KVCacheBlocks,
        *,
        hit_len: int,
        cached_prefix_len: int,
    ) -> KVCacheBlocks:
        if cached_prefix_len == hit_len:
            return blocks
        clipped_groups: list[list[KVCacheBlock]] = []
        for group_blocks, cache_group in zip(
            blocks.blocks,
            self.native_manager.kv_cache_config.kv_cache_groups,
            strict=True,
        ):
            block_size = int(cache_group.kv_cache_spec.block_size)
            keep_blocks = (cached_prefix_len + block_size - 1) // block_size
            if keep_blocks > len(group_blocks):
                raise RuntimeError(
                    "Native prefix lookup returned too few blocks for the common CFG boundary: "
                    f"required={keep_blocks}, returned={len(group_blocks)}"
                )
            clipped_groups.append(list(group_blocks[:keep_blocks]))
        return self.native_manager.create_kv_cache_blocks(tuple(clipped_groups))

    def _safe_cached_prefix_len(self, request: DiffusionKVRequest, native_hit_len: int) -> int:
        """Keep a native hit on a computation-closed dependency boundary.

        Fine-grained lookup normally returns an exact image boundary. The
        rollback also keeps coarser hash configurations conservative. vLLM's
        partial-hit path privately copies a shared physical tail page before
        the changed suffix is overwritten.
        """

        safe_hit_len = native_hit_len
        for offset, _ in request.cache_dependencies:
            activation_offset = (offset + self.hash_block_size - 1) // self.hash_block_size * self.hash_block_size
            if activation_offset == native_hit_len:
                safe_hit_len = min(safe_hit_len, offset)
            elif activation_offset > native_hit_len:
                break
        for start, end in request.atomic_cache_spans:
            if start < safe_hit_len < end:
                safe_hit_len = start
                break
        return safe_hit_len

    def _discard_pending_allocation_metadata(self) -> None:
        _, retained_blocks = self.native_manager.take_kv_cache_block_copies()
        if retained_blocks:
            self.native_manager.block_pool.free_blocks(retained_blocks)
        # Full-attention COW copies overwrite the complete destination page, so
        # these native zeroing records have no Worker-side consumer. Drain them
        # explicitly instead of retaining one Python integer per COW forever.
        self.native_manager.take_new_block_ids()

    @staticmethod
    def _unique_blocks(block_sets: Sequence[KVCacheBlocks]) -> list[KVCacheBlock]:
        unique: dict[int, KVCacheBlock] = {}
        for blocks in block_sets:
            for group in blocks.blocks:
                for block in group:
                    if not block.is_null:
                        unique[id(block)] = block
        return list(unique.values())

    def _get_empty_pool_required_blocks(
        self,
        requests: Sequence[DiffusionKVRequest],
        computed_blocks: Sequence[KVCacheBlocks],
        cached_prefix_len: int,
    ) -> int:
        """Return the minimum physical capacity for one public request.

        Cached CFG rows may point to the same physical blocks. Count those hit
        blocks once, then ask the native coordinator how many additional blocks
        each row needs for its suffix. This gives an admission bound that still
        works for warm requests and does not over-count shared prefix pages.
        """

        coordinator = self.native_manager.coordinator
        unique_hit_blocks = len(self._unique_blocks(computed_blocks))
        suffix_blocks = 0
        for request, hit in zip(requests, computed_blocks, strict=True):
            native_required = coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=request.num_tokens,
                new_computed_blocks=hit.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=cached_prefix_len,
                num_local_computed_tokens=cached_prefix_len,
                num_tokens_main_model=request.num_tokens,
                apply_admission_cap=True,
            )
            evictable_hits = sum(block.ref_cnt == 0 and not block.is_null for group in hit.blocks for block in group)
            suffix_blocks += max(native_required - evictable_hits, 0)
        # ``get_num_blocks_to_allocate`` includes evictable hit pages because
        # the live free-capacity check must first remove them from the free
        # queue. They are already part of the pool's physical capacity, so do
        # not count them a second time in this empty-pool bound.
        return unique_hit_blocks + suffix_blocks

    def has_request(self, public_request_id: str) -> bool:
        return public_request_id in self._requests

    def reserve_request(
        self,
        public_request_id: str,
        kv_requests: Sequence[DiffusionKVRequest],
    ) -> DiffusionKVMetadata | None:
        """Reserve all CFG sequences or roll back the complete public request."""

        if not public_request_id:
            raise ValueError("public_request_id must be non-empty")
        if public_request_id in self._requests:
            raise ValueError(f"Diffusion KV request {public_request_id!r} is already allocated")
        requests = tuple(kv_requests)
        if not requests:
            raise ValueError("Diffusion KV allocation requires at least one sequence")

        sequence_ids = [request.sequence_id for request in requests]
        internal_ids = [request.request_id for request in requests]
        expected_sequence_ids = list(range(len(requests)))
        if sequence_ids != expected_sequence_ids:
            raise ValueError(
                "Diffusion KV requests must be ordered by contiguous sequence_id values: "
                f"expected={expected_sequence_ids}, got={sequence_ids}"
            )
        if len(internal_ids) != len(set(internal_ids)):
            raise ValueError(f"Diffusion KV internal request IDs must be unique, got {internal_ids}")
        conflicts = self._internal_request_ids.intersection(internal_ids)
        if conflicts:
            raise ValueError(f"Diffusion KV internal request IDs are already allocated: {sorted(conflicts)}")
        for request in requests:
            if request.kv_contexts:
                raise DiffusionKVAdmissionError("independent DiffusionKVContext allocation is not implemented yet")
            if request.seq_len > self.max_model_len:
                raise DiffusionKVAdmissionError(
                    f"Diffusion KV sequence {request.request_id!r} exceeds max_model_len: "
                    f"seq_len={request.seq_len}, max_model_len={self.max_model_len}"
                )

        self._prepare_block_hashes(requests)
        hit_results = [self.native_manager.get_computed_blocks(request) for request in requests]
        safe_hit_lens = [
            self._safe_cached_prefix_len(request, result[1])
            for request, result in zip(requests, hit_results, strict=True)
        ]
        cached_prefix_len = min(safe_hit_lens)
        computed_blocks = [
            self._clip_computed_blocks(
                result[0],
                hit_len=result[1],
                cached_prefix_len=cached_prefix_len,
            )
            for result in hit_results
        ]
        for request, result in zip(requests, hit_results, strict=True):
            request.shared_prefix_boundary = result[2]

        required_blocks = self._get_empty_pool_required_blocks(
            requests,
            computed_blocks,
            cached_prefix_len,
        )
        if required_blocks > self._empty_pool_num_free_blocks:
            raise DiffusionKVAdmissionError(
                f"Diffusion KV request {public_request_id!r} cannot fit even when the block pool is empty: "
                f"required_blocks={required_blocks}, available_blocks={self._empty_pool_num_free_blocks}; "
                "increase KV cache capacity or reduce the request sequence count/length"
            )

        allocated: list[DiffusionKVRequest] = []
        sequence_metadata: list[DiffusionKVSequenceMetadata] = []
        block_copies: list[KVCacheBlockCopy] = []
        cow_retained_blocks: list[KVCacheBlock] = []
        pinned_blocks = self._unique_blocks(computed_blocks)
        if pinned_blocks:
            # All CFG lookups happen before allocation so they can share one
            # execution boundary. Pin every distinct hit until each native
            # allocation has taken its own reference; otherwise an earlier
            # branch's miss allocation could evict a later branch's hit.
            self.native_manager.block_pool.touch(pinned_blocks)
        try:
            for request, request_computed_blocks in zip(requests, computed_blocks, strict=True):
                blocks = self.native_manager.allocate_slots(
                    request,
                    num_new_tokens=request.seq_len - cached_prefix_len,
                    num_new_computed_tokens=cached_prefix_len,
                    new_computed_blocks=request_computed_blocks,
                    delay_cache_blocks=True,
                    full_sequence_must_fit=True,
                )
                if blocks is None:
                    self._rollback(allocated)
                    self._discard_pending_allocation_metadata()
                    allocated.clear()
                    return None
                allocated.append(request)
                request.num_computed_tokens = cached_prefix_len
                sequence_metadata.append(
                    DiffusionKVSequenceMetadata(
                        sequence_id=request.sequence_id,
                        prefix_len=request.prefix_len,
                        target_len=request.target_len,
                        seq_len=request.seq_len,
                        block_ids=self.native_manager.get_block_ids(request.request_id),
                        cached_prefix_len=cached_prefix_len,
                    )
                )
            block_copies, cow_retained_blocks = self.native_manager.take_kv_cache_block_copies()
            self.native_manager.take_new_block_ids()
        except Exception:
            self._rollback(allocated)
            self._discard_pending_allocation_metadata()
            raise
        finally:
            if pinned_blocks:
                self.native_manager.block_pool.free_blocks(reversed(pinned_blocks))

        metadata = DiffusionKVMetadata(
            request_id=public_request_id,
            allocation_generation=self._next_allocation_generation,
            sequences=tuple(sequence_metadata),
            block_copies=tuple(block_copies),
        )
        self._next_allocation_generation += 1
        self._requests[public_request_id] = requests
        self._metadata[public_request_id] = metadata
        if cow_retained_blocks:
            self._cow_retained_blocks[public_request_id] = cow_retained_blocks
        self._internal_request_ids.update(internal_ids)
        return metadata

    def complete_block_copies(self, public_request_id: str) -> None:
        """Release COW endpoints after every Worker has completed its copy."""

        retained_blocks = self._cow_retained_blocks.pop(public_request_id, ())
        if retained_blocks:
            self.native_manager.block_pool.free_blocks(retained_blocks)

    def _cache_atomic_boundaries(
        self,
        request: DiffusionKVRequest,
        publishable_prefix_len: int,
    ) -> None:
        boundaries = sorted(
            {
                boundary
                for span in request.atomic_cache_spans
                for boundary in span
                if 0 < boundary <= publishable_prefix_len
            }
        )
        if not boundaries:
            return
        for cache_group_manager in self.native_manager.coordinator.single_type_managers:
            block_size = int(cache_group_manager.block_size)
            if block_size % self.hash_block_size:
                raise RuntimeError(
                    "Atomic prefix hash unit must divide every physical cache block: "
                    f"hash_block_size={self.hash_block_size}, block_size={block_size}"
                )
            blocks = cache_group_manager.req_to_blocks[request.request_id]
            for boundary in boundaries:
                if boundary % self.hash_block_size or boundary % block_size == 0:
                    continue
                block_idx = boundary // block_size
                if block_idx >= len(blocks):
                    raise RuntimeError(
                        "Atomic cache boundary has no allocated physical block: "
                        f"boundary={boundary}, block_size={block_size}, blocks={len(blocks)}"
                    )
                self.native_manager.block_pool.cache_partial_block(
                    request=request,
                    block=blocks[block_idx],
                    num_tokens=boundary,
                    kv_cache_group_id=cache_group_manager.kv_cache_group_id,
                    block_size=block_size,
                )

    def publish_request(self, public_request_id: str) -> None:
        """Publish hash-granular stable prefixes and atomic boundary aliases."""

        if not self.enable_prefix_caching:
            return
        try:
            requests = self._requests[public_request_id]
        except KeyError as exc:
            raise KeyError(f"Diffusion KV request {public_request_id!r} is not allocated") from exc
        for request in requests:
            publishable_prefix_len = min(
                request.prefix_len // self.hash_block_size * self.hash_block_size,
                len(request.block_hashes) * self.hash_block_size,
            )
            if publishable_prefix_len > 0:
                self.native_manager.cache_blocks(request, publishable_prefix_len)
                self._cache_atomic_boundaries(request, publishable_prefix_len)

    def get_metadata(self, public_request_id: str) -> DiffusionKVMetadata:
        try:
            return self._metadata[public_request_id]
        except KeyError as exc:
            raise KeyError(f"Diffusion KV request {public_request_id!r} is not allocated") from exc

    def free_request(self, public_request_id: str) -> None:
        self.complete_block_copies(public_request_id)
        requests = self._requests.pop(public_request_id, ())
        self._metadata.pop(public_request_id, None)
        for request in reversed(requests):
            free_kv_blocks_in_physical_order(self.native_manager, request)
            request.num_computed_tokens = 0
            request.shared_prefix_boundary = 0
            self._internal_request_ids.discard(request.request_id)

    def close(self) -> None:
        for public_request_id in tuple(self._requests):
            self.free_request(public_request_id)

    def _rollback(self, requests: Sequence[DiffusionKVRequest]) -> None:
        for request in reversed(requests):
            free_kv_blocks_in_physical_order(self.native_manager, request)
            request.num_computed_tokens = 0
            request.shared_prefix_boundary = 0
