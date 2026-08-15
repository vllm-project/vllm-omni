# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig

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
    ) -> None:
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        if kv_cache_config.num_blocks <= 0 or not kv_cache_config.kv_cache_groups:
            raise ValueError("Diffusion KVCacheConfig must contain a positive block pool and at least one group")
        if scheduler_block_size <= 0 or hash_block_size <= 0:
            raise ValueError("scheduler_block_size and hash_block_size must be positive")
        if max_in_flight_tokens is not None and max_in_flight_tokens <= 0:
            raise ValueError(f"max_in_flight_tokens must be positive when provided, got {max_in_flight_tokens}")
        self.native_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            max_in_flight_tokens=max_in_flight_tokens,
            enable_caching=False,
        )
        self.max_model_len = max_model_len
        # Native vLLM may reserve a null block, so an idle BlockPool does not
        # necessarily report ``kv_cache_config.num_blocks`` free blocks.
        self._empty_pool_num_free_blocks = self.native_manager.block_pool.get_num_free_blocks()
        self._requests: dict[str, tuple[DiffusionKVRequest, ...]] = {}
        self._metadata: dict[str, DiffusionKVMetadata] = {}
        self._internal_request_ids: set[str] = set()
        self._next_allocation_generation = 1

    def _get_empty_pool_required_blocks(self, requests: Sequence[DiffusionKVRequest]) -> int:
        """Return the native full-sequence reservation for one public request.

        vLLM performs this calculation inside ``allocate_slots`` when
        ``full_sequence_must_fit`` is enabled. Diffusion CFG adds one semantic
        requirement that native requests do not have: every execution sequence
        must fit atomically. Sum the native per-sequence result so a request
        that can never fit is rejected independently of current pool usage.
        """

        coordinator = self.native_manager.coordinator
        empty_blocks = self.native_manager.empty_kv_cache_blocks.blocks
        return sum(
            coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=request.num_tokens,
                new_computed_blocks=empty_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=request.num_tokens,
                apply_admission_cap=True,
            )
            for request in requests
        )

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

        required_blocks = self._get_empty_pool_required_blocks(requests)
        if required_blocks > self._empty_pool_num_free_blocks:
            raise DiffusionKVAdmissionError(
                f"Diffusion KV request {public_request_id!r} cannot fit even when the block pool is empty: "
                f"required_blocks={required_blocks}, available_blocks={self._empty_pool_num_free_blocks}; "
                "increase KV cache capacity or reduce the request sequence count/length"
            )

        allocated: list[DiffusionKVRequest] = []
        sequence_metadata: list[DiffusionKVSequenceMetadata] = []
        try:
            for request in requests:
                blocks = self.native_manager.allocate_slots(
                    request,
                    num_new_tokens=request.seq_len,
                    delay_cache_blocks=True,
                    full_sequence_must_fit=True,
                )
                if blocks is None:
                    self._rollback(allocated)
                    allocated.clear()
                    return None
                allocated.append(request)
                sequence_metadata.append(
                    DiffusionKVSequenceMetadata(
                        sequence_id=request.sequence_id,
                        prefix_len=request.prefix_len,
                        target_len=request.target_len,
                        seq_len=request.seq_len,
                        block_ids=blocks.get_block_ids(),
                    )
                )
        except Exception:
            self._rollback(allocated)
            raise

        metadata = DiffusionKVMetadata(
            request_id=public_request_id,
            allocation_generation=self._next_allocation_generation,
            sequences=tuple(sequence_metadata),
        )
        self._next_allocation_generation += 1
        self._requests[public_request_id] = requests
        self._metadata[public_request_id] = metadata
        self._internal_request_ids.update(internal_ids)
        return metadata

    def get_metadata(self, public_request_id: str) -> DiffusionKVMetadata:
        try:
            return self._metadata[public_request_id]
        except KeyError as exc:
            raise KeyError(f"Diffusion KV request {public_request_id!r} is not allocated") from exc

    def free_request(self, public_request_id: str) -> None:
        requests = self._requests.pop(public_request_id, ())
        self._metadata.pop(public_request_id, None)
        for request in reversed(requests):
            self.native_manager.free(request)
            self._internal_request_ids.discard(request.request_id)

    def close(self) -> None:
        for public_request_id in tuple(self._requests):
            self.free_request(public_request_id)

    def _rollback(self, requests: Sequence[DiffusionKVRequest]) -> None:
        for request in reversed(requests):
            self.native_manager.free(request)
