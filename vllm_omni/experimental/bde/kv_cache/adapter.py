# SPDX-License-Identifier: Apache-2.0
"""Adapt a BDE diffusion request to vLLM's ``KVCacheManager`` request surface."""

from __future__ import annotations

from vllm.v1.request import RequestStatus


class BDERequestAdapter:
    """Duck-types the subset of ``vllm.v1.request.Request`` that the
    ``KVCacheManager`` reads (``allocate_slots`` / ``get_computed_blocks`` /
    ``free`` and the coordinator they call into).

    It is intentionally NOT a full ``Request``. The conformance test exercises a
    real ``KVCacheManager`` against this adapter so the surface cannot silently
    drift across vLLM versions.

    A BDE request advances one *chunk* at a time: ``allocate_slots`` is called
    once per chunk and ``num_computed_tokens`` advances only when a chunk is
    committed (:meth:`on_chunk_committed`), so the ``T`` denoise steps of a chunk
    reuse the same slots.
    """

    def __init__(
        self,
        request_id: str,
        *,
        chunk_size: int,
        prefill_prefix_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self._chunk_size = chunk_size
        self._prefill = prefill_prefix_tokens
        self._completed_chunks = 0
        # Filled only when cross-request prefix reuse is enabled (Phase 3).
        self.block_hashes: list = []
        self.skip_reading_prefix_cache = True
        self.num_preemptions = 0
        # vLLM watermark gate reads this; map the request lifecycle onto it.
        self.status = RequestStatus.WAITING

    @property
    def num_computed_tokens(self) -> int:
        """Persistent KV already materialized (committed chunks + prefill)."""
        return self._prefill + self._completed_chunks * self._chunk_size

    @property
    def num_tokens(self) -> int:
        """Total tokens once the in-flight chunk is committed."""
        return self._prefill + (self._completed_chunks + 1) * self._chunk_size

    @property
    def num_prompt_tokens(self) -> int:
        """The prefill prefix length (read by ``cache_blocks`` when caching)."""
        return self._prefill

    @property
    def completed_chunks(self) -> int:
        return self._completed_chunks

    def on_chunk_committed(self) -> None:
        """Advance by one chunk. Call once per chunk, not per denoise step."""
        self._completed_chunks += 1
