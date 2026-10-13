# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transactional raw K/V access for models that transform history on read.

Storage has no positional-encoding semantics. Model adapters may apply their
own transforms after gathering a window without changing committed pool data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .paged_attention import ARDiffusionPagedLayerContext
    from .state import ARDiffusionKVState


class RawKVWindow:
    """One model block backed by the runner's existing paged session pool."""

    def __init__(self, state: ARDiffusionKVState, branch: str, *, frames_per_chunk: int, history_chunks: int):
        if frames_per_chunk <= 0 or history_chunks <= 0:
            raise ValueError("Raw KV chunk and history sizes must be positive")
        self.state = state
        self.branch = branch
        self.frames_per_chunk = frames_per_chunk
        self.history_chunks = history_chunks
        self.rows = state.kv_cache.spec.chunk_size * frames_per_chunk
        self._contexts: list[ARDiffusionPagedLayerContext] | None = None
        self._staged: set[int] = set()

    def _check_chunk(self, chunk_index: int) -> None:
        committed = self.state.adapter(self.branch).completed_chunks
        if committed != chunk_index * self.frames_per_chunk:
            raise ValueError(f"Raw KV chunk {chunk_index} differs from committed frame count {committed}")

    def history(self, layer: int, chunk_index: int) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        self._check_chunk(chunk_index)
        if not 0 <= layer < self.state.num_layers:
            raise ValueError("Raw KV layer index is out of bounds")
        adapter = self.state.adapter(self.branch)
        count = min(chunk_index, self.history_chunks)
        if count == 0:
            return []
        pool = self.state.kv_cache
        stop = adapter.num_computed_tokens
        start = stop - count * self.rows
        block_size = pool.block_size
        first_block, last_block = start // block_size, (stop - 1) // block_size
        block_ids = pool.block_ids_at(adapter, range(first_block, last_block + 1))
        if pool.null_block_id in block_ids:
            raise RuntimeError("Raw KV history contains an evicted block")
        device = pool.key_cache(layer).device
        positions = torch.arange(start, stop, device=device)
        blocks = torch.tensor(block_ids, device=device, dtype=torch.long)
        physical = blocks[positions.div(block_size, rounding_mode="floor") - first_block]
        offsets = positions.remainder(block_size)
        key = pool.key_cache(layer)[physical, offsets].unsqueeze(0)
        value = pool.value_cache(layer)[physical, offsets].unsqueeze(0)
        return [
            (
                chunk_index - count + i,
                key[:, i * self.rows : (i + 1) * self.rows],
                value[:, i * self.rows : (i + 1) * self.rows],
            )
            for i in range(count)
        ]

    def stage(self, layer: int, chunk_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self._check_chunk(chunk_index)
        pool = self.state.kv_cache
        expected = (1, self.rows, pool.num_kv_heads, pool.head_size)
        if key.shape != expected or value.shape != expected:
            raise ValueError(f"Raw KV must have shape {expected}")
        if not 0 <= layer < self.state.num_layers or layer in self._staged:
            raise ValueError("Raw KV layer is invalid or already staged")
        key_pool, value_pool = pool.key_cache(layer), pool.value_cache(layer)
        if key.dtype != key_pool.dtype or value.dtype != value_pool.dtype:
            raise ValueError("Raw KV dtype must match the runner pool")
        if key.device != key_pool.device or value.device != value_pool.device:
            raise ValueError("Raw KV device must match the runner pool")
        if self._contexts is None:
            self._contexts = self.state.prepare_paged_context(self.branch, self.rows, commit_current=True)
        context = self._contexts[layer].forward_ctx
        context.ensure_video_slots(key.device)
        slots = context.current_video_slot_mapping
        assert slots is not None
        blocks = slots.div(pool.block_size, rounding_mode="floor")
        offsets = slots.remainder(pool.block_size)
        key_pool[blocks, offsets] = key.detach()[0]
        value_pool[blocks, offsets] = value.detach()[0]
        self._staged.add(layer)

    def commit_chunk(self, chunk_index: int) -> None:
        self._check_chunk(chunk_index)
        if self._staged != set(range(self.state.num_layers)):
            raise RuntimeError("Raw KV commit requires every layer to be staged")
        self.state.commit_paged_context(self.branch)
        self._contexts = None
        self._staged.clear()

    def abort(self) -> None:
        # The runner closes the session on failure; partially written managed
        # slots must never be retried as though the failed chunk had committed.
        self._contexts = None
        self._staged.clear()
