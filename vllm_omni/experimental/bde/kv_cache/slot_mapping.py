# SPDX-License-Identifier: Apache-2.0
"""Slot mapping + block-table access for the BDE engine.

Phase 1 is single-request and owns its KV layout — it gathers the resident window
into DreamZero's existing attention rather than calling vLLM's paged kernel — so a
thin slot-mapping helper is used instead of vLLM's batch/Triton ``BlockTables``.
The layout is the standard PagedAttention one:

    slot(pos) = block_id(pos) * block_size + (pos % block_size)

Write (commit) and read (gather) use the same formula, so the layout is
self-consistent regardless of vLLM's internal kernel layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_slot_mapping(
    block_ids: Sequence[int],
    positions: torch.Tensor | Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Map absolute token positions to physical KV-cache slots.

    Args:
        block_ids: physical block id per block index (the request's block table).
        positions: absolute token positions to map (1-D).
        block_size: tokens per block.

    Returns:
        ``LongTensor`` of physical slots, one per position.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    table = torch.as_tensor(block_ids, dtype=torch.long)
    pos = torch.as_tensor(positions, dtype=torch.long)
    block_index = torch.div(pos, block_size, rounding_mode="floor")
    offset = pos % block_size
    return table[block_index] * block_size + offset


def chunk_slot_mapping(
    block_ids: Sequence[int],
    num_computed_tokens: int,
    chunk_size: int,
    block_size: int,
) -> torch.Tensor:
    """Slot mapping for the in-flight chunk's tokens (the commit write target).

    The chunk occupies absolute positions
    ``[num_computed_tokens, num_computed_tokens + chunk_size)``.
    """
    positions = torch.arange(
        num_computed_tokens,
        num_computed_tokens + chunk_size,
        dtype=torch.long,
    )
    return compute_slot_mapping(block_ids, positions, block_size)


def resident_block_ids(block_ids: Sequence[int], null_block_id: int) -> list[int]:
    """Real (non-null) blocks currently resident, in table order.

    These are the blocks the read path gathers the attention window from;
    out-of-window positions are the shared ``null_block`` and are excluded.
    """
    return [int(b) for b in block_ids if int(b) != null_block_id]
