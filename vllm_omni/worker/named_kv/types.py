# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared data types for named causal KV branches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NamedKVBranchStep:
    """Metadata returned to a model while one branch append is active.

    This is the original step type used by ``append_and_enter*`` context
    managers.  It carries a GPU position tensor and the current sequence
    length so the model can issue its forward pass.
    """

    position: torch.Tensor
    sequence_length: int


@dataclass(frozen=True)
class NamedKVAppendBatch:
    """Snapshot of one append batch for the graph-capable executor path.

    Unlike :class:`NamedKVBranchStep`, this type does not carry GPU tensors.
    All fields are plain Python tuples so the object is safe to construct
    outside a compilation region and pass across the eager/graph boundary.

    ``block_ids`` is a tuple of tuples (one per request) so the snapshot
    is immutable even though the runtime's internal state uses lists.
    """

    request_ids: tuple[str, ...]
    positions: tuple[int, ...]
    slot_values: tuple[int, ...]
    seq_lens: tuple[int, ...]
    block_ids: tuple[tuple[int, ...], ...]


__all__ = [
    "NamedKVAppendBatch",
    "NamedKVBranchStep",
]
