# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract between block selection and sparse attention execution."""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import torch


class BlockSelection(NamedTuple):
    """Selected logical KV blocks, without changing attention mathematics.

    indices: contiguous int32 [B, Hq, ceil(Sq / Bq), capacity].
    counts: contiguous int32 [B, Hq, ceil(Sq / Bq)].
    Both tensors use the input device. Each row's first counts entries are
    sorted, unique, in-range KV block IDs; remaining entries are ignored.
    Every row must contain at least one block. Selectors must include every
    block intersecting the protected KV token prefix, or reject the request.
    Sequence tails are bounded by input lengths. No K/V head expansion occurs.
    """

    indices: torch.Tensor
    counts: torch.Tensor


def validate_protected_kv_prefix(prefix: int, key_length: int) -> None:
    """Require a token prefix within the key sequence, including either endpoint."""
    if isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 <= prefix <= key_length:
        raise ValueError("protected_kv_prefix must be an integer within the key sequence")


class BlockSelector(ABC):
    """Prepared selection policy; request tensors must remain invocation-local.

    Implementations own scoring/approximation, budgets, rounding and geometry
    restrictions. A scorer configuration is optional and interpreted by the
    implementation; fused selection need not materialize a score matrix.
    Algorithms requiring corrections for omitted keys need a separate attention
    method, rather than extending this selected-key normalization contract.
    """

    @abstractmethod
    def __init__(self, options: dict[str, Any]) -> None:
        """Bind strategy options; prepare() binds execution geometry separately."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def normalize_config(cls, options: dict[str, Any]) -> dict[str, Any]:
        """Normalize strategy-owned serializable options without executing GPU work."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self, block_size: tuple[int, int], head_size: int, device: torch.device) -> None:
        """Reject unsupported geometry/device choices before execution."""
        raise NotImplementedError

    @abstractmethod
    def validate_request(self, query: torch.Tensor, key: torch.Tensor, protected_prefix: int) -> None:
        """Check strategy input constraints and mandatory-block feasibility.

        Called by capability resolution before execution, and by select() for
        direct callers. Shared layout, head mapping and device checks belong
        to the orchestration layer; prepared geometry is checked in prepare().
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, query: torch.Tensor, key: torch.Tensor, scale: float, protected_prefix: int) -> BlockSelection:
        """Return a pattern meeting BlockSelection, including nonempty rows."""
        raise NotImplementedError
