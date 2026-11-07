# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR-Diffusion execution envelopes.

This layer separates request/session lifecycle from strategy-specific execution.
Paged AR-Diffusion may carry several request records in one envelope; the Noisy
cell path runs each request continuously through its ``CellPlan``, so a batch wave
carries the records that wave owns and a request-mode call carries one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class ARDiffusionExecutionMode(str, Enum):
    """Execution contract an AR batch follows."""

    PAGED_CONTINUOUS = "paged_continuous"
    NOISY_CONTINUOUS = "noisy_continuous"


@dataclass
class ARDiffusionRequestRecord:
    """One request's lifecycle envelope inside an AR-Diffusion batch.

    ``kv_id`` is the identity the strategy uses: the outer ``session_id`` for a
    retained Noisy session, the request id for an ephemeral one, and the request
    id on the paged path.
    """

    kv_id: str
    session_id: str | None = None
    strategy_payload: object | None = None
    owned_kv_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kv_id:
            raise ValueError("ARDiffusionRequestRecord.kv_id must be non-empty")
        if not self.owned_kv_ids:
            self.owned_kv_ids = (self.kv_id,)


@dataclass
class ARDiffusionBatch:
    """Strategy-neutral AR execution envelope.

    The envelope unifies request identity, session metadata, strategy payload,
    ownership, cleanup and output attribution. It does not impose one tensor
    batching implementation on paged and Noisy strategies, and it does not decide
    how many requests a strategy may run: a paged wave may carry several records,
    a Noisy request-mode call carries one, and a Noisy batch wave carries the
    records that wave owns.
    """

    mode: ARDiffusionExecutionMode
    records: list[ARDiffusionRequestRecord]
    metadata: dict[str, object] = field(default_factory=dict)
    outputs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("ARDiffusionBatch requires at least one request record")
        ids = [record.kv_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ARDiffusionBatch request ids must be unique, got {ids}")

    @classmethod
    def _build(
        cls,
        mode: ARDiffusionExecutionMode,
        kv_ids: Sequence[str],
        session_ids: Sequence[str | None] | None,
    ) -> ARDiffusionBatch:
        ids = tuple(kv_ids)
        if not ids:
            raise ValueError("ARDiffusionBatch requires at least one request record")
        sessions = tuple(session_ids) if session_ids is not None else (None,) * len(ids)
        if len(sessions) != len(ids):
            raise ValueError("session_ids must align with kv_ids")
        return cls(
            mode=mode,
            records=[
                ARDiffusionRequestRecord(kv_id=kv_id, session_id=session_id)
                for kv_id, session_id in zip(ids, sessions, strict=True)
            ],
        )

    @classmethod
    def for_noisy(
        cls,
        kv_ids: Sequence[str],
        *,
        session_ids: Sequence[str | None] | None = None,
    ) -> ARDiffusionBatch:
        """Envelope for the cell path: one record per request in this call."""
        return cls._build(ARDiffusionExecutionMode.NOISY_CONTINUOUS, kv_ids, session_ids)

    @classmethod
    def for_paged(
        cls,
        request_ids: Sequence[str],
        *,
        session_ids: Sequence[str | None] | None = None,
    ) -> ARDiffusionBatch:
        """Envelope for the paged strategy, which may carry several records."""
        return cls._build(ARDiffusionExecutionMode.PAGED_CONTINUOUS, request_ids, session_ids)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(record.kv_id for record in self.records)

    @property
    def record(self) -> ARDiffusionRequestRecord:
        if len(self.records) != 1:
            raise ValueError("ARDiffusionBatch.record requires exactly one request record")
        return self.records[0]

    def set_strategy_payload(self, kv_id: str, payload: object) -> None:
        for record in self.records:
            if record.kv_id == kv_id:
                record.strategy_payload = payload
                return
        raise KeyError(f"request {kv_id!r} is not part of the AR-Diffusion batch")

    def set_output(self, kv_id: str, output: object) -> None:
        if kv_id not in self.request_ids:
            raise KeyError(f"request {kv_id!r} is not part of the AR-Diffusion batch")
        self.outputs[kv_id] = output
