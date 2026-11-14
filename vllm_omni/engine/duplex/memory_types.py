# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable inputs and receipts for session-owned memory cards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudgetLimits

DuplexMemoryAdmissionStatus = Literal["applied", "no_change", "stale", "invalid", "capacity"]


@dataclass(frozen=True)
class DuplexMemoryEvidenceSpan:
    """A range in versioned text, bound by the session rather than the model."""

    source_id: int
    revision: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class DuplexMemoryCardOperation:
    """A proposed upsert, or a retraction when ``text`` is None."""

    fact_key: str = field(repr=False)
    expected_generation: int
    evidence: tuple[DuplexMemoryEvidenceSpan, ...]
    text: str | None = field(repr=False)
    parents: tuple[int, ...] = ()


@dataclass(frozen=True)
class DuplexMemoryCardBatch:
    operations: tuple[DuplexMemoryCardOperation, ...]
    catalog_revision: int


@dataclass(frozen=True)
class DuplexMemoryCardMetadata:
    fact_key: str = field(repr=False)
    generation: int
    evidence_order_floor: int
    supersedes_node_id: int | None
    evidence_spans: tuple[DuplexMemoryEvidenceSpan, ...]


@dataclass(frozen=True)
class DuplexMemoryCardHead:
    """A bounded index entry; an empty head retains the replacement fence."""

    fact_key: str = field(repr=False)
    generation: int
    node_id: int | None
    evidence_order_floor: int


@dataclass(frozen=True)
class DuplexMemoryAdmission:
    status: DuplexMemoryAdmissionStatus
    node_ids: tuple[int, ...] = ()
    reason_code: str | None = None


@dataclass(frozen=True)
class DuplexContextAuxiliaryPolicy:
    """Cheap opt-in gate, evaluated before capturing any history evidence."""

    max_sources: int = 8
    max_chars: int = 8192
    allow_during_response: bool = False
    compute_budget: DuplexAuxiliaryBudgetLimits | None = None


@dataclass(frozen=True, order=True)
class DuplexMemoryCursor:
    slot_order: int = 0
    char_offset: int = 0


@dataclass(frozen=True)
class DuplexContextAuxiliaryRechunk:
    """A plugin asks the owner for smaller work; it conveys no coverage."""


@dataclass(frozen=True)
class DuplexContextEvidence:
    handle: int
    source_id: int
    revision: int
    slot_order: int
    role: Literal["user", "assistant"]
    text: str = field(repr=False)
    char_start: int
    char_end: int


@dataclass(frozen=True)
class DuplexContextAuxiliaryWork:
    scope_id: str = field(repr=False)
    epoch: int
    config_revision: int
    progress_generation: int
    start: DuplexMemoryCursor
    end: DuplexMemoryCursor
    read_set: tuple[DuplexMemoryEvidenceSpan, ...]
    catalog_revision: int
    pass_id: int = 0
    catalog_page: int = 0
    suppression_version: int = 0
    reconciliation_id: int | None = None
    reconciliation_target: int | None = None
    rewrite_epoch: int = 0


@dataclass(frozen=True)
class DuplexContextAuxiliaryInput:
    work: DuplexContextAuxiliaryWork
    evidence: tuple[DuplexContextEvidence, ...]
    catalog: tuple[DuplexMemoryCardHead, ...]
    history_revision: int
    barriers: tuple[DuplexContextBarrier, ...] = ()

    @property
    def work_key(self) -> tuple[object, ...]:
        """Coverage identity, independent of execution epochs and pass nonces."""
        work = self.work
        return (
            work.scope_id,
            work.config_revision,
            work.progress_generation,
            work.start,
            work.end,
            work.read_set,
            work.catalog_revision,
            work.catalog_page,
            work.suppression_version,
            work.reconciliation_id,
            work.reconciliation_target,
            work.rewrite_epoch,
        )


@dataclass(frozen=True)
class DuplexMemoryEvidenceSelection:
    handle: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class DuplexMemoryProposal:
    fact_key: str = field(repr=False)
    text: str | None = field(repr=False)
    evidence: tuple[DuplexMemoryEvidenceSelection, ...]


@dataclass(frozen=True)
class DuplexMemoryProposalBatch:
    operations: tuple[DuplexMemoryProposal, ...]
    resolutions: tuple[DuplexMemoryResolution, ...] = ()
    reconciled: bool = False


@dataclass(frozen=True)
class DuplexContextBarrier:
    handle: int
    barrier_id: int
    generation: int
    keys: frozenset[str] = field(repr=False)
    evidence: tuple[DuplexMemoryEvidenceSelection, ...]


@dataclass(frozen=True)
class DuplexMemoryResolution:
    handle: int
    keys: frozenset[str] = field(repr=False)
    evidence: tuple[DuplexMemoryEvidenceSelection, ...]
    disposition: Literal["replace", "retract", "no_conflict"] = "no_conflict"
