# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Immutable inputs and receipts for session-owned memory cards."""

from dataclasses import dataclass, field
from typing import Literal

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
