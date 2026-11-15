# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Derived context nodes over authoritative duplex history sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from sys import getsizeof
from types import MappingProxyType
from unicodedata import normalize

from vllm_omni.engine.duplex.memory_types import (
    DuplexMemoryAdmission,
    DuplexMemoryCardBatch,
    DuplexMemoryCardHead,
    DuplexMemoryCardMetadata,
    DuplexMemoryCardOperation,
    DuplexMemoryEvidenceSpan,
)
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot, copy_history_message
from vllm_omni.engine.duplex.session.memory_lexical import lexemes

_MAX_CARD_LEXEME_BYTES = 4 * 1024 * 1024


def _lexeme_bytes(terms: frozenset[str]) -> int:
    # Account retained term objects; the mapping is separately bounded by the
    # card-head limit. This is not a process RSS or allocator-arena limit.
    return getsizeof(terms) + sum(getsizeof(term) for term in terms)


class DuplexContextLevel(IntEnum):
    """Derived node kinds: MEDIA_DROPPED is the L1 view; SUMMARY stores summaries."""

    MEDIA_DROPPED = 1
    SUMMARY = 2
    MEMORY_CARD = 3


@dataclass(frozen=True)
class DuplexSourceRef:
    """A source ID and revision scoped to one live session."""

    source_id: int
    revision: int


@dataclass(frozen=True)
class DuplexContextNode:
    """A rendered message derived from history sources and earlier nodes.

    Parents always have smaller node IDs, so ascending node ID order is a
    topological order of the dependency graph. Message containers are owned
    by the ledger; readers must treat them and opaque payloads as read-only.
    """

    node_id: int
    level: DuplexContextLevel
    sources: tuple[DuplexSourceRef, ...]
    parents: tuple[int, ...]
    message: Mapping[str, object]
    card: DuplexMemoryCardMetadata | None = None


class DuplexContextNodeRejectedError(ValueError):
    """The ledger refused a node that is not valid against the given snapshot."""


def source_refs(snapshot: DuplexHistorySnapshot, *source_ids: int) -> tuple[DuplexSourceRef, ...]:
    revisions = _current_revisions(snapshot)
    return tuple(DuplexSourceRef(source_id, revisions[source_id]) for source_id in source_ids)


def _current_revisions(snapshot: DuplexHistorySnapshot) -> dict[int, int]:
    return dict(zip(snapshot.source_ids, snapshot.source_revisions, strict=True))


def _sources_current(sources: tuple[DuplexSourceRef, ...], current: Mapping[int, int]) -> bool:
    return all(current.get(ref.source_id) == ref.revision for ref in sources)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _bounded_text(value: object, *, chars: int, byte_limit: int) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > chars:
        return False
    try:
        return len(value.encode("utf-8")) <= byte_limit
    except UnicodeEncodeError:
        return False


def _valid_card_operation(operation: DuplexMemoryCardOperation) -> bool:
    key = operation.fact_key
    if (
        not _bounded_text(key, chars=128, byte_limit=128)
        or normalize("NFKC", key).casefold() != key
        or any(not (char.isalnum() or char in "/_.:-") for char in key)
        or not _nonnegative_int(operation.expected_generation)
        or (operation.text is not None and not _bounded_text(operation.text, chars=512, byte_limit=2048))
        or not isinstance(operation.evidence, tuple)
        or not 1 <= len(operation.evidence) <= 8
        or not isinstance(operation.parents, tuple)
        or len(operation.parents) > 8
        or any(not _nonnegative_int(parent) or parent == 0 for parent in operation.parents)
        or len(set(operation.parents)) != len(operation.parents)
    ):
        return False
    for span in operation.evidence:
        if not isinstance(span, DuplexMemoryEvidenceSpan) or not all(
            _nonnegative_int(value) for value in (span.source_id, span.revision, span.char_start, span.char_end)
        ):
            return False
        if span.char_end <= span.char_start:
            return False
    return len(set(operation.evidence)) == len(operation.evidence)


class DuplexContextLedger:
    """Session-owned derived nodes that never write to history.

    A node is valid while every recorded source has the recorded revision in
    the snapshot and every parent node is valid. ``refresh`` prunes nodes that
    are no longer valid. Node IDs are never reused.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, DuplexContextNode] = {}
        self._media_dropped_by_source: dict[int, int] = {}
        self._next_node_id = 1
        self._card_heads: dict[str, DuplexMemoryCardHead] = {}
        self._card_catalog_revision = 0
        self._card_lexemes: dict[int, frozenset[str]] | None = None
        self._card_lexeme_bytes = 0
        self._retrieval_cache_closed = False

    @property
    def card_catalog_revision(self) -> int:
        return self._card_catalog_revision

    def card_catalog(self) -> tuple[DuplexMemoryCardHead, ...]:
        """Return bounded heads, including tombstones that fence old work."""
        return tuple(self._card_heads.values())

    def card_candidates(self) -> tuple[DuplexContextNode, ...]:
        """Return active cards as of the latest authoritative refresh."""
        return tuple(self._nodes[head.node_id] for head in self._card_heads.values() if head.node_id is not None)

    def card_lexemes(self, node_id: int) -> frozenset[str]:
        """Reuse terms for an active immutable card after authoritative refresh.

        This index grants no source validity: retrieval must still refresh.
        Closed-owner diagnostics compute without retaining new terms.
        """
        node = self._nodes.get(node_id) if type(node_id) is int and node_id > 0 else None
        if node is None or node.card is None or node.level is not DuplexContextLevel.MEMORY_CARD:
            raise DuplexContextNodeRejectedError("memory_card_not_active")
        head = self._card_heads.get(node.card.fact_key)
        text = node.message.get("content")
        if head is None or head.node_id != node_id or not isinstance(text, str):
            raise DuplexContextNodeRejectedError("memory_card_not_active")
        if self._card_lexemes is not None and node_id in self._card_lexemes:
            return self._card_lexemes[node_id]
        terms = lexemes(node.card.fact_key + " " + text)
        cost = _lexeme_bytes(terms)
        if not self._retrieval_cache_closed and self._card_lexeme_bytes + cost <= _MAX_CARD_LEXEME_BYTES:
            if self._card_lexemes is None:
                self._card_lexemes = {}
            self._card_lexemes[node_id] = terms
            self._card_lexeme_bytes += cost
        return terms

    def close_retrieval_cache(self) -> None:
        """Drop derived text terms and disable retention after session close."""
        self._card_lexemes = None
        self._card_lexeme_bytes = 0
        self._retrieval_cache_closed = True

    def admit_card_batch(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        source_orders: Mapping[int, int],
        batch: DuplexMemoryCardBatch,
    ) -> DuplexMemoryAdmission:
        prepared = self._prepare_card_batch(snapshot, source_orders=source_orders, batch=batch)
        return prepared if isinstance(prepared, DuplexMemoryAdmission) else self._commit_card_batch(prepared)

    def _prepare_card_batch(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        source_orders: Mapping[int, int],
        batch: DuplexMemoryCardBatch,
    ) -> _CardAdmissionPlan | DuplexMemoryAdmission:
        """Preflight the entire batch before changing nodes, heads or IDs.

        The owner supplies a current snapshot and source slot orders. Evidence
        spans must already be bound to legal text by the owner, not the model.
        """
        if (
            not isinstance(batch, DuplexMemoryCardBatch)
            or not isinstance(batch.operations, tuple)
            or not _nonnegative_int(batch.catalog_revision)
        ):
            return DuplexMemoryAdmission("invalid", reason_code="invalid_batch")
        if batch.catalog_revision != self._card_catalog_revision:
            return DuplexMemoryAdmission("stale", reason_code="catalog_changed")
        if len(batch.operations) > 8:
            return DuplexMemoryAdmission("capacity", reason_code="batch_limit")
        if not batch.operations:
            return DuplexMemoryAdmission("no_change")
        current = _current_revisions(snapshot)
        keys: set[str] = set()
        retire_roots: set[int] = set()
        prepared: list[tuple[DuplexMemoryCardOperation, tuple[DuplexSourceRef, ...], int]] = []
        memo: dict[int, bool] = {}
        for operation in batch.operations:
            if not isinstance(operation, DuplexMemoryCardOperation) or not _valid_card_operation(operation):
                return DuplexMemoryAdmission("invalid", reason_code="invalid_card")
            key = operation.fact_key
            if key in keys:
                return DuplexMemoryAdmission("invalid", reason_code="duplicate_fact_key")
            keys.add(key)
            head = self._card_heads.get(key)
            if operation.expected_generation != (0 if head is None else head.generation):
                return DuplexMemoryAdmission("stale", reason_code="card_generation_changed")
            refs = tuple(dict.fromkeys(DuplexSourceRef(span.source_id, span.revision) for span in operation.evidence))
            if not _sources_current(refs, current) or any(ref.source_id not in source_orders for ref in refs):
                return DuplexMemoryAdmission("stale", reason_code="source_changed")
            order = max(source_orders[ref.source_id] for ref in refs)
            if not _nonnegative_int(order) or (head is not None and order < head.evidence_order_floor):
                return DuplexMemoryAdmission("stale", reason_code="older_fact_evidence")
            if not all(self._valid_against(parent, current, memo) for parent in operation.parents):
                return DuplexMemoryAdmission("stale", reason_code="parent_not_valid")
            if head is not None and head.node_id is not None:
                retire_roots.add(head.node_id)
            prepared.append((operation, refs, order))
        if len(self._card_heads) + len(keys.difference(self._card_heads)) > 256:
            return DuplexMemoryAdmission("capacity", reason_code="fact_key_limit")
        retired = self._retired_with_descendants(retire_roots)
        if any(parent in retired for operation, _, _ in prepared for parent in operation.parents):
            return DuplexMemoryAdmission("invalid", reason_code="parent_retired_by_batch")

        # Allocate every record before mutating the graph. Replacement edges
        # are metadata, so a new fact never inherits the replaced fact's sources.
        nodes: list[DuplexContextNode] = []
        heads: list[DuplexMemoryCardHead] = []
        for operation, refs, order in prepared:
            old = self._card_heads.get(operation.fact_key)
            generation = operation.expected_generation + 1
            node_id = None
            if operation.text is not None:
                node_id = self._next_node_id + len(nodes)
                nodes.append(
                    DuplexContextNode(
                        node_id=node_id,
                        level=DuplexContextLevel.MEMORY_CARD,
                        sources=refs,
                        parents=operation.parents,
                        message=MappingProxyType({"role": "user", "content": operation.text}),
                        card=DuplexMemoryCardMetadata(
                            fact_key=operation.fact_key,
                            generation=generation,
                            evidence_order_floor=order,
                            supersedes_node_id=None if old is None else old.node_id,
                            evidence_spans=operation.evidence,
                        ),
                    )
                )
            heads.append(DuplexMemoryCardHead(operation.fact_key, generation, node_id, order))
        projected = dict(self._card_heads)
        for key, head in projected.items():
            if head.node_id in retired:
                projected[key] = DuplexMemoryCardHead(key, head.generation + 1, None, head.evidence_order_floor)
        projected.update((head.fact_key, head) for head in heads)
        return _CardAdmissionPlan(
            self, self._card_catalog_revision, self._next_node_id, retired, tuple(nodes), tuple(projected.values())
        )

    def _commit_card_batch(self, plan: _CardAdmissionPlan) -> DuplexMemoryAdmission:
        """Owner-only: commit immediately after preflight, without yielding."""
        if (
            plan.owner is not self
            or plan.catalog_revision != self._card_catalog_revision
            or plan.next_node_id != self._next_node_id
        ):
            return DuplexMemoryAdmission("stale", reason_code="catalog_changed")
        self._remove_nodes(plan.retired)
        for node in plan.nodes:
            self._nodes[node.node_id] = node
        self._card_heads = {head.fact_key: head for head in plan.catalog}
        self._next_node_id += len(plan.nodes)
        self._card_catalog_revision += 1
        return DuplexMemoryAdmission("applied", node_ids=tuple(node.node_id for node in plan.nodes))

    def _retired_with_descendants(self, roots: set[int]) -> set[int]:
        retired = set(roots)
        for node in self._nodes.values():
            if any(parent in retired for parent in node.parents):
                retired.add(node.node_id)
        return retired

    def _remove_nodes(self, retired: set[int]) -> bool:
        changed_cards = False
        for node_id in retired:
            node = self._nodes.pop(node_id)
            if node.level is DuplexContextLevel.MEDIA_DROPPED:
                source_id = node.sources[0].source_id
                if self._media_dropped_by_source.get(source_id) == node_id:
                    del self._media_dropped_by_source[source_id]
            if node.card is not None:
                if self._card_lexemes is not None:
                    retired_terms = self._card_lexemes.pop(node_id, None)
                    if retired_terms is not None:
                        self._card_lexeme_bytes -= _lexeme_bytes(retired_terms)
                    if not self._card_lexemes:
                        self._card_lexemes = None
                head = self._card_heads[node.card.fact_key]
                if head.node_id == node_id:
                    self._card_heads[head.fact_key] = DuplexMemoryCardHead(
                        head.fact_key, head.generation + 1, None, head.evidence_order_floor
                    )
                    changed_cards = True
        return changed_cards

    def add_node(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        level: DuplexContextLevel,
        sources: tuple[DuplexSourceRef, ...],
        parents: tuple[int, ...] = (),
        message: Mapping[str, object],
    ) -> DuplexContextNode:
        """Admit a node only if it is valid against ``snapshot``.

        Validation does not prune: a snapshot older than the last refresh must
        not discard nodes built from newer revisions or displace their L1 lookup.
        At most one L1 node is admitted per source revision between refreshes.
        """
        return self._add_node(
            _current_revisions(snapshot),
            level=level,
            sources=sources,
            parents=parents,
            message=copy_history_message(dict(message)),
        )

    def add_l1_nodes(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        messages: Mapping[int, Mapping[str, object]],
    ) -> tuple[DuplexContextNode, ...]:
        """Admit L1 messages in mapping iteration order.

        Unknown sources or existing valid L1 nodes reject the entire batch.
        """
        if not messages:
            return ()
        current = _current_revisions(snapshot)
        if any(source_id not in current for source_id in messages):
            raise DuplexContextNodeRejectedError("unknown_source_id")
        for source_id in messages:
            if self._has_l1_revision(source_id, current[source_id]):
                raise DuplexContextNodeRejectedError("source_already_has_media_dropped_node")
        # Copy the whole batch before writing any node or consuming an ID.
        prepared = tuple((source_id, copy_history_message(dict(message))) for source_id, message in messages.items())
        return tuple(
            self._add_node(
                current,
                level=DuplexContextLevel.MEDIA_DROPPED,
                sources=(DuplexSourceRef(source_id, current[source_id]),),
                message=message,
            )
            for source_id, message in prepared
        )

    def _add_node(
        self,
        current: Mapping[int, int],
        *,
        level: DuplexContextLevel,
        sources: tuple[DuplexSourceRef, ...],
        parents: tuple[int, ...] = (),
        message: Mapping[str, object],
    ) -> DuplexContextNode:
        """Validate and store a message already detached by a public admission API."""
        if level is DuplexContextLevel.MEMORY_CARD:
            raise DuplexContextNodeRejectedError("memory_card_requires_batch_admission")
        if not sources and not parents:
            raise DuplexContextNodeRejectedError("node_has_no_dependencies")
        if not _sources_current(sources, current):
            raise DuplexContextNodeRejectedError("stale_source_revision")
        if level is DuplexContextLevel.MEDIA_DROPPED:
            if len(sources) != 1 or parents:
                raise DuplexContextNodeRejectedError("media_dropped_node_needs_exactly_one_source")
            if self._has_l1_revision(sources[0].source_id, sources[0].revision):
                raise DuplexContextNodeRejectedError("source_already_has_media_dropped_node")
        memo: dict[int, bool] = {}
        if not all(self._valid_against(parent, current, memo) for parent in parents):
            raise DuplexContextNodeRejectedError("parent_not_valid")
        node = DuplexContextNode(
            node_id=self._next_node_id,
            level=level,
            sources=tuple(sources),
            parents=tuple(parents),
            message=message,
        )
        self._next_node_id += 1
        self._nodes[node.node_id] = node
        if level is DuplexContextLevel.MEDIA_DROPPED:
            existing = self.l1_node_for(sources[0].source_id)
            if existing is None or existing.sources[0].revision < sources[0].revision:
                self._media_dropped_by_source[sources[0].source_id] = node.node_id
        return node

    def _has_l1_revision(self, source_id: int, revision: int) -> bool:
        existing = self.l1_node_for(source_id)
        if existing is None or existing.sources[0].revision < revision:
            return False
        if existing.sources[0].revision == revision:
            return True
        # Only out-of-order historical admission needs a scan. The synchronous
        # current-snapshot renderer uses the indexed newest revision above.
        return any(
            node.level is DuplexContextLevel.MEDIA_DROPPED and node.sources[0] == DuplexSourceRef(source_id, revision)
            for node in self._nodes.values()
        )

    def refresh(self, snapshot: DuplexHistorySnapshot) -> tuple[int, ...]:
        """Prune against a current authoritative snapshot; return IDs in order."""
        current = _current_revisions(snapshot)
        removed: list[int] = []
        retired: set[int] = set()
        # Insertion order is ascending node ID, so parents are decided first.
        for node in tuple(self._nodes.values()):
            if _sources_current(node.sources, current) and all(
                parent in self._nodes and parent not in retired for parent in node.parents
            ):
                continue
            retired.add(node.node_id)
            removed.append(node.node_id)
        if self._remove_nodes(retired):
            self._card_catalog_revision += 1
        return tuple(removed)

    def discard(self, node_id: int) -> tuple[int, ...]:
        """Retire a derived node and its descendants; return IDs in order."""
        if node_id not in self._nodes:
            return ()
        retired = self._retired_with_descendants({node_id})
        removed = tuple(candidate for candidate in self._nodes if candidate in retired)
        if self._remove_nodes(retired):
            self._card_catalog_revision += 1
        return removed

    def valid(self, node_id: int) -> bool:
        """Whether the node survived the latest refresh."""
        return node_id in self._nodes

    def nodes(self) -> tuple[DuplexContextNode, ...]:
        return tuple(self._nodes.values())

    def l1_node_for(self, source_id: int) -> DuplexContextNode | None:
        node_id = self._media_dropped_by_source.get(source_id)
        return None if node_id is None else self._nodes.get(node_id)

    def _valid_against(self, node_id: int, current: Mapping[int, int], memo: dict[int, bool]) -> bool:
        if node_id not in memo:
            node = self._nodes.get(node_id)
            memo[node_id] = (
                node is not None
                and _sources_current(node.sources, current)
                and all(self._valid_against(parent, current, memo) for parent in node.parents)
            )
        return memo[node_id]


@dataclass(frozen=True)
class _CardAdmissionPlan:
    owner: DuplexContextLedger
    catalog_revision: int
    next_node_id: int
    retired: set[int]
    nodes: tuple[DuplexContextNode, ...]
    catalog: tuple[DuplexMemoryCardHead, ...]
