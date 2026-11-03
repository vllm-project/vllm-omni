# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded lexical selection and session-owned pending-correction fences."""

from dataclasses import dataclass, field, replace
from unicodedata import normalize

from vllm_omni.engine.duplex.memory_types import DuplexMemoryCardHead, DuplexMemoryEvidenceSpan
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, DuplexSourceRef
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot
from vllm_omni.engine.duplex.session.memory_lexical import lexemes as _lexemes


@dataclass(frozen=True)
class DuplexRetrievedCard:
    node_id: int
    fact_key: str = field(repr=False)
    generation: int
    sources: tuple[DuplexSourceRef, ...]
    text: str = field(repr=False)
    score: tuple[int, int]


def retrieve_memory_cards(
    *,
    snapshot: DuplexHistorySnapshot,
    ledger: DuplexContextLedger,
    query: str,
    suppressed_keys: frozenset[str],
    suppress_all: bool,
    max_candidates: int,
) -> tuple[DuplexRetrievedCard, ...]:
    """Return ranked candidates; the consumer owns final template/token cost.

    Source validity is refreshed once, never once per card. The caller must
    also fence eligibility after asynchronous prompt preparation.
    """
    if type(max_candidates) is not int or not 0 <= max_candidates <= 16:
        raise ValueError("invalid_memory_candidate_limit")
    if suppress_all or max_candidates == 0 or not query.strip() or len(query) > 4096:
        return ()
    try:
        if len(query.encode("utf-8")) > 16384:
            return ()
    except UnicodeEncodeError:
        return ()
    query = normalize("NFKC", query).casefold()
    terms = frozenset(sorted(_lexemes(query))[:256])
    ledger.refresh(snapshot)
    ranked: list[DuplexRetrievedCard] = []
    for node in ledger.card_candidates():
        metadata = node.card
        if metadata is None or metadata.fact_key in suppressed_keys:
            continue
        text = node.message.get("content")
        if not isinstance(text, str):
            continue
        score = (int(metadata.fact_key in query), len(terms.intersection(ledger.card_lexemes(node.node_id))))
        if score == (0, 0):
            continue
        ranked.append(
            DuplexRetrievedCard(node.node_id, metadata.fact_key, metadata.generation, node.sources, text, score)
        )
    ranked.sort(key=lambda card: (-card.score[0], -card.score[1], -card.node_id, card.fact_key))
    return tuple(ranked[:max_candidates])


@dataclass(frozen=True)
class DuplexMemoryBarrier:
    barrier_id: int
    generation: int
    evidence: tuple[DuplexMemoryEvidenceSpan, ...]
    keys: frozenset[str] | None = field(repr=False)
    required_keys: frozenset[str] = field(repr=False)


def _current_evidence(snapshot: DuplexHistorySnapshot, evidence: tuple[DuplexMemoryEvidenceSpan, ...]) -> bool:
    current = dict(zip(snapshot.source_ids, snapshot.source_revisions, strict=True))
    return (
        isinstance(evidence, tuple)
        and 1 <= len(evidence) <= 8
        and all(
            isinstance(span, DuplexMemoryEvidenceSpan)
            and all(type(value) is int and value >= 0 for value in (span.source_id, span.revision, span.char_start))
            and type(span.char_end) is int
            and span.char_end > span.char_start
            and current.get(span.source_id) == span.revision
            for span in evidence
        )
    )


def _merge_spans(spans: tuple[DuplexMemoryEvidenceSpan, ...]) -> tuple[DuplexMemoryEvidenceSpan, ...]:
    merged: list[DuplexMemoryEvidenceSpan] = []
    for span in sorted(spans, key=lambda value: (value.source_id, value.revision, value.char_start, value.char_end)):
        if (
            merged
            and (merged[-1].source_id, merged[-1].revision) == (span.source_id, span.revision)
            and span.char_start <= merged[-1].char_end
        ):
            previous = merged[-1]
            merged[-1] = replace(previous, char_end=max(previous.char_end, span.char_end))
        else:
            merged.append(span)
    return tuple(merged)


def _covers(expected: tuple[DuplexMemoryEvidenceSpan, ...], actual: tuple[DuplexMemoryEvidenceSpan, ...]) -> bool:
    return all(
        any(
            (span.source_id, span.revision) == (target.source_id, target.revision)
            and span.char_start <= target.char_start
            and span.char_end >= target.char_end
            for span in actual
        )
        for target in expected
    )


class DuplexMemorySuppression:
    """Persistent barriers, resolved only by explicit owner-validated proof.

    This tracks deterministic eligibility, not recognition of natural-language
    contradictions. An empty card extraction never calls ``resolve``. Invalid
    trigger evidence stays blocked until a later authoritative reconciliation.
    """

    def __init__(self) -> None:
        self._barriers: dict[int, DuplexMemoryBarrier] = {}
        self._coverage: dict[int, dict[str | None, tuple[DuplexMemoryEvidenceSpan, ...]]] = {}
        self._coverage_generations: dict[int, dict[str | None, int]] = {}
        self._next_id = 1
        self._version = 0
        self._overflowed = False
        self._overflow_generation = 0
        self._reconciling = False

    @property
    def version(self) -> int:
        return self._version

    @property
    def suppress_all(self) -> bool:
        return self._reconciling or self._overflowed or any(barrier.keys is None for barrier in self._barriers.values())

    @property
    def suppressed_keys(self) -> frozenset[str]:
        return frozenset(key for barrier in self._barriers.values() if barrier.keys is not None for key in barrier.keys)

    def barriers(self) -> tuple[DuplexMemoryBarrier, ...]:
        return tuple(self._barriers.values())

    def reconciliation_required(self, snapshot: DuplexHistorySnapshot) -> bool:
        return (
            self._reconciling
            or self._overflowed
            or any(not _current_evidence(snapshot, barrier.evidence) for barrier in self._barriers.values())
        )

    def begin_reconciliation(self) -> tuple[int, int]:
        if not self._reconciling:
            self._reconciling = True
            self._version += 1
        return self._next_id - 1, self._overflow_generation

    def finish_reconciliation(self, *, barrier_cutoff: int, overflow_generation: int) -> None:
        """Owner-only: all current legal text/pages need explicit confirmation."""
        for barrier_id in tuple(self._barriers):
            if barrier_id <= barrier_cutoff:
                del self._barriers[barrier_id]
                del self._coverage[barrier_id]
                del self._coverage_generations[barrier_id]
        if self._overflow_generation == overflow_generation:
            self._overflowed = False
        self._reconciling = False
        self._version += 1

    def stage(self) -> "DuplexMemorySuppression":
        """Detach bounded proof metadata for an owner-only admission transaction."""
        staged = DuplexMemorySuppression()
        staged._barriers = self._barriers.copy()
        staged._coverage = {key: value.copy() for key, value in self._coverage.items()}
        staged._coverage_generations = {key: value.copy() for key, value in self._coverage_generations.items()}
        staged._next_id = self._next_id
        staged._version = self._version
        staged._overflowed = self._overflowed
        staged._overflow_generation = self._overflow_generation
        staged._reconciling = self._reconciling
        return staged

    def commit(self, staged: "DuplexMemorySuppression") -> None:
        """Commit validated metadata synchronously with its card admission."""
        self._barriers = staged._barriers
        self._coverage = staged._coverage
        self._coverage_generations = staged._coverage_generations
        self._next_id = staged._next_id
        self._version = staged._version
        self._overflowed = staged._overflowed
        self._overflow_generation = staged._overflow_generation
        self._reconciling = staged._reconciling

    def block(
        self,
        *,
        snapshot: DuplexHistorySnapshot,
        evidence: tuple[DuplexMemoryEvidenceSpan, ...],
        keys: frozenset[str] | None,
        catalog: tuple[DuplexMemoryCardHead, ...],
    ) -> DuplexMemoryBarrier | None:
        if not _current_evidence(snapshot, evidence) or (
            keys is not None and (not isinstance(keys, frozenset) or not 1 <= len(keys) <= 256)
        ):
            raise ValueError("invalid_memory_barrier")
        for barrier in self._barriers.values():
            if barrier.evidence == evidence and barrier.keys == keys:
                return barrier
        if len(self._barriers) == 256:
            self._overflowed = True
            self._overflow_generation += 1
            self._version += 1
            return None
        self._version += 1
        required_keys = frozenset(head.fact_key for head in catalog) if keys is None else keys
        barrier = DuplexMemoryBarrier(self._next_id, self._version, evidence, keys, required_keys)
        self._next_id += 1
        self._barriers[barrier.barrier_id] = barrier
        self._coverage[barrier.barrier_id] = {}
        self._coverage_generations[barrier.barrier_id] = {}
        return barrier

    def resolve(
        self,
        *,
        snapshot: DuplexHistorySnapshot,
        catalog: tuple[DuplexMemoryCardHead, ...],
        barrier_id: int,
        generation: int,
        evidence: tuple[DuplexMemoryEvidenceSpan, ...],
        keys: frozenset[str],
    ) -> bool:
        """Accept a proof fragment; retain the barrier until all scope is covered.

        The owner calls this only for an explicit semantic resolution that
        passed the same execution/admission fences as its candidate batch.
        """
        if type(barrier_id) is not int or type(generation) is not int:
            return False
        barrier = self._barriers.get(barrier_id)
        if (
            barrier is None
            or barrier.generation != generation
            or not isinstance(keys, frozenset)
            or not _current_evidence(snapshot, barrier.evidence)
            or not _current_evidence(snapshot, evidence)
        ):
            return False
        # A proof fragment must lie inside the exact captured trigger ranges.
        if not all(_covers((span,), barrier.evidence) for span in evidence):
            return False
        required = barrier.required_keys
        if barrier.keys is None:
            required |= frozenset(head.fact_key for head in catalog)
        if not keys.issubset(required) or (required and not keys):
            return False
        current_generations: dict[str | None, int] = {head.fact_key: head.generation for head in catalog}
        previous_generations = self._coverage_generations[barrier_id]
        coverage = {
            key: spans
            for key, spans in self._coverage[barrier_id].items()
            if previous_generations.get(key) == current_generations.get(key, 0)
        }
        targets: frozenset[str | None] = frozenset(keys) if keys else frozenset({None})
        for target in targets:
            coverage[target] = _merge_spans(coverage.get(target, ()) + evidence)
        if sum(len(spans) for spans in coverage.values()) > 2048:
            return False
        all_targets: frozenset[str | None] = frozenset(required) if required else frozenset({None})
        if all(_covers(barrier.evidence, coverage.get(target, ())) for target in all_targets):
            del self._barriers[barrier_id]
            del self._coverage[barrier_id]
            del self._coverage_generations[barrier_id]
        else:
            self._barriers[barrier_id] = replace(barrier, required_keys=required)
            self._coverage[barrier_id] = coverage
            self._coverage_generations[barrier_id] = {key: current_generations.get(key, 0) for key in coverage}
        self._version += 1
        return True
