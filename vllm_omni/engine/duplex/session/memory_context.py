# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Owner-issued text evidence and incremental progress for context work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.memory_types import (
    DuplexContextAuxiliaryInput,
    DuplexContextAuxiliaryWork,
    DuplexContextBarrier,
    DuplexContextEvidence,
    DuplexMemoryAdmission,
    DuplexMemoryCardBatch,
    DuplexMemoryCardOperation,
    DuplexMemoryCursor,
    DuplexMemoryEvidenceSelection,
    DuplexMemoryEvidenceSpan,
    DuplexMemoryProposal,
    DuplexMemoryProposalBatch,
    DuplexMemoryResolution,
)

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot


def _text(message: dict[str, object]) -> str:
    """Read text/transcripts without stringifying opaque media or values."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    audio_transcript = False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(value := part.get("text"), str):
            parts.append(value)
        elif part.get("type") == "audio_url" and isinstance(value := part.get("transcript"), str):
            parts.append(value)
            audio_transcript = True
    if not audio_transcript and isinstance(value := message.get("transcript"), str):
        parts.append(value)
    return "\n".join(parts)


def _legal_text(session: DuplexEngineSession, item_id: str | None, message: dict[str, object]) -> tuple[str, bool]:
    """Return legal text and whether a live settlement boundary is pending."""
    role = message.get("role")
    if role not in {"user", "assistant"}:
        return "", False
    if role == "user":
        return _text(message), False
    # Unbound assistant text cannot prove which response owned its audio.
    if item_id is None:
        return "", False
    response_id = item_id.removeprefix("item_") if item_id.startswith("item_") else None
    if response_id is not None and (
        response_id == session.active_response_id or session.response_has_draining_request(response_id)
    ):
        return "", True
    text = _text(message)
    if not session.history_item_has_audio(item_id):
        return text, False
    prefix = session._conversation.heard_prefixes.get(item_id)
    if prefix is not None:
        accepted = prefix[0][: prefix[1]].rstrip()
        # Authoritative text may be even shorter; never expand it from retained
        # generated text or estimate a prefix from character/audio ratios.
        if accepted.startswith(text):
            return text, False
        return "", True
    if response_id is None or response_id in session._conversation.assistant_response_snapshots:
        return "", True
    # Finalized, acknowledged audio has released its original generation. The
    # stored history is the committed prefix and retains audio provenance.
    return text, False


def omitted_user_sources_covered(session: DuplexEngineSession, source_ids: tuple[int, ...]) -> bool:
    """A pending or partial window cannot certify omitted user text as processed.

    Read only the completed frontier, including edits waiting to rewind it.
    This does not invalidate cards on tail append when that text is retained
    in the foreground prompt, nor advance/refund any background work.
    """
    state = session._memory_context
    frontier = 0 if state is None else state._cursor.slot_order
    if state is not None and state._dirty_order is not None:
        frontier = min(frontier, state._dirty_order)
    omitted = frozenset(source_ids)
    return all(
        source.slot_order < frontier or source.message.get("role") != "user" or not _text(source.message)
        for source in session._conversation.sources.values()
        if source.source_id in omitted
    )


class DuplexMemoryContextState:
    """One bounded pending window; no history, media or node store of its own."""

    def __init__(self) -> None:
        self._scope = uuid4().hex
        self._cursor = DuplexMemoryCursor()
        self._generation = 0
        self._dirty_order: int | None = None
        self._pending: DuplexContextAuxiliaryInput | None = None
        self._limits: tuple[int, int, frozenset[str]] | None = None
        self._pass: _CatalogPass | None = None
        self._next_pass_id = 1
        self._recovery: _ReconciliationPass | None = None
        self._next_recovery_id = 1
        self._force_recovery = False
        self._catalog_width = 16
        self._adaptive_chars = 8192

    @property
    def retrieval_blocked(self) -> bool:
        """An explicit restart blocks old cards before the next capture."""
        return self._force_recovery or self._recovery is not None

    def note_change(self, slot_order: int) -> None:
        if self._recovery is not None and slot_order < self._recovery.target:
            self._recovery.rewrite_pending = True
        end = self._pending.work.end if self._pending is not None else self._cursor
        frontier = max(self._cursor, end)
        if slot_order < frontier.slot_order or (slot_order == frontier.slot_order and frontier.char_offset > 0):
            self._dirty_order = slot_order if self._dirty_order is None else min(slot_order, self._dirty_order)

    def restart_reconciliation(self) -> None:
        """Explicitly reschedule a parked pass; ordinary traffic cannot do so."""
        self._force_recovery = True
        self._recovery = None
        self._pending = None
        self._pass = None

    def _sync_recovery(self, session: DuplexEngineSession, snapshot: DuplexHistorySnapshot) -> None:
        if self._recovery is None:
            if not self._force_recovery and not session.memory_suppression.reconciliation_required(snapshot):
                return
            cutoff, overflow_generation = session.memory_suppression.begin_reconciliation()
            self._recovery = _ReconciliationPass(
                reconciliation_id=self._next_recovery_id,
                target=session._conversation.next_slot_order,
                barrier_cutoff=cutoff,
                overflow_generation=overflow_generation,
                config_revision=session.config_revision,
                catalog_revision=session.context_ledger.card_catalog_revision,
                keys=tuple(head.fact_key for head in session.context_ledger.card_catalog()),
            )
            self._next_recovery_id += 1
            self._force_recovery = False
            self._cursor = DuplexMemoryCursor()
            self._pass = None
            self._pending = None
            self._dirty_order = None
        recovery = self._recovery
        if (
            recovery.rewrite_pending
            or recovery.catalog_revision != session.context_ledger.card_catalog_revision
            or recovery.config_revision != session.config_revision
        ):
            recovery.rewrite_epoch += 1
            recovery.rewrite_pending = False
            recovery.config_revision = session.config_revision
            recovery.catalog_revision = session.context_ledger.card_catalog_revision
            recovery.keys = tuple(head.fact_key for head in session.context_ledger.card_catalog())
            recovery.known_keys = set(recovery.keys)
            recovery.replay_keys.clear()
            recovery.confirmed = True
            recovery.parked = False
            self._cursor = DuplexMemoryCursor()
            self._pass = None
            self._pending = None
            self._dirty_order = None

    def capture(
        self,
        session: DuplexEngineSession,
        *,
        max_sources: int,
        max_chars: int,
        excluded_item_ids: frozenset[str],
    ) -> DuplexContextAuxiliaryInput | None:
        if type(max_sources) is not int or not 1 <= max_sources <= 8:
            raise ValueError("invalid_memory_source_limit")
        if type(max_chars) is not int or not 1 <= max_chars <= 8192:
            raise ValueError("invalid_memory_text_limit")
        if session.state is not DuplexSessionState.OPEN:
            return None
        max_chars = min(max_chars, self._adaptive_chars)
        limits = (max_sources, max_chars, excluded_item_ids)
        if (
            self._pending is not None
            and self._dirty_order is None
            and limits == self._limits
            and self.matches(session, self._pending)
        ):
            return self._pending
        self._pending = None
        if limits != self._limits:
            self._pass = None
        self._limits = limits
        if self._dirty_order is not None:
            self._cursor = min(self._cursor, DuplexMemoryCursor(self._dirty_order))
            self._dirty_order = None
            self._pass = None
        snapshot = session.history_snapshot()
        session.context_ledger.refresh(snapshot)
        self._sync_recovery(session, snapshot)
        if self._recovery is not None and self._recovery.parked:
            return None
        catalog = session.context_ledger.card_catalog()
        if self._pass is not None:
            if (
                self._pass.catalog_revision != session.context_ledger.card_catalog_revision
                or self._pass.config_revision != session.config_revision
                or not self._evidence_matches(session, self._pass.evidence)
            ):
                self._pass = None
            else:
                return self._capture_page(session, snapshot.history_revision)
        sources = {source.source_id: source for source in session._conversation.sources.values()}
        evidence: list[DuplexContextEvidence] = []
        remaining = max_chars
        end = self._cursor
        for message, item_id, source_id, revision in zip(
            snapshot.messages, snapshot.item_ids, snapshot.source_ids, snapshot.source_revisions, strict=True
        ):
            source = sources[source_id]
            if self._recovery is not None and source.slot_order >= self._recovery.target:
                end = max(end, DuplexMemoryCursor(self._recovery.target))
                break
            if source.slot_order < self._cursor.slot_order:
                continue
            if item_id in excluded_item_ids:
                break
            text, pending = _legal_text(session, item_id, message)
            if pending:
                break
            offset = self._cursor.char_offset if source.slot_order == self._cursor.slot_order else 0
            if not text or offset >= len(text):
                end = DuplexMemoryCursor(source.slot_order + 1)
                continue
            stop = min(len(text), offset + remaining)
            role: Literal["user", "assistant"] = "user" if message.get("role") == "user" else "assistant"
            evidence.append(
                DuplexContextEvidence(
                    len(evidence), source_id, revision, source.slot_order, role, text[offset:stop], offset, stop
                )
            )
            remaining -= stop - offset
            end = (
                DuplexMemoryCursor(source.slot_order + 1)
                if stop == len(text)
                else DuplexMemoryCursor(source.slot_order, stop)
            )
            if remaining == 0 or len(evidence) == max_sources:
                break
        else:
            if self._recovery is not None:
                end = max(end, DuplexMemoryCursor(self._recovery.target))
        if not evidence:
            self._cursor = end
            if self._recovery is None or end < DuplexMemoryCursor(self._recovery.target):
                return None
        keys = tuple(head.fact_key for head in catalog) if self._recovery is None else self._recovery.keys
        self._pass = _CatalogPass(
            pass_id=self._next_pass_id,
            config_revision=session.config_revision,
            start=self._cursor,
            end=end,
            evidence=tuple(evidence),
            pages=[keys[index : index + self._catalog_width] for index in range(0, len(keys), self._catalog_width)]
            or [()],
            catalog_revision=session.context_ledger.card_catalog_revision,
        )
        self._next_pass_id += 1
        return self._capture_page(session, snapshot.history_revision)

    def _capture_page(self, session: DuplexEngineSession, history_revision: int) -> DuplexContextAuxiliaryInput:
        current_pass = self._pass
        assert current_pass is not None
        heads = {head.fact_key: head for head in session.context_ledger.card_catalog()}
        work = DuplexContextAuxiliaryWork(
            scope_id=self._scope,
            epoch=session.epoch,
            config_revision=session.config_revision,
            progress_generation=self._generation,
            start=current_pass.start,
            end=current_pass.end,
            read_set=tuple(
                DuplexMemoryEvidenceSpan(e.source_id, e.revision, e.char_start, e.char_end)
                for e in current_pass.evidence
            ),
            catalog_revision=session.context_ledger.card_catalog_revision,
            pass_id=current_pass.pass_id,
            catalog_page=current_pass.page,
            suppression_version=session.memory_suppression.version,
            reconciliation_id=None if self._recovery is None else self._recovery.reconciliation_id,
            reconciliation_target=None if self._recovery is None else self._recovery.target,
            rewrite_epoch=0 if self._recovery is None else self._recovery.rewrite_epoch,
        )
        page_keys = frozenset(current_pass.pages[current_pass.page])
        grants: list[DuplexContextBarrier] = []
        for barrier in session.memory_suppression.barriers():
            selections: list[DuplexMemoryEvidenceSelection] = []
            for span in barrier.evidence:
                for item in current_pass.evidence:
                    if (item.source_id, item.revision) != (span.source_id, span.revision):
                        continue
                    start, end = max(item.char_start, span.char_start), min(item.char_end, span.char_end)
                    if start < end:
                        selections.append(
                            DuplexMemoryEvidenceSelection(item.handle, start - item.char_start, end - item.char_start)
                        )
            if selections:
                allowed_keys = page_keys if barrier.keys is None else page_keys.intersection(barrier.keys)
                if not allowed_keys and (barrier.required_keys or page_keys):
                    continue
                grants.append(
                    DuplexContextBarrier(
                        len(grants), barrier.barrier_id, barrier.generation, allowed_keys, tuple(selections)
                    )
                )
        self._pending = DuplexContextAuxiliaryInput(
            work,
            current_pass.evidence,
            tuple(heads[key] for key in current_pass.pages[current_pass.page]),
            history_revision,
            tuple(grants),
        )
        return self._pending

    def matches(self, session: DuplexEngineSession, context: DuplexContextAuxiliaryInput) -> bool:
        if (
            context is not self._pending
            or session.state is not DuplexSessionState.OPEN
            or context.work.scope_id != self._scope
            or context.work.epoch != session.epoch
            or context.work.config_revision != session.config_revision
            or context.work.catalog_revision != session.context_ledger.card_catalog_revision
            or context.work.suppression_version != session.memory_suppression.version
            or (self._recovery is not None and self._recovery.rewrite_pending)
        ):
            return False
        return self._evidence_matches(session, context.evidence)

    def rechunk(self, session: DuplexEngineSession, context: DuplexContextAuxiliaryInput) -> bool:
        """Replace retained work with a smaller capture without acknowledging it.

        Ceilings only shrink during a session, so an unfit minimum parks rather
        than repeatedly re-tokenizing. Restarting the current catalogue pass may
        replay pages, but cannot advance past an unexamined text range.
        """
        if not self.matches(session, context):
            return False
        if len(context.catalog) > 1:
            self._catalog_width = min(self._catalog_width, len(context.catalog) // 2)
        else:
            chars = sum(len(item.text) for item in context.evidence)
            if chars <= 1:
                return False
            self._adaptive_chars = min(self._adaptive_chars, chars // 2)
        self._pending = None
        self._pass = None
        self._generation += 1
        return True

    def _evidence_matches(self, session: DuplexEngineSession, evidence: tuple[DuplexContextEvidence, ...]) -> bool:
        sources = {source.source_id: source for source in session._conversation.sources.values()}
        for item in evidence:
            source = sources.get(item.source_id)
            if source is None or source.revision != item.revision or source.slot_order != item.slot_order:
                return False
            text, pending = _legal_text(session, source.item_id, source.message)
            if pending or text[item.char_start : item.char_end] != item.text:
                return False
        return True

    @staticmethod
    def _bind_selections(
        context: DuplexContextAuxiliaryInput, selections: tuple[DuplexMemoryEvidenceSelection, ...]
    ) -> tuple[DuplexMemoryEvidenceSpan, ...] | None:
        if not isinstance(selections, tuple) or not 1 <= len(selections) <= 8:
            return None
        spans: list[DuplexMemoryEvidenceSpan] = []
        for selection in selections:
            if not isinstance(selection, DuplexMemoryEvidenceSelection) or not all(
                type(value) is int for value in (selection.handle, selection.char_start, selection.char_end)
            ):
                return None
            if not 0 <= selection.handle < len(context.evidence):
                return None
            item = context.evidence[selection.handle]
            if not 0 <= selection.char_start < selection.char_end <= len(item.text):
                return None
            spans.append(
                DuplexMemoryEvidenceSpan(
                    item.source_id,
                    item.revision,
                    item.char_start + selection.char_start,
                    item.char_start + selection.char_end,
                )
            )
        return tuple(spans)

    def admit(
        self, session: DuplexEngineSession, *, context: DuplexContextAuxiliaryInput, proposal: DuplexMemoryProposalBatch
    ) -> DuplexMemoryAdmission:
        snapshot = session.history_snapshot()
        session.context_ledger.refresh(snapshot)
        if not self.matches(session, context):
            return DuplexMemoryAdmission("stale", reason_code="memory_context_changed")
        if (
            not isinstance(proposal, DuplexMemoryProposalBatch)
            or not isinstance(proposal.operations, tuple)
            or not isinstance(proposal.resolutions, tuple)
            or type(proposal.reconciled) is not bool
            or (proposal.reconciled and context.work.reconciliation_id is None)
        ):
            return DuplexMemoryAdmission("invalid", reason_code="invalid_memory_proposal")
        if len(proposal.operations) > 8 or len(proposal.resolutions) > 8:
            return DuplexMemoryAdmission("capacity", reason_code="batch_limit")
        generations = {head.fact_key: head.generation for head in context.catalog}
        operations: list[DuplexMemoryCardOperation] = []
        for operation in proposal.operations:
            if (
                not isinstance(operation, DuplexMemoryProposal)
                or not isinstance(operation.fact_key, str)
                or not isinstance(operation.evidence, tuple)
                or not 1 <= len(operation.evidence) <= 8
            ):
                return DuplexMemoryAdmission("invalid", reason_code="invalid_memory_proposal")
            spans = self._bind_selections(context, operation.evidence)
            if spans is None:
                return DuplexMemoryAdmission("invalid", reason_code="invalid_evidence_selection")
            operations.append(
                DuplexMemoryCardOperation(
                    operation.fact_key, generations.get(operation.fact_key, 0), spans, operation.text
                )
            )
        ledger = session.context_ledger
        orders = {source.source_id: source.slot_order for source in session._conversation.sources.values()}
        prepared = ledger._prepare_card_batch(
            snapshot,
            source_orders=orders,
            batch=DuplexMemoryCardBatch(tuple(operations), context.work.catalog_revision),
        )
        if isinstance(prepared, DuplexMemoryAdmission) and prepared.status != "no_change":
            return prepared
        staged = session.memory_suppression.stage() if proposal.resolutions else None
        projected_catalog = ledger.card_catalog() if isinstance(prepared, DuplexMemoryAdmission) else prepared.catalog
        by_key = {operation.fact_key: operation for operation in operations}
        for resolution in proposal.resolutions:
            if (
                not isinstance(resolution, DuplexMemoryResolution)
                or type(resolution.handle) is not int
                or not 0 <= resolution.handle < len(context.barriers)
                or not isinstance(resolution.keys, frozenset)
                or not all(isinstance(key, str) for key in resolution.keys)
                or not isinstance(resolution.disposition, str)
                or resolution.disposition not in {"replace", "retract", "no_conflict"}
            ):
                return DuplexMemoryAdmission("invalid", reason_code="invalid_memory_resolution")
            grant = context.barriers[resolution.handle]
            spans = self._bind_selections(context, resolution.evidence)
            if spans is None or not resolution.keys.issubset(grant.keys):
                return DuplexMemoryAdmission("invalid", reason_code="invalid_memory_resolution")
            if resolution.disposition != "no_conflict" and (
                not resolution.keys
                or any(
                    key not in by_key or (by_key[key].text is None) != (resolution.disposition == "retract")
                    for key in resolution.keys
                )
            ):
                return DuplexMemoryAdmission("invalid", reason_code="missing_resolution_operation")
            assert staged is not None
            if not staged.resolve(
                snapshot=snapshot,
                catalog=projected_catalog,
                barrier_id=grant.barrier_id,
                generation=grant.generation,
                evidence=spans,
                keys=resolution.keys,
            ):
                return DuplexMemoryAdmission("invalid", reason_code="invalid_memory_resolution")
        result = prepared if isinstance(prepared, DuplexMemoryAdmission) else ledger._commit_card_batch(prepared)
        if result.status in {"applied", "no_change"}:
            if staged is not None:
                session.memory_suppression.commit(staged)
            current_pass = self._pass
            assert current_pass is not None
            known = {key for page in current_pass.pages for key in page}
            new_keys = tuple(
                head.fact_key for head in session.context_ledger.card_catalog() if head.fact_key not in known
            )
            if self._recovery is not None:
                recovery = self._recovery
                recovery.confirmed = recovery.confirmed and proposal.reconciled
                recovery.catalog_revision = session.context_ledger.card_catalog_revision
                discovered = {head.fact_key for head in session.context_ledger.card_catalog()} - recovery.known_keys
                new_keys = tuple(sorted(discovered))
                recovery.replay_keys.update(discovered)
                recovery.known_keys.update(discovered)
            current_pass.pages.extend(
                new_keys[index : index + self._catalog_width] for index in range(0, len(new_keys), self._catalog_width)
            )
            current_pass.catalog_revision = session.context_ledger.card_catalog_revision
            current_pass.page += 1
            if current_pass.page == len(current_pass.pages):
                self._cursor = context.work.end
                self._pass = None
            if self._dirty_order is not None:
                self._cursor = min(self._cursor, DuplexMemoryCursor(self._dirty_order))
                self._dirty_order = None
                self._pass = None
            self._pending = None
            self._generation += 1
            if (
                self._recovery is not None
                and self._pass is None
                and self._cursor >= DuplexMemoryCursor(self._recovery.target)
            ):
                recovery = self._recovery
                if recovery.replay_keys:
                    recovery.keys = tuple(sorted(recovery.replay_keys))
                    recovery.replay_keys.clear()
                    self._cursor = DuplexMemoryCursor()
                elif recovery.confirmed:
                    session.memory_suppression.finish_reconciliation(
                        barrier_cutoff=recovery.barrier_cutoff,
                        overflow_generation=recovery.overflow_generation,
                    )
                    self._recovery = None
                else:
                    recovery.parked = True
        return result


@dataclass
class _CatalogPass:
    """Bounded key membership for one source window, independent of live heads."""

    pass_id: int
    config_revision: int
    start: DuplexMemoryCursor
    end: DuplexMemoryCursor
    evidence: tuple[DuplexContextEvidence, ...]
    pages: list[tuple[str, ...]]
    catalog_revision: int
    page: int = 0


@dataclass
class _ReconciliationPass:
    reconciliation_id: int
    target: int
    barrier_cutoff: int
    overflow_generation: int
    config_revision: int
    catalog_revision: int
    keys: tuple[str, ...]
    rewrite_epoch: int = 0
    rewrite_pending: bool = False
    confirmed: bool = True
    parked: bool = False
    known_keys: set[str] = field(default_factory=set)
    replay_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.known_keys = set(self.keys)
