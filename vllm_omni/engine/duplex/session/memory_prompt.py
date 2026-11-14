# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Owner-only receipts for memory selected during asynchronous prompt work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.session.memory_context import omitted_user_sources_covered
from vllm_omni.engine.duplex.session.memory_retrieval import DuplexRetrievedCard, retrieve_memory_cards

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
    from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot


@dataclass(frozen=True)
class DuplexMemoryPromptCapture:
    """Retained by the owner; only detached cards go to a formatter."""

    scope: object = field(repr=False)
    epoch: int
    config_revision: int
    suppression_version: int
    cards: tuple[DuplexRetrievedCard, ...] = field(repr=False)


class DuplexMemoryPromptState:
    def __init__(self) -> None:
        self._scope = object()

    @staticmethod
    def _eligible(session: DuplexEngineSession, snapshot: DuplexHistorySnapshot) -> bool:
        return (
            session.state is DuplexSessionState.OPEN
            and session.lease.terminal_reason is None
            and not session.memory_suppression.reconciliation_required(snapshot)
            and not (session._memory_context is not None and session._memory_context.retrieval_blocked)
        )

    def capture(self, session: DuplexEngineSession, *, query: str, max_candidates: int) -> DuplexMemoryPromptCapture:
        if not isinstance(query, str):
            raise ValueError("invalid_memory_prompt_query")
        snapshot = session.history_snapshot()
        suppression = session.memory_suppression
        cards = retrieve_memory_cards(
            snapshot=snapshot,
            ledger=session.context_ledger,
            query=query,
            suppressed_keys=suppression.suppressed_keys,
            suppress_all=suppression.suppress_all or not self._eligible(session, snapshot),
            max_candidates=max_candidates,
        )
        return DuplexMemoryPromptCapture(
            self._scope, session.epoch, session.config_revision, suppression.version, cards
        )

    def matches(
        self,
        session: DuplexEngineSession,
        capture: DuplexMemoryPromptCapture,
        node_ids: tuple[int, ...],
        *,
        omitted_source_ids: tuple[int, ...] = (),
    ) -> bool:
        if (
            not isinstance(capture, DuplexMemoryPromptCapture)
            or capture.scope is not self._scope
            or capture.epoch != session.epoch
            or capture.config_revision != session.config_revision
            or capture.suppression_version != session.memory_suppression.version
            or not isinstance(node_ids, tuple)
            or len(node_ids) > 16
            or any(type(node_id) is not int or node_id < 0 for node_id in node_ids)
            or len(set(node_ids)) != len(node_ids)
        ):
            return False
        snapshot = session.history_snapshot()
        if (
            not isinstance(omitted_source_ids, tuple)
            or len(omitted_source_ids) > len(snapshot.source_ids)
            or any(type(source_id) is not int or source_id < 0 for source_id in omitted_source_ids)
            or len(set(omitted_source_ids)) != len(omitted_source_ids)
            or not set(omitted_source_ids).issubset(snapshot.source_ids)
        ):
            return False
        if node_ids and not omitted_user_sources_covered(session, omitted_source_ids):
            return False
        if not self._eligible(session, snapshot):
            return False
        session.context_ledger.refresh(snapshot)
        candidates = {card.node_id: card for card in capture.cards}
        current = {node.node_id: node for node in session.context_ledger.card_candidates()}
        for node_id in node_ids:
            card, node = candidates.get(node_id), current.get(node_id)
            if (
                card is None
                or node is None
                or node.card is None
                or card.fact_key in session.memory_suppression.suppressed_keys
                or session.memory_suppression.suppress_all
                or (card.fact_key, card.generation, card.sources, card.text)
                != (node.card.fact_key, node.card.generation, node.sources, node.message.get("content"))
            ):
                return False
        return True
