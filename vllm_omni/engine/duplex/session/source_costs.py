# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-supplied token costs keyed by public source revisions and media view."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from vllm_omni.engine.duplex.session.context_ledger import DuplexSourceRef


@dataclass(frozen=True)
class SourceTokenCost:
    sources: tuple[DuplexSourceRef, ...]
    view: Hashable
    tokens: int


@dataclass
class SourceTokenCache:
    """One scalar cost per live source group; never retains text or media.

    The model supplies a view key that changes when a source is rendered as
    original media, transcript, summary reference or omission. Repartitioned,
    revised and retired groups are removed by the owner on reconciliation.
    """

    _costs: dict[int, SourceTokenCost] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._costs)

    def get(self, sources: tuple[DuplexSourceRef, ...], view: Hashable) -> int | None:
        entry = self._costs.get(sources[0].source_id) if sources else None
        return entry.tokens if entry is not None and entry.sources == sources and entry.view == view else None

    def put(self, sources: tuple[DuplexSourceRef, ...], view: Hashable, tokens: int) -> None:
        if sources:
            self._costs[sources[0].source_id] = SourceTokenCost(sources, view, tokens)

    def retain(self, groups: Iterable[tuple[DuplexSourceRef, ...]]) -> None:
        live = {sources[0].source_id: sources for sources in groups if sources}
        self._costs = {key: entry for key, entry in self._costs.items() if live.get(key) == entry.sources}
