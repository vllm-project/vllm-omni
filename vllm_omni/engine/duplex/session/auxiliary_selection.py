# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Two bounded work identities for the session's single auxiliary slot."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Literal

DuplexAuxiliaryKind = Literal["summary", "card"]
_KINDS: tuple[DuplexAuxiliaryKind, ...] = ("summary", "card")


@dataclass
class _WorkState:
    source: Hashable
    failures: int = 0
    exhausted: bool = False
    parked: bool = False
    ready_at: float = 0.0


class DuplexAuxiliarySelector:
    """Alternate eligible kinds without forgetting a temporarily idle failure."""

    def __init__(self) -> None:
        self._states: dict[DuplexAuxiliaryKind, _WorkState] = {}
        self._next: DuplexAuxiliaryKind = "summary"

    def choose(self, sources: Mapping[DuplexAuxiliaryKind, Hashable], *, now: float) -> DuplexAuxiliaryKind | None:
        for kind in _KINDS:
            if kind not in sources:
                continue
            source = sources[kind]
            if kind not in self._states or self._states[kind].source != source:
                self._states[kind] = _WorkState(source)
        order: tuple[DuplexAuxiliaryKind, ...] = (self._next, "card" if self._next == "summary" else "summary")
        for kind in order:
            if kind not in sources:
                continue
            state = self._states[kind]
            if state.parked or state.exhausted or state.ready_at > now:
                continue
            state.parked = True
            self._next = "card" if kind == "summary" else "summary"
            return kind
        return None

    def release(self, kind: DuplexAuxiliaryKind, source: Hashable, *, prefer: bool = False) -> None:
        state = self._states.get(kind)
        if state is not None and state.source == source and not state.exhausted:
            state.parked = False
            if prefer:
                self._next = kind

    def succeeded(self, kind: DuplexAuxiliaryKind, source: Hashable) -> None:
        state = self._states[kind]
        if state.source == source:
            state.failures, state.exhausted, state.parked = 0, False, True
            state.ready_at = 0.0

    def failed(
        self, kind: DuplexAuxiliaryKind, source: Hashable, *, delays: tuple[float, ...], now: float
    ) -> tuple[int, bool]:
        state = self._states[kind]
        if state.source == source:
            state.failures = min(state.failures + 1, len(delays) + 1)
            state.exhausted = state.failures > len(delays)
            state.parked = state.exhausted
            state.ready_at = 0.0 if state.exhausted else now + delays[state.failures - 1]
        return state.failures, state.exhausted

    def deadline(self, sources: Mapping[DuplexAuxiliaryKind, Hashable], *, now: float) -> float | None:
        return min(
            (
                state.ready_at
                for kind, state in self._states.items()
                if kind in sources
                and sources[kind] == state.source
                and not state.parked
                and not state.exhausted
                and state.ready_at > now
            ),
            default=None,
        )

    def clear(self) -> None:
        self._states.clear()
