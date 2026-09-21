"""Translation boundary between vLLM runner state and the omni prefix cache.

The adapter is the only prefix-cache component that interprets scheduler
objects.  Its outputs are immutable value objects so the manager never needs
to retain or inspect upstream scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PrefixCacheEventKind(str, Enum):
    STARTED = "started"
    EXTENDED = "extended"
    RESUMED = "resumed"
    FINISHED = "finished"
    ABORTED = "aborted"
    # Reserved for the content-identity work in item 5.  This adapter does
    # not infer replacement from a reset or a prompt mutation.
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class PrefixCacheRequestEvent:
    req_id: str
    kind: PrefixCacheEventKind
    hit_start: int = 0
    hit_end: int = 0
    block_ids: tuple[tuple[int, ...], ...] = ()
    scheduled_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PrefixCacheStep:
    events: tuple[PrefixCacheRequestEvent, ...]
    scheduled_tokens: tuple[tuple[str, int], ...]

    def __iter__(self):
        return iter(self.events)


@dataclass(frozen=True, slots=True)
class PrefixCacheWrite:
    req_id: str
    row_start: int
    row_end: int
    slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixCacheWriteLayout:
    writes: tuple[PrefixCacheWrite, ...]
    total_rows: int


class PrefixCacheSchedulerAdapter:
    """Translate scheduler output and post-update batch state.

    ``aborted_req_ids`` is deliberately optional: current main does not carry
    an explicit abort side channel, and finished IDs must never be guessed to
    be aborted.
    """

    def __init__(self) -> None:
        self._observed_req_ids: set[str] = set()

    @staticmethod
    def _req_id(data: Any) -> str:
        value = getattr(data, "req_id", None)
        if value is None:
            value = getattr(data, "request_id", None)
        if value is None:
            raise AttributeError("scheduled request data has no req_id/request_id")
        return str(value)

    @staticmethod
    def _blocks(data: Any) -> tuple[tuple[int, ...], ...]:
        blocks = getattr(data, "block_ids", None)
        if blocks is None:
            return ()
        if blocks and isinstance(blocks[0], int):
            return (tuple(int(block) for block in blocks),)
        return tuple(tuple(int(block) for block in group) for group in blocks)

    def translate_scheduler_output(self, scheduler_output: Any) -> tuple[PrefixCacheRequestEvent, ...]:
        events: list[PrefixCacheRequestEvent] = []
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = set(getattr(cached, "resumed_req_ids", ()) or ()) if cached is not None else set()
        aborted = set(getattr(scheduler_output, "aborted_req_ids", ()) or ())
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

        for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._req_id(data)
            kind = PrefixCacheEventKind.EXTENDED if req_id in self._observed_req_ids else PrefixCacheEventKind.STARTED
            self._observed_req_ids.add(req_id)
            computed = int(getattr(data, "num_computed_tokens", 0) or 0)
            blocks = self._blocks(data)
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    kind,
                    0,
                    computed,
                    blocks,
                    int(scheduled_tokens.get(req_id, 0)),
                )
            )

        for req_id in resumed:
            req_id = str(req_id)
            self._observed_req_ids.add(req_id)
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED,
                    scheduled_tokens=int(scheduled_tokens.get(req_id, 0)),
                )
            )

        finished = set(getattr(scheduler_output, "finished_req_ids", ()) or ())
        for req_id in sorted(finished | aborted):
            req_id = str(req_id)
            kind = PrefixCacheEventKind.ABORTED if req_id in aborted else PrefixCacheEventKind.FINISHED
            events.append(PrefixCacheRequestEvent(req_id, kind))
            self._observed_req_ids.discard(req_id)
        return tuple(events)

    def translate_step(self, scheduler_output: Any) -> PrefixCacheStep:
        events = self.translate_scheduler_output(scheduler_output)
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        return PrefixCacheStep(events, tuple((str(req_id), int(count)) for req_id, count in scheduled.items()))

    def build_write_layout(
        self,
        group_view: Any,
        *,
        num_scheduled_tokens: Mapping[str, int],
    ) -> PrefixCacheWriteLayout:
        req_order = tuple(group_view.batch_req_ids())
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for req_id in req_order:
            count = max(0, int(num_scheduled_tokens.get(req_id, 0)))
            offsets[req_id] = (cursor, cursor + count)
            cursor += count
        slots = group_view.step_slots_cpu(list(req_order), dict(num_scheduled_tokens))
        writes: list[PrefixCacheWrite] = []
        cursor = 0
        for req_id in req_order:
            start, end = offsets[req_id]
            count = end - start
            writes.append(
                PrefixCacheWrite(
                    req_id,
                    start,
                    end,
                    tuple(int(x) for x in slots[cursor : cursor + count].tolist()),
                )
            )
            cursor += count
        return PrefixCacheWriteLayout(tuple(writes), cursor)
