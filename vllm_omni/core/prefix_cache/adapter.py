# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Translation boundary between vLLM runner state and the omni prefix cache.

The adapter is the only prefix-cache component that interprets scheduler
objects.  Its outputs are immutable value objects so the manager never needs
to retain or inspect upstream scheduler state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


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
    num_output_tokens: int = 0


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


@dataclass(frozen=True, slots=True)
class PrefixCacheWriteLayout:
    writes: tuple[PrefixCacheWrite, ...]
    total_rows: int
    slots_cpu: torch.Tensor | None = None


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
        return PrefixCacheSchedulerAdapter._blocks_value(getattr(data, "block_ids", None))

    @staticmethod
    def _blocks_value(blocks: Any) -> tuple[tuple[int, ...], ...]:
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
        finished = set(scheduler_output.finished_req_ids)
        terminal_ids = {str(req_id) for req_id in finished | aborted}

        cached_by_id: dict[str, tuple[int, Any, int]] = {}
        if cached is not None:
            req_ids = tuple(getattr(cached, "req_ids", ()) or ())
            computed = tuple(getattr(cached, "num_computed_tokens", ()) or ())
            new_blocks = tuple(getattr(cached, "new_block_ids", ()) or ())
            output_tokens = tuple(getattr(cached, "num_output_tokens", ()) or ())
            for index, req_id in enumerate(req_ids):
                cached_by_id[str(req_id)] = (
                    int(computed[index]) if index < len(computed) else 0,
                    new_blocks[index] if index < len(new_blocks) else None,
                    int(output_tokens[index]) if index < len(output_tokens) else 0,
                )

        for req_id in finished | aborted:
            self._observed_req_ids.discard(str(req_id))

        for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._req_id(data)
            kind: PrefixCacheEventKind = (
                PrefixCacheEventKind.STARTED
                if req_id in terminal_ids or req_id not in self._observed_req_ids
                else PrefixCacheEventKind.EXTENDED
            )
            self._observed_req_ids.add(req_id)
            computed_tokens = int(getattr(data, "num_computed_tokens", 0) or 0)
            blocks = self._blocks(data)
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    kind,
                    0,
                    computed_tokens,
                    blocks,
                    int(scheduled_tokens.get(req_id, 0)),
                )
            )

        for req_id in resumed:
            req_id = str(req_id)
            self._observed_req_ids.add(req_id)
            hit_end, resumed_blocks, num_output_tokens = cached_by_id.get(req_id, (0, None, 0))
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED,
                    hit_end=hit_end,
                    block_ids=self._blocks_value(resumed_blocks),
                    scheduled_tokens=int(scheduled_tokens.get(req_id, 0)),
                    num_output_tokens=num_output_tokens,
                )
            )

        for req_id in sorted(finished | aborted):
            req_id = str(req_id)
            kind = PrefixCacheEventKind.ABORTED if req_id in aborted else PrefixCacheEventKind.FINISHED
            events.append(PrefixCacheRequestEvent(req_id, kind))
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
        for req_id in req_order:
            start, end = offsets[req_id]
            writes.append(PrefixCacheWrite(req_id, start, end))
        return PrefixCacheWriteLayout(tuple(writes), cursor, slots)
