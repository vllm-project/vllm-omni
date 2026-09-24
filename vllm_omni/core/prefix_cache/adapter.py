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
        return PrefixCacheSchedulerAdapter._blocks_value(getattr(data, "block_ids", None))

    @staticmethod
    def _blocks_value(blocks: Any) -> tuple[tuple[int, ...], ...]:
        if blocks is None:
            return ()
        if not isinstance(blocks, (list, tuple)):
            raise TypeError(f"block_ids must be a sequence, got {type(blocks).__name__}")
        if blocks and isinstance(blocks[0], int):
            return (tuple(int(block) for block in blocks),)
        groups = []
        for group in blocks:
            if not isinstance(group, (list, tuple)):
                raise TypeError("block_ids must be a flat sequence or a sequence of groups")
            groups.append(tuple(int(block) for block in group))
        return tuple(groups)

    @staticmethod
    def _cached_field(cached: Any, name: str, req_ids: tuple[Any, ...]) -> tuple[Any, ...]:
        """Read one parallel cached-request field with an early contract check."""
        value = tuple(getattr(cached, name, ()) or ())
        # Empty optional fields are valid in synthetic/older scheduler
        # outputs. Once a field is present, however, it must describe every
        # cached request; silently defaulting a missing row changes ownership.
        if value and len(value) != len(req_ids):
            raise ValueError(
                f"scheduled_cached_reqs.{name} has {len(value)} entries for {len(req_ids)} req_ids"
            )
        return value

    def translate_scheduler_output(self, scheduler_output: Any) -> tuple[PrefixCacheRequestEvent, ...]:
        events: list[PrefixCacheRequestEvent] = []
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = {str(req_id) for req_id in (getattr(cached, "resumed_req_ids", ()) or ())} if cached is not None else set()
        aborted = set(getattr(scheduler_output, "aborted_req_ids", ()) or ())
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        terminal_ids = {
            str(req_id)
            for req_id in (set(getattr(scheduler_output, "finished_req_ids", ()) or ()) | aborted)
        }

        cached_by_id: dict[str, tuple[int, Any, int]] = {}
        if cached is not None:
            req_ids = tuple(getattr(cached, "req_ids", ()) or ())
            computed = self._cached_field(cached, "num_computed_tokens", req_ids)
            new_blocks = self._cached_field(cached, "new_block_ids", req_ids)
            output_tokens = self._cached_field(cached, "num_output_tokens", req_ids)
            for index, req_id in enumerate(req_ids):
                cached_by_id[str(req_id)] = (
                    int(computed[index]) if computed else 0,
                    new_blocks[index] if new_blocks else None,
                    int(output_tokens[index]) if output_tokens else 0,
                )

        missing_resumed = resumed - set(cached_by_id)
        if missing_resumed:
            raise ValueError(
                "scheduled_cached_reqs.resumed_req_ids missing from req_ids: "
                f"{sorted(str(req_id) for req_id in missing_resumed)}"
            )

        for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._req_id(data)
            kind = (
                PrefixCacheEventKind.STARTED
                if req_id in terminal_ids or req_id not in self._observed_req_ids
                else PrefixCacheEventKind.EXTENDED
            )
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
            self._observed_req_ids.add(req_id)
            hit_end, block_ids, num_output_tokens = cached_by_id.get(req_id, (0, None, 0))
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED,
                    hit_end=hit_end,
                    block_ids=self._blocks_value(block_ids),
                    scheduled_tokens=int(scheduled_tokens.get(req_id, 0)),
                    num_output_tokens=num_output_tokens,
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
        expected_rows = cursor
        actual_rows = int(slots.numel())
        if actual_rows != expected_rows:
            raise ValueError(
                f"write layout slot count {actual_rows} does not match scheduled row count {expected_rows}"
            )
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
