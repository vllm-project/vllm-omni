# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Translation boundary between vLLM runner state and the omni prefix cache.

The adapter is the only prefix-cache component that interprets scheduler
objects and owns started-request observations. Its outputs are immutable
value objects so the manager never needs to retain upstream scheduler state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError

if TYPE_CHECKING:
    import torch


class PrefixCacheEventKind(str, Enum):
    STARTED = "started"
    EXTENDED = "extended"
    RESUMED = "resumed"
    FINISHED = "finished"
    ABORTED = "aborted"
    # Reserved for the content-identity work in item 5. This adapter does
    # not infer replacement from a reset or a prompt mutation.
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class PrefixCacheRequestEvent:
    req_id: str
    kind: PrefixCacheEventKind
    # Current supported hits are prefixes [0, hit_end). Arbitrary ranges
    # and resume delivery watermarks belong to the follow-up items.
    hit_end: int = 0
    block_ids: tuple[tuple[int, ...], ...] = ()
    scheduled_tokens: int = 0
    num_output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PrefixCacheStep:
    events: tuple[PrefixCacheRequestEvent, ...]
    scheduled_tokens: tuple[tuple[str, int], ...]
    # EXTENDED has no payload beyond counts already captured above. Avoid
    # allocating one event per running request on every decode step.
    extended_req_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrefixCacheWrite:
    req_id: str
    row_start: int
    row_end: int


@dataclass(frozen=True, slots=True)
class PrefixCacheWriteLayout:
    writes: tuple[PrefixCacheWrite, ...]
    total_rows: int
    # A fresh CPU tensor produced by the group view, shared by row-range
    # writes rather than converted to Python integers and rebuilt.
    slots_cpu: torch.Tensor | None = None


class PrefixCacheSchedulerAdapter:
    """Translate scheduler output and post-update batch state.

    Terminal events precede arrivals so same-step request-ID reuse retires
    the old request before starting the new one. EXTENDED covers ordinary
    decode, chunked prefill and a live ID re-entering scheduled_new_reqs.

    ``aborted_req_ids`` is deliberately optional: current vLLM/Omni scheduler
    outputs do not carry an explicit abort side channel yet. Finished IDs are
    never guessed to be aborted; ABORTED requires an explicit producer.
    """

    def __init__(self) -> None:
        self._observed_req_ids: set[str] = set()

    @staticmethod
    def _req_id(data: Any) -> str:
        value = getattr(data, "req_id", None)
        if value is None:
            value = getattr(data, "request_id", None)
        if value is None:
            raise OmniPrefixCacheUnmatchError("scheduled request data has no req_id/request_id")
        return str(value)

    @staticmethod
    def _blocks_value(blocks: Any) -> tuple[tuple[int, ...], ...]:
        if blocks is None:
            return ()
        if blocks and isinstance(blocks[0], int):
            return (tuple(int(block) for block in blocks),)
        return tuple(tuple(int(block) for block in group) for group in blocks)

    def translate_step(self, scheduler_output: Any) -> PrefixCacheStep:
        events: list[PrefixCacheRequestEvent] = []
        extended: list[str] = []
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = {str(req_id) for req_id in (getattr(cached, "resumed_req_ids", ()) or ())}
        finished = {str(req_id) for req_id in (getattr(scheduler_output, "finished_req_ids", ()) or ())}
        aborted = {str(req_id) for req_id in (getattr(scheduler_output, "aborted_req_ids", ()) or ())}
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        terminal_ids = finished | aborted

        self._observed_req_ids.difference_update(terminal_ids)
        for req_id in sorted(terminal_ids):
            kind = PrefixCacheEventKind.ABORTED if req_id in aborted else PrefixCacheEventKind.FINISHED
            events.append(PrefixCacheRequestEvent(req_id, kind))

        emitted: set[str] = set()
        for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._req_id(data)
            if req_id in emitted:
                continue
            emitted.add(req_id)
            if req_id in self._observed_req_ids:
                extended.append(req_id)
                continue
            self._observed_req_ids.add(req_id)
            computed_tokens = int(getattr(data, "num_computed_tokens", 0) or 0)
            blocks = self._blocks_value(getattr(data, "block_ids", None)) if computed_tokens > 0 else ()
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.STARTED,
                    hit_end=computed_tokens,
                    block_ids=blocks,
                    scheduled_tokens=int(scheduled_tokens.get(req_id, 0)),
                )
            )

        # Only resumed requests need a cached-request payload snapshot.
        # Iterate req_ids (scheduler order), not the resumed membership set.
        req_ids = (getattr(cached, "req_ids", ()) or ()) if resumed else ()
        computed = getattr(cached, "num_computed_tokens", ()) or ()
        new_blocks = getattr(cached, "new_block_ids", ()) or ()
        output_tokens = getattr(cached, "num_output_tokens", ()) or ()
        for index, raw_req_id in enumerate(req_ids):
            req_id = str(raw_req_id)
            if req_id not in resumed or req_id in emitted:
                continue
            self._observed_req_ids.add(req_id)
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED,
                    hit_end=int(computed[index]) if index < len(computed) else 0,
                    block_ids=self._blocks_value(new_blocks[index]) if index < len(new_blocks) else (),
                    scheduled_tokens=int(scheduled_tokens.get(req_id, 0)),
                    num_output_tokens=int(output_tokens[index]) if index < len(output_tokens) else 0,
                )
            )
            emitted.add(req_id)

        # Missing payloads are tolerated for explicit resume signals, as
        # before; their order is deterministic even for a partial test stub.
        for req_id in sorted(resumed - emitted):
            self._observed_req_ids.add(req_id)
            events.append(
                PrefixCacheRequestEvent(
                    req_id, PrefixCacheEventKind.RESUMED, scheduled_tokens=int(scheduled_tokens.get(req_id, 0))
                )
            )
            emitted.add(req_id)

        counts: list[tuple[str, int]] = []
        for raw_req_id, raw_count in scheduled_tokens.items():
            req_id, count = str(raw_req_id), int(raw_count)
            if count < 0:
                raise OmniPrefixCacheUnmatchError(f"negative scheduled token count for req {req_id}: {count}")
            counts.append((req_id, count))
            if count > 0 and req_id in self._observed_req_ids and req_id not in emitted:
                extended.append(req_id)
        return PrefixCacheStep(tuple(events), tuple(counts), tuple(extended))

    def build_write_layout(
        self,
        group_view: Any,
        *,
        num_scheduled_tokens: Mapping[str, int],
    ) -> PrefixCacheWriteLayout:
        req_order = tuple(group_view.batch_req_ids())
        writes: list[PrefixCacheWrite] = []
        cursor = 0
        for req_id in req_order:
            count = int(num_scheduled_tokens.get(req_id, 0))
            if count < 0:
                raise OmniPrefixCacheUnmatchError(f"negative scheduled token count for req {req_id}: {count}")
            writes.append(PrefixCacheWrite(req_id, cursor, cursor + count))
            cursor += count
        slots = group_view.step_slots_cpu(list(req_order), dict(num_scheduled_tokens))
        return PrefixCacheWriteLayout(tuple(writes), cursor, slots)
