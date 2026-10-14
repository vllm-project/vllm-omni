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
    # Replacement is explicit; never infer it from a reset or prompt mutation.
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True, order=True)
class PrefixCacheRequestOwner:
    """Admission and content generation; distinct from a reusable cache key."""

    admission_id: int
    generation: int = 0


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
    owner: PrefixCacheRequestOwner | None = None
    lookup_complete: bool = True


@dataclass(frozen=True, slots=True)
class PrefixCacheStep:
    events: tuple[PrefixCacheRequestEvent, ...]
    scheduled_tokens: tuple[tuple[str, int], ...]
    # EXTENDED has no payload beyond counts already captured above. Avoid
    # allocating one event per running request on every decode step.
    extended_req_ids: tuple[str, ...] = ()
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class PrefixCacheWrite:
    req_id: str
    row_start: int
    row_end: int
    # Item 4 delivery ranges are captured after batch reordering, not inferred
    # from packed rows or read again by the asynchronous output builder.
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class PrefixCacheWriteLayout:
    writes: tuple[PrefixCacheWrite, ...]
    total_rows: int
    # A fresh CPU tensor produced by the group view, shared by row-range
    # writes rather than converted to Python integers and rebuilt.
    slots_cpu: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _RequestObservation:
    owner: PrefixCacheRequestOwner | None = None
    # A control-only replacement must still emit its first cached/new lookup.
    lookup_pending: bool = False


class PrefixCacheSchedulerAdapter:
    """Translate scheduler output and post-update batch state.

    Terminal events precede arrivals so same-step request-ID reuse retires
    the old request before starting the new one. EXTENDED covers ordinary
    decode, chunked prefill and a live ID re-entering scheduled_new_reqs.

    ABORTED requires an explicit optional cancellation field:
    ``discarded_req_ids`` (the producer proposed in PR #7681) or
    ``aborted_req_ids``. Neither is produced by the current scheduler
    contract; until a producer lands, production terminals remain FINISHED.
    These fields are accepted for forward compatibility, not inferred.
    """

    def __init__(self) -> None:
        self._observed: dict[str, _RequestObservation] = {}
        self._last_step_sequence = -1

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
        sequence = getattr(scheduler_output, "prefix_cache_step_sequence", None)
        if sequence is not None:
            if sequence <= self._last_step_sequence:
                return PrefixCacheStep((), (), sequence=sequence)
            self._last_step_sequence = sequence
        events: list[PrefixCacheRequestEvent] = []
        extended: list[str] = []
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = {str(req_id) for req_id in (getattr(cached, "resumed_req_ids", ()) or ())}
        finished = {str(req_id) for req_id in (getattr(scheduler_output, "finished_req_ids", ()) or ())}
        aborted = {str(req_id) for req_id in (getattr(scheduler_output, "aborted_req_ids", ()) or ())}
        aborted.update(str(req_id) for req_id in (getattr(scheduler_output, "discarded_req_ids", ()) or ()))
        owners = getattr(scheduler_output, "prefix_cache_owners", {}) or {}
        terminal_owners = getattr(scheduler_output, "prefix_cache_terminal_owners", {}) or {}
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        terminal_ids = finished | aborted

        for req_id in sorted(terminal_ids):
            terminal_observation = self._observed.get(req_id, _RequestObservation())
            current_owner = terminal_observation.owner
            terminal_owner = terminal_owners.get(req_id, current_owner)
            if current_owner is None or terminal_owner is None or terminal_owner >= current_owner:
                self._observed.pop(req_id, None)
            kind = PrefixCacheEventKind.ABORTED if req_id in aborted else PrefixCacheEventKind.FINISHED
            events.append(PrefixCacheRequestEvent(req_id, kind, owner=terminal_owner))

        # A control-only replacement fences old work before the new lookup.
        for event in getattr(scheduler_output, "prefix_cache_replacements", ()) or ():
            if event.kind is not PrefixCacheEventKind.REPLACED or event.owner is None:
                raise ValueError("prefix-cache replacement control requires REPLACED and an owner")
            current_owner = self._observed.get(event.req_id, _RequestObservation()).owner
            if current_owner is not None and event.owner < current_owner:
                continue
            self._observed[event.req_id] = _RequestObservation(event.owner, not event.lookup_complete)
            events.append(event)

        emitted: set[str] = set()
        for data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._req_id(data)
            if req_id in emitted:
                continue
            emitted.add(req_id)
            owner = owners.get(req_id)
            observed = self._observed.get(req_id)
            current_owner = observed.owner if observed is not None else None
            if owner is not None and current_owner is not None and owner < current_owner:
                continue
            kind = PrefixCacheEventKind.STARTED
            if observed is not None and (owner is None or owner == current_owner):
                if not observed.lookup_pending:
                    extended.append(req_id)
                    continue
                kind = PrefixCacheEventKind.EXTENDED
            self._observed[req_id] = _RequestObservation(owner if owner is not None else current_owner)
            computed_tokens = int(getattr(data, "num_computed_tokens", 0) or 0)
            blocks = self._blocks_value(getattr(data, "block_ids", None)) if computed_tokens > 0 else ()
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    kind,
                    hit_end=computed_tokens,
                    block_ids=blocks,
                    scheduled_tokens=max(0, int(scheduled_tokens.get(req_id, 0))),
                    owner=owner,
                )
            )

        # Ordinary decode only needs counts. Controls awaiting their first
        # lookup and explicit resumes additionally need the cached payload.
        req_ids = getattr(cached, "req_ids", ()) or ()
        computed = getattr(cached, "num_computed_tokens", ()) or ()
        new_blocks = getattr(cached, "new_block_ids", ()) or ()
        output_tokens = getattr(cached, "num_output_tokens", ()) or ()
        for index, raw_req_id in enumerate(req_ids):
            req_id = str(raw_req_id)
            observed = self._observed.get(req_id)
            if req_id in emitted or (req_id not in resumed and (observed is None or not observed.lookup_pending)):
                continue
            owner = owners.get(req_id)
            current_owner = observed.owner if observed is not None else None
            if owner is not None and current_owner is not None and owner < current_owner:
                emitted.add(req_id)
                continue
            self._observed[req_id] = _RequestObservation(owner if owner is not None else current_owner)
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED if req_id in resumed else PrefixCacheEventKind.EXTENDED,
                    hit_end=int(computed[index]) if index < len(computed) else 0,
                    block_ids=self._blocks_value(new_blocks[index]) if index < len(new_blocks) else (),
                    scheduled_tokens=max(0, int(scheduled_tokens.get(req_id, 0))),
                    num_output_tokens=int(output_tokens[index]) if index < len(output_tokens) else 0,
                    owner=owner,
                )
            )
            emitted.add(req_id)

        # Missing payloads are tolerated for explicit resume signals, as
        # before; their order is deterministic even for a partial test stub.
        for req_id in sorted(resumed - emitted):
            self._observed[req_id] = _RequestObservation(
                owners.get(req_id, self._observed.get(req_id, _RequestObservation()).owner)
            )
            events.append(
                PrefixCacheRequestEvent(
                    req_id,
                    PrefixCacheEventKind.RESUMED,
                    scheduled_tokens=max(0, int(scheduled_tokens.get(req_id, 0))),
                    owner=owners.get(req_id),
                )
            )
            emitted.add(req_id)

        counts: list[tuple[str, int]] = []
        for raw_req_id, raw_count in scheduled_tokens.items():
            req_id, count = str(raw_req_id), int(raw_count)
            if count < 0:
                if any(e.req_id == req_id and e.kind is PrefixCacheEventKind.STARTED and e.hit_end > 0 for e in events):
                    raise OmniPrefixCacheUnmatchError(f"negative scheduled token count for req {req_id}: {count}")
                # #5269: ordinary non-positive spans are no-ops. A STARTED
                # hit cannot use a negative count to size its prefetch buffer.
                count = 0
            counts.append((req_id, count))
            if count > 0 and req_id in self._observed and req_id not in emitted:
                extended.append(req_id)
        return PrefixCacheStep(tuple(events), tuple(counts), tuple(extended), sequence)

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
            start, end = group_view.token_range(req_id, count)
            writes.append(PrefixCacheWrite(req_id, cursor, cursor + count, start, end))
            cursor += count
        slots = group_view.step_slots_cpu(list(req_order), dict(num_scheduled_tokens))
        return PrefixCacheWriteLayout(tuple(writes), cursor, slots)
