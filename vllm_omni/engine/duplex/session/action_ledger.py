# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session capture ledger: assembles validated unit payloads into sealed
``DuplexActionRecordSet`` s, one per scope (response or listening period).

Validation failure is session-fatal; sealed sets spill into the manager's terminal
cache; close/interrupt/
cancel seal partial sets. Mutated only on the orchestrator loop — no locking.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from vllm_omni.protocol.duplex.action_records import (
    ACTION_CAPTURE_RECORD_VERSION,
    ACTION_SAMPLED,
    INPUT_ORIGINS,
    ORIGIN_ENGINE,
    STATUS_COMPLETED,
    STATUS_TRUNCATED_CANCEL,
    STATUS_TRUNCATED_CLOSE,
    STATUS_TRUNCATED_INTERRUPTION,
    DuplexActionRecordSet,
    DuplexActionStream,
    decode_unit_payload,
)

# Retention window: the open scope plus this many sealed, uncollected sets.
RETAINED_SEALED_SETS = 1

CODE_INVALID = "action_capture_invalid"
CODE_RETENTION_OVERFLOW = "action_capture_retention_overflow"
CODE_NOT_RETAINED = "action_capture_not_retained"


class ActionLedgerError(ValueError):
    """A capture-ledger invariant was violated (session-fatal unless noted)."""

    def __init__(self, message: str, *, code: str = CODE_INVALID) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class _UnitRow:
    stage_id: int
    seq: int
    unit_index: int
    offset: int
    input_modality: str
    input_origin: str
    input_commit_ns: int | None
    emission_ns: int | None
    sampling: Mapping[str, Any]
    terminators: tuple[Mapping[str, Any], ...] = ()


@dataclass
class _OpenScope:
    response_id: str | None
    epoch: int
    turn_id: int
    # Flat token-level columns.
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float | None] = field(default_factory=list)
    action_kinds: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    stream_names: list[str] = field(default_factory=list)
    # Unit-level rows, one per model unit, in open order.
    units: list[_UnitRow] = field(default_factory=list)
    streams: dict[tuple[str, int, str], DuplexActionStream] = field(default_factory=dict)
    effective_seed: int | None = None
    # The submit-time response index the scope's draws used. ``None`` until a
    # payload carries one. Stragglers match this, not the seed alone.
    response_index: int | None = None
    model: dict[str, Any] = field(default_factory=dict)
    last_seq: int | None = None
    num_sampled: int = 0

    @property
    def is_listening(self) -> bool:
        return self.response_id is None


class SessionActionLedger:
    """Bounded per-session store of action-record sets."""

    def __init__(self, *, session_id: str, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._session_id = session_id
        self._clock = clock
        self._open: _OpenScope | None = None
        # scope key -> sealed set; at most RETAINED_SEALED_SETS entries.
        self._sealed: OrderedDict[str, DuplexActionRecordSet] = OrderedDict()
        # Submit-order input annotations (modality, origin, commit time) per epoch.
        # Claimed one per new unit_index. The capture seq is not the key: one
        # unit emits several payloads, so that counter runs ahead of append seqs.
        self._input_annotations: dict[int, deque[tuple[str, str, int | None]]] = {}
        # Scope keys already sealed once (ingest accounting that outlives
        # collection): a payload for one of these is invalid, never a gap.
        self._sealed_keys: set[str] = set()
        # ``(last_seq, last_unit_index, response_index)`` of each sealed scope.
        self._sealed_cursors: dict[str, tuple[int | None, int | None, int | None]] = {}

    # ---- scope keys ----

    @staticmethod
    def _scope_key(response_id: str | None, epoch: int, turn_id: int) -> str:
        return response_id if response_id is not None else f"~listen-{epoch}-{turn_id}"

    # ---- engine-side annotations (no payload involved) ----

    def note_input_unit(
        self, epoch: int, seq: int, *, modality: str, origin: str, commit_ns: int | None = None
    ) -> None:
        """Queue one append's input annotation, in submit order.

        ``seq`` is the append reservation. It is not the lookup key: a unit's
        capture seq is the model's own counter and runs ahead of these
        reservations. Unknown origins are rejected.
        """
        if origin not in INPUT_ORIGINS:
            raise ActionLedgerError(f"unknown capture input origin: {origin!r}")
        self._input_annotations.setdefault(epoch, deque()).append((str(modality), str(origin), commit_ns))

    def claim_input_unit(self, epoch: int) -> tuple[str, str, int | None]:
        """The next submit-order annotation, or engine-origin when none is waiting.

        One claim per new ``unit_index``. A model-minted continuation has no
        append behind it, so the empty queue is engine-origin.
        """
        queue = self._input_annotations.get(epoch)
        if not queue:
            return ("audio", ORIGIN_ENGINE, None)
        return queue.popleft()

    # ---- scope lifecycle ----

    def bind_response(self, response_id: str, *, epoch: int, turn_id: int) -> None:
        """Open a response scope, sealing any open scope as completed."""
        if self._open is not None and self._open.response_id == response_id:
            return
        self._seal_open(STATUS_COMPLETED)
        self._open = _OpenScope(response_id=response_id, epoch=epoch, turn_id=turn_id)

    def bind_listening(self, *, epoch: int, turn_id: int) -> None:
        """Open (or keep) the listening scope for units sampled with no active response."""
        if self._open is not None and self._open.is_listening and self._open.epoch == epoch:
            return
        self._seal_open(STATUS_COMPLETED)
        self._open = _OpenScope(response_id=None, epoch=epoch, turn_id=turn_id)

    def _require_open(self, *, epoch: int, response_id: str | None, turn_id: int) -> _OpenScope:
        if self._open is None:
            # Capture may have been turned on mid-session (``session.update``):
            # the first payload opens the scope for its response (or listening).
            if response_id is not None:
                self._open = _OpenScope(response_id=response_id, epoch=epoch, turn_id=turn_id)
            else:
                self._open = _OpenScope(response_id=None, epoch=epoch, turn_id=turn_id)
            return self._open
        if self._open.epoch != epoch:
            raise ActionLedgerError(
                f"capture payload for epoch {epoch} while scope epoch is {self._open.epoch} "
                f"(open={self._open.response_id or 'listening'})"
            )
        if response_id is not None and self._open.response_id != response_id:
            if self._open.is_listening:
                # The response binding trails the unit that started speaking;
                # promote the listening scope so ordering holds.
                self._open.response_id = response_id
                self._open.turn_id = turn_id
            else:
                raise ActionLedgerError(
                    f"capture payload for response {response_id} while scope {self._open.response_id} is open"
                )
        return self._open

    # ---- ingest ----

    def ingest(
        self,
        payload: bytes | Mapping[str, Any],
        *,
        epoch: int,
        turn_id: int,
        response_id: str | None,
        stage_id: int,
        input_modality: str = "",
        input_commit_ns: int | None = None,
    ) -> int:
        """Validate and fold one capture payload into the scope's units; returns the scope unit index.

        ``decode_unit_payload`` owns schema validation; this method owns ordering: the
        model-asserted ``seq`` is strictly monotone per scope and ``unit_index`` monotone
        non-decreasing. Consecutive payloads sharing a ``unit_index`` are one model unit's
        decode steps: the first opens the unit's row, the rest extend its columns, and the
        unit keeps its first step's input annotation and emission time.
        """
        record = decode_unit_payload(payload)
        if int(record["epoch"]) != int(epoch):
            raise ActionLedgerError(f"capture payload epoch {record['epoch']} != delivery epoch {epoch}")
        if int(record["stage_id"]) != int(stage_id):
            raise ActionLedgerError(f"capture payload stage {record['stage_id']} != delivery stage {stage_id}")
        absorbed = self._absorb_listening_straggler(
            record, epoch=int(epoch), response_id=response_id, stage_id=int(stage_id)
        )
        if absorbed is not None:
            return absorbed
        seq = int(record["seq"])
        scope = self._require_open(epoch=epoch, response_id=response_id, turn_id=turn_id)
        payload_index = record.get("response_index")
        if payload_index is not None and scope.response_index is None:
            scope.response_index = int(payload_index)
        if scope.last_seq is not None and seq <= scope.last_seq:
            raise ActionLedgerError(
                f"capture payload seq {seq} is not after {scope.last_seq} in scope {scope.response_id or 'listening'}"
            )
        unit_index = int(record["unit_index"])
        last = scope.units[-1] if scope.units else None
        if last is not None and unit_index < last.unit_index:
            raise ActionLedgerError(
                f"capture payload unit_index {unit_index} is behind the scope's unit {last.unit_index}"
            )
        scope.last_seq = seq
        offset = len(scope.token_ids)
        count = len(record["token_ids"])
        scope.token_ids.extend(int(value) for value in record["token_ids"])
        scope.logprobs.extend(record["logprobs"])
        scope.action_kinds.extend(record["action_kinds"])
        reasons = list(record.get("reason_codes") or ())[:count]
        names = list(record.get("stream_names") or ())[:count]
        reasons += [""] * (count - len(reasons))
        names += [""] * (count - len(names))
        scope.reason_codes.extend(reasons)
        scope.stream_names.extend(names)
        for stream in record.get("streams") or ():
            key = (str(stream["name"]), int(stream["stage"]), str(stream["modality"]))
            scope.streams.setdefault(key, DuplexActionStream(name=key[0], stage=key[1], modality=key[2]))
        if any(kind == ACTION_SAMPLED for kind in record["action_kinds"]):
            scope.num_sampled += 1
        if record.get("effective_seed") is not None:
            seed = int(record["effective_seed"])
            if scope.effective_seed is not None and scope.effective_seed != seed:
                raise ActionLedgerError(f"capture payload seed {seed} contradicts scope seed {scope.effective_seed}")
            scope.effective_seed = seed
        model_extension = record.get("model") or {}
        if model_extension:
            scope.model.update(model_extension)
        if last is not None and unit_index == last.unit_index:
            last.terminators += tuple(dict(term) for term in record.get("terminators") or ())
            return len(scope.units) - 1
        modality, origin, commit_ns_from_input = self.claim_input_unit(epoch)
        scope.units.append(
            _UnitRow(
                stage_id=int(stage_id),
                seq=seq,
                unit_index=unit_index,
                offset=offset,
                input_modality=str(input_modality) or modality,
                input_origin=origin,
                input_commit_ns=input_commit_ns if input_commit_ns is not None else commit_ns_from_input,
                emission_ns=self._clock(),
                sampling=dict(record.get("sampling_params") or {}),
                terminators=tuple(dict(term) for term in record.get("terminators") or ()),
            )
        )
        return len(scope.units) - 1

    def _absorb_listening_straggler(
        self,
        record: Mapping[str, Any],
        *,
        epoch: int,
        response_id: str | None,
        stage_id: int,
    ) -> int | None:
        """Fold a payload drawn under a just-sealed listening scope back into that scope.

        Per-unit generation leaves one unit in flight when the response bind
        seals listening. Its seed is the listening scope's. ``None`` means this
        payload is not that straggler. Only a sealed set still in the retention
        window can be reached.
        """
        if response_id is None or self._open is None or self._open.is_listening:
            return None
        payload_index = record.get("response_index")
        if self._open.response_id != response_id or record.get("effective_seed") is None or payload_index is None:
            return None
        seed = int(record["effective_seed"])
        payload_index = int(payload_index)
        # A payload already consistent with the open scope is not a straggler.
        if self._open.response_index == payload_index:
            return None
        seq = int(record["seq"])
        unit_index = int(record["unit_index"])
        for key in reversed(self._sealed):
            sealed = self._sealed[key]
            last_seq, last_unit, scope_index = self._sealed_cursors.get(key, (None, None, None))
            if (
                sealed.response_id is not None
                or sealed.epoch != epoch
                or sealed.effective_seed != seed
                or scope_index != payload_index
            ):
                continue
            if last_seq is None or last_unit is None or seq <= last_seq or unit_index < last_unit:
                raise ActionLedgerError(
                    f"capture payload seq {seq} unit_index {unit_index} is not a monotone continuation "
                    f"of sealed listening scope {key}"
                )
            self._sealed[key] = self._extend_sealed(
                sealed, record, epoch=epoch, seq=seq, unit_index=unit_index, stage_id=stage_id, last_unit=last_unit
            )
            self._sealed_cursors[key] = (seq, unit_index, payload_index)
            return self._sealed[key].num_units() - 1
        return None

    def _extend_sealed(
        self,
        sealed: DuplexActionRecordSet,
        record: Mapping[str, Any],
        *,
        epoch: int,
        seq: int,
        unit_index: int,
        stage_id: int,
        last_unit: int,
    ) -> DuplexActionRecordSet:
        """Append one payload's columns onto a sealed set, folding a repeated unit index."""
        count = len(record["token_ids"])
        reasons = list(record.get("reason_codes") or ())[:count]
        names = list(record.get("stream_names") or ())[:count]
        reasons += [""] * (count - len(reasons))
        names += [""] * (count - len(names))
        token_ids = sealed.token_ids + tuple(int(value) for value in record["token_ids"])
        logprobs = sealed.logprobs + tuple(record["logprobs"])
        action_kinds = sealed.action_kinds + tuple(str(value) for value in record["action_kinds"])
        reason_codes = sealed.reason_codes + tuple(reasons)
        stream_names = sealed.stream_names + tuple(names)
        terminators = tuple(dict(term) for term in record.get("terminators") or ())
        streams = {(stream.name, stream.stage, stream.modality): stream for stream in sealed.streams}
        for stream in record.get("streams") or ():
            key = (str(stream["name"]), int(stream["stage"]), str(stream["modality"]))
            streams.setdefault(key, DuplexActionStream(name=key[0], stage=key[1], modality=key[2]))
        model = dict(sealed.model)
        model.update(record.get("model") or {})
        stream_values = tuple(streams.values())
        if unit_index == last_unit:
            offsets = sealed.unit_offsets[:-1] + (sealed.unit_offsets[-1] + count,)
            unit_terminators = sealed.terminators[:-1] + (sealed.terminators[-1] + terminators,)
            return replace(
                sealed,
                token_ids=token_ids,
                logprobs=logprobs,
                action_kinds=action_kinds,
                reason_codes=reason_codes,
                stream_names=stream_names,
                unit_offsets=offsets,
                terminators=unit_terminators,
                streams=stream_values,
                model=model,
            )
        modality, origin, commit_ns = self.claim_input_unit(epoch)
        return replace(
            sealed,
            token_ids=token_ids,
            logprobs=logprobs,
            action_kinds=action_kinds,
            reason_codes=reason_codes,
            stream_names=stream_names,
            unit_offsets=sealed.unit_offsets + (len(token_ids),),
            stage_ids=sealed.stage_ids + (stage_id,),
            input_modalities=sealed.input_modalities + (modality,),
            input_origins=sealed.input_origins + (origin,),
            input_commit_ns=sealed.input_commit_ns + (commit_ns,),
            emission_ns=sealed.emission_ns + (self._clock(),),
            sampling_params=sealed.sampling_params + (dict(record.get("sampling_params") or {}),),
            terminators=sealed.terminators + (terminators,),
            streams=stream_values,
            model=model,
        )

    # ---- sealing ----

    def seal_active_response(self, response_id: str | None, status: str = STATUS_COMPLETED) -> None:
        """Seal the open scope if it is ``response_id`` (or the listening scope when None)."""
        if self._open is not None and self._open.response_id == response_id:
            self._seal_open(status)

    def seal_for_close(self, status: str = STATUS_TRUNCATED_CLOSE) -> None:
        """Seal the open scope on close; bypasses the retention guard (the
        terminal cache drains everything, so no overflow is reported)."""
        self._seal_open(status, force=True)

    def seal_for_interruption(self) -> None:
        """Seal the open scope when a barge-in advances the epoch."""
        self._seal_open(STATUS_TRUNCATED_INTERRUPTION)

    def seal_for_cancel(self) -> None:
        """Seal the open scope when the client cancels the response."""
        self._seal_open(STATUS_TRUNCATED_CANCEL)

    def _seal_open(self, status: str, *, force: bool = False) -> None:
        scope = self._open
        if scope is None:
            return
        self._open = None
        key = self._scope_key(scope.response_id, scope.epoch, scope.turn_id)
        if key in self._sealed_keys:
            raise ActionLedgerError(f"capture scope {key} was sealed twice")
        if not force and len(self._sealed) >= RETAINED_SEALED_SETS:
            held = next(iter(self._sealed))
            raise ActionLedgerError(
                f"action records for {held} were not spilled before the next scope completed",
                code=CODE_RETENTION_OVERFLOW,
            )
        self._sealed[key] = self._freeze(scope, status)
        last = scope.units[-1] if scope.units else None
        self._sealed_cursors[key] = (
            scope.last_seq,
            None if last is None else last.unit_index,
            scope.response_index,
        )
        self._sealed_keys.add(key)

    def _freeze(self, scope: _OpenScope, status: str) -> DuplexActionRecordSet:
        return DuplexActionRecordSet(
            session_id=self._session_id,
            response_id=scope.response_id,
            epoch=scope.epoch,
            turn_id=scope.turn_id,
            status=status,
            schema_version=ACTION_CAPTURE_RECORD_VERSION,
            token_ids=tuple(scope.token_ids),
            logprobs=tuple(scope.logprobs),
            action_kinds=tuple(scope.action_kinds),
            reason_codes=tuple(scope.reason_codes),
            stream_names=tuple(scope.stream_names),
            unit_offsets=tuple([unit.offset for unit in scope.units] + [len(scope.token_ids)]),
            stage_ids=tuple(unit.stage_id for unit in scope.units),
            input_modalities=tuple(unit.input_modality for unit in scope.units),
            input_origins=tuple(unit.input_origin for unit in scope.units),
            input_commit_ns=tuple(unit.input_commit_ns for unit in scope.units),
            emission_ns=tuple(unit.emission_ns for unit in scope.units),
            sampling_params=tuple(dict(unit.sampling) for unit in scope.units),
            terminators=tuple(unit.terminators for unit in scope.units),
            effective_seed=scope.effective_seed,
            streams=tuple(scope.streams.values()),
            model=dict(scope.model),
        )

    # ---- retention / retrieval ----

    def pop_oldest_sealed(self) -> DuplexActionRecordSet | None:
        """Spill support: pop the oldest sealed set once the window is full.

        ``None`` means the window has room for the next seal; the session's
        retention sink moves non-``None`` sets into the manager's terminal
        cache. Spilled scope keys stay sealed — their payloads remain invalid.
        """
        if len(self._sealed) < RETAINED_SEALED_SETS:
            return None
        key = next(iter(self._sealed))
        return self._sealed.pop(key)

    def take_terminal_sets(self) -> list[DuplexActionRecordSet]:
        """Drain every sealed set for the manager's terminal cache (post-close)."""
        sets = list(self._sealed.values())
        self._sealed.clear()
        self._open = None
        return sets

    @property
    def open_scope_response_id(self) -> str | None:
        return self._open.response_id if self._open is not None else None

    @property
    def has_open_scope(self) -> bool:
        return self._open is not None


__all__ = [
    "ActionLedgerError",
    "CODE_INVALID",
    "CODE_NOT_RETAINED",
    "CODE_RETENTION_OVERFLOW",
    "RETAINED_SEALED_SETS",
    "SessionActionLedger",
]
