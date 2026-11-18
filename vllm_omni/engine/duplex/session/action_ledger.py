# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session capture ledger: assembles validated unit payloads into sealed
``DuplexActionRecordSet`` s, one per scope (response or listening period).

Validation failure is session-fatal; retention is draining; close/interrupt/
cancel seal partial sets. Mutated only on the orchestrator loop — no locking.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.protocol.duplex.action_records import (
    ACTION_CAPTURE_RECORD_VERSION,
    ACTION_SAMPLED,
    INPUT_ORIGINS,
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
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
    # Unit-level rows, in ingest order.
    units: list[_UnitRow] = field(default_factory=list)
    streams: dict[tuple[str, int, str], DuplexActionStream] = field(default_factory=dict)
    effective_seed: int | None = None
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
        # Appended-unit input annotations (modality, origin, commit time) per epoch.
        self._input_units: dict[int, dict[int, tuple[str, str, int | None]]] = {}
        # Scope keys already sealed once (ingest accounting that outlives
        # collection): a payload for one of these is invalid, never a gap.
        self._sealed_keys: set[str] = set()

    # ---- scope keys ----

    @staticmethod
    def _scope_key(response_id: str | None, epoch: int, turn_id: int) -> str:
        return response_id if response_id is not None else f"~listen-{epoch}-{turn_id}"

    # ---- engine-side annotations (no payload involved) ----

    def note_input_unit(
        self, epoch: int, seq: int, *, modality: str, origin: str, commit_ns: int | None = None
    ) -> None:
        """Annotate one appended unit's input side; unknown origins are rejected."""
        if origin not in INPUT_ORIGINS:
            raise ActionLedgerError(f"unknown capture input origin: {origin!r}")
        self._input_units.setdefault(epoch, {})[seq] = (str(modality), str(origin), commit_ns)

    def input_unit(self, epoch: int, seq: int) -> tuple[str, str, int | None]:
        """The ``(modality, origin, commit_ns)`` annotation for one unit."""
        return self._input_units.get(epoch, {}).get(seq, ("audio", "client", None))

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
        """Validate and append one unit payload; returns the scope unit index.

        ``decode_unit_payload`` owns schema validation; this method owns
        ordering (model-asserted ``seq``, strictly monotone per scope)."""
        record = decode_unit_payload(payload)
        if int(record["epoch"]) != int(epoch):
            raise ActionLedgerError(f"capture payload epoch {record['epoch']} != delivery epoch {epoch}")
        if int(record["stage_id"]) != int(stage_id):
            raise ActionLedgerError(f"capture payload stage {record['stage_id']} != delivery stage {stage_id}")
        seq = int(record["seq"])
        scope = self._require_open(epoch=epoch, response_id=response_id, turn_id=turn_id)
        if scope.last_seq is not None and seq <= scope.last_seq:
            raise ActionLedgerError(
                f"capture payload seq {seq} is not after {scope.last_seq} in scope {scope.response_id or 'listening'}"
            )
        scope.last_seq = seq
        modality, origin, commit_ns_from_input = self.input_unit(epoch, seq)
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
        scope.units.append(
            _UnitRow(
                stage_id=int(stage_id),
                seq=seq,
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
                f"action records for {held} were not collected before the next scope completed",
                code=CODE_RETENTION_OVERFLOW,
            )
        self._sealed[key] = self._freeze(scope, status)
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

    def collect(self, response_id: str | None = None) -> DuplexActionRecordSet:
        """Return and clear a retained set; ``None`` selects the most recent
        (sealed, or the open scope drained as an ``in_progress`` snapshot)."""
        if response_id is None:
            if self._sealed:
                key = next(reversed(self._sealed))
                return self._sealed.pop(key)
            if self._open is not None and (self._open.units or self._open.token_ids):
                return self._drain_open()
            raise ActionLedgerError("no action records retained", code=CODE_NOT_RETAINED)
        if response_id in self._sealed:
            return self._sealed.pop(response_id)
        if (
            self._open is not None
            and self._open.response_id == response_id
            and (self._open.units or self._open.token_ids)
        ):
            return self._drain_open()
        raise ActionLedgerError(
            f"action records for {response_id} are outside the retention window",
            code=CODE_NOT_RETAINED,
        )

    def _drain_open(self) -> DuplexActionRecordSet:
        scope = self._open
        assert scope is not None
        snapshot = self._freeze(scope, STATUS_IN_PROGRESS)
        # Drained units leave the scope; its sealed successor carries only
        # post-drain units (the trainer concatenates drained + sealed).
        scope.token_ids = []
        scope.logprobs = []
        scope.action_kinds = []
        scope.reason_codes = []
        scope.stream_names = []
        scope.units = []
        return snapshot

    def take_terminal_sets(self) -> list[DuplexActionRecordSet]:
        """Drain every sealed set for the manager's terminal cache (post-close)."""
        sets = list(self._sealed.values())
        self._sealed.clear()
        self._open = None
        return sets

    @property
    def open_scope_response_id(self) -> str | None:
        return self._open.response_id if self._open is not None else None


__all__ = [
    "ActionLedgerError",
    "CODE_INVALID",
    "CODE_NOT_RETAINED",
    "CODE_RETENTION_OVERFLOW",
    "RETAINED_SEALED_SETS",
    "SessionActionLedger",
]
