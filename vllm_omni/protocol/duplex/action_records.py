# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action-record vocabulary for duplex sessions: the sealed columnar
``DuplexActionRecordSet`` (with its per-unit ``DuplexActionRecord`` view) and the
model -> engine wire payload contract. The engine records provenance only."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Wire-payload schema version of the model -> engine capture hop.
ACTION_CAPTURE_WIRE_VERSION = 1

# Record-set schema version of the engine -> trainer container.
ACTION_CAPTURE_RECORD_VERSION = 1

# ---- Action kinds ----

# Drawn from the policy distribution (including post-interruption tokens).
ACTION_SAMPLED = "sampled"
# Engine/env-resolved rather than sampled; ``reason`` says which forcing class.
ACTION_FORCED = "forced"
# Protocol re-emission of an earlier action (terminator lookahead): no draw.
ACTION_REEMIT = "reemit"
ACTION_KINDS = frozenset({ACTION_SAMPLED, ACTION_FORCED, ACTION_REEMIT})

# ---- Record-set statuses ----

STATUS_COMPLETED = "completed"
STATUS_TRUNCATED_CLOSE = "truncated_close"
STATUS_TRUNCATED_INTERRUPTION = "truncated_interruption"
STATUS_TRUNCATED_CANCEL = "truncated_cancel"
RECORD_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_TRUNCATED_CLOSE, STATUS_TRUNCATED_INTERRUPTION, STATUS_TRUNCATED_CANCEL}
)
# A drained mid-flight snapshot of the still-open scope (collection is
# draining: the eventual sealed set carries only the post-drain units).
STATUS_IN_PROGRESS = "in_progress"

# ---- Input origins ----

# Fed by the consumer of the session.
ORIGIN_CLIENT = "client"
# Injected by the runtime's silence-continuation clock.
ORIGIN_ENGINE = "engine"
INPUT_ORIGINS = frozenset({ORIGIN_CLIENT, ORIGIN_ENGINE})


class ActionPayloadError(ValueError):
    """A capture payload failed schema validation (``action_capture_invalid``)."""


@dataclass(frozen=True, slots=True)
class DuplexActionStream:
    """One named token sequence per (stage, modality); names flat-array positions."""

    name: str
    stage: int
    modality: str


@dataclass(frozen=True, slots=True)
class DuplexAction:
    """One action token with its exact-distribution log-probability and kind."""

    token_id: int
    logprob: float | None
    action_kind: str
    reason: str | None = None
    stream: str = ""


@dataclass(frozen=True, slots=True)
class DuplexActionInputChunk:
    """The unit's input chunk description (modality and origin)."""

    modality: str
    origin: str
    commit_ns: int | None = None


@dataclass(frozen=True, slots=True)
class DuplexActionRecord:
    """The derived per-unit view over a :class:`DuplexActionRecordSet`."""

    session_id: str
    response_id: str | None
    epoch: int
    turn_id: int
    unit_index: int
    stage_id: int
    input_chunk: DuplexActionInputChunk
    actions: tuple[DuplexAction, ...]
    emission_ns: int | None
    sampling: Mapping[str, Any]
    effective_seed: int | None
    model: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DuplexActionRecordSet:
    """Sealed columnar records of one capture scope: flat parallel arrays + unit offsets.

    Status is final once sealed: ``completed``, or ``truncated_close`` /
    ``truncated_interruption`` / ``truncated_cancel`` on an early end.
    ``response_id`` is ``None`` for a listening scope (units sampled while no
    engine response exists — still policy, never dropped).
    """

    session_id: str
    response_id: str | None
    epoch: int
    turn_id: int
    status: str
    schema_version: int = ACTION_CAPTURE_RECORD_VERSION
    # Token-level flat arrays (length == number of action tokens).
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float | None, ...] = ()
    action_kinds: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    stream_names: tuple[str, ...] = ()
    # Unit-level columns (length == number of units).
    unit_offsets: tuple[int, ...] = (0,)
    stage_ids: tuple[int, ...] = ()
    input_modalities: tuple[str, ...] = ()
    input_origins: tuple[str, ...] = ()
    input_commit_ns: tuple[int | None, ...] = ()
    emission_ns: tuple[int | None, ...] = ()
    sampling_params: tuple[Mapping[str, Any], ...] = ()
    # Per-unit model-sampled terminator tokens: ``{token_id, kind, reason?}``.
    terminators: tuple[tuple[Mapping[str, Any], ...], ...] = ()
    # Response-level columns.
    effective_seed: int | None = None
    streams: tuple[DuplexActionStream, ...] = ()
    # Namespaced model extension, engine-opaque, merged last-write-wins:
    # every non-``SamplingParams`` distribution input rides here.
    model: Mapping[str, Any] = field(default_factory=dict)

    def num_units(self) -> int:
        return len(self.stage_ids)

    def num_tokens(self) -> int:
        return len(self.token_ids)

    def records(self) -> Iterator[DuplexActionRecord]:
        """Derive the per-unit records view."""
        for unit_index in range(self.num_units()):
            start, end = self.unit_offsets[unit_index], self.unit_offsets[unit_index + 1]
            actions = tuple(
                DuplexAction(
                    token_id=self.token_ids[offset],
                    logprob=self.logprobs[offset],
                    action_kind=self.action_kinds[offset],
                    reason=self.reason_codes[offset] or None,
                    stream=self.stream_names[offset],
                )
                for offset in range(start, end)
            )
            yield DuplexActionRecord(
                session_id=self.session_id,
                response_id=self.response_id,
                epoch=self.epoch,
                turn_id=self.turn_id,
                unit_index=unit_index,
                stage_id=self.stage_ids[unit_index],
                input_chunk=DuplexActionInputChunk(
                    modality=self.input_modalities[unit_index],
                    origin=self.input_origins[unit_index],
                    commit_ns=self.input_commit_ns[unit_index],
                ),
                actions=actions,
                emission_ns=self.emission_ns[unit_index],
                sampling=self.sampling_params[unit_index],
                effective_seed=self.effective_seed,
                model=self.model,
            )

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe mapping for the reply event / client surface."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "response_id": self.response_id,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "status": self.status,
            "token_ids": list(self.token_ids),
            "logprobs": list(self.logprobs),
            "action_kinds": list(self.action_kinds),
            "reason_codes": list(self.reason_codes),
            "stream_names": list(self.stream_names),
            "unit_offsets": list(self.unit_offsets),
            "stage_ids": list(self.stage_ids),
            "input_modalities": list(self.input_modalities),
            "input_origins": list(self.input_origins),
            "input_commit_ns": list(self.input_commit_ns),
            "emission_ns": list(self.emission_ns),
            "sampling_params": [dict(params) for params in self.sampling_params],
            "terminators": [list(unit) for unit in self.terminators],
            "effective_seed": self.effective_seed,
            "streams": [
                {"name": stream.name, "stage": stream.stage, "modality": stream.modality} for stream in self.streams
            ],
            "model": dict(self.model),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> DuplexActionRecordSet:
        streams = tuple(
            DuplexActionStream(name=str(stream["name"]), stage=int(stream["stage"]), modality=str(stream["modality"]))
            for stream in payload.get("streams") or ()
        )
        return cls(
            session_id=str(payload["session_id"]),
            response_id=payload.get("response_id"),
            epoch=int(payload["epoch"]),
            turn_id=int(payload["turn_id"]),
            status=str(payload["status"]),
            schema_version=int(payload.get("schema_version", ACTION_CAPTURE_RECORD_VERSION)),
            token_ids=tuple(int(value) for value in payload.get("token_ids") or ()),
            logprobs=tuple(None if value is None else float(value) for value in payload.get("logprobs") or ()),
            action_kinds=tuple(str(value) for value in payload.get("action_kinds") or ()),
            reason_codes=tuple(str(value) for value in payload.get("reason_codes") or ()),
            stream_names=tuple(str(value) for value in payload.get("stream_names") or ()),
            unit_offsets=tuple(int(value) for value in payload.get("unit_offsets") or (0,)),
            stage_ids=tuple(int(value) for value in payload.get("stage_ids") or ()),
            input_modalities=tuple(str(value) for value in payload.get("input_modalities") or ()),
            input_origins=tuple(str(value) for value in payload.get("input_origins") or ()),
            input_commit_ns=tuple(
                None if value is None else int(value) for value in payload.get("input_commit_ns") or ()
            ),
            emission_ns=tuple(None if value is None else int(value) for value in payload.get("emission_ns") or ()),
            sampling_params=tuple(
                dict(params) if isinstance(params, Mapping) else {} for params in payload.get("sampling_params") or ()
            ),
            terminators=tuple(
                tuple(term for term in unit if isinstance(term, Mapping)) for unit in payload.get("terminators") or ()
            ),
            effective_seed=payload.get("effective_seed"),
            streams=streams,
            model=dict(payload.get("model") or {}),
        )


# ---- Model -> engine wire payload (one per model unit) ----

# Top-level keys the wire payload accepts. Unknown top-level keys are a
# validation error; unknown keys *inside* ``model`` pass through untouched.
_UNIT_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "stage_id",
        "unit_index",
        "seq",
        "epoch",
        "token_ids",
        "logprobs",
        "action_kinds",
        "reason_codes",
        "stream_names",
        "terminators",
        "sampling_params",
        "effective_seed",
        "streams",
        "model",
    }
)


def build_unit_payload(
    *,
    session_id: str,
    stage_id: int,
    unit_index: int,
    seq: int,
    epoch: int,
    token_ids: Sequence[int],
    logprobs: Sequence[float | None],
    action_kinds: Sequence[str],
    reason_codes: Sequence[str | None] | None = None,
    stream_names: Sequence[str] | None = None,
    streams: Sequence[Mapping[str, Any]] | None = None,
    terminators: Sequence[Mapping[str, Any]] | None = None,
    sampling_params: Mapping[str, Any] | None = None,
    effective_seed: int | None = None,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one unit's capture payload (producer side).

    Args:
        session_id: Session the unit belongs to.
        stage_id / unit_index / seq / epoch: Delivery identity; ``seq`` is the
            model-asserted unit key.
        token_ids / logprobs / action_kinds: Parallel columns; ``logprobs[i]``
            is the exact-distribution logprob of ``token_ids[i]``, ``None``
            only for draws that did not happen (``reemit``).
        reason_codes / stream_names / streams: Per-token reasons and stream
            labels (padded to the token count when omitted).
        terminators: Model-sampled terminator tokens with their kinds.
        sampling_params: Effective per-unit sampling parameters.
        effective_seed: The derived per-response seed, when set.
        model: Namespaced engine-opaque extension.

    Returns:
        The wire-ready payload mapping.

    Raises:
        ActionPayloadError: On column-length mismatch.
        ValueError: On non-finite logprobs (JSON encoding).
    """
    count = len(token_ids)
    if not (len(logprobs) == len(action_kinds) == count):
        raise ActionPayloadError(
            f"action capture column mismatch: {count} tokens, {len(logprobs)} logprobs, {len(action_kinds)} kinds"
        )
    if stream_names is not None and len(stream_names) != count:
        raise ActionPayloadError(f"stream_names length {len(stream_names)} != {count} tokens")
    return {
        "schema_version": ACTION_CAPTURE_WIRE_VERSION,
        "session_id": str(session_id),
        "stage_id": int(stage_id),
        "unit_index": int(unit_index),
        "seq": int(seq),
        "epoch": int(epoch),
        "token_ids": [int(value) for value in token_ids],
        "logprobs": [None if value is None else float(value) for value in logprobs],
        "action_kinds": [str(value) for value in action_kinds],
        "reason_codes": (["" if value is None else str(value) for value in (reason_codes or ())] + [""] * count)[
            :count
        ],
        "stream_names": ([str(value) for value in (stream_names or ())] + [""] * count)[:count],
        "streams": [
            {"name": str(stream["name"]), "stage": int(stream["stage"]), "modality": str(stream["modality"])}
            for stream in (streams or ())
        ],
        "terminators": [
            {
                "token_id": int(term["token_id"]),
                "kind": str(term["kind"]),
                **({"reason": str(term["reason"])} if term.get("reason") is not None else {}),
            }
            for term in (terminators or ())
        ],
        "sampling_params": dict(sampling_params or {}),
        "effective_seed": None if effective_seed is None else int(effective_seed),
        "model": dict(model or {}),
    }


def encode_unit_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialize one unit payload onto the bytes carrier.

    Returns:
        The compact UTF-8 JSON encoding of ``payload``.

    Raises:
        ValueError: On non-finite logprobs (``allow_nan=False``).
    """
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_unit_payload(data: bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Decode and strictly validate one unit payload at ledger ingest.

    Args:
        data: The bytes carrier's payload (or an already-decoded mapping).

    Returns:
        The validated payload mapping; unknown ``model`` keys pass through.

    Raises:
        ActionPayloadError: On unknown schema version, unknown or missing
            universal keys, wrong column lengths or element types, invalid
            action kinds, or non-finite logprobs.
    """
    try:
        payload = json.loads(data) if isinstance(data, (bytes, bytearray)) else dict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionPayloadError(f"action capture payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionPayloadError("action capture payload must be a JSON object")
    version = payload.get("schema_version")
    if version != ACTION_CAPTURE_WIRE_VERSION:
        raise ActionPayloadError(f"unsupported action capture schema_version: {version!r}")
    unknown = set(payload) - _UNIT_PAYLOAD_KEYS
    if unknown:
        raise ActionPayloadError(f"unknown action capture fields: {sorted(unknown)}")

    def require(name: str, types: tuple[type, ...]) -> Any:
        value = payload.get(name)
        if not isinstance(value, types) or isinstance(value, bool) and bool not in types:
            raise ActionPayloadError(f"action capture field {name!r} has invalid type {type(value).__name__}")
        return value

    token_ids = require("token_ids", (list,))
    logprobs = require("logprobs", (list,))
    kinds = require("action_kinds", (list,))
    if not (len(token_ids) == len(logprobs) == len(kinds)):
        raise ActionPayloadError("action capture token/logprob/kind columns must be parallel")
    for value in token_ids:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ActionPayloadError("action capture token_ids entries must be integers")
    for name in ("stage_id", "unit_index", "seq", "epoch"):
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ActionPayloadError(f"action capture field {name!r} must be an integer")
    for value in kinds:
        if value not in ACTION_KINDS:
            raise ActionPayloadError(f"invalid action kind: {value!r}")
    for value in logprobs:
        if value is not None:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ActionPayloadError("action capture logprob must be a number or null")
            if value != value or value in (float("inf"), float("-inf")):
                raise ActionPayloadError("action capture logprob must be finite")
    for name in ("reason_codes", "stream_names"):
        column = payload.get(name)
        if column is not None and len(column) != len(token_ids):
            raise ActionPayloadError(f"action capture column {name!r} length mismatch")
    for stream in payload.get("streams") or ():
        if not isinstance(stream, dict) or not isinstance(stream.get("name"), str):
            raise ActionPayloadError("action capture stream entries must name a string name")
    for term in payload.get("terminators") or ():
        if (
            not isinstance(term, dict)
            or not isinstance(term.get("token_id"), int)
            or not isinstance(term.get("kind"), str)
        ):
            raise ActionPayloadError("action capture terminator entries must carry int token_id and str kind")
    if not isinstance(payload.get("model", {}), dict):
        raise ActionPayloadError("action capture 'model' extension must be an object")
    if not isinstance(payload.get("sampling_params", {}), dict):
        raise ActionPayloadError("action capture 'sampling_params' must be an object")
    return payload


__all__ = [
    "ACTION_CAPTURE_RECORD_VERSION",
    "ACTION_CAPTURE_WIRE_VERSION",
    "ACTION_FORCED",
    "ACTION_KINDS",
    "ACTION_REEMIT",
    "ACTION_SAMPLED",
    "ActionPayloadError",
    "DuplexAction",
    "DuplexActionInputChunk",
    "DuplexActionRecord",
    "DuplexActionRecordSet",
    "DuplexActionStream",
    "INPUT_ORIGINS",
    "ORIGIN_CLIENT",
    "ORIGIN_ENGINE",
    "RECORD_STATUSES",
    "STATUS_COMPLETED",
    "STATUS_IN_PROGRESS",
    "STATUS_TRUNCATED_CANCEL",
    "STATUS_TRUNCATED_CLOSE",
    "STATUS_TRUNCATED_INTERRUPTION",
    "build_unit_payload",
    "decode_unit_payload",
    "encode_unit_payload",
]
