# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded synthetic replay plans and an in-process session-owner adapter.

The runner supplies session lifetime, idle scheduling, live resume and query
execution. This module does not create a server, model, judge or metric system.
Calibrated text fixtures explicitly do not establish audio/serving coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

_MAX_FILE_BYTES = 8 * 1024 * 1024
_TIMINGS = frozenset({"fixed_idle", "immediate_after_mutation"})
_EVENT_FIELDS = {
    "message": ({"type", "item_id", "label", "role", "text"}, set()),
    "edit": ({"type", "item_id", "label", "replacement_text"}, set()),
    "delete": ({"type", "item_id"}, set()),
    "idle": ({"type", "duration_ms"}, set()),
    "resume_live_session": ({"type", "same_session_object"}, set()),
    "assistant_playback": (
        {"type", "item_id", "label", "generated_text", "heard_text", "heard_char_end", "calibration"},
        {"played_audio_ms", "barge_in"},
    ),
}


@dataclass(frozen=True)
class MemoryReplayEvent:
    event_type: str
    payload: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True)
class MemoryReplayQuery:
    text: str = field(repr=False)
    gold_json: str = field(repr=False)


@dataclass(frozen=True)
class MemoryReplayCase:
    case_id: str = field(repr=False)
    events: tuple[MemoryReplayEvent, ...] = field(repr=False)
    queries: tuple[MemoryReplayQuery, ...] = field(repr=False)
    timing_variants: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True)
class MemoryReplayVariant:
    """Execution inputs only: evaluation answers never reach runner callbacks."""

    case_id: str = field(repr=False)
    query_index: int
    timing: str
    events: tuple[MemoryReplayEvent, ...] = field(repr=False)
    query_text: str = field(repr=False)
    source_sha256: str


@dataclass(frozen=True)
class MemoryReplaySource:
    """Observation at one event; presence does not imply query-time validity."""

    event_index: int
    event_type: str
    label: str | None = field(repr=False)
    fixture_item_id: str = field(repr=False)
    native_item_id: str = field(repr=False)
    source_id: int | None
    revision: int | None
    present: bool


@dataclass(frozen=True)
class MemoryReplayResult:
    case_id: str = field(repr=False)
    query_index: int
    timing: str
    source_sha256: str
    source_trace: tuple[MemoryReplaySource, ...] = field(repr=False)
    output: object = field(repr=False)
    calibrated_text_fixture_only: bool


def _invalid(*_: object) -> NoReturn:
    raise ValueError("invalid_memory_replay")


def _text(value: object, *, empty: bool = False, limit: int = 16384) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError("invalid_memory_replay")
    try:
        if len(value.encode("utf-8")) > 4 * limit:
            _invalid()
    except UnicodeEncodeError:
        raise ValueError("invalid_memory_replay") from None
    return value


def _integer(value: object, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid_memory_replay")
    return value


def _object(value: object, required: set[str], optional: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise ValueError("invalid_memory_replay")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _gold(value: object, depth: int = 0) -> None:
    # Gold is bounded, detached judge input. It is not an executable event or
    # a prompt field, and its semantic scoring belongs to the existing judge.
    if depth > 6:
        _invalid()
    if isinstance(value, dict):
        if len(value) > 32:
            _invalid()
        for key, item in value.items():
            _text(key, limit=128)
            _gold(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 32:
            _invalid()
        for item in value:
            _gold(item, depth + 1)
    elif isinstance(value, str):
        _text(value, empty=True)
    elif isinstance(value, (int, float)):
        try:
            if not math.isfinite(value):
                _invalid()
        except OverflowError:
            raise ValueError("invalid_memory_replay") from None
    elif value is not None and type(value) is not bool:
        _invalid()


def _event(value: object, live: set[str], labels: set[str]) -> MemoryReplayEvent:
    if not isinstance(value, dict) or value.get("type") not in _EVENT_FIELDS:
        raise ValueError("invalid_memory_replay")
    kind = str(value["type"])
    required, optional = _EVENT_FIELDS[kind]
    event = dict(_object(value, required, optional))
    if "item_id" in event:
        item_id = _text(event["item_id"], limit=128)
        if kind in {"message", "assistant_playback"}:
            if item_id in live:
                _invalid()
            live.add(item_id)
        elif item_id not in live:
            _invalid()
        elif kind == "delete":
            live.remove(item_id)
    if "label" in event:
        label = _text(event["label"], limit=128)
        if label in labels:
            _invalid()
        labels.add(label)
    if kind == "message":
        if event["role"] not in {"user", "assistant"}:
            _invalid()
        text = _text(event["text"])
        if event["role"] == "assistant" and text != text.strip():
            _invalid()
    elif kind == "edit":
        _text(event["replacement_text"])
    elif kind == "idle":
        _integer(event["duration_ms"], 600000)
    elif kind == "resume_live_session":
        if event["same_session_object"] is not True:
            _invalid()
    elif kind == "assistant_playback":
        generated = _text(event["generated_text"])
        heard = _text(event["heard_text"], empty=True)
        end = _integer(event["heard_char_end"], len(generated))
        played = _integer(event.get("played_audio_ms", 1000 if heard else 0), 2000)
        expected = "accepted_prefix_fixture_only" if heard else "accepted_empty_prefix_fixture_only"
        if generated != generated.strip() or heard != generated[:end] or (heard and not played):
            _invalid()
        if event["calibration"] != expected or type(event.get("barge_in", False)) is not bool:
            _invalid()
    return MemoryReplayEvent(kind, MappingProxyType(event))


def load_memory_replay_cases(path: Path) -> tuple[MemoryReplayCase, ...]:
    """Read a small curated JSONL fixture, rejecting malformed data up front."""
    with path.open("rb") as source:
        raw = source.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        _invalid()
    digest = hashlib.sha256(raw).hexdigest()
    cases: list[MemoryReplayCase] = []
    identities: set[str] = set()
    metadata = {
        "language",
        "title",
        "status",
        "padding_user_assistant_pairs",
        "mutation_coverage_requires_prior_card",
        "requires_media_adapter",
    }
    try:
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid)
            row = _object(value, {"case_id", "synthetic", "events", "queries", "timing_variants"}, metadata)
            identity = _text(row["case_id"], limit=128)
            if identity in identities or row["synthetic"] is not True or len(cases) >= 128:
                _invalid()
            identities.add(identity)
            values, queries, timings = row["events"], row["queries"], row["timing_variants"]
            if not isinstance(values, list) or not 1 <= len(values) <= 512:
                _invalid()
            if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
                _invalid()
            if (
                not isinstance(timings, list)
                or not timings
                or len(timings) > 2
                or any(t not in _TIMINGS for t in timings)
                or len(set(timings)) != len(timings)
            ):
                _invalid()
            live: set[str] = set()
            labels: set[str] = set()
            events = tuple(_event(event, live, labels) for event in values)
            idles = [_integer(event.payload["duration_ms"], 600000) for event in events if event.event_type == "idle"]
            if sum(idles) > 3600000:
                _invalid()
            prepared_queries = []
            for query in queries:
                item = _object(query, {"text", "gold"})
                if not isinstance(item["gold"], dict):
                    _invalid()
                _gold(item["gold"])
                prepared_queries.append(
                    MemoryReplayQuery(_text(item["text"]), json.dumps(item["gold"], ensure_ascii=False, sort_keys=True))
                )
            cases.append(MemoryReplayCase(identity, events, tuple(prepared_queries), tuple(timings), digest))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError):
        raise ValueError("invalid_memory_replay") from None
    if not cases:
        _invalid()
    return tuple(cases)


def memory_replay_variants(cases: Sequence[MemoryReplayCase]) -> tuple[MemoryReplayVariant, ...]:
    variants = []
    for case in cases:
        for index, query in enumerate(case.queries):
            for timing in case.timing_variants:
                events = case.events
                if timing == "immediate_after_mutation":
                    last_change = max(
                        (
                            i
                            for i, event in enumerate(events)
                            if event.event_type not in {"idle", "resume_live_session"}
                        ),
                        default=-1,
                    )
                    tail_idles = [i for i, event in enumerate(events) if event.event_type == "idle" and i > last_change]
                    if tail_idles:
                        events = tuple(event for i, event in enumerate(events) if i != tail_idles[-1])
                variants.append(
                    MemoryReplayVariant(case.case_id, index, timing, events, query.text, case.source_sha256)
                )
    return tuple(variants)


def _source(session: DuplexEngineSession, native_id: str) -> tuple[int | None, int | None]:
    snapshot = session.history_snapshot()
    if native_id not in snapshot.item_ids:
        return None, None
    index = snapshot.item_ids.index(native_id)
    return snapshot.source_ids[index], snapshot.source_revisions[index]


def _require_open(session: DuplexEngineSession) -> None:
    if session.state.value != "open" or session.lease.terminal_reason is not None:
        raise ValueError("invalid_memory_replay_closed_session")


def _assistant(session: DuplexEngineSession, text: str, playback: Mapping[str, object] | None) -> str:
    if session.active_response_id is not None:
        raise ValueError("invalid_memory_replay_active_response")
    response_id = session.begin_response()
    native_id = f"item_{response_id}"
    session.reserve_history_item(native_id)
    session.append_assistant_text(text)
    if playback is not None:
        if session.config.playback_commit_policy != "ack_only":
            raise ValueError("invalid_memory_replay_playback_policy")
        session.mark_audio_sent(duration_ms=2000, text_chars=len(text))
        heard = str(playback["heard_text"])
        played = _integer(playback.get("played_audio_ms", 1000 if heard else 0), 2000)
        session.acknowledge_playback(played_ms=played)
    session.register_history_item(native_id, session.end_response())
    if playback is not None and playback["heard_text"]:
        if not session.apply_heard_text(
            response_id, text=text, audio_end_ms=played, char_end=_integer(playback["heard_char_end"], len(text))
        ):
            raise ValueError("invalid_memory_replay_calibration")
    if playback is not None and playback.get("barge_in", False):
        session.truncate_history_item(native_id, audio_end_ms=played, hard=True)
    return native_id


async def replay_memory_cases(
    cases: Sequence[MemoryReplayCase],
    *,
    session_factory: Callable[[MemoryReplayVariant], AbstractAsyncContextManager[DuplexEngineSession]],
    idle: Callable[[DuplexEngineSession, int], Awaitable[None]],
    resume: Callable[[DuplexEngineSession], Awaitable[DuplexEngineSession]],
    query: Callable[[DuplexEngineSession, str], Awaitable[object]],
    allow_calibrated_text_fixtures: bool = False,
) -> tuple[MemoryReplayResult, ...]:
    """Replay committed text into fresh owners; callbacks reuse runner services.

    Idle is fixed-duration, never a wait-until-extraction-succeeds barrier.
    The caller owns generation/judging. Prefix fixtures use owner calibration,
    not an audio decoder; the default rejects them before opening sessions.
    """
    variants = memory_replay_variants(cases)
    if type(allow_calibrated_text_fixtures) is not bool:
        _invalid()
    if not allow_calibrated_text_fixtures and any(
        event.event_type == "assistant_playback" for variant in variants for event in variant.events
    ):
        raise ValueError("invalid_memory_replay_requires_calibrated_text_opt_in")
    results = []
    owners: list[weakref.ReferenceType[DuplexEngineSession]] = []
    for variant in variants:
        async with session_factory(variant) as session:
            if (
                any(owner() is session for owner in owners)
                or session.state.value != "open"
                or session.lease.terminal_reason is not None
                or session.active_response_id is not None
                or session.history_snapshot().messages
                or session.context_ledger.nodes()
                or session.context_ledger.card_catalog()
            ):
                raise ValueError("invalid_memory_replay_requires_fresh_session")
            owners.append(weakref.ref(session))
            aliases: dict[str, str] = {}
            trace = []
            used_playback_fixture = False
            for index, event in enumerate(variant.events):
                _require_open(session)
                p, kind = event.payload, event.event_type
                if kind == "idle":
                    await idle(session, _integer(p["duration_ms"], 600000))
                    continue
                if kind == "resume_live_session":
                    if await resume(session) is not session:
                        raise ValueError("invalid_memory_replay_replaced_live_session")
                    continue
                fixture_id = str(p["item_id"])
                if kind == "message":
                    if p["role"] == "assistant":
                        native_id = _assistant(session, str(p["text"]), None)
                    else:
                        native_id = fixture_id
                        session.register_history_item(native_id, {"role": "user", "content": str(p["text"])})
                    aliases[fixture_id] = native_id
                elif kind == "assistant_playback":
                    native_id = _assistant(session, str(p["generated_text"]), p)
                    aliases[fixture_id] = native_id
                    used_playback_fixture = True
                else:
                    native_id = aliases[fixture_id]
                    if kind == "delete":
                        source_id, revision = _source(session, native_id)
                        if not session.delete_history_item(native_id):
                            raise ValueError("invalid_memory_replay_missing_item")
                        trace.append(
                            MemoryReplaySource(index, kind, None, fixture_id, native_id, source_id, revision, False)
                        )
                        continue
                    snapshot = session.history_snapshot()
                    position = snapshot.item_ids.index(native_id)
                    session.register_history_item(
                        native_id, {"role": snapshot.messages[position]["role"], "content": str(p["replacement_text"])}
                    )
                source_id, revision = _source(session, native_id)
                trace.append(
                    MemoryReplaySource(
                        index, kind, str(p["label"]), fixture_id, native_id, source_id, revision, source_id is not None
                    )
                )
            _require_open(session)
            output = await query(session, variant.query_text)
            results.append(
                MemoryReplayResult(
                    variant.case_id,
                    variant.query_index,
                    variant.timing,
                    variant.source_sha256,
                    tuple(trace),
                    output,
                    used_playback_fixture,
                )
            )
    return tuple(results)
