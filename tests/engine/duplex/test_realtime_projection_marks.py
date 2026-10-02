# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""An audio delta updates the in-progress item's marks without redoing the earlier ones.

The projection is replayed once as it is and once with the previous
``_remember_response_audio_metadata`` (a full re-sort per delta, and so a full
copy of the item's marks); both must give the same events and items at every
step.
"""

from __future__ import annotations

import base64
import dataclasses
import random
from collections.abc import Mapping

import pytest

from vllm_omni.engine.duplex import realtime_events
from vllm_omni.engine.duplex.realtime_events import RealtimeProjectionState, project_internal_event

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PCM = base64.b64encode(b"\x00\x10" * 8).decode("ascii")


def _legacy_remember_response_audio_metadata(
    state: RealtimeProjectionState, response_id: object, event: Mapping[str, object]
) -> None:
    """``_remember_response_audio_metadata`` before in-order marks were appended."""
    projection = realtime_events._response_state(state, response_id)
    if projection is None:
        return
    duration = event.get("audio_duration_ms")
    playback = event.get("playback")
    if not isinstance(duration, int | float) and isinstance(playback, dict):
        duration = playback.get("sent_ms") or playback.get("generated_ms")
    if isinstance(duration, int | float):
        projection.audio_duration_ms = max(projection.audio_duration_ms or 0, int(duration))
    marks = event.get("audio_text_marks")
    if not isinstance(marks, list):
        return
    clean_marks: list[dict[str, int]] = []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        text_chars = mark.get("text_chars")
        audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
        if not isinstance(text_chars, int | float) or not isinstance(audio_end_ms, int | float):
            continue
        clean_marks.append({"text_chars": max(0, int(text_chars)), "audio_end_ms": max(0, int(audio_end_ms))})
    if clean_marks:
        merged = list(projection.audio_text_marks)
        merged.extend(clean_marks)
        deduped: dict[tuple[int, int], dict[str, int]] = {}
        for mark in merged:
            deduped[(int(mark["audio_end_ms"]), int(mark["text_chars"]))] = mark
        projection.audio_text_marks = sorted(
            deduped.values(), key=lambda mark: (mark["audio_end_ms"], mark["text_chars"])
        )
        projection.item_audio_text_marks = []


def _delta_marks(rng: random.Random, audio_ms: int, text_chars: int, last: tuple[int, int]) -> object:
    """The ``audio_text_marks`` of one delta: mostly one mark past the last, sometimes not."""
    roll = rng.random()
    if roll < 0.70:
        return [{"text_chars": text_chars, "audio_end_ms": audio_ms}]
    if roll < 0.75:
        return [{"text_chars": last[1], "audio_end_ms": last[0]}]  # repeats the last mark
    if roll < 0.80:
        return [{"text_chars": max(0, text_chars - 7), "audio_end_ms": max(0, audio_ms - 900)}]  # sorts earlier
    if roll < 0.85:
        return [
            {"text_chars": text_chars, "audio_end_ms": audio_ms - 40},
            {"text_chars": text_chars, "audio_end_ms": audio_ms},
        ]
    if roll < 0.88:
        return [{"text_chars": text_chars, "audio_end_ms": audio_ms}, {"text_chars": 0, "audio_end_ms": 0}]
    if roll < 0.91:
        return [{"text_chars": float(text_chars), "audio_ms": audio_ms + 0.5}]
    if roll < 0.93:
        return ["junk", {"text_chars": "x", "audio_end_ms": audio_ms}, {"text_chars": text_chars}]
    if roll < 0.95:
        return [{"text_chars": -3, "audio_end_ms": -10}]
    if roll < 0.97:
        return []
    return None


def _event_sequence(seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    events: list[dict[str, object]] = []
    for response in range(3):
        response_id = f"resp_{seed}_{response}"
        events.append({"type": "response.created", "response_id": response_id, "modalities": ["audio", "text"]})
        audio_ms = 0
        text_chars = 0
        last = (0, 0)
        for index in range(rng.randrange(40, 120)):
            audio_ms += 80 * rng.randrange(1, 6)
            text = rng.choice(["", "", "ab", "hello ", "x"])
            text_chars += len(text)
            event: dict[str, object] = {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "audio": _PCM if rng.random() > 0.05 else "",
                "text": text,
                "format": "pcm16",
            }
            if rng.random() > 0.1:
                event["audio_duration_ms"] = audio_ms
            else:
                event["playback"] = {"sent_ms": audio_ms, "generated_ms": audio_ms}
            marks = _delta_marks(rng, audio_ms, text_chars, last)
            if marks is not None:
                event["audio_text_marks"] = marks
            events.append(event)
            last = (audio_ms, text_chars)
            if index == 30 and response == 1:
                # The client truncates the in-progress item; the refresh keeps applying it.
                events.append({"type": "_truncate", "item_id": f"item_{response_id}", "audio_end_ms": audio_ms // 2})
        events.append({"type": "response.done", "response_id": response_id})
    return events


def _replay(events: list[dict[str, object]]) -> list[tuple[list[object], dict[str, object]]]:
    state = RealtimeProjectionState(session_id="duplex-marks", model="test-model")
    steps: list[tuple[list[object], dict[str, object]]] = []
    for event in events:
        if event["type"] == "_truncate":
            state.item_truncation_cursors[str(event["item_id"])] = (0, int(event["audio_end_ms"]))  # type: ignore[call-overload]
            continue
        projected = [_without_event_id(projected) for projected in project_internal_event(state, event)]
        items = {item_id: _plain(item) for item_id, item in state.conversation_items.items()}
        marks = {key: _plain(projection.audio_text_marks) for key, projection in state.response_states.items()}
        steps.append((projected, {"items": items, "marks": marks}))
    return steps


def _without_event_id(event: object) -> object:
    """Blank ``event_id``: each event gets a fresh uuid, so two replays can agree only on the rest."""
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return dataclasses.replace(event, event_id="")
    return event


def _plain(value: object) -> object:
    """A structural copy, so a later step cannot change what an earlier one recorded."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def test_items_and_events_match_the_previous_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _event_sequence(seed=0)
    current = _replay(events)

    monkeypatch.setattr(realtime_events, "_remember_response_audio_metadata", _legacy_remember_response_audio_metadata)
    legacy = _replay(events)

    assert len(current) == len(legacy)
    for step, ((events_now, state_now), (events_before, state_before)) in enumerate(zip(current, legacy)):
        assert events_now == events_before, f"events differ at step {step}"
        assert state_now == state_before, f"items or marks differ at step {step}"
