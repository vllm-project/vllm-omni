"""Strict MiniCPM-o 4.5 Realtime duplex E2E scenario harness.

This script is intentionally scenario-based instead of a generic chat client.
It has two explicit validation contracts:

1. ``model-policy`` accepts either a model-owned listen decision or a complete
   spoken response for each streamed user turn;
2. ``response-required`` uses a known fixture and requires a complete audio
   response with an independent transcript for every requested turn.

Explicit serving-side barge-in is intentionally not part of this smoke test.
MiniCPM-o native duplex currently exposes model-owned listen/speak switching,
not a separate model-level barge-in contract.

Run only after a MiniCPM-o 4.5 vLLM-Omni server is up:

  python tests/e2e/online_serving/minicpmo_realtime_duplex_scenarios.py \
      --url ws://localhost:8099/v1/realtime?duplex=1 \
      --model openbmb/MiniCPM-o-4_5 \
      --input-wav input_16k_mono_pcm16.wav \
      --validation-mode model-policy \
      --output-dir /tmp/minicpmo_duplex_demo
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import hashlib
import inspect
import json
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover - demo dependency.
    raise SystemExit("Install websockets first: pip install websockets") from exc

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
)

_url_with_model = build_realtime_url
_read_wav_pcm16 = read_pcm16_wav


def _ref_audio_data_url(path: str | None) -> str | None:
    if path is None:
        return None
    ref_path = Path(path).expanduser()
    return "data:audio/wav;base64," + base64.b64encode(ref_path.read_bytes()).decode("ascii")


@dataclass
class DemoState:
    events: list[dict[str, object]] = field(default_factory=list)
    timing_events: RealtimeEventCollector = field(default_factory=RealtimeEventCollector)
    audio_deltas: list[bytes] = field(default_factory=list)
    response_audio_deltas: dict[str, list[bytes]] = field(default_factory=dict)
    response_ids: list[str] = field(default_factory=list)
    assistant_item_ids: list[str] = field(default_factory=list)
    done_count: int = 0
    cancelled_count: int = 0
    listen_count: int = 0
    model_listen_count: int = 0
    buffering_listen_count: int = 0
    model_speak_event_count: int = 0
    model_speak_delta_count: int = 0
    playback_ack_count: int = 0
    playback_history_committed_count: int = 0
    truncate_count: int = 0
    input_transcription_count: int = 0
    audio_marks_seen: bool = False
    overlap_decisions: list[dict[str, object]] = field(default_factory=list)
    output_sample_rate_hz: int = 24000
    input_commit_sent_at_s: list[float] = field(default_factory=list)

    def add(self, event: dict[str, object], *, received_at_s: float | None = None) -> None:
        self.events.append(event)
        self.timing_events.add(event, received_at_s=received_at_s)
        event_type = event.get("type")
        if event_type == "response.created":
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
            if isinstance(response_id, str) and response_id not in self.response_ids:
                self.response_ids.append(response_id)
        elif event_type == "conversation.item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("role") == "assistant":
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id not in self.assistant_item_ids:
                    self.assistant_item_ids.append(item_id)
        elif event_type == "response.audio.delta":
            delta = event.get("delta") or event.get("audio")
            if isinstance(delta, str) and delta:
                try:
                    decoded = base64.b64decode(delta)
                    self.audio_deltas.append(decoded)
                    response_id = self._event_response_id(event)
                    if isinstance(response_id, str):
                        self.response_audio_deltas.setdefault(response_id, []).append(decoded)
                except Exception:
                    pass
            metadata = event.get("metadata")
            if isinstance(metadata, dict):
                if metadata.get("model_speak") is True:
                    self.model_speak_delta_count += 1
                if isinstance(metadata.get("audio_text_marks"), list):
                    self.audio_marks_seen = True
            sample_rate_hz = event.get("sample_rate_hz")
            if isinstance(sample_rate_hz, int) and sample_rate_hz > 0:
                self.output_sample_rate_hz = sample_rate_hz
        elif event_type == "response.done":
            self.done_count += 1
            response = event.get("response")
            if isinstance(response, dict) and response.get("status") == "cancelled":
                self.cancelled_count += 1
        elif event_type in {"audio.cancelled", "input.cancelled"}:
            self.cancelled_count += 1
        elif event_type == "response.listen":
            self.listen_count += 1
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            if isinstance(metadata, dict) and metadata.get("model_listen") is True:
                self.model_listen_count += 1
            if isinstance(metadata, dict) and metadata.get("buffering") is True:
                self.buffering_listen_count += 1
        elif event_type == "response.speak":
            self.model_speak_event_count += 1
        elif event_type == "overlap.decision":
            self.overlap_decisions.append(event)
        elif event_type == "playback.acknowledged":
            self.playback_ack_count += 1
            payload = event.get("event")
            if isinstance(payload, dict) and payload.get("history_committed") is True:
                self.playback_history_committed_count += 1
        elif event_type == "conversation.item.truncated":
            self.truncate_count += 1
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self.input_transcription_count += 1

    def count(self, event_type: str) -> int:
        return sum(1 for event in self.events if event.get("type") == event_type)

    def response_timing_summaries(self) -> dict[str, dict[str, object]]:
        summaries: dict[str, dict[str, object]] = {}
        for response_id in self.response_ids:
            created_at_s = next(
                (
                    received_at_s
                    for event, received_at_s in zip(
                        self.timing_events.events,
                        self.timing_events.event_received_at_s,
                        strict=True,
                    )
                    if event.get("type") == "response.created" and self._event_response_id(event) == response_id
                ),
                None,
            )
            if created_at_s is None:
                continue
            input_committed_at_s = next(
                (
                    committed_at_s
                    for committed_at_s in reversed(self.input_commit_sent_at_s)
                    if committed_at_s <= created_at_s
                ),
                None,
            )
            timing = self.timing_events.timing_summary(
                after_s=created_at_s,
                input_committed_at_s=input_committed_at_s,
                response_id=response_id,
            )
            timing["measurement_origin"] = {
                "response": "response.created client receive",
                "input": "input_audio_buffer.commit client send" if input_committed_at_s is not None else None,
            }
            summaries[response_id] = timing
        return summaries

    def first_index(self, event_type: str, predicate=None) -> int | None:
        for index, event in enumerate(self.events):
            if event.get("type") != event_type:
                continue
            if predicate is not None and not predicate(event):
                continue
            return index
        return None

    @staticmethod
    def _event_response_id(event: dict[str, object]) -> str | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        response = event.get("response")
        if isinstance(response, dict):
            response_id = response.get("id")
            if isinstance(response_id, str) and response_id:
                return response_id
        return None

    @staticmethod
    def _event_item_id(event: dict[str, object]) -> str | None:
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            return item_id
        item = event.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                return item_id
        return None

    def first_response_lifecycle_indices(self) -> dict[str, int]:
        response_created_index = self.first_index("response.created")
        if response_created_index is None:
            return {}
        response_id = self._event_response_id(self.events[response_created_index])
        if not response_id:
            return {}
        item_id = f"item_{response_id}"
        indices: dict[str, int] = {"response.created": response_created_index}
        for event_type in (
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ):
            index = self.first_index(
                event_type,
                lambda event, event_type=event_type: (
                    self._event_item_id(event) == item_id
                    if event_type == "conversation.item.added"
                    else self._event_response_id(event) == response_id
                ),
            )
            if index is None and event_type not in {
                "response.speak",
                "response.audio.delta",
                "response.audio.done",
            }:
                return {}
            if index is not None:
                indices[event_type] = index
        return indices

    def event_order_ok(self, *, require_input_commit: bool = True) -> bool:
        if not self.events or self.events[0].get("type") != "session.created":
            return False
        first_commit_index = self.first_index("input_audio_buffer.committed")
        first_response_index = self.first_index("response.created")
        if first_response_index is None:
            return False
        if require_input_commit and (first_commit_index is None or first_commit_index > first_response_index):
            return False
        indices_by_type = self.first_response_lifecycle_indices()
        if not indices_by_type:
            return False
        required_types = [
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        if any(event_type not in indices_by_type for event_type in required_types):
            return False
        ordered_types = [
            "response.created",
            "conversation.item.added",
            "response.output_item.added",
            "response.content_part.added",
            "response.speak",
            "response.audio.delta",
            "response.audio.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ]
        if "response.audio.delta" not in indices_by_type:
            listen_index = self.first_index(
                "response.listen",
                lambda event: (
                    isinstance(event.get("response"), dict)
                    and isinstance(event["response"].get("metadata"), dict)
                    and event["response"]["metadata"].get("model_listen") is True
                ),
            )
            return (
                listen_index is not None
                and indices_by_type["response.created"] < listen_index < indices_by_type["response.done"]
            )
        if any(event_type not in indices_by_type for event_type in ordered_types):
            return False
        indices = [indices_by_type[event_type] for event_type in ordered_types]
        return indices == sorted(indices)

    def model_policy_event_order_ok(
        self,
        *,
        expected_turns: int,
        require_input_commit: bool = True,
    ) -> bool:
        if not self.events or self.events[0].get("type") != "session.created":
            return False
        commit_indices = [
            index for index, event in enumerate(self.events) if event.get("type") == "input_audio_buffer.committed"
        ]
        decision_indices = [
            index
            for index, event in enumerate(self.events)
            if event.get("type") == "response.created"
            or (
                event.get("type") == "response.listen"
                and isinstance(event.get("response"), dict)
                and isinstance(event["response"].get("metadata"), dict)
                and event["response"]["metadata"].get("model_listen") is True
            )
        ]
        first_input_index = self.first_index("input_audio_buffer.speech_started")
        if first_input_index is None:
            first_input_index = commit_indices[0] if commit_indices else None
        if first_input_index is None and not require_input_commit:
            first_input_index = 0
        if (
            first_input_index is None
            or (require_input_commit and len(commit_indices) < expected_turns)
            or len(decision_indices) < expected_turns
        ):
            return False
        return decision_indices[0] > first_input_index

    def model_speak_before_audio_ok(self) -> bool:
        speak_index = self.first_index("response.speak")
        audio_index = self.first_index("response.audio.delta")
        return speak_index is not None and audio_index is not None and speak_index < audio_index

    def response_done(self, response_id: str | None) -> bool:
        if not response_id:
            return False
        return any(
            event.get("type") == "response.done" and self._event_response_id(event) == response_id
            for event in self.events
        )

    def response_audio_delta_count(self, response_id: str | None) -> int:
        if not response_id:
            return 0
        return sum(
            1
            for event in self.events
            if event.get("type") == "response.audio.delta" and self._event_response_id(event) == response_id
        )

    def response_playback_sent_ms(self, response_id: str | None) -> int:
        if not response_id:
            return 0
        for event in reversed(self.events):
            if event.get("type") != "response.done" or self._event_response_id(event) != response_id:
                continue
            response = event.get("response")
            metadata = response.get("metadata") if isinstance(response, dict) else None
            playback = metadata.get("playback") if isinstance(metadata, dict) else event.get("playback")
            sent_ms = playback.get("sent_ms") if isinstance(playback, dict) else None
            if isinstance(sent_ms, int | float):
                return max(0, int(sent_ms))
        return 0

    def response_playback_history_committed(self, response_id: str) -> bool:
        item_id = f"item_{response_id}"
        for event in self.events:
            if event.get("type") != "playback.acknowledged":
                continue
            payload = event.get("event")
            if not isinstance(payload, dict):
                continue
            if payload.get("item_id") == item_id and payload.get("history_committed") is True:
                return True
        return False

    def response_transcript_delta(self, response_id: str) -> str:
        return "".join(
            str(event.get("delta", ""))
            for event in self.events
            if event.get("type") == "response.audio_transcript.delta" and self._event_response_id(event) == response_id
        )

    def response_transcript_done(self, response_id: str) -> list[str]:
        return [
            str(event.get("transcript", ""))
            for event in self.events
            if event.get("type") == "response.audio_transcript.done" and self._event_response_id(event) == response_id
        ]

    def completed_response_ids(self) -> list[str]:
        response_ids: list[str] = []
        for event in self.events:
            if event.get("type") != "response.done":
                continue
            response_id = self._event_response_id(event)
            if isinstance(response_id, str):
                response_ids.append(response_id)
        return response_ids

    def stale_audio_delta_count(self) -> int:
        cancelled_epochs_by_index: list[tuple[int, int]] = []
        for index, event in enumerate(self.events):
            if event.get("type") != "response.done":
                continue
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                continue
            metadata = response.get("metadata")
            if not isinstance(metadata, dict):
                continue
            cancelled_epoch = metadata.get("cancelled_epoch")
            if isinstance(cancelled_epoch, int):
                cancelled_epochs_by_index.append((index, cancelled_epoch))
        if not cancelled_epochs_by_index:
            return 0
        stale = 0
        for index, event in enumerate(self.events):
            if event.get("type") != "response.audio.delta":
                continue
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue
            event_epoch = metadata.get("epoch")
            for cancel_index, cancelled_epoch in cancelled_epochs_by_index:
                if index > cancel_index and event_epoch == cancelled_epoch:
                    stale += 1
                    break
        return stale


def _session_update_event(args: argparse.Namespace) -> dict[str, object]:
    session_payload: dict[str, object] = {
        "model": args.model,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": args.output_audio_format,
        "turn_detection": None,
        "overlap_policy": "listen_only",
        "overlap_short_ack_ms": args.short_ack_ms,
        "playback_commit_policy": "ack_only",
        "extra_body": {
            "auto_response": True,
            "minicpmo45_native_duplex": True,
            "force_listen_count": 0,
        },
    }
    temperature = getattr(args, "temperature", None)
    if temperature is None and getattr(args, "validation_mode", None) == "response-required":
        temperature = 0.0
    if temperature is not None:
        session_payload["temperature"] = temperature
    ref_audio = _ref_audio_data_url(getattr(args, "ref_audio", None))
    if ref_audio is not None:
        session_payload["ref_audio"] = ref_audio
    event: dict[str, object] = {
        "type": "session.update",
        "session": session_payload,
    }
    session_id = getattr(args, "session_id", None)
    if session_id:
        session_payload["session_id"] = session_id
    return event


def _turn_input_paths(primary: Path, additional: list[str], *, turns: int) -> list[Path]:
    turn_count = max(1, turns)
    if not additional:
        return [primary] * turn_count
    if len(additional) != turn_count - 1:
        raise ValueError(
            f"provide exactly one --turn-input-wav for each turn after the first "
            f"(expected {turn_count - 1}, got {len(additional)})"
        )
    return [primary, *(Path(path) for path in additional)]


def _turn_inputs_are_distinct(paths: list[Path], pcm16_inputs: list[bytes]) -> bool:
    if len(paths) != len(pcm16_inputs):
        return False
    distinct_paths = len({str(path.resolve()) for path in paths}) == len(paths)
    distinct_audio = len({hashlib.sha256(pcm16).digest() for pcm16 in pcm16_inputs}) == len(pcm16_inputs)
    return distinct_paths and distinct_audio


def _turn_durations(
    explicit: list[int],
    *,
    turns: int,
    first_turn_ms: int,
) -> list[int | None]:
    turn_count = max(1, turns)
    if not explicit:
        return [first_turn_ms, *([min(first_turn_ms, 1200)] * (turn_count - 1))]
    if len(explicit) != turn_count:
        raise ValueError(
            f"provide exactly one --turn-duration-ms for each turn (expected {turn_count}, got {len(explicit)})"
        )
    if any(duration_ms < 0 for duration_ms in explicit):
        raise ValueError("--turn-duration-ms values must be non-negative")
    return [None if duration_ms == 0 else duration_ms for duration_ms in explicit]


def _turn_transcripts(first: str, *, turns: int) -> list[str]:
    turn_count = max(1, turns)
    transcripts = [first, "继续", "再说一次"][:turn_count]
    transcripts.extend(f"turn-{turn_index + 1}" for turn_index in range(len(transcripts), turn_count))
    return transcripts


def _write_wav(path: Path, pcm_bytes: bytes, *, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)


def _write_demo_artifacts(state: DemoState, output_dir: Path, *, output_audio_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if state.audio_deltas and output_audio_format == "pcm16":
        _write_wav(
            output_dir / "joined_audio_deltas.wav",
            b"".join(state.audio_deltas),
            sample_rate_hz=state.output_sample_rate_hz,
        )
    elif state.audio_deltas:
        (output_dir / "joined_audio_deltas.bin").write_bytes(b"".join(state.audio_deltas))
    for index, response_id in enumerate(state.response_ids, start=1):
        response_audio = state.response_audio_deltas.get(response_id, [])
        if not response_audio:
            continue
        payload = b"".join(response_audio)
        if output_audio_format == "pcm16":
            _write_wav(
                output_dir / f"response_{index:02d}.wav",
                payload,
                sample_rate_hz=state.output_sample_rate_hz,
            )
        else:
            (output_dir / f"response_{index:02d}.bin").write_bytes(payload)
    (output_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in state.events) + "\n",
        encoding="utf-8",
    )


def _pcm16_silence(duration_ms: int) -> bytes:
    samples = PCM16_SAMPLE_RATE * max(0, duration_ms) // 1000
    return b"\x00\x00" * samples


def _pcm16_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    return pcm16[: min(len(pcm16), byte_count)]


def _pcm16_active_slice(pcm16: bytes, duration_ms: int) -> bytes:
    byte_count = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * max(1, duration_ms) // 1000
    byte_count = min(len(pcm16), max(PCM16_BYTES_PER_SAMPLE, byte_count))
    byte_count -= byte_count % PCM16_BYTES_PER_SAMPLE
    if byte_count <= 0:
        return _pcm16_slice(pcm16, duration_ms)
    step = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * 20 // 1000, PCM16_BYTES_PER_SAMPLE)
    step -= step % PCM16_BYTES_PER_SAMPLE
    best_offset = 0
    best_energy = -1.0
    for offset in range(0, max(1, len(pcm16) - byte_count + 1), max(PCM16_BYTES_PER_SAMPLE, step)):
        chunk = pcm16[offset : offset + byte_count]
        samples = array.array("h")
        samples.frombytes(chunk)
        if not samples:
            continue
        energy = sum(abs(sample) for sample in samples) / len(samples)
        if energy > best_energy:
            best_energy = energy
            best_offset = offset
    return pcm16[best_offset : best_offset + byte_count]


def _select_turn_audio(pcm16: bytes, duration_ms: int | None) -> bytes:
    if duration_ms is None:
        return pcm16
    return _pcm16_active_slice(pcm16, duration_ms)


def _canonical_transcript(text: str) -> str:
    return "".join(text.split())


def _reuses_previous_turn_tail(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if len(previous) >= 4 and previous in current:
        return True
    max_overlap = min(len(previous), len(current))
    return any(previous.endswith(current[:overlap]) for overlap in range(max_overlap, 2, -1))


def _has_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip("\"'”’）)]} ")
    return bool(stripped) and (len(stripped) <= 4 or stripped[-1] in "。！？!?…")


def _all_audio_responses_have_transcript(state: DemoState, response_ids: list[str]) -> bool:
    return all(
        state.response_audio_delta_count(response_id) == 0
        or bool(_canonical_transcript(state.response_transcript_delta(response_id)))
        for response_id in response_ids
    )


def _all_response_playback_history_committed(state: DemoState, response_ids: list[str]) -> bool:
    return all(
        state.response_playback_history_committed(response_id)
        for response_id in response_ids
        if state.response_playback_sent_ms(response_id) > 0
    )


def _response_cardinality_ok(
    completed_response_ids: list[str],
    *,
    expected_turns: int,
    validation_mode: str,
) -> bool:
    if validation_mode == "response-required":
        return len(completed_response_ids) == expected_turns
    return True


def _requires_cross_turn_independence(
    *,
    validation_mode: str,
    distinct_inputs_required: bool,
) -> bool:
    return distinct_inputs_required and validation_mode == "response-required"


def _input_transcription_ok(count: int, *, transcript_hints_enabled: bool) -> bool:
    return count > 0 if transcript_hints_enabled else True


def _unexpected_error_events(state: DemoState) -> list[dict[str, object]]:
    return [event for event in state.events if event.get("type") == "error"]


def _evaluate_response_speak_contract(state: DemoState) -> dict[str, object]:
    speak_counts: dict[str, int] = {}
    invalid_response_speak_count = 0
    text_bearing_response_speak_count = 0

    for event in state.events:
        if event.get("type") != "response.speak":
            continue
        response_id = state._event_response_id(event)
        if not response_id or response_id not in state.response_ids:
            invalid_response_speak_count += 1
        else:
            speak_counts[response_id] = speak_counts.get(response_id, 0) + 1
        if "text" in event:
            text_bearing_response_speak_count += 1

    duplicate_response_speak_ids = sorted(response_id for response_id, count in speak_counts.items() if count > 1)
    return {
        "response_speak_contract_ok": not duplicate_response_speak_ids
        and invalid_response_speak_count == 0
        and text_bearing_response_speak_count == 0,
        "duplicate_response_speak_ids": duplicate_response_speak_ids,
        "invalid_response_speak_count": invalid_response_speak_count,
        "text_bearing_response_speak_count": text_bearing_response_speak_count,
    }


def _evaluate_transcript_integrity(
    state: DemoState,
    response_ids: list[str],
    *,
    expected_empty_response_ids: set[str],
    require_cross_turn_independence: bool,
    require_terminal_punctuation: bool = False,
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    transcripts: list[str] = []
    transcript_delta_done_ok = True
    empty_turns_ok = True
    nonempty_audio_has_transcript_ok = True
    terminal_punctuation_ok = True

    for response_id in response_ids:
        transcript = state.response_transcript_delta(response_id)
        done_transcripts = state.response_transcript_done(response_id)
        canonical_transcript = _canonical_transcript(transcript)
        response_delta_done = (
            len(done_transcripts) == 1 and _canonical_transcript(done_transcripts[0]) == canonical_transcript
        ) or (not done_transcripts and not canonical_transcript)
        expected_empty = response_id in expected_empty_response_ids
        audio_delta_count = state.response_audio_delta_count(response_id)
        response_empty_ok = not expected_empty or (not canonical_transcript and audio_delta_count == 0)
        response_audio_has_transcript = expected_empty or audio_delta_count == 0 or bool(canonical_transcript)
        response_terminal_punctuation_ok = (
            expected_empty
            or not canonical_transcript
            or not require_terminal_punctuation
            or _has_terminal_punctuation(canonical_transcript)
        )
        transcript_delta_done_ok = transcript_delta_done_ok and response_delta_done
        empty_turns_ok = empty_turns_ok and response_empty_ok
        nonempty_audio_has_transcript_ok = nonempty_audio_has_transcript_ok and response_audio_has_transcript
        terminal_punctuation_ok = terminal_punctuation_ok and response_terminal_punctuation_ok
        transcripts.append(canonical_transcript)
        details.append(
            {
                "response_id": response_id,
                "transcript": transcript,
                "delta_done_ok": response_delta_done,
                "expected_empty": expected_empty,
                "empty_ok": response_empty_ok,
                "audio_delta_count": audio_delta_count,
                "audio_has_transcript": response_audio_has_transcript,
                "terminal_punctuation_ok": response_terminal_punctuation_ok,
            }
        )

    cross_turn_independent_ok = True
    if require_cross_turn_independence:
        for index, current in enumerate(transcripts):
            for previous in transcripts[:index]:
                if _reuses_previous_turn_tail(previous, current):
                    cross_turn_independent_ok = False
                    break
            if not cross_turn_independent_ok:
                break

    return {
        "transcript_delta_done_ok": transcript_delta_done_ok,
        "cross_turn_independent_ok": cross_turn_independent_ok,
        "empty_turns_ok": empty_turns_ok,
        "nonempty_audio_has_transcript_ok": nonempty_audio_has_transcript_ok,
        "terminal_punctuation_ok": terminal_punctuation_ok,
        "transcript_integrity": details,
    }


async def _reader(ws, state: DemoState, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if isinstance(event, dict):
                state.add(event)
    except ConnectionClosed:
        return


async def _send_pcm16(
    ws,
    pcm16: bytes,
    *,
    chunk_ms: int,
    realtime_delay: bool,
    hints: dict[str, object] | None = None,
    first_chunk_hints: dict[str, object] | None = None,
    on_model_unit_ready=None,
    frame_b64: str | None = None,
) -> None:
    hints = hints or {}
    first_chunk_hints = first_chunk_hints or {}
    chunk_bytes = max(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000, PCM16_BYTES_PER_SAMPLE)
    model_unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    audio_ms = 0
    next_model_unit_bytes = model_unit_bytes
    frames_sent = 0
    for offset in range(0, len(pcm16), chunk_bytes):
        chunk = pcm16[offset : offset + chunk_bytes]
        duration_ms = int(len(chunk) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE) * 1000)
        audio_ms += duration_ms
        chunk_hints = dict(hints)
        if offset == 0:
            chunk_hints.update(first_chunk_hints)
        if frame_b64 is not None and audio_ms > frames_sent * 1000:
            # Omni duplex cadence: one camera frame per 1 s of audio.
            chunk_hints["video_frames"] = [frame_b64]
            frames_sent += 1
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "input_audio_format": "pcm16",
                    "sample_rate_hz": PCM16_SAMPLE_RATE,
                    "duration_ms": duration_ms,
                    "audio_end_ms": audio_ms,
                    **chunk_hints,
                }
            )
        )
        while offset + len(chunk) >= next_model_unit_bytes:
            next_model_unit_bytes += model_unit_bytes
            if on_model_unit_ready is None:
                continue
            should_continue = on_model_unit_ready()
            if inspect.isawaitable(should_continue):
                should_continue = await should_continue
            if should_continue is False:
                return
        if realtime_delay:
            await asyncio.sleep(duration_ms / 1000)


async def _wait_for(
    state: DemoState,
    predicate,
    *,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


async def _wait_for_model_policy_outcome(
    state: DemoState,
    *,
    before_created: int,
    before_model_listen: int,
    timeout_s: float,
    listen_settle_s: float,
    label: str,
) -> str:
    deadline = time.monotonic() + timeout_s
    listen_deadline: float | None = None
    while time.monotonic() < deadline:
        if state.count("response.created") > before_created:
            return "speak"
        if state.model_listen_count > before_model_listen:
            if listen_deadline is None:
                listen_deadline = time.monotonic() + max(0.0, listen_settle_s)
            if time.monotonic() >= listen_deadline:
                return "listen"
        await asyncio.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {label}")


async def _send_clean_turn(
    ws,
    state: DemoState,
    pcm16: bytes,
    *,
    transcript: str,
    duration_ms: int | None,
    chunk_ms: int,
    timeout_s: float,
    require_audio: bool,
    validation_mode: str,
    send_transcript_hint: bool = True,
    realtime_input: bool = False,
    model_policy_settle_s: float = 2.0,
    commit_input: bool = True,
    frame_b64: str | None = None,
) -> tuple[str | None, str]:
    before_created = state.count("response.created")
    before_model_listen = state.model_listen_count

    def eligible_response_id() -> str | None:
        for response_id in state.response_ids[before_created:]:
            if not state.response_done(response_id):
                continue
            if require_audio and state.response_audio_delta_count(response_id) == 0:
                continue
            if require_audio and not _canonical_transcript(state.response_transcript_delta(response_id)):
                continue
            return response_id
        return None

    await _send_pcm16(
        ws,
        _select_turn_audio(pcm16, duration_ms),
        chunk_ms=chunk_ms,
        realtime_delay=realtime_input,
        hints={"transcript": transcript} if send_transcript_hint else {},
        frame_b64=frame_b64,
    )
    if commit_input:
        state.input_commit_sent_at_s.append(time.monotonic())
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
    if validation_mode == "model-policy":
        outcome = await _wait_for_model_policy_outcome(
            state,
            before_created=before_created,
            before_model_listen=before_model_listen,
            timeout_s=timeout_s,
            listen_settle_s=model_policy_settle_s,
            label=f"{transcript} model speak/listen decision",
        )
        if outcome == "listen":
            return None, "listen"
    else:
        await _wait_for(
            state,
            lambda: state.count("response.created") > before_created,
            timeout_s=timeout_s,
            label=f"{transcript} response.created",
        )
        await _wait_for(
            state,
            lambda: eligible_response_id() is not None,
            timeout_s=timeout_s,
            label=f"{transcript} completed response",
        )
    response_id = (
        eligible_response_id()
        if validation_mode == "response-required"
        else (state.response_ids[before_created] if len(state.response_ids) > before_created else None)
    )
    if require_audio:
        await _wait_for(
            state,
            lambda: state.response_audio_delta_count(response_id) > 0,
            timeout_s=timeout_s,
            label=f"{transcript} response.audio.delta",
        )
    await _wait_for(
        state,
        lambda: state.response_done(response_id),
        timeout_s=timeout_s,
        label=f"{transcript} response.done",
    )
    if not isinstance(response_id, str):
        raise TimeoutError(f"Missing response id for {transcript}")
    await _ack_response_playback(
        ws,
        state,
        response_id,
        timeout_s=timeout_s,
        label=transcript,
    )
    return response_id, "speak"


async def _ack_response_playback(
    ws,
    state: DemoState,
    response_id: str,
    *,
    timeout_s: float,
    label: str,
) -> None:
    played_ms = state.response_playback_sent_ms(response_id)
    if played_ms <= 0:
        return
    before_ack = state.playback_ack_count
    await ws.send(
        json.dumps(
            {
                "type": "playback.ack",
                "response_id": response_id,
                "item_id": f"item_{response_id}",
                "played_ms": played_ms,
                "committed_ms": played_ms,
            }
        )
    )
    await _wait_for(
        state,
        lambda: state.playback_ack_count > before_ack and state.response_playback_history_committed(response_id),
        timeout_s=timeout_s,
        label=f"{label} playback history committed",
    )


async def _ack_all_completed_response_playback(
    ws,
    state: DemoState,
    *,
    timeout_s: float,
) -> None:
    for response_id in state.completed_response_ids():
        if state.response_playback_sent_ms(response_id) <= 0:
            continue
        if state.response_playback_history_committed(response_id):
            continue
        await _ack_response_playback(
            ws,
            state,
            response_id,
            timeout_s=timeout_s,
            label=response_id,
        )


async def _drain_model_policy_responses(
    ws,
    state: DemoState,
    *,
    timeout_s: float,
    settle_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    settle_s = max(0.0, settle_s)
    observed_created_count = state.count("response.created")
    quiet_since: float | None = None

    while True:
        remaining_s = deadline - time.monotonic()
        active_response_ids = [
            response_id for response_id in state.response_ids if not state.response_done(response_id)
        ]
        if remaining_s <= 0:
            raise TimeoutError(f"Timed out draining model-policy responses; active response ids: {active_response_ids}")

        await _ack_all_completed_response_playback(
            ws,
            state,
            timeout_s=remaining_s,
        )

        now = time.monotonic()
        created_count = state.count("response.created")
        if created_count != observed_created_count:
            observed_created_count = created_count
            quiet_since = None

        active_response_ids = [
            response_id for response_id in state.response_ids if not state.response_done(response_id)
        ]
        if active_response_ids:
            quiet_since = None
        else:
            if quiet_since is None:
                quiet_since = now
            if now - quiet_since >= settle_s:
                return

        await asyncio.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


def _event_index_for_response(state: DemoState, event_type: str, response_id: str) -> int | None:
    return state.first_index(
        event_type,
        lambda event: state._event_response_id(event) == response_id,
    )


def _nth_event_index(state: DemoState, event_type: str, occurrence: int) -> int | None:
    seen = 0
    for index, event in enumerate(state.events):
        if event.get("type") != event_type:
            continue
        seen += 1
        if seen == occurrence:
            return index
    return None


def _continuous_overlap_terminal_is_outcome(
    state: DemoState,
    *,
    validation_mode: str,
    model_unit_ready_while_active: bool,
    before_created: int,
    before_model_listen: int,
) -> bool:
    """Accept current-response termination as a model-policy overlap outcome."""
    return (
        validation_mode == "model-policy"
        and model_unit_ready_while_active
        and state.count("response.created") == before_created
        and state.model_listen_count == before_model_listen
    )


async def _send_listen_only_overlap_pair(
    ws,
    state: DemoState,
    first_pcm16: bytes,
    second_pcm16: bytes,
    *,
    transcripts: tuple[str, str],
    durations_ms: tuple[int | None, int | None],
    chunk_ms: int,
    timeout_s: float,
    send_transcript_hint: bool,
    realtime_input: bool,
    model_policy_settle_s: float,
    validation_mode: str,
    silence_ms: int,
    frame_b64: str | None = None,
) -> tuple[list[str | None], list[str], bool]:
    if not realtime_input:
        raise ValueError("listen-only-overlap requires --realtime-input")

    before_first_created = state.count("response.created")
    observed_first_created = before_first_created
    observed_first_listen = state.model_listen_count

    async def continue_until_first_speak() -> bool:
        nonlocal observed_first_created, observed_first_listen
        outcome = await _wait_for_model_policy_outcome(
            state,
            before_created=observed_first_created,
            before_model_listen=observed_first_listen,
            timeout_s=timeout_s,
            listen_settle_s=model_policy_settle_s,
            label=f"{transcripts[0]} model-unit speak/listen decision",
        )
        observed_first_created = state.count("response.created")
        observed_first_listen = state.model_listen_count
        return outcome != "speak"

    first_input = _select_turn_audio(first_pcm16, durations_ms[0]) + _pcm16_silence(silence_ms)
    await _send_pcm16(
        ws,
        first_input,
        chunk_ms=chunk_ms,
        realtime_delay=True,
        hints={"transcript": transcripts[0]} if send_transcript_hint else {},
        on_model_unit_ready=continue_until_first_speak,
        frame_b64=frame_b64,
    )
    state.input_commit_sent_at_s.append(time.monotonic())
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
    await _wait_for(
        state,
        lambda: state.count("response.created") > before_first_created,
        timeout_s=timeout_s,
        label=f"{transcripts[0]} response.created",
    )
    first_response_id = state.response_ids[before_first_created]
    await _wait_for(
        state,
        lambda: state.response_audio_delta_count(first_response_id) > 0,
        timeout_s=timeout_s,
        label=f"{transcripts[0]} response.audio.delta",
    )
    if state.response_done(first_response_id):
        raise RuntimeError("first response completed before overlap input could start")

    before_second_created = state.count("response.created")
    before_second_listen = state.model_listen_count
    overlap_started_while_active = not state.response_done(first_response_id)
    model_unit_ready_while_active = False

    def record_model_unit_ready() -> None:
        nonlocal model_unit_ready_while_active
        model_unit_ready_while_active = not state.response_done(first_response_id)

    await _send_pcm16(
        ws,
        _select_turn_audio(second_pcm16, durations_ms[1]),
        chunk_ms=chunk_ms,
        realtime_delay=True,
        hints={"transcript": transcripts[1]} if send_transcript_hint else {},
        on_model_unit_ready=record_model_unit_ready,
        frame_b64=frame_b64,
    )
    state.input_commit_sent_at_s.append(time.monotonic())
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

    await _wait_for(
        state,
        lambda: state.response_done(first_response_id),
        timeout_s=timeout_s,
        label=f"{transcripts[0]} response.done",
    )
    await _ack_response_playback(
        ws,
        state,
        first_response_id,
        timeout_s=timeout_s,
        label=transcripts[0],
    )

    if _continuous_overlap_terminal_is_outcome(
        state,
        validation_mode=validation_mode,
        model_unit_ready_while_active=model_unit_ready_while_active,
        before_created=before_second_created,
        before_model_listen=before_second_listen,
    ):
        second_outcome = "continuous_terminal"
    else:
        second_outcome = await _wait_for_model_policy_outcome(
            state,
            before_created=before_second_created,
            before_model_listen=before_second_listen,
            timeout_s=timeout_s,
            listen_settle_s=model_policy_settle_s,
            label=f"{transcripts[1]} model speak/listen decision",
        )
    second_response_id: str | None = None
    if second_outcome == "speak":
        second_response_id = state.response_ids[before_second_created]
        await _wait_for(
            state,
            lambda: state.response_done(second_response_id),
            timeout_s=timeout_s,
            label=f"{transcripts[1]} response.done",
        )
        await _ack_response_playback(
            ws,
            state,
            second_response_id,
            timeout_s=timeout_s,
            label=transcripts[1],
        )

    second_speech_index = _nth_event_index(state, "input_audio_buffer.speech_started", 2)
    first_done_index = _event_index_for_response(state, "response.done", first_response_id)
    overlap_ok = (
        overlap_started_while_active
        and model_unit_ready_while_active
        and second_speech_index is not None
        and first_done_index is not None
        and second_speech_index < first_done_index
    )
    return [first_response_id, second_response_id], ["speak", second_outcome], overlap_ok


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    demo_frame_b64: str | None = None
    if getattr(args, "frame_image", None):
        demo_frame_b64 = base64.b64encode(Path(args.frame_image).read_bytes()).decode("ascii")
    turn_input_paths = _turn_input_paths(
        Path(args.input_wav),
        list(getattr(args, "turn_input_wav", []) or []),
        turns=args.turns,
    )
    turn_pcm16 = [_read_wav_pcm16(path) for path in turn_input_paths]
    if any(not pcm16 for pcm16 in turn_pcm16):
        raise ValueError("input WAV has no audio")
    turn_durations = _turn_durations(
        list(getattr(args, "turn_duration_ms", []) or []),
        turns=args.turns,
        first_turn_ms=args.first_turn_ms,
    )
    expected_empty_turns = set(getattr(args, "expect_empty_turn", []) or [])
    invalid_empty_turns = sorted(
        turn_number for turn_number in expected_empty_turns if turn_number < 1 or turn_number > max(1, args.turns)
    )
    if invalid_empty_turns:
        raise ValueError(
            f"--expect-empty-turn values are 1-based and must refer to an existing turn: {invalid_empty_turns}"
        )
    distinct_turn_inputs = _turn_inputs_are_distinct(turn_input_paths, turn_pcm16)
    if getattr(args, "require_distinct_inputs", False) and not distinct_turn_inputs:
        raise ValueError("--require-distinct-inputs requires a different WAV path and audio payload for every turn")
    url = _url_with_model(
        args.url,
        args.model,
        autostart=False if getattr(args, "ref_audio", None) else None,
        session_id=getattr(args, "session_id", None),
    )
    state = DemoState()
    stop = asyncio.Event()
    output_dir = Path(args.output_dir)
    turn_response_ids: list[str | None] = []
    turn_outcomes: list[str] = []
    validation_mode = getattr(args, "validation_mode", "response-required")
    scenario = getattr(args, "scenario", "sequential")
    transcript_hints_enabled = not getattr(args, "omit_transcript_hints", False)
    realtime_input = getattr(args, "realtime_input", False)
    continuous_input = getattr(args, "continuous_input", False)
    listen_only_overlap_ok = scenario != "listen-only-overlap"
    if scenario == "listen-only-overlap" and args.turns < 2:
        raise ValueError("listen-only-overlap requires at least two turns")

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        reader = asyncio.create_task(_reader(ws, state, stop))
        try:
            await ws.send(json.dumps(_session_update_event(args)))
            await _wait_for(state, lambda: state.count("session.created") > 0, timeout_s=20, label="session.created")

            turn_transcripts = _turn_transcripts(args.first_turn_transcript, turns=args.turns)
            turn_specs = list(zip(turn_transcripts, turn_durations, strict=True))
            if scenario == "listen-only-overlap":
                turn_response_ids, turn_outcomes, listen_only_overlap_ok = await _send_listen_only_overlap_pair(
                    ws,
                    state,
                    turn_pcm16[0],
                    turn_pcm16[1],
                    transcripts=(turn_specs[0][0], turn_specs[1][0]),
                    durations_ms=(turn_specs[0][1], turn_specs[1][1]),
                    chunk_ms=args.chunk_ms,
                    timeout_s=args.timeout_s,
                    send_transcript_hint=transcript_hints_enabled,
                    realtime_input=realtime_input,
                    model_policy_settle_s=max(0.0, args.model_policy_settle_ms / 1000),
                    validation_mode=validation_mode,
                    silence_ms=max(0, args.silence_ms),
                    frame_b64=demo_frame_b64,
                )
                for turn_index in range(2, len(turn_specs)):
                    transcript, duration_ms = turn_specs[turn_index]
                    response_id, outcome = await _send_clean_turn(
                        ws,
                        state,
                        turn_pcm16[turn_index],
                        transcript=transcript,
                        duration_ms=duration_ms,
                        chunk_ms=args.chunk_ms,
                        timeout_s=args.timeout_s,
                        require_audio=args.require_audio and (turn_index + 1) not in expected_empty_turns,
                        validation_mode=validation_mode,
                        send_transcript_hint=transcript_hints_enabled,
                        realtime_input=realtime_input,
                        model_policy_settle_s=max(0.0, args.model_policy_settle_ms / 1000),
                        commit_input=not continuous_input,
                        frame_b64=demo_frame_b64,
                    )
                    turn_response_ids.append(response_id)
                    turn_outcomes.append(outcome)
            else:
                for turn_index, (transcript, duration_ms) in enumerate(turn_specs):
                    response_id, outcome = await _send_clean_turn(
                        ws,
                        state,
                        turn_pcm16[turn_index],
                        transcript=transcript,
                        duration_ms=duration_ms,
                        chunk_ms=args.chunk_ms,
                        timeout_s=args.timeout_s,
                        require_audio=args.require_audio and (turn_index + 1) not in expected_empty_turns,
                        validation_mode=validation_mode,
                        send_transcript_hint=transcript_hints_enabled,
                        realtime_input=realtime_input,
                        model_policy_settle_s=max(0.0, args.model_policy_settle_ms / 1000),
                        commit_input=not continuous_input,
                        frame_b64=demo_frame_b64,
                    )
                    turn_response_ids.append(response_id)
                    turn_outcomes.append(outcome)

            if validation_mode == "model-policy":
                await _drain_model_policy_responses(
                    ws,
                    state,
                    timeout_s=args.timeout_s,
                    settle_s=max(0.0, args.model_policy_settle_ms / 1000),
                )
            await _ack_all_completed_response_playback(
                ws,
                state,
                timeout_s=args.timeout_s,
            )
            await ws.send(json.dumps({"type": "session.close"}))
            await _wait_for(state, lambda: state.count("session.closed") > 0, timeout_s=20, label="session.closed")
        finally:
            stop.set()
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
            _write_demo_artifacts(state, output_dir, output_audio_format=args.output_audio_format)

    overlap_barge_in = any(decision.get("action") == "barge_in" for decision in state.overlap_decisions)
    expected_turns = max(1, args.turns)
    event_order_ok = (
        state.model_policy_event_order_ok(
            expected_turns=expected_turns,
            require_input_commit=not continuous_input,
        )
        if validation_mode == "model-policy"
        else state.event_order_ok(require_input_commit=not continuous_input)
    )
    input_transcription_ok = _input_transcription_ok(
        state.input_transcription_count,
        transcript_hints_enabled=transcript_hints_enabled and not continuous_input,
    )
    model_speak_event_ok = state.model_speak_before_audio_ok()
    realtime_audio_lifecycle_ok = state.count("response.audio.delta") > 0 and state.count("response.audio.done") > 0
    completed_response_ids = state.completed_response_ids()
    observed_turn_response_ids = [response_id for response_id in turn_response_ids if isinstance(response_id, str)]
    expected_empty_response_ids = {
        response_id
        for turn_number in expected_empty_turns
        if turn_number <= len(turn_response_ids)
        and isinstance((response_id := turn_response_ids[turn_number - 1]), str)
    }
    transcript_integrity = _evaluate_transcript_integrity(
        state,
        observed_turn_response_ids,
        expected_empty_response_ids=expected_empty_response_ids,
        require_cross_turn_independence=_requires_cross_turn_independence(
            validation_mode=validation_mode,
            distinct_inputs_required=getattr(args, "require_distinct_inputs", False),
        ),
        require_terminal_punctuation=getattr(args, "require_distinct_inputs", False),
    )
    response_speak_contract = _evaluate_response_speak_contract(state)
    expected_audio_turns = expected_turns - len(expected_empty_turns)
    lifecycle_counts_ok = (
        state.count("response.created") == state.count("response.done")
        and len(state.response_ids) == len(completed_response_ids)
        and len(completed_response_ids) == len(set(completed_response_ids))
        and state.count("response.audio.done") <= state.count("response.done")
        and _response_cardinality_ok(
            completed_response_ids,
            expected_turns=expected_turns,
            validation_mode=validation_mode,
        )
    )
    turn_outcomes_ok = len(turn_outcomes) == expected_turns and (
        validation_mode == "model-policy" or all(outcome == "speak" for outcome in turn_outcomes)
    )
    clean_turn_audio_ok = all(
        state.response_audio_delta_count(response_id) > 0
        for response_id in observed_turn_response_ids
        if response_id not in expected_empty_response_ids
    )
    playback_history_committed_ok = _all_response_playback_history_committed(
        state,
        completed_response_ids,
    )
    if validation_mode == "response-required":
        full_audio_response_ok = (
            len(observed_turn_response_ids) == expected_audio_turns + len(expected_empty_turns)
            and clean_turn_audio_ok
            and model_speak_event_ok
            and state.model_speak_delta_count > 0
            and realtime_audio_lifecycle_ok
            and state.audio_marks_seen
        )
    else:
        full_audio_response_ok = not observed_turn_response_ids or (
            realtime_audio_lifecycle_ok and state.model_speak_delta_count > 0
        )
    stale_audio_delta_count = state.stale_audio_delta_count()
    terminal_activity_ok = state.count("response.done") > 0 or (
        validation_mode == "model-policy" and state.model_listen_count > 0
    )
    all_audio_responses_have_transcript = (
        validation_mode != "response-required"
        or _all_audio_responses_have_transcript(
            state,
            completed_response_ids,
        )
    )
    unexpected_error_events = _unexpected_error_events(state)
    continuous_input_ok = not continuous_input or state.count("input_audio_buffer.committed") == 0
    result = {
        "ok": terminal_activity_ok
        and state.count("session.closed") > 0
        and state.cancelled_count == 0
        and not overlap_barge_in
        and state.truncate_count == 0
        and event_order_ok
        and input_transcription_ok
        and stale_audio_delta_count == 0
        and lifecycle_counts_ok
        and turn_outcomes_ok
        and full_audio_response_ok,
        "event_counts": {
            event_type: state.count(event_type)
            for event_type in sorted({str(event.get("type")) for event in state.events})
        },
        "audio_delta_count": len(state.audio_deltas),
        "done_count": state.done_count,
        "cancelled_count": state.cancelled_count,
        "listen_count": state.listen_count,
        "model_listen_count": state.model_listen_count,
        "buffering_listen_count": state.buffering_listen_count,
        "model_speak_event_count": state.model_speak_event_count,
        "model_speak_delta_count": state.model_speak_delta_count,
        "playback_ack_count": state.playback_ack_count,
        "playback_history_committed_count": state.playback_history_committed_count,
        "playback_history_committed_ok": playback_history_committed_ok,
        "truncate_count": state.truncate_count,
        "input_transcription_count": state.input_transcription_count,
        "audio_marks_seen": state.audio_marks_seen,
        "overlap_decisions": state.overlap_decisions,
        "overlap_barge_in": overlap_barge_in,
        "event_order_ok": event_order_ok,
        "terminal_activity_ok": terminal_activity_ok,
        "input_transcription_ok": input_transcription_ok,
        "completed_response_ids": completed_response_ids,
        "response_timings": state.response_timing_summaries(),
        "lifecycle_counts_ok": lifecycle_counts_ok,
        "validation_mode": validation_mode,
        "scenario": scenario,
        "listen_only_overlap_ok": listen_only_overlap_ok,
        "transcript_hints_enabled": transcript_hints_enabled,
        "realtime_input": realtime_input,
        "continuous_input": continuous_input,
        "continuous_input_ok": continuous_input_ok,
        "input_chunk_ms": args.chunk_ms,
        "model_policy_settle_ms": args.model_policy_settle_ms,
        "turn_outcomes": turn_outcomes,
        "turn_outcomes_ok": turn_outcomes_ok,
        "clean_turn_audio_ok": clean_turn_audio_ok,
        "full_audio_response_ok": full_audio_response_ok,
        "model_speak_event_ok": model_speak_event_ok,
        "realtime_audio_lifecycle_ok": realtime_audio_lifecycle_ok,
        "stale_audio_delta_count": stale_audio_delta_count,
        "all_audio_responses_have_transcript": all_audio_responses_have_transcript,
        "distinct_turn_inputs": distinct_turn_inputs,
        "error_count": len(unexpected_error_events),
        "errors": unexpected_error_events,
        **response_speak_contract,
        **transcript_integrity,
        "turn_inputs": [
            {
                "path": str(path),
                "source_duration_ms": len(pcm16) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
                "requested_duration_ms": duration_ms,
                "sent_duration_ms": len(_select_turn_audio(pcm16, duration_ms))
                * 1000
                // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
            }
            for path, pcm16, duration_ms in zip(turn_input_paths, turn_pcm16, turn_durations, strict=True)
        ],
        "output_dir": str(output_dir),
    }
    result["ok"] = bool(
        result["ok"]
        and transcript_integrity["transcript_delta_done_ok"]
        and transcript_integrity["cross_turn_independent_ok"]
        and transcript_integrity["empty_turns_ok"]
        and transcript_integrity["nonempty_audio_has_transcript_ok"]
        and transcript_integrity["terminal_punctuation_ok"]
        and all_audio_responses_have_transcript
        and playback_history_committed_ok
        and listen_only_overlap_ok
        and not unexpected_error_events
        and response_speak_contract["response_speak_contract_ok"]
        and continuous_input_ok
        and (distinct_turn_inputs or not getattr(args, "require_distinct_inputs", False))
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--session-id", help="Use an explicit public session ID, including for close/reopen tests.")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--ref-audio", help="Optional WAV used as the MiniCPM-o voice prompt.")
    parser.add_argument(
        "--frame-image",
        default=None,
        help="Optional image file sent as an omni-duplex camera frame (one per 1 s of audio)",
    )
    parser.add_argument(
        "--turn-input-wav",
        action="append",
        default=[],
        help="WAV for each turn after the first; repeat exactly turns-1 times.",
    )
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument(
        "--output-audio-format",
        default="pcm16",
        choices=["pcm16", "wav", "g711_ulaw", "g711_alaw"],
    )
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument(
        "--model-policy-settle-ms",
        type=int,
        default=2000,
        help="Wait for a delayed response.created after model-listen and before closing the session.",
    )
    parser.add_argument("--first-turn-ms", type=int, default=1400)
    parser.add_argument(
        "--turn-duration-ms",
        action="append",
        type=int,
        default=[],
        help="Audio duration for each turn; repeat turns times. Use 0 to send the complete WAV.",
    )
    parser.add_argument("--first-turn-transcript", default="demo input speech")
    parser.add_argument(
        "--omit-transcript-hints",
        action="store_true",
        help="Keep local turn labels but do not send transcript hints to the server.",
    )
    parser.add_argument(
        "--realtime-input",
        action="store_true",
        help="Pace each input chunk according to its audio duration.",
    )
    parser.add_argument(
        "--continuous-input",
        action="store_true",
        help="Model-owned browser mode: stream PCM continuously without input_audio_buffer.commit.",
    )
    parser.add_argument(
        "--scenario",
        choices=["sequential", "listen-only-overlap"],
        default="sequential",
        help="Run sequential turns or stream turn 2 while turn 1 audio is active without interruption.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["model-policy", "response-required"],
        default="response-required",
        help="Allow model listen decisions, or require at least one response for every input turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Stage0 sampling temperature; response-required defaults to deterministic sampling.",
    )
    parser.add_argument(
        "--require-distinct-inputs",
        action="store_true",
        help="Require distinct WAV paths and audio payloads, and reject cross-turn transcript reuse.",
    )
    parser.add_argument(
        "--expect-empty-turn",
        action="append",
        type=int,
        default=[],
        help="1-based turn expected to end without text or audio; repeat for multiple turns.",
    )
    parser.add_argument("--short-ack-ms", type=int, default=350)
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--playback-ack-ms", type=int, default=500)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Fail if the first native response listens instead of producing audio.",
    )
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
