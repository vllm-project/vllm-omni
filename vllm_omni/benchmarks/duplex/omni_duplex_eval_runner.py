# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-sample duplex generation.

Model-native duplex models (MiniCPM-o) hear the whole clip and are committed
once at the end. Turn-based models (Qwen3-Omni) run with ``server_vad`` turn
detection: the server cuts the question audio into turns, and the runner waits
for every turn to be answered instead of committing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pybase64 as base64

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex_session_inputs import (
    FRAME_TRANSPORT_IMAGE_ITEMS,
    SERVER_VAD_TAIL_S,
    TURN_DETECTION_MODES,
    TURN_DETECTION_NONE,
    TURN_DETECTION_SERVER_VAD,
    ImageItemWindow,
    select_frame_transport,
    server_turns_pending,
    session_turn_detection,
    turn_activity_count,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    wait_for,
)

if TYPE_CHECKING:
    from vllm_omni.clients.duplex import EventCollector

from .omni_duplex_eval_clock import extract_timed_sentences
from .omni_duplex_eval_media import iter_av_units, iter_jpegs, materialize_media, read_audio_pcm16, video_duration

#: MiniCPM-o's native duplex system prompt, the historical default.
DEFAULT_INSTRUCTIONS = "Streaming Omni Conversation."
_SERVER_TURN_SETTLE_S = 2.0


@dataclass(frozen=True)
class GenerateSampleResult:
    """Timed-sentence path plus duplex metrics for one generate sample."""

    output: Path
    request_metrics: list[dict[str, object]] = field(default_factory=list)
    session_metrics: dict[str, object] = field(default_factory=dict)


def _ref_audio(path: str | Path) -> str:
    value = Path(path).expanduser().read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(value).decode("ascii")


def _event_collector_from_realtime(source: RealtimeEventCollector) -> EventCollector:
    """Replay experimental collector events onto the public EventCollector."""
    from vllm_omni.clients.duplex import EventCollector

    collector = EventCollector()
    for event, received_at_s in zip(source.events, source.event_received_at_s, strict=True):
        collector.add(event, received_at_s=received_at_s)
    return collector


def _with_sample_identity(payload: dict[str, object], sample: DuplexSample) -> dict[str, object]:
    return {"sample_id": sample.id, "split": sample.split, **payload}


def _session_capabilities(client: RealtimeDuplexClient) -> dict[str, object]:
    for event in client.events.events:
        session = event.get("session") if event.get("type") == "session.created" else None
        if isinstance(session, dict) and isinstance(session.get("capabilities"), dict):
            return session["capabilities"]
    return {}


async def _wait_for_server_turns(client: RealtimeDuplexClient, *, timeout_s: float) -> None:
    """Wait until no server-detected speech or response is open and turns are quiet."""
    deadline = time.monotonic() + timeout_s
    activity = turn_activity_count(client.events.events)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        client.raise_if_reader_stopped()
        current = turn_activity_count(client.events.events)
        if current != activity:
            activity, stable_since = current, time.monotonic()
        if not server_turns_pending(client.events.events) and time.monotonic() - stable_since >= _SERVER_TURN_SETTLE_S:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for server-detected turns to finish")


async def generate_sample(
    sample: DuplexSample,
    *,
    url: str,
    model: str,
    ref_audio: str | Path | None,
    output_root: str | Path,
    fps: float = 1.0,
    mix: str = "question",
    pace: str = "realtime",
    clock: str = "media",
    overwrite: bool = False,
    unit_ms: int = 1000,
    instructions: str | None = DEFAULT_INSTRUCTIONS,
    turn_detection: str = TURN_DETECTION_NONE,
) -> GenerateSampleResult:
    """Generate one sample's timed sentences and collect duplex session metrics.

    Timed sentences and ``*.meta.json`` stay the quality-eval artifacts. Duplex
    request/session metrics are returned for the CLI to write
    ``duplex_metrics.json``; they are not written into meta.
    """
    output = Path(output_root) / sample.split / f"{sample.id}.json"
    meta_path = output.with_name(output.stem + ".meta.json")
    if output.exists() and not overwrite:
        return GenerateSampleResult(output=output)
    if mix != "question":
        raise NotImplementedError("v1 supports mix=question; soundtrack mixing is reserved for P1")
    if turn_detection not in TURN_DETECTION_MODES:
        raise ValueError(f"turn_detection must be one of {TURN_DETECTION_MODES}")
    server_turns = turn_detection == TURN_DETECTION_SERVER_VAD
    media_dir = output.parent / ".media"
    audio_path = materialize_media(sample.question_audio, media_dir, f"{sample.id}_question", ".wav")
    video_path = materialize_media(sample.video, media_dir, sample.id, ".mp4")
    pcm = read_audio_pcm16(audio_path)
    if server_turns:
        # No final commit: trailing silence lets the VAD close the last turn.
        pcm += bytes(round(SERVER_VAD_TAIL_S * PCM16_SAMPLE_RATE) * PCM16_BYTES_PER_SAMPLE)
    duration = sample.video_duration or video_duration(video_path)
    frames = iter_jpegs(video_path, fps=fps, duration=duration)
    realtime = pace == "realtime"
    if pace not in {"realtime", "as-fast-as-possible"}:
        raise ValueError("pace must be realtime or as-fast-as-possible")
    client = RealtimeDuplexClient(build_realtime_url(url, model, autostart=False))
    response_done = False
    drain_timeout = None
    close_timeout = None
    stream_start: float | None = None
    async with client:
        await client.configure(
            model,
            ref_audio=_ref_audio(ref_audio) if ref_audio else None,
            instructions=instructions,
            auto_response=not server_turns,
            turn_detection=session_turn_detection(turn_detection),
        )
        frame_transport = select_frame_transport(_session_capabilities(client))
        image_items = (
            ImageItemWindow(client.send, item_prefix=f"duplexeval_frame_{sample.id}")
            if frame_transport == FRAME_TRANSPORT_IMAGE_ITEMS
            else None
        )
        ack_task = asyncio.create_task(_ack_playback(client))
        try:
            stream_start = time.monotonic()
            await client.stream_av_units(
                iter_av_units(pcm, frames, unit_ms=unit_ms),
                realtime=realtime,
                frame_sink=image_items.push if image_items is not None else None,
            )
            try:
                if server_turns:
                    await _wait_for_server_turns(client, timeout_s=max(20.0, duration + 20.0))
                    response_done = client.events.count("response.done") > 0
                else:
                    await client.commit()
                    await wait_for(
                        lambda: client.events.count("response.done") > 0,
                        timeout_s=max(20.0, duration + 20.0),
                        label="response.done",
                    )
                    response_done = True
            except TimeoutError as exc:
                drain_timeout = str(exc)
        finally:
            ack_task.cancel()
            try:
                await ack_task
            except asyncio.CancelledError:
                pass
        await client.acknowledge_playback()
        try:
            await client.close_session(timeout_s=20.0)
        except TimeoutError as exc:
            close_timeout = str(exc)
        events = list(client.events.events)
        collector = _event_collector_from_realtime(client.events)
    timed = [sentence.as_dict() for sentence in extract_timed_sentences(events, clock=clock)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timed, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "id": sample.id,
        "split": sample.split,
        "clock": clock if realtime else "invalid",
        "pace": pace,
        "mix": mix,
        "fps": fps,
        "unit_ms": unit_ms,
        "model": model,
        "turn_detection": turn_detection,
        "frame_transport": frame_transport,
        "instructions": instructions,
        "response_done": response_done,
        "drain_timeout": drain_timeout,
        "close_timeout": close_timeout,
        "ref_audio_sha256": hashlib.sha256(Path(ref_audio).read_bytes()).hexdigest() if ref_audio else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics

    bundle = collect_duplex_session_metrics(
        collector,
        stream_start=stream_start if stream_start is not None else 0.0,
        session_id=sample.id,
    )
    return GenerateSampleResult(
        output=output,
        request_metrics=[_with_sample_identity(metric, sample) for metric in bundle.request_metrics],
        session_metrics=_with_sample_identity(bundle.session_metrics, sample),
    )


async def _ack_playback(client: RealtimeDuplexClient) -> None:
    while True:
        await asyncio.sleep(0.25)
        await client.acknowledge_playback()
