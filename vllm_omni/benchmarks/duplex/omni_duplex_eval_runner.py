# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-sample MiniCPM-o native-duplex generation."""

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
from vllm_omni.experimental.fullduplex.client import (
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    wait_for,
)

if TYPE_CHECKING:
    from vllm_omni.clients.duplex import EventCollector

from .omni_duplex_eval_clock import extract_timed_sentences
from .omni_duplex_eval_media import iter_av_units, iter_jpegs, materialize_media, read_audio_pcm16, video_duration


@dataclass(frozen=True)
class GenerateSampleResult:
    """Timed-sentence path plus duplex metrics for one generate sample."""

    output: Path
    request_metrics: list[dict[str, object]] = field(default_factory=list)
    session_metrics: dict[str, object] = field(default_factory=dict)
    timed_sentences: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    output_tokens: int = 0
    audio_bytes: int = 0
    audio_sample_rate: int = 24_000
    error: str = ""


@dataclass(frozen=True)
class PreparedSample:
    """Decoded input, reusable across readiness, warmup and measured sessions."""

    pcm: bytes = field(repr=False)
    frames: tuple[tuple[float, bytes], ...] = field(repr=False)
    duration: float
    ref_audio: str = field(repr=False)
    ref_audio_sha256: str


def prepare_sample(sample: DuplexSample, *, media_dir: Path, ref_audio: str | Path, fps: float) -> PreparedSample:
    audio_path = materialize_media(sample.question_audio, media_dir, f"{sample.id}_question", ".wav")
    video_path = materialize_media(sample.video, media_dir, sample.id, ".mp4")
    duration = sample.video_duration or video_duration(video_path)
    return PreparedSample(
        pcm=read_audio_pcm16(audio_path),
        frames=tuple(iter_jpegs(video_path, fps=fps, duration=duration)),
        duration=duration,
        ref_audio=_ref_audio(ref_audio),
        ref_audio_sha256=hashlib.sha256(Path(ref_audio).read_bytes()).hexdigest(),
    )


def write_sample_result(result: GenerateSampleResult) -> None:
    """Publish the existing judge input format; serving calls this after timing."""
    output = result.output
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_name(output.stem + ".meta.json")
    # Publish the judge-visible JSON last. A failed write cannot leave a partial
    # response that the evaluator would mistake for a complete sample.
    meta_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    temporary = output.with_suffix(".json.part")
    try:
        temporary.write_text(json.dumps(result.timed_sentences, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


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


async def generate_sample(
    sample: DuplexSample,
    *,
    url: str,
    model: str,
    ref_audio: str | Path,
    output_root: str | Path,
    fps: float = 1.0,
    mix: str = "question",
    pace: str = "realtime",
    clock: str = "media",
    overwrite: bool = False,
    unit_ms: int = 1000,
    prepared: PreparedSample | None = None,
    defer_artifacts: bool = False,
    additional_headers: dict[str, str] | None = None,
    extra_body: dict[str, object] | None = None,
) -> GenerateSampleResult:
    """Generate one sample's timed sentences and collect duplex session metrics.

    Timed sentences and ``*.meta.json`` stay the quality-eval artifacts. Duplex
    request/session metrics are returned for the CLI to write
    ``duplex_metrics.json``; they are not written into meta.
    """
    output = Path(output_root) / sample.split / f"{sample.id}.json"
    if not defer_artifacts and output.exists() and not overwrite:
        return GenerateSampleResult(output=output)
    if mix != "question":
        raise NotImplementedError("v1 supports mix=question; soundtrack mixing is reserved for P1")
    if prepared is None:
        # Standalone generation shares its loop with other live sessions;
        # decoding a whole clip must not pause their input or playback ACKs.
        prepared = await asyncio.to_thread(
            prepare_sample, sample, media_dir=output.parent / ".media", ref_audio=ref_audio, fps=fps
        )
    realtime = pace == "realtime"
    if pace not in {"realtime", "as-fast-as-possible"}:
        raise ValueError("pace must be realtime or as-fast-as-possible")
    client = RealtimeDuplexClient(
        build_realtime_url(url, model, autostart=False), additional_headers=additional_headers
    )
    response_done = False
    drain_timeout = None
    close_timeout = None
    stream_start: float | None = None
    async with client:
        await client.configure(
            model, ref_audio=prepared.ref_audio, instructions="Streaming Omni Conversation.", extra_body=extra_body
        )
        ack_task = asyncio.create_task(_ack_playback(client))
        try:
            stream_start = time.monotonic()
            await client.stream_av_units(
                iter_av_units(prepared.pcm, iter(prepared.frames), unit_ms=unit_ms), realtime=realtime
            )
            await client.commit()
            try:
                await wait_for(
                    lambda: client.events.count("response.done") > 0,
                    timeout_s=max(20.0, prepared.duration + 20.0),
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
    meta = {
        "id": sample.id,
        "split": sample.split,
        "clock": clock if realtime else "invalid",
        "pace": pace,
        "mix": mix,
        "fps": fps,
        "unit_ms": unit_ms,
        "model": model,
        "response_done": response_done,
        "drain_timeout": drain_timeout,
        "close_timeout": close_timeout,
        "ref_audio_sha256": prepared.ref_audio_sha256,
    }
    from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics

    bundle = collect_duplex_session_metrics(
        collector,
        stream_start=stream_start if stream_start is not None else 0.0,
        session_id=sample.id,
    )
    errors = []
    for event in client.events.errors():
        error = event.get("error")
        code = error.get("code") if isinstance(error, dict) else event.get("code")
        # As in OmniInteract, an ACK racing a later commit is harmless. All
        # other server errors still make the serving sample unsuccessful.
        if code != "playback_ack_too_late":
            errors.append(str(event))
    errors.extend(str(error) for error in (drain_timeout, close_timeout) if error)
    completed = set()
    for event in events:
        response = event.get("response")
        if event.get("type") == "response.done" and isinstance(response, dict):
            if response.get("status", "completed") == "completed":
                completed.add(response.get("id"))
            else:
                errors.append(f"Response did not complete: {response}")
    if set(client.events.response_ids) - completed:
        errors.append("Session closed with unfinished responses")
    result = GenerateSampleResult(
        output=output,
        request_metrics=[_with_sample_identity(metric, sample) for metric in bundle.request_metrics],
        session_metrics=_with_sample_identity(bundle.session_metrics, sample),
        timed_sentences=timed,
        metadata=meta,
        output_tokens=bundle.output_tokens,
        audio_bytes=len(client.events.audio_bytes()),
        audio_sample_rate=client.events.output_sample_rate_hz,
        error="; ".join(errors),
    )
    if not defer_artifacts:
        write_sample_result(result)
    return result


async def _ack_playback(client: RealtimeDuplexClient) -> None:
    while True:
        await asyncio.sleep(0.25)
        await client.acknowledge_playback()
