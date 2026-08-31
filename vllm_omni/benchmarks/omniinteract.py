# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract session execution and artifacts for serving benchmarks."""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlsplit

import pybase64 as base64

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
    DEFAULT_MAX_VIDEO_DURATION_S,
    OmniInteractCase,
    OmniInteractPreparedInput,
    case_manifest,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    chunk_period_ms,
    has_residual_model_unit,
    reference_audio_data_url,
    summarize_session_request_metrics,
    write_pcm16_wav,
)

OUTPUT_SAMPLE_RATE = 24_000
SUCCESS_ARTIFACTS = (".done", "output.wav", "wav_transcript.json", "events.json", "result.json")
BATCH_ARTIFACTS = ("batch_summary.json", "official_eval_manifest.jsonl")
ARTIFACT_LOCK_FILE = ".omniinteract.lock"
_INPUT_CHUNK_MS = 200
VIDEO_FPS = 1.0
_COMPLETION_SETTLE_S = 2.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OmniInteractBenchmarkConfig:
    model: str
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/v1/realtime"
    output_root: Path = Path("omniinteract-output")
    timeout_s: float = 900.0
    media_timeout_s: float = 600.0
    max_video_duration_s: float = DEFAULT_MAX_VIDEO_DURATION_S
    ref_audio: str | None = None
    require_response: bool = False
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, object] | None = None


@dataclass
class OmniInteractCaseResult:
    subset: str
    video: str
    output_dir: str
    success: bool = False
    error: str = ""
    session_id: str = ""
    latency_s: float = 0.0
    input_audio_chunks: int = 0
    input_video_frames: int = 0
    pacing_mean_lag_s: float = 0.0
    pacing_max_lag_s: float = 0.0
    responses: int = 0
    audio_bytes: int = 0
    audio_clipped_bytes: int = 0
    transcript: str = ""
    eligible_for_official_eval: bool = False
    official_eval_ineligible_reasons: list[str] = field(default_factory=list)
    artifact_warnings: list[str] = field(default_factory=list)
    output_tokens: int = 0
    duplex_request_metrics: list[dict[str, object]] = field(default_factory=list)
    duplex_session_metrics: dict[str, object] = field(default_factory=dict)
    _artifact_context: _DeferredArtifactContext | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy({key: value for key, value in vars(self).items() if not key.startswith("_")})


@dataclass(frozen=True)
class _DeferredArtifactContext:
    horizon_bytes: int
    spans: tuple[tuple[int, bytes], ...]
    chunks: list[dict[str, object]]
    events: list[dict[str, object]]


def benchmark_summary(results: list[OmniInteractCaseResult]) -> dict[str, Any]:
    succeeded = sum(result.success for result in results)
    eligible = sum(result.success and result.eligible_for_official_eval for result in results)
    return {
        "total": len(results),
        "success": succeeded,
        "failed": len(results) - succeeded,
        "eligible_for_official_eval": eligible,
        "successful_but_ineligible": succeeded - eligible,
        "audio_clipped_bytes": sum(result.audio_clipped_bytes for result in results),
        "results": [result.as_dict() for result in results],
    }


def _run_media_command(
    command: list[str],
    *,
    timeout_s: float,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=text,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Media command timed out after {timeout_s:g}s: {command[0]}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required media command is unavailable: {command[0]}") from exc
    if result.returncode:
        error = result.stderr if text else result.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"{command[0]} failed: {error.strip()}")
    return result


def prepare_media(
    video: Path,
    fps: float,
    *,
    timeout_s: float,
    max_duration_s: float = DEFAULT_MAX_VIDEO_DURATION_S,
) -> tuple[float, bytes, list[str | None]]:
    probe = [
        "ffprobe",
        *"-v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split(),
        str(video),
    ]
    duration = float(_run_media_command(probe, text=True, timeout_s=timeout_s).stdout.strip())
    if not math.isfinite(duration) or duration <= 0 or duration > max_duration_s:
        raise ValueError(f"Invalid video duration for {video}: {duration!r}")
    target_bytes = math.ceil(duration) * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    audio = [
        "ffmpeg",
        *"-loglevel error -i".split(),
        str(video),
        "-t",
        str(duration),
        *f"-vn -f s16le -ac 1 -ar {PCM16_SAMPLE_RATE} -fs".split(),
        str(target_bytes),
        "pipe:1",
    ]
    pcm = _run_media_command(audio, timeout_s=timeout_s).stdout
    pcm = (pcm + bytes(max(0, target_bytes - len(pcm))))[:target_bytes]

    frame_count = math.ceil(duration * fps)
    frames: list[str | None] = [None] * frame_count
    with tempfile.TemporaryDirectory(prefix="vllm-omni-frames-") as temp_dir:
        output_pattern = Path(temp_dir) / "frame-%06d.jpg"
        frame_filter = f"select=gte(t\\,(selected_n+0.5)/{fps}),scale=640:640:force_original_aspect_ratio=decrease"
        frame_command = [
            "ffmpeg",
            *"-loglevel error -i".split(),
            str(video),
            "-an",
            "-vf",
            frame_filter,
            *f"-frames:v {frame_count} -vsync vfr -q:v 5".split(),
            str(output_pattern),
        ]
        _run_media_command(frame_command, timeout_s=timeout_s)
        for index, frame_path in enumerate(sorted(Path(temp_dir).glob("frame-*.jpg"))[:frame_count]):
            frames[index] = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    return duration, pcm, frames


@dataclass(frozen=True)
class _AudioSegment:
    event_index: int
    response_id: str
    start_s: float
    end_s: float
    pcm16: bytes


class _Playback:
    def __init__(self) -> None:
        self.cursor = 0
        self.end_s = 0.0
        self.segments: list[_AudioSegment] = []
        self.completed: set[str] = set()
        self.completion_acked: set[str] = set()
        self.warnings: list[str] = []
        self._total_samples: dict[str, int] = {}
        self._response_end_s: dict[str, float] = {}

    def _warn_once(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def ingest(self, events: RealtimeEventCollector) -> None:
        while self.cursor < len(events.events):
            index, self.cursor = self.cursor, self.cursor + 1
            event = events.events[index]
            response_id = events.response_id(event)
            if event.get("type") == "response.done":
                if response_id:
                    self.completed.add(response_id)
                continue
            if event.get("type") != "response.audio.delta":
                continue
            encoded = event.get("delta") or event.get("audio")
            if not response_id:
                raise ValueError("response audio has no response_id")
            if not isinstance(encoded, str):
                raise ValueError("response audio payload is missing")
            if event.get("format") is None:
                self._warn_once("response.audio.delta omitted format; assumed pcm16")
            elif event.get("format") != "pcm16":
                raise ValueError("OmniInteract output must be pcm16")
            rate = event.get("sample_rate_hz")
            if rate is None:
                rate = events.output_sample_rate_hz or OUTPUT_SAMPLE_RATE
                self._warn_once(f"response.audio.delta omitted sample_rate_hz; assumed {rate}")
            if rate != OUTPUT_SAMPLE_RATE:
                raise ValueError(f"OmniInteract output must use {OUTPUT_SAMPLE_RATE} Hz audio")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("response audio is not valid base64") from exc
            if not raw or len(raw) % PCM16_BYTES_PER_SAMPLE:
                raise ValueError("response audio is empty or not PCM16 aligned")
            start = max(events.event_received_at_s[index], self.end_s)
            samples = len(raw) // PCM16_BYTES_PER_SAMPLE
            segment = _AudioSegment(index, response_id, start, start + samples / rate, raw)
            self.segments.append(segment)
            self.end_s = segment.end_s
            self._total_samples[response_id] = self._total_samples.get(response_id, 0) + samples
            self._response_end_s[response_id] = segment.end_s

    async def acknowledge(self, client: RealtimeDuplexClient, now: float | None = None) -> None:
        events = client.events
        self.ingest(events)
        now = time.monotonic() if now is None else now
        for response_id in self.completed - self.completion_acked:
            total_samples = self._total_samples.get(response_id, 0)
            if not total_samples or now < self._response_end_s[response_id]:
                continue
            played_ms = total_samples * 1000 // OUTPUT_SAMPLE_RATE
            await client.send_playback_ack(response_id, played_ms)
            self.completion_acked.add(response_id)


async def stream_inputs(
    client: RealtimeDuplexClient,
    pcm: bytes,
    frames: Sequence[str | None],
    playback: _Playback,
) -> tuple[int, int, float, float]:
    bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    chunk_bytes = bytes_per_second * _INPUT_CHUNK_MS // 1000
    started_at = time.monotonic()
    frame_cursor = sent_frames = 0
    lags: list[float] = []
    for chunks, offset in enumerate(range(0, len(pcm), chunk_bytes), start=1):
        end = min(offset + chunk_bytes, len(pcm))
        end_ms = end * 1000 // bytes_per_second
        ready: list[str] = []
        while frame_cursor < len(frames) and end_ms >= (frame_cursor + 0.5) * 1000 / VIDEO_FPS:
            if frames[frame_cursor]:
                ready.append(frames[frame_cursor] or "")
            frame_cursor += 1
        sent_frames += len(ready)
        event: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm[offset:end]).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": PCM16_SAMPLE_RATE,
            "duration_ms": (end - offset) * 1000 // bytes_per_second,
            "audio_end_ms": end_ms,
        }
        if ready:
            event["video_frames"] = ready
        lags.append(max(0.0, time.monotonic() - started_at - offset / bytes_per_second))
        await client.send(event)
        await playback.acknowledge(client)
        await asyncio.sleep(max(0.0, started_at + end_ms / 1000 - time.monotonic()))
    return chunks if pcm else 0, sent_frames, sum(lags) / len(lags) if lags else 0.0, max(lags, default=0.0)


def _response_status(event: dict[str, object]) -> str | None:
    response = event.get("response")
    if isinstance(response, dict) and response.get("status"):
        return str(response["status"])
    status = event.get("status")
    return str(status) if status else None


def response_ledger(
    collector: RealtimeEventCollector,
    *,
    end: int | None = None,
) -> tuple[set[str], set[str]]:
    created: set[str] = set()
    done: set[str] = set()
    for event in collector.events[:end]:
        event_type = event.get("type")
        if event_type not in {"response.created", "response.done"}:
            continue
        response_id = collector.response_id(event)
        if not response_id:
            raise ValueError(f"{event_type} has no response_id")
        if event_type == "response.created":
            if response_id in created:
                raise ValueError(f"duplicate response.created for {response_id}")
            created.add(response_id)
            continue
        if response_id not in created:
            raise ValueError(f"response.done without response.created for {response_id}")
        if response_id in done:
            raise ValueError(f"duplicate response.done for {response_id}")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        details = event.get("status_details") if isinstance(event.get("status_details"), dict) else {}
        nested = response.get("status_details") if isinstance(response.get("status_details"), dict) else {}
        if "failed" in {event.get("status"), response.get("status"), details.get("type"), nested.get("type")}:
            raise ValueError(f"response.done reports failure for {response_id}")
        done.add(response_id)
    return created, done


def _is_model_listen_decision(event: dict[str, object]) -> bool:
    response = event.get("response")
    metadata = response.get("metadata") if isinstance(response, dict) else event.get("metadata")
    return isinstance(metadata, dict) and metadata.get("buffering") is not True and metadata.get("model_listen") is True


def _has_post_commit_decision(
    collector: RealtimeEventCollector,
    events: list[dict[str, object]],
    *,
    prior_response_ids: set[str],
) -> bool:
    pending_prior = set(prior_response_ids)
    for event in events:
        event_type = event.get("type")
        if event_type == "response.done":
            response_id = collector.response_id(event)
            if response_id in pending_prior:
                pending_prior.remove(response_id)
                continue
            if not pending_prior and _response_status(event) != "cancelled":
                return True
        elif event_type == "response.listen" and not pending_prior and _is_model_listen_decision(event):
            return True
    return False


def _raise_if_session_terminated(
    collector: RealtimeEventCollector,
    from_index: int,
    *,
    explicit_close_from: int | None = None,
) -> None:
    errors = collector.errors()
    if errors:
        raise RuntimeError(str(errors[-1]))
    for index, event in enumerate(collector.events[from_index:], start=from_index):
        event_type = event.get("type")
        if event_type not in {"session.expired", "session.closed"}:
            continue
        reason = event.get("reason")
        nested = event.get("event")
        if reason is None and isinstance(nested, dict):
            reason = nested.get("reason")
        expected = (
            event_type == "session.closed"
            and explicit_close_from is not None
            and index >= explicit_close_from
            and reason is None
        )
        if expected:
            continue
        detail = f": {reason}" if reason else ""
        prefix = "Unexpected " if explicit_close_from is not None else ""
        raise RuntimeError(f"{prefix}{event_type}{detail}")


def _ensure_final_commit_tail(pcm: bytes, events: list[dict[str, object]]) -> bytes:
    """Reserve one PCM sample so an exact model unit is flushed by commit.

    A complete unit is otherwise emitted before commit, so its asynchronous
    decision cannot be correlated with the accepted final input. Keeping an
    almost-full unit buffered makes the server's final residual flush carry
    that correlation; the server pads the missing sample back to a full unit.
    """

    period_ms = chunk_period_ms(events)
    if len(pcm) >= PCM16_BYTES_PER_SAMPLE and not has_residual_model_unit(pcm, chunk_period_ms=period_ms):
        return pcm[:-PCM16_BYTES_PER_SAMPLE]
    return pcm


async def wait_for_session_completion(
    client: RealtimeDuplexClient,
    playback: _Playback,
    *,
    commit_from: int,
    session_from: int | None = None,
    timeout_s: float,
    settle_s: float,
) -> int:
    deadline = time.monotonic() + timeout_s
    committed_index: int | None = None
    prior_response_ids: set[str] = set()
    last_event_count = len(client.events.events)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        client.raise_if_reader_stopped()
        _raise_if_session_terminated(client.events, commit_from if session_from is None else session_from)
        await playback.acknowledge(client)
        if len(client.events.events) != last_event_count:
            last_event_count = len(client.events.events)
            stable_since = time.monotonic()
        if committed_index is None:
            committed_index = next(
                (
                    index
                    for index in range(commit_from, len(client.events.events))
                    if client.events.events[index].get("type") == "input_audio_buffer.committed"
                ),
                None,
            )
            if committed_index is not None:
                committed_event = client.events.events[committed_index]
                committed = committed_event.get("event")
                if isinstance(committed, dict) and committed.get("overlap_deferred") is True:
                    prior_created, prior_done = response_ledger(client.events, end=committed_index)
                    prior_response_ids = prior_created - prior_done
                stable_since = time.monotonic()
        if committed_index is not None:
            created, done = response_ledger(client.events)
            post_commit = client.events.events[committed_index + 1 :]
            decision = _has_post_commit_decision(
                client.events,
                post_commit,
                prior_response_ids=prior_response_ids,
            )
            if created == done and decision and time.monotonic() - stable_since >= settle_s:
                return committed_index
        await asyncio.sleep(0.05)
    missing: set[str] = set()
    with contextlib.suppress(ValueError):
        created, done = response_ledger(client.events)
        missing = created - done
    raise TimeoutError(
        "Timed out waiting for committed input and stable responses"
        + (f"; unfinished response_ids={sorted(missing)}" if missing else "")
    )


def _output_dir(root: Path, case: OmniInteractCase) -> Path:
    relative = case.video_rel.replace("\\", "/")
    stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
    digest = hashlib.sha256(f"{case.subset}/{relative}".encode()).hexdigest()[:8]
    return root / case.subset / f"{stem}--{digest}"


@contextlib.contextmanager
def omniinteract_output_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ARTIFACT_LOCK_FILE).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_replace(path, lambda temporary: temporary.write_text(value, encoding="utf-8"))


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_wav(path: Path, pcm: bytes | bytearray, rate: int) -> None:
    _atomic_replace(
        path,
        lambda temporary: write_pcm16_wav(
            temporary,
            cast(bytes, pcm),  # wave.writeframes accepts bytearray.
            sample_rate_hz=rate,
        ),
    )


def _clear_artifacts(directory: Path, names: Sequence[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).unlink(missing_ok=True)


def _artifact_summary(case: OmniInteractCase, result: OmniInteractCaseResult) -> dict[str, Any]:
    return {
        **result.as_dict(),
        "annotation": str(case.annotation_path),
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
    }


def _publish_success_artifacts(
    directory: Path,
    result: OmniInteractCaseResult,
    context: _DeferredArtifactContext,
    summary: dict[str, Any],
) -> None:
    pcm = bytearray(context.horizon_bytes)
    for offset, chunk in context.spans:
        pcm[offset : offset + len(chunk)] = chunk
    try:
        _clear_artifacts(directory, (*SUCCESS_ARTIFACTS, ".failed.json"))
        _atomic_write_wav(directory / "output.wav", pcm, OUTPUT_SAMPLE_RATE)
        _atomic_write_json(
            directory / "wav_transcript.json",
            {
                "text": result.transcript,
                "chunks": context.chunks,
                "timestamp_semantics": "serialized playback queue time relative to input streaming start",
            },
        )
        _atomic_write_json(directory / "events.json", context.events)
        _atomic_write_json(directory / "result.json", result.as_dict())
        _atomic_write_json(directory / ".done", summary)
    except Exception:
        _clear_artifacts(directory, SUCCESS_ARTIFACTS)
        result.success = result.eligible_for_official_eval = False
        if "artifact_write_failed" not in result.official_eval_ineligible_reasons:
            result.official_eval_ineligible_reasons.append("artifact_write_failed")
        raise


def _collect_output(
    collector: RealtimeEventCollector,
    playback: _Playback,
    *,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
    retain_events: bool,
    result: OmniInteractCaseResult,
) -> _DeferredArtifactContext:
    playback.ingest(collector)
    created, done = response_ledger(collector)
    if created != done:
        raise ValueError(f"unfinished response_ids: {sorted(created - done)}")
    done_events = [event for event in collector.events if event.get("type") == "response.done"]
    cancelled = any(_response_status(event) == "cancelled" for event in done_events)
    terminal_responses = {
        response_id
        for event in done_events
        if _response_status(event) != "cancelled" and (response_id := collector.response_id(event))
    }
    horizon_s = math.ceil(video_duration_s)
    horizon_bytes = horizon_s * OUTPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    spans: list[tuple[int, bytes]] = []
    clipped_bytes = 0
    response_times: dict[str, list[float]] = {}
    responses_with_audio: set[str] = set()
    for segment in playback.segments:
        response_id = segment.response_id
        if response_id not in created:
            raise ValueError("response audio has no matching response.created")
        start_s = max(0.0, segment.start_s - stream_start)
        end_s = max(start_s, segment.end_s - stream_start)
        offset = round(start_s * OUTPUT_SAMPLE_RATE) * PCM16_BYTES_PER_SAMPLE
        writable = max(0, min(len(segment.pcm16), horizon_bytes - offset))
        clipped_bytes += len(segment.pcm16) - writable
        if writable:
            clipped = segment.pcm16[:writable]
            spans.append((offset, clipped))
            if any(clipped):
                responses_with_audio.add(response_id)
        timing = response_times.setdefault(response_id, [start_s, end_s])
        timing[0], timing[1] = min(timing[0], start_s), max(timing[1], end_s)

    text_event_types = {"response.audio_transcript.delta", "response.output_text.delta", "response.text.delta"}
    text_ids = {collector.response_id(event) for event in collector.events if event.get("type") in text_event_types}
    if None in text_ids or not text_ids <= created:
        raise ValueError("response transcript has no matching response.created")
    chunks: list[dict[str, object]] = []
    texts: list[str] = []
    for response_id in collector.response_ids:
        text = collector.response_text(response_id).strip()
        if not text:
            continue
        response_timing = response_times.get(response_id)
        if response_timing is None or response_timing[0] >= horizon_s:
            continue
        texts.append(text)
        chunks.append(
            {
                "response_id": response_id,
                "text": text,
                "timestamp": [round(response_timing[0], 6), round(min(response_timing[1], horizon_s), 6)],
            }
        )
    complete_outputs = terminal_responses & responses_with_audio & {str(chunk["response_id"]) for chunk in chunks}
    if require_response and not complete_outputs:
        raise ValueError("OmniInteract E2E requires a response with audio and transcript")
    result.audio_bytes, result.audio_clipped_bytes = (
        sum(len(segment.pcm16) for segment in playback.segments),
        clipped_bytes,
    )
    result.transcript, result.responses, result.success = " ".join(texts).strip(), len(created), True
    result.artifact_warnings = list(playback.warnings)
    result.official_eval_ineligible_reasons = []
    if clipped_bytes:
        result.official_eval_ineligible_reasons.append("audio_clipped")
    if cancelled:
        result.official_eval_ineligible_reasons.append("cancelled_response")
    result.eligible_for_official_eval = not result.official_eval_ineligible_reasons
    audio_sizes = {segment.event_index: len(segment.pcm16) for segment in playback.segments}
    events = (
        [
            {
                **{key: value for key, value in event.items() if key not in {"delta", "audio"}},
                "audio_bytes": audio_sizes[index],
            }
            if event.get("type") == "response.audio.delta"
            else dict(event)
            for index, event in enumerate(collector.events)
        ]
        if retain_events
        else []
    )
    return _DeferredArtifactContext(horizon_bytes, tuple(spans), chunks, events)


def prepare_success_artifacts(
    root: Path,
    case: OmniInteractCase,
    collector: RealtimeEventCollector,
    *,
    playback: _Playback | None = None,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
    result: OmniInteractCaseResult,
    capture_artifacts: bool = True,
) -> dict[str, Any]:
    directory = _output_dir(root, case)
    context = _collect_output(
        collector,
        playback or _Playback(),
        stream_start=stream_start,
        video_duration_s=video_duration_s,
        require_response=require_response,
        retain_events=capture_artifacts,
        result=result,
    )
    result.output_dir = str(directory.resolve())
    summary = _artifact_summary(case, result)
    if capture_artifacts:
        result._artifact_context = context
    return summary


def write_failure_artifacts(root: Path, case: OmniInteractCase, result: OmniInteractCaseResult) -> None:
    directory = _output_dir(root, case)
    result.output_dir = str(directory.resolve())
    result.success = result.eligible_for_official_eval = False
    _clear_artifacts(directory, SUCCESS_ARTIFACTS)
    _atomic_write_json(directory / ".failed.json", _artifact_summary(case, result))


def publish_deferred_case_artifacts(
    root: Path,
    case: OmniInteractCase,
    result: OmniInteractCaseResult,
) -> None:
    context = result._artifact_context
    result._artifact_context = None
    if not result.success:
        write_failure_artifacts(root, case, result)
        return
    if context is None:
        raise RuntimeError("OmniInteract benchmark output lost its deferred artifact context")
    directory = _output_dir(root, case)
    _publish_success_artifacts(directory, result, context, _artifact_summary(case, result))


def clear_case_artifacts(root: Path, case: OmniInteractCase) -> None:
    directory = _output_dir(root, case)
    _clear_artifacts(directory, (*SUCCESS_ARTIFACTS, ".failed.json"))


def clear_batch_artifacts(root: Path) -> None:
    _clear_artifacts(root, BATCH_ARTIFACTS)


def write_batch_artifacts(
    root: Path,
    cases: list[OmniInteractCase],
    results: list[OmniInteractCaseResult],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        case_manifest(case, _output_dir(root, case))
        for case, result in zip(cases, results, strict=True)
        if result.success and result.eligible_for_official_eval
    ]
    ineligible = [result for result in results if result.success and not result.eligible_for_official_eval]
    if ineligible:
        reasons = Counter(
            reason for result in ineligible for reason in (result.official_eval_ineligible_reasons or ["unspecified"])
        )
        logger.warning(
            "%d successful OmniInteract cases were excluded from official evaluation: %s",
            len(ineligible),
            ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())),
        )
    _atomic_write_json(root / "batch_summary.json", benchmark_summary(results))
    _atomic_write_text(
        root / "official_eval_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _websocket_url(config: OmniInteractBenchmarkConfig, session_id: str) -> str:
    endpoint = (
        config.endpoint
        if urlsplit(config.endpoint).scheme
        else urljoin(config.base_url.rstrip("/") + "/", config.endpoint.lstrip("/"))
    )
    return build_realtime_url(
        endpoint,
        config.model,
        autostart=False,
        native_duplex=True,
        session_id=session_id,
    )


def _populate_response_metrics(
    result: OmniInteractCaseResult,
    collector: RealtimeEventCollector,
    *,
    stream_start: float,
) -> None:
    measurement_origin = {
        "ttft": "response.created client receive to first non-empty text delta",
        "ttfp": "response.created client receive to first audio packet",
        "rtf": "response.created client receive to last audio packet divided by emitted audio duration",
    }
    request_metrics: list[dict[str, object]] = []
    output_tokens = 0
    for request_index, response_id in enumerate(collector.response_ids):
        timing = collector.timing_summary(
            after_s=stream_start,
            input_committed_at_s=None,
            response_id=response_id,
            measurement_origin=measurement_origin,
        )
        raw_metric = timing.get("request_metrics")
        stage0 = timing.get("stage0_tokens")
        metric = {
            "session_id": result.session_id,
            "request_index": request_index,
            "response_id": response_id,
            **(raw_metric if isinstance(raw_metric, dict) else {}),
        }
        if isinstance(stage0, dict):
            metric["stage0_tokens"] = dict(stage0)
            output_tokens += int(stage0.get("output_token_count") or 0)
        if isinstance(raw_metric, dict) or isinstance(stage0, dict):
            request_metrics.append(metric)
    result.output_tokens = output_tokens
    result.duplex_request_metrics = request_metrics
    result.duplex_session_metrics = summarize_session_request_metrics(
        request_metrics,
        session_id=result.session_id,
    )


async def run_omniinteract_case(
    case: OmniInteractCase,
    config: OmniInteractBenchmarkConfig,
    *,
    request_index: int | str,
    capture_artifacts: bool = True,
    prepared_input: OmniInteractPreparedInput | None = None,
) -> OmniInteractCaseResult:
    if not config.ref_audio:
        raise ValueError("ref_audio is required for MiniCPM-o native-duplex audio output")
    for name, value in (
        ("timeout_s", config.timeout_s),
        ("media_timeout_s", config.media_timeout_s),
        ("max_video_duration_s", config.max_video_duration_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    session_id = f"omniinteract:{case.subset}:{request_index}:{time.monotonic_ns()}"
    result = OmniInteractCaseResult(
        case.subset,
        str(case.video_path),
        str(_output_dir(config.output_root, case).resolve()),
        session_id=session_id,
    )
    started_at = time.monotonic()
    try:
        if prepared_input is None:
            reference_audio = reference_audio_data_url(config.ref_audio)
            duration, pcm, frames = await asyncio.to_thread(
                prepare_media,
                case.video_path,
                VIDEO_FPS,
                timeout_s=config.media_timeout_s,
                max_duration_s=config.max_video_duration_s,
            )
        else:
            reference_audio = prepared_input.ref_audio_data_url
            duration = prepared_input.duration_s
            pcm = prepared_input.pcm16
            frames = prepared_input.video_frames
        if not any(frames):
            raise ValueError(f"No video frames were decoded from {case.video_path}")
        async with RealtimeDuplexClient(
            _websocket_url(config, session_id), additional_headers=config.extra_headers
        ) as client:
            session_from = len(client.events.events)
            await client.configure(
                config.model,
                ref_audio=reference_audio,
                session_id=session_id,
                extra_body=config.extra_body,
                idle_timeout_s=config.timeout_s,
                timeout_s=min(config.timeout_s, 20.0),
            )
            pcm = _ensure_final_commit_tail(pcm, client.events.events)
            playback = _Playback()
            stream_start = time.monotonic()
            try:
                input_duration_s = len(pcm) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
                upload_timeout_s = config.timeout_s + input_duration_s
                try:
                    chunks, frame_count, mean_lag, max_lag = await asyncio.wait_for(
                        stream_inputs(client, pcm, frames, playback), timeout=upload_timeout_s
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f"Realtime upload timed out after {upload_timeout_s:g}s") from exc
                commit_from = len(client.events.events)
                await client.commit()
                await wait_for_session_completion(
                    client,
                    playback,
                    commit_from=commit_from,
                    session_from=session_from,
                    timeout_s=config.timeout_s,
                    settle_s=_COMPLETION_SETTLE_S,
                )
                await playback.acknowledge(client, playback.end_s)
                _raise_if_session_terminated(client.events, session_from)
                close_from = len(client.events.events)
                await client.close_session(timeout_s=min(config.timeout_s, 20.0))
            except Exception:
                with contextlib.suppress(Exception):
                    await client.close_session(timeout_s=min(config.timeout_s, 20.0))
                raise
            errors = client.events.errors()
            if errors:
                raise RuntimeError(str(errors[-1]))
            _raise_if_session_terminated(client.events, session_from, explicit_close_from=close_from)
            result.latency_s = time.monotonic() - started_at
            result.input_audio_chunks, result.input_video_frames = chunks, frame_count
            result.pacing_mean_lag_s, result.pacing_max_lag_s = mean_lag, max_lag
            _populate_response_metrics(result, client.events, stream_start=stream_start)
            await asyncio.to_thread(
                prepare_success_artifacts,
                config.output_root,
                case,
                client.events,
                playback=playback,
                stream_start=stream_start,
                video_duration_s=duration,
                require_response=config.require_response,
                result=result,
                capture_artifacts=capture_artifacts,
            )
    except Exception as exc:
        result.success = result.eligible_for_official_eval = False
        if "case_failed" not in result.official_eval_ineligible_reasons:
            result.official_eval_ineligible_reasons.append("case_failed")
        result.error = str(exc)
        result.latency_s = max(0.0, time.monotonic() - started_at)
    return result
