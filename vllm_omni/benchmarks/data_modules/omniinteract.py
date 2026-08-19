"""OmniInteract full-video benchmark for MiniCPM native duplex serving."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    _event_stage_metrics,
    build_realtime_url,
    wait_for,
)

_SUBSETS = {"1q1a", "1q1a_math", "1qna"}
_OUTPUT_RATE = 24_000


@dataclass(frozen=True)
class OmniInteractConfig:
    chunk_ms: int = 200
    video_fps: float = 1.0
    ref_audio: str | None = None
    pace: bool = True
    timeout_s: float = 900.0
    output_root: str | None = None


@dataclass(frozen=True)
class OmniInteractCase:
    subset: str
    video_rel: str
    video_path: str
    annotation_path: str
    scene_type: str
    config: OmniInteractConfig


@dataclass
class OmniInteractSampleRequest(SampleRequest):
    omniinteract: OmniInteractCase | None = None


@dataclass
class OmniInteractResult:
    success: bool = False
    error: str = ""
    latency_s: float = 0.0
    ttft_s: float = 0.0
    audio_rtf: float = 0.0
    generated_text: str = ""
    pacing_mean_lag_s: float = 0.0
    pacing_max_lag_s: float = 0.0
    turn_metrics: list[dict[str, Any]] = field(default_factory=list)
    official_summary: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _data_dir(root: Path) -> Path:
    for candidate in (root, root / "data"):
        if (candidate / "1q1a").is_dir():
            return candidate
    raise FileNotFoundError(f"OmniInteract data not found under {root}")


def _safe_extract(handle: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = handle.getmembers()
    for member in members:
        path = Path(member.name)
        destination = (root / path).resolve()
        if (
            path.is_absolute()
            or ".." in path.parts
            or (destination != root and root not in destination.parents)
            or not (member.isdir() or member.isfile())
        ):
            raise ValueError(f"Unsafe path in OmniInteract archive: {member.name!r}")
    handle.extractall(target, members=members)


def _extract(archive: Path, target: Path) -> Path:
    marker = target / ".source"
    fingerprint = f"{archive.stat().st_size}:{archive.stat().st_mtime_ns}"
    if marker.is_file() and marker.read_text().strip() == fingerprint:
        try:
            return _data_dir(target)
        except FileNotFoundError:
            pass
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    with tarfile.open(archive, "r:*") as handle:
        _safe_extract(handle, target)
    marker.write_text(fingerprint)
    return _data_dir(target)


def resolve_omniinteract_root(dataset_path: str | None, explicit_root: str | None = None) -> Path:
    value = explicit_root or dataset_path
    if not value:
        value = "lucky-lance/OmniInteract"
    local = Path(value).expanduser()
    if explicit_root and not local.is_dir():
        raise FileNotFoundError(f"--omniinteract-root is not a directory: {local.resolve()}")
    if local.is_dir():
        try:
            return _data_dir(local.resolve())
        except FileNotFoundError:
            for name in ("data.tar.gz", "data.tar"):
                archive = local / name
                if archive.is_file():
                    return _extract(archive, local / ".vllm_omni_extracted")
            raise
    from huggingface_hub import hf_hub_download

    archive: Path | None = None
    for name in ("data.tar.gz", "data.tar"):
        try:
            archive = Path(hf_hub_download(repo_id=value, filename=name, repo_type="dataset"))
            break
        except Exception:
            continue
    if archive is None:
        raise FileNotFoundError(f"Could not download OmniInteract data archive from {value!r}")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "vllm_omni" / "omniinteract"
    return _extract(archive, cache / value.replace("/", "__"))


class OmniInteractDataset(BenchmarkDataset):
    SUPPORTED_DATASET_PATHS = {"lucky-lance/OmniInteract"}
    DEFAULT_OUTPUT_LEN = 1
    IS_MULTIMODAL = True

    def __init__(
        self,
        dataset_path: str | None,
        *,
        data_root: str | None,
        subsets: list[str],
        config: OmniInteractConfig,
        random_seed: int = 0,
        disable_shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        invalid = set(subsets) - _SUBSETS
        if invalid:
            raise ValueError(f"Unsupported OmniInteract subsets: {sorted(invalid)}")
        self.config = config
        self.subsets = subsets
        self.root = resolve_omniinteract_root(dataset_path, data_root)
        super().__init__(
            dataset_path=dataset_path or str(self.root),
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            **kwargs,
        )
        self.data = self._cases()
        if not self.data:
            raise ValueError(f"No OmniInteract sessions found under {self.root}")
        if not disable_shuffle:
            random.Random(random_seed).shuffle(self.data)

    def _cases(self) -> list[OmniInteractCase]:
        cases: list[OmniInteractCase] = []
        for subset in self.subsets:
            root = self.root / subset
            if subset != "1qna":
                mapping = root / "video_json_map.json"
                entries = json.loads(mapping.read_text()).get("entries", []) if mapping.is_file() else []
                for row in entries:
                    video_rel, annotation_rel = str(row.get("video") or ""), str(row.get("annotation") or "")
                    video, annotation = root / video_rel, root / annotation_rel
                    if video.is_file() and annotation.is_file():
                        cases.append(
                            OmniInteractCase(
                                subset,
                                video_rel,
                                str(video.resolve()),
                                str(annotation.resolve()),
                                str(row.get("scene_type") or "multi_turn").lower(),
                                self.config,
                            )
                        )
                continue
            videos, annotations = root / "videos_bench", root / "annotations"
            for video in sorted(videos.rglob("*.mp4")) if videos.is_dir() else []:
                relative = video.relative_to(videos)
                annotation = (annotations / relative).with_suffix(".json")
                if annotation.is_file():
                    cases.append(
                        OmniInteractCase(
                            subset,
                            str(video.relative_to(root)),
                            str(video.resolve()),
                            str(annotation.resolve()),
                            "1qna",
                            self.config,
                        )
                    )
        return cases

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **_: Any,
    ) -> list[SampleRequest]:
        tokenizer = get_cached_tokenizer(tokenizer)
        requests: list[SampleRequest] = []
        for index, case in enumerate(self.data[:num_requests]):
            prompt = f"OmniInteract: {case.subset}/{case.video_rel}"
            requests.append(
                OmniInteractSampleRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer.encode(prompt)),
                    expected_output_len=output_len or self.DEFAULT_OUTPUT_LEN,
                    request_id=f"{request_id_prefix}{index}",
                    omniinteract=case,
                )
            )
        self.maybe_oversample_requests(requests, num_requests, request_id_prefix, no_oversample)
        return requests


def validate_config(config: OmniInteractConfig) -> None:
    if not 0 < config.chunk_ms <= 1000:
        raise ValueError("OmniInteract chunk_ms must be in [1, 1000]")
    if not math.isfinite(config.video_fps) or not 0 < config.video_fps <= 1:
        raise ValueError("MiniCPM duplex supports video_fps in (0, 1]")
    if config.output_root and (not config.pace or config.chunk_ms != 200 or config.video_fps != 1):
        raise ValueError("Official OmniInteract output requires realtime pacing, 200 ms audio, and 1 FPS video")


def _run(command: list[str], *, text: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, check=False, text=text)
    if result.returncode:
        error = result.stderr if text else result.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"Command failed: {error.strip()}")
    return result


def prepare_media(video: Path, fps: float) -> tuple[float, bytes, list[str | None]]:
    duration = float(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            text=True,
        ).stdout.strip()
    )
    pcm = _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(PCM16_SAMPLE_RATE),
            "pipe:1",
        ]
    ).stdout
    target = math.ceil(duration) * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    pcm = (pcm + bytes(max(0, target - len(pcm))))[:target]

    import imageio.v3 as iio
    from PIL import Image

    source_fps = float(iio.immeta(str(video)).get("fps") or 30)
    indices = [int((index + 0.5) * source_fps / fps) for index in range(math.ceil(duration * fps))]
    frames: list[str | None] = [None] * len(indices)
    cursor = 0
    for index, frame in enumerate(iio.imiter(str(video))):
        if cursor == len(indices):
            break
        if index < indices[cursor]:
            continue
        image = Image.fromarray(frame)
        image.thumbnail((640, 640))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=85)
        while cursor < len(indices) and index >= indices[cursor]:
            frames[cursor] = base64.b64encode(output.getvalue()).decode()
            cursor += 1
    return duration, pcm, frames


@dataclass(frozen=True)
class _AudioSegment:
    response_id: str
    start_s: float
    samples: int
    rate: int


class _Playback:
    def __init__(self) -> None:
        self.cursor = 0
        self.end_s = 0.0
        self.segments: list[_AudioSegment] = []
        self.acked: dict[str, int] = {}
        self.completed: set[str] = set()
        self.completion_acked: set[str] = set()

    async def acknowledge(self, client: RealtimeDuplexClient, now: float | None = None) -> None:
        events = client.events
        while self.cursor < len(events.events):
            index, self.cursor = self.cursor, self.cursor + 1
            event = events.events[index]
            if event.get("type") == "response.done":
                if response_id := events.response_id(event):
                    self.completed.add(response_id)
                continue
            if event.get("type") != "response.audio.delta":
                continue
            response_id, encoded = events.response_id(event), event.get("delta") or event.get("audio")
            if not response_id or not isinstance(encoded, str):
                continue
            samples = len(base64.b64decode(encoded)) // PCM16_BYTES_PER_SAMPLE
            rate = int(event.get("sample_rate_hz") or events.output_sample_rate_hz or _OUTPUT_RATE)
            start = max(events.event_received_at_s[index], self.end_s)
            self.segments.append(_AudioSegment(response_id, start, samples, rate))
            self.end_s = start + samples / rate
        now = time.monotonic() if now is None else now
        played: dict[str, int] = {}
        for segment in self.segments:
            samples = min(segment.samples, max(0, round((now - segment.start_s) * segment.rate)))
            played[segment.response_id] = played.get(segment.response_id, 0) + samples * 1000 // segment.rate
        for response_id, played_ms in played.items():
            completion_due = response_id in self.completed and response_id not in self.completion_acked
            if played_ms <= self.acked.get(response_id, -1) and not completion_due:
                continue
            await client.send(
                {
                    "type": "playback.ack",
                    "response_id": response_id,
                    "item_id": f"item_{response_id}",
                    "played_ms": played_ms,
                    "committed_ms": played_ms,
                }
            )
            self.acked[response_id] = played_ms
            if response_id in self.completed:
                self.completion_acked.add(response_id)


async def _stream(
    client: RealtimeDuplexClient,
    pcm: bytes,
    frames: list[str | None],
    config: OmniInteractConfig,
    playback: _Playback,
) -> tuple[int, int, float, float]:
    chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * config.chunk_ms // 1000
    bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    start, frame_cursor, sent_frames = time.monotonic(), 0, 0
    lags: list[float] = []
    for offset in range(0, len(pcm), chunk_bytes):
        end = min(offset + chunk_bytes, len(pcm))
        end_ms = end * 1000 // bytes_per_second
        lags.append(max(0.0, time.monotonic() - (start + offset / bytes_per_second)))
        ready: list[str] = []
        while frame_cursor < len(frames) and end_ms >= (frame_cursor + 0.5) * 1000 / config.video_fps:
            if frames[frame_cursor]:
                ready.append(frames[frame_cursor] or "")
            frame_cursor += 1
        payload: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm[offset:end]).decode(),
            "input_audio_format": "pcm16",
            "sample_rate_hz": PCM16_SAMPLE_RATE,
            "duration_ms": (end - offset) * 1000 // bytes_per_second,
            "audio_end_ms": end_ms,
        }
        if ready:
            payload["video_frames"] = ready
        await client.send(payload)
        sent_frames += len(ready)
        if config.pace:
            await playback.acknowledge(client)
            await asyncio.sleep(max(0.0, start + end_ms / 1000 - time.monotonic()))
    lags.append(max(0.0, time.monotonic() - (start + len(pcm) / bytes_per_second)))
    return math.ceil(len(pcm) / chunk_bytes), sent_frames, sum(lags) / len(lags), max(lags)


def _identity(event: dict[str, object], prefix: str) -> tuple[str, int, int] | None:
    session_id, epoch, seq = event.get("session_id"), event.get("epoch"), event.get(f"{prefix}_input_seq")
    if (
        isinstance(session_id, str)
        and bool(session_id)
        and isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and isinstance(seq, int)
        and not isinstance(seq, bool)
        and seq > 0
    ):
        return session_id, epoch, seq
    return None


def _done_for(collector: RealtimeEventCollector, response_id: str) -> dict[str, object] | None:
    return next(
        (
            event
            for event in collector.events
            if event.get("type") == "response.done" and collector.response_id(event) == response_id
        ),
        None,
    )


def _needs_legacy_drain(pcm: bytes, events: list[dict[str, object]]) -> bool:
    period_ms = 1000
    for event in reversed(events):
        session = event.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        value = capabilities.get("chunk_period_ms") if isinstance(capabilities, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            period_ms = value
            break
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * period_ms // 1000
    active = sum(event.get("type") == "response.created" for event in events) > sum(
        event.get("type") == "response.done" for event in events
    )
    return active or bool(unit_bytes and len(pcm) % unit_bytes)


def _legacy_decision(events: list[dict[str, object]], committed_index: int) -> bool:
    return any(
        event.get("type") == "response.listen"
        or (event.get("type") == "response.done" and _status(event) != "cancelled")
        for event in events[committed_index + 1 :]
    )


def _status(event: dict[str, object]) -> str | None:
    response = event.get("response")
    return str(response.get("status")) if isinstance(response, dict) and response.get("status") else event.get("status")  # type: ignore[return-value]


def _raise_if_session_terminated(events: list[dict[str, object]], from_index: int) -> None:
    for event in events[from_index:]:
        event_type = event.get("type")
        if event_type not in {"session.expired", "session.closed"}:
            continue
        nested = event.get("event")
        reason = event.get("reason")
        if reason is None and isinstance(nested, dict):
            reason = nested.get("reason")
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{event_type}{detail}")


async def _wait_final(collector: RealtimeEventCollector, commit_from: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        errors = collector.errors()
        if errors:
            raise RuntimeError(str(errors[-1]))
        _raise_if_session_terminated(collector.events, commit_from)
        commit = next(
            (event for event in collector.events[commit_from:] if event.get("type") == "input_audio_buffer.committed"),
            None,
        )
        key = _identity(commit, "accepted") if commit else None
        if commit is not None and key is None:
            raise RuntimeError("input_audio_buffer.committed must include session_id, epoch, and accepted_input_seq")
        if key is None:
            await asyncio.sleep(0.02)
            continue
        processed = next(
            (
                event
                for event in collector.events
                if event.get("type") == "input_audio_buffer.processed" and _identity(event, "processed") == key
            ),
            None,
        )
        if processed:
            outcome = processed.get("outcome")
            if outcome == "failed":
                raise RuntimeError("final input processing failed")
            if outcome == "listen":
                return
            if outcome != "speak":
                raise RuntimeError(f"invalid final input processing outcome: {outcome!r}")
            response_id = processed.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                raise RuntimeError("processed speak outcome has no response_id")
            done = _done_for(collector, response_id)
            if done:
                if _status(done) != "completed":
                    raise RuntimeError(f"final response status is {_status(done)!r}")
                if not collector.audio_bytes(response_id):
                    raise RuntimeError("processed speak response has no audio")
                return
        await asyncio.sleep(0.02)
    raise TimeoutError("final accepted input was not processed before timeout")


async def _wait_committed(collector: RealtimeEventCollector, commit_from: int, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        errors = collector.errors()
        if errors:
            raise RuntimeError(str(errors[-1]))
        _raise_if_session_terminated(collector.events, commit_from)
        for index in range(commit_from, len(collector.events)):
            if collector.events[index].get("type") == "input_audio_buffer.committed":
                return index
        await asyncio.sleep(0.02)
    raise TimeoutError("input_audio_buffer.committed was not received before timeout")


def _response_text(collector: RealtimeEventCollector, response_id: str) -> str:
    return "".join(
        str(event.get("delta") or "")
        for event in collector.events
        if collector.response_id(event) == response_id
        and event.get("type") in {"response.audio_transcript.delta", "response.output_text.delta"}
    )


def _positive_ms(value: object) -> float:
    try:
        milliseconds = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return milliseconds / 1000 if math.isfinite(milliseconds) and milliseconds > 0 else 0.0


def _turn_metrics(collector: RealtimeEventCollector, stream_start: float) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    rate = collector.output_sample_rate_hz or _OUTPUT_RATE
    for turn_index, response_id in enumerate(collector.response_ids):
        created: float | None = None
        first_text: float | None = None
        first_audio: float | None = None
        done: float | None = None
        stage0_metrics: dict[str, object] | None = None
        for event, received in zip(collector.events, collector.event_received_at_s, strict=True):
            if collector.response_id(event) != response_id:
                continue
            event_type = event.get("type")
            if event_type == "response.created" and created is None:
                created = received
            if event_type in {"response.output_text.delta", "response.audio_transcript.delta"} and first_text is None:
                first_text = received
            if event_type == "response.audio.delta" and first_audio is None:
                first_audio = received
            if event_type == "response.done":
                done = received
            stage_metrics = _event_stage_metrics(event)
            stage0 = stage_metrics.get("0") if isinstance(stage_metrics, dict) else None
            if isinstance(stage0, dict):
                stage0_metrics = stage0
        origin = created if created is not None else stream_start
        audio_s = len(collector.audio_bytes(response_id)) / (rate * PCM16_BYTES_PER_SAMPLE)
        generation_s = max(0.0, done - created) if done is not None and created is not None else 0.0
        first_output = first_text if first_text is not None else first_audio
        observed_ttft = max(0.0, first_output - origin) if first_output is not None else 0.0
        stage_ttft = _positive_ms((stage0_metrics or {}).get("vllm_ttft_ms"))
        stage_tpot = _positive_ms((stage0_metrics or {}).get("vllm_tpot_ms"))
        metrics.append(
            {
                "turn_index": turn_index,
                "response_id": response_id,
                "video_time_s": max(0.0, origin - stream_start),
                "ttft_s": stage_ttft or observed_ttft,
                "tpot_s": stage_tpot,
                "rtf": generation_s / audio_s if generation_s > 0 and audio_s > 0 else 0.0,
                "audio_duration_s": audio_s,
                "response_generation_s": generation_s,
                "generated_text": _response_text(collector, response_id),
                "success": bool(_response_text(collector, response_id).strip() or audio_s > 0),
                "error": "",
            }
        )
    return metrics


def validate_output(collector: RealtimeEventCollector) -> int:
    sample_rate: int | None = None
    created = {
        response_id
        for event in collector.events
        if event.get("type") == "response.created" and (response_id := collector.response_id(event)) is not None
    }
    for event in collector.events:
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        details = event.get("status_details") if isinstance(event.get("status_details"), dict) else {}
        response_details = response.get("status_details") if isinstance(response.get("status_details"), dict) else {}
        states = {event.get("status"), response.get("status"), details.get("type"), response_details.get("type")}
        if event.get("type") == "response.done" and "failed" in states:
            raise ValueError("response.done reports failure")
        if event.get("type") != "response.audio.delta":
            continue
        response_id = collector.response_id(event)
        if not response_id or response_id not in created:
            raise ValueError("response audio has no matching response.created")
        if event.get("format") != "pcm16":
            raise ValueError("official output requires pcm16 audio")
        rate = event.get("sample_rate_hz")
        if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0 or sample_rate not in (None, rate):
            raise ValueError("response audio sample rate is missing or inconsistent")
        sample_rate = rate
        encoded = event.get("delta") or event.get("audio")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("response audio payload is missing")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("response audio is not valid base64") from exc
        if len(raw) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("response audio is not PCM16 aligned")
    return sample_rate or _OUTPUT_RATE


def _output_dir(root: Path, case: OmniInteractCase) -> Path:
    relative = case.video_rel.replace("\\", "/")
    if case.subset == "1qna" and relative.startswith("videos_bench/"):
        relative = relative.removeprefix("videos_bench/")
    return root / case.subset / relative.removesuffix(".mp4").replace("/", "__")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setparams((1, PCM16_BYTES_PER_SAMPLE, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)


def write_artifacts(
    root: Path,
    case: OmniInteractCase,
    collector: RealtimeEventCollector,
    stream_start: float,
    duration: float,
    inference_s: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    directory = _output_dir(root, case)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (".done", ".failed.json", "output.json", "wav_transcript_aligned.json", "precise_truncation.json"):
        (directory / name).unlink(missing_ok=True)
    try:
        rate = validate_output(collector)
        horizon = math.ceil(duration)
        output = bytearray(horizon * rate * PCM16_BYTES_PER_SAMPLE)
        cursor_s = 0.0
        response_times: dict[str, list[float]] = {}
        for event, received in zip(collector.events, collector.event_received_at_s, strict=True):
            if event.get("type") != "response.audio.delta":
                continue
            response_id = collector.response_id(event)
            assert response_id is not None
            raw = base64.b64decode(str(event.get("delta") or event.get("audio")), validate=True)
            start = max(received - stream_start, cursor_s)
            end = start + len(raw) / (rate * PCM16_BYTES_PER_SAMPLE)
            offset = round(start * rate) * PCM16_BYTES_PER_SAMPLE
            output[offset : min(len(output), offset + len(raw))] = raw[: max(0, len(output) - offset)]
            cursor_s = end
            timing = response_times.setdefault(response_id, [start, end])
            timing[0], timing[1] = min(timing[0], start), max(timing[1], end)
        chunks = []
        for response_id in collector.response_ids:
            text, timing = _response_text(collector, response_id), response_times.get(response_id)
            if text and timing and timing[0] < horizon:
                chunks.append(
                    {
                        "text": text,
                        "timestamp": [round(timing[0], 6), round(min(timing[1], horizon), 6)],
                    }
                )
        _write_wav(directory / "output.wav", bytes(output), rate)
        _write_json(directory / "wav_transcript.json", {"text": " ".join(c["text"] for c in chunks), "chunks": chunks})
        summary = {
            "video": str(Path(case.video_path).resolve()),
            "output_dir": str(directory.resolve()),
            "annotation": str(Path(case.annotation_path).resolve()),
            "subset": case.subset,
            "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
            "duration_sec": round(duration, 6),
            "inference_sec": round(inference_s, 6),
            "paced_e2e_ratio": round(inference_s / duration, 6),
            "status": "ok",
            **stats,
        }
        _write_json(directory / ".done", summary)
        return summary
    except Exception as exc:
        failure_summary(root, case, str(exc))
        raise ValueError(str(exc)) from exc


def failure_summary(root: Path, case: OmniInteractCase, error: str) -> dict[str, Any]:
    directory = _output_dir(root, case)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (".done", "output.json", "wav_transcript_aligned.json", "precise_truncation.json"):
        (directory / name).unlink(missing_ok=True)
    summary = {
        "video": case.video_path,
        "output_dir": str(directory.resolve()),
        "annotation": case.annotation_path,
        "subset": case.subset,
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
        "status": "error",
        "error": error,
    }
    _write_json(directory / ".failed.json", summary)
    return summary


def write_batch(root: Path, summaries: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    successes = [row for row in summaries if row.get("status") == "ok"]
    _write_json(
        root / "batch_summary.json",
        {
            "total": len(summaries),
            "success": len(successes),
            "failed": len(summaries) - len(successes),
            "results": summaries,
        },
    )
    rows = [
        {
            "sample_id": f"{row['subset']}__{Path(row['output_dir']).name}",
            "gt_json": row["annotation"],
            "model_json": str((Path(row["output_dir"]) / "wav_transcript.json").resolve()),
            "scene_type": row["scene_type"],
        }
        for row in successes
    ]
    (root / "official_eval_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def _ws_url(api_url: str, model: str, session_id: str) -> str:
    parts = urlsplit(api_url)
    if parts.scheme in {"http", "https"}:
        parts = parts._replace(scheme="ws" if parts.scheme == "http" else "wss")
    return build_realtime_url(urlunsplit(parts), model, autostart=False, session_id=session_id)


def _ref_audio(config: OmniInteractConfig) -> str | None:
    if not config.ref_audio:
        return None
    path = Path(config.ref_audio).expanduser()
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


async def probe_omniinteract(case: OmniInteractCase, api_url: str, model: str) -> None:
    """Check the Realtime endpoint without replaying an entire benchmark video."""
    validate_config(case.config)
    session_id = f"omniinteract:probe:{time.monotonic_ns()}"
    async with RealtimeDuplexClient(_ws_url(api_url, model, session_id)) as client:
        await client.configure(
            model,
            ref_audio=_ref_audio(case.config),
            session_id=session_id,
            idle_timeout_s=case.config.timeout_s,
            timeout_s=min(case.config.timeout_s, 20),
        )
        await client.close_session(timeout_s=min(case.config.timeout_s, 20))


async def run_omniinteract(case: OmniInteractCase, api_url: str, model: str, request_id: str) -> OmniInteractResult:
    result = OmniInteractResult()
    validate_config(case.config)
    output_root = Path(case.config.output_root) if case.config.output_root and request_id else None
    session_started: float | None = None
    try:
        preprocess = time.monotonic()
        duration, pcm, frames = await asyncio.to_thread(prepare_media, Path(case.video_path), case.config.video_fps)
        preprocess_s = time.monotonic() - preprocess
        session_started = time.monotonic()
        session_id = f"omniinteract:{case.subset}:{request_id or 'probe'}"
        async with RealtimeDuplexClient(_ws_url(api_url, model, session_id)) as client:
            await client.configure(
                model,
                ref_audio=_ref_audio(case.config),
                session_id=session_id,
                idle_timeout_s=case.config.timeout_s,
                timeout_s=case.config.timeout_s,
            )
            try:
                stream_start, playback = time.monotonic(), _Playback()
                chunks, frame_count, mean_lag, max_lag = await _stream(client, pcm, frames, case.config, playback)
                commit_from = len(client.events.events)
                await client.commit()
                if output_root:
                    await _wait_final(client.events, commit_from, case.config.timeout_s)
                else:
                    committed_index = await _wait_committed(client.events, commit_from, case.config.timeout_s)
                    if _needs_legacy_drain(pcm, client.events.events[: committed_index + 1]):
                        await wait_for(
                            lambda: _legacy_decision(client.events.events, committed_index),
                            timeout_s=case.config.timeout_s,
                            label="post-commit model decision or response drain",
                        )
                await wait_for(
                    lambda: client.events.count("response.created") <= client.events.count("response.done"),
                    timeout_s=case.config.timeout_s,
                    label="responses drained",
                )
                finished = time.monotonic()
                if case.config.pace:
                    await playback.acknowledge(client, finished)
                else:
                    await client.acknowledge_playback()
            except Exception:
                with contextlib.suppress(Exception):
                    await client.close_session(timeout_s=min(case.config.timeout_s, 20))
                raise
            await client.close_session(timeout_s=min(case.config.timeout_s, 20))
            result.latency_s = finished - session_started
            result.pacing_mean_lag_s, result.pacing_max_lag_s = mean_lag, max_lag
            result.turn_metrics = _turn_metrics(client.events, stream_start)
            result.ttft_s = result.turn_metrics[0]["ttft_s"] if result.turn_metrics else 0.0
            rtfs = [metric["rtf"] for metric in result.turn_metrics if metric["rtf"] > 0]
            result.audio_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
            result.generated_text = "\n".join(
                text
                for response_id in client.events.response_ids
                if (text := _response_text(client.events, response_id))
            )
            result.success = not client.events.errors()
            if not result.success:
                result.error = str(client.events.errors()[-1])
            if output_root:
                writer = write_artifacts if result.success else failure_summary
                args = (
                    (
                        output_root,
                        case,
                        client.events,
                        stream_start,
                        duration,
                        finished - stream_start,
                        {
                            "preprocess_sec": round(preprocess_s, 6),
                            "input_audio_chunks": chunks,
                            "input_video_frames": frame_count,
                            "pacing_mean_lag_sec": round(mean_lag, 6),
                            "pacing_max_lag_sec": round(max_lag, 6),
                        },
                    )
                    if result.success
                    else (output_root, case, result.error)
                )
                result.official_summary = await asyncio.to_thread(writer, *args)
    except Exception as exc:
        result.error, result.success = str(exc), False
        if session_started is not None:
            result.latency_s = max(0.0, time.monotonic() - session_started)
        if output_root:
            result.official_summary = await asyncio.to_thread(failure_summary, output_root, case, result.error)
    return result
