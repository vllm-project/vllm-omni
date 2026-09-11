# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict PersonaPlex validation for the unified OpenAI Realtime duplex path."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import time
import uuid
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError as exc:  # pragma: no cover - driver dependency.
    raise SystemExit("Install websockets first: pip install websockets") from exc

from vllm_omni.clients.duplex import (
    AudioDelta,
    EventCollector,
    wrap_event,
    write_pcm16_wav,
)
from vllm_omni.clients.duplex import (
    wait_for_condition as wait_for,
)

SAMPLE_RATE_HZ = 24_000
FRAME_SAMPLES = 1_920
FRAME_PERIOD_S = FRAME_SAMPLES / SAMPLE_RATE_HZ


class _ProbeTransport(Protocol):
    async def recv(self) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    async def close(self) -> None: ...


class RawRealtimeProbe:
    """Raw-event Realtime probe: one websocket plus an :class:`EventCollector`.

    This driver speaks the wire protocol directly (it validates handshake
    errors such as admission limits, which the public
    :class:`vllm_omni.clients.duplex.DuplexClient` treats as failures), so it
    keeps a transport-only client and reads everything from ``.events``.
    """

    def __init__(self, url: str, *, max_size: int = 64 * 1024 * 1024) -> None:
        self.url = url
        self.max_size = max_size
        self.events = EventCollector()
        self._ws: _ProbeTransport | None = None
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> RawRealtimeProbe:
        self._ws = await websockets.connect(self.url, max_size=self.max_size)
        self._reader_task = asyncio.create_task(self._read_events())
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    async def _read_events(self) -> None:
        assert self._ws is not None
        try:
            while True:
                raw = await self._ws.recv()
                if not isinstance(raw, str):
                    continue
                event = json.loads(raw)
                if isinstance(event, dict):
                    self.events.add(event)
        except ConnectionClosed:
            return

    async def send(self, event: dict[str, object]) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(event))


def _input_identity(
    path: Path,
    *,
    expected_sha256: str | None,
) -> dict[str, str]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise ValueError(
            f"input WAV SHA-256 mismatch: expected={expected_sha256.lower()}, actual={actual}, path={resolved}"
        )
    return {"path": str(resolved), "sha256": actual}


def _realtime_url(base_url: str, model: str, session_id: str) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(duplex="1", model=model, autostart="0", session_id=session_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _read_wav_as_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    dtypes = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
    if sample_width not in dtypes:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    pcm = np.frombuffer(frames, dtype=dtypes[sample_width]).astype(np.float32)
    if sample_width == 1:
        pcm = (pcm - 128.0) / 128.0
    else:
        pcm /= float(1 << (sample_width * 8 - 1))
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if source_rate != SAMPLE_RATE_HZ and pcm.size:
        output_samples = max(1, round(pcm.size * SAMPLE_RATE_HZ / source_rate))
        pcm = np.interp(
            np.linspace(0, pcm.size - 1, output_samples),
            np.arange(pcm.size),
            pcm,
        ).astype(np.float32)
    pcm = np.ascontiguousarray(pcm, dtype="<f4")
    if not pcm.size or not np.isfinite(pcm).all():
        raise ValueError("input WAV must contain finite audio")
    return pcm


def _events(client: RawRealtimeProbe, event_type: str) -> list[dict[str, object]]:
    return [event for event in client.events.events if event.get("type") == event_type]


async def _open_session(
    args: argparse.Namespace,
    *,
    session_id: str,
    persona: str,
    expect_error: bool = False,
) -> tuple[RawRealtimeProbe, dict[str, object]]:
    client = RawRealtimeProbe(_realtime_url(args.url, args.model, session_id))
    await client.__aenter__()
    await client.send(_session_update(args, session_id=session_id, persona=persona))
    event_type = "error" if expect_error else "session.created"
    await wait_for(
        lambda: client.events.count(event_type) > 0,
        timeout_s=args.timeout_s,
        label=event_type,
    )
    return client, _events(client, event_type)[-1]


def _session_update(args: argparse.Namespace, *, session_id: str, persona: str) -> dict[str, object]:
    return {
        "type": "session.update",
        "session": {
            "session_id": session_id,
            "model": args.model,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "voice": args.voice,
            "instructions": persona,
            "turn_detection": None,
            "extra_body": {"auto_response": True},
        },
    }


async def _close_session(client: RawRealtimeProbe, *, timeout_s: float) -> None:
    await client.send({"type": "session.close"})
    await wait_for(
        lambda: client.events.count("session.closed") > 0,
        timeout_s=timeout_s,
        label="session.closed",
    )
    await client.__aexit__(None, None, None)


async def _stream_frames(
    client: RawRealtimeProbe,
    pcm: np.ndarray,
    *,
    max_frames: int | None = None,
) -> int:
    frame_count = math.ceil(pcm.size / FRAME_SAMPLES)
    if max_frames is not None:
        if max_frames < 0:
            raise ValueError(f"max_frames must be non-negative, got {max_frames}")
        if max_frames > 0:
            frame_count = min(frame_count, max_frames)
    for seq in range(frame_count):
        frame = pcm[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES]
        await client.send(_frame_event(frame, seq))
        await asyncio.sleep(FRAME_PERIOD_S)
    return frame_count


def _frame_event(frame: np.ndarray, seq: int) -> dict[str, object]:
    frame = np.ascontiguousarray(np.pad(frame, (0, FRAME_SAMPLES - frame.size)), dtype="<f4")
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(frame).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_ms": 80,
        "audio_end_ms": (seq + 1) * 80,
        "is_speech": bool(np.sqrt(np.mean(np.square(frame))) >= 1e-4),
    }


def _audio_frame_stats(
    raw: bytes,
    *,
    input_frames: int,
    voiced_frame_rms_threshold: float,
) -> dict[str, int | float]:
    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size % FRAME_SAMPLES:
        raise AssertionError(
            "PersonaPlex output must contain whole 80 ms codec frames: "
            f"samples={pcm.size}, samples_per_frame={FRAME_SAMPLES}"
        )
    output_frames = pcm.size // FRAME_SAMPLES
    frames = pcm.astype(np.float32).reshape(output_frames, FRAME_SAMPLES) / 32768.0
    voiced_frames = int(np.count_nonzero(np.sqrt(np.mean(np.square(frames), axis=1)) >= voiced_frame_rms_threshold))
    return {
        "output_frames": int(output_frames),
        "voiced_frames": voiced_frames,
        "silent_frames": int(output_frames - voiced_frames),
        "frame_deficit": int(input_frames - output_frames),
        "frame_coverage_ratio": float(output_frames / input_frames),
    }


def _response_ids(client: RawRealtimeProbe) -> set[str]:
    return {
        str(event["response_id"])
        for event in _events(client, "response.output_audio.delta")
        if isinstance(event.get("response_id"), str)
    }


def _session_result(
    client: RawRealtimeProbe,
    *,
    input_frames: int,
    args: argparse.Namespace,
    minimum_chunks: int,
) -> tuple[bytes, set[str], dict[str, object]]:
    chunks = sum(len(items) for items in client.events.response_audio.values())
    raw = client.events.audio_bytes()
    if chunks < minimum_chunks or not raw or len(raw) % 2:
        raise AssertionError(f"invalid PCM16 output: chunks={chunks}, bytes={len(raw)}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(pcm))))
    # This low floor only catches corrupt, non-finite, or effectively zero PCM.
    # Audible speech is checked per frame below with the higher 1e-3 default.
    if not np.isfinite(pcm).all() or rms <= 1e-5:
        raise AssertionError(f"output audio is non-finite or silent: samples={pcm.size}, rms={rms}")
    rates = {
        rate
        for event in _events(client, "response.output_audio.delta")
        if isinstance(rate := event.get("sample_rate_hz"), int)
    }
    if rates != {SAMPLE_RATE_HZ}:
        raise AssertionError(f"unexpected output sample rates: {sorted(rates)}")
    stats = _audio_frame_stats(
        raw,
        input_frames=input_frames,
        voiced_frame_rms_threshold=args.voiced_frame_rms_threshold,
    )
    if not 0 <= int(stats["frame_deficit"]) <= args.max_frame_deficit:
        raise AssertionError(f"invalid synchronous frame accounting: input={input_frames}, stats={stats}")
    min_voiced_frames = int(args.min_voiced_frames)
    if min_voiced_frames < 1:
        raise ValueError(f"min_voiced_frames must be positive, got {min_voiced_frames}")
    if int(stats["voiced_frames"]) < min_voiced_frames:
        raise AssertionError(
            f"output contained insufficient audible speech: minimum_voiced_frames={min_voiced_frames}, stats={stats}"
        )
    runtime_metadata = [
        metadata["vllm_omni"]
        for event in _events(client, "response.output_audio.delta")
        if isinstance((metadata := event.get("metadata")), dict) and isinstance(metadata.get("vllm_omni"), dict)
    ]
    if not any(
        item.get("runtime_impl") == "scheduler_data_plane"
        and item.get("uses_model_runner_scheduler") is True
        and item.get("runner_kv_backed") is True
        for item in runtime_metadata
    ):
        raise AssertionError(f"audio events did not prove the scheduler data plane: {runtime_metadata}")
    response_ids = _response_ids(client)
    if len(response_ids) != 1:
        raise AssertionError(f"continuous stream used multiple responses: {sorted(response_ids)}")
    return (
        raw,
        response_ids,
        {
            **stats,
            "audio_chunks": chunks,
            "audio_samples": len(raw) // 2,
            "audio_rms": rms,
            "response_ids": sorted(response_ids),
        },
    )


def _capabilities(created: dict[str, object]) -> dict[str, object]:
    session = created.get("session")
    capabilities = session.get("capabilities") if isinstance(session, dict) else None
    if not isinstance(capabilities, dict):
        raise AssertionError(f"session.created omitted capabilities: {created}")
    return capabilities


def _save(output_dir: Path, name: str, client: RawRealtimeProbe, raw: bytes) -> None:
    (output_dir / f"{name}-events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
        encoding="utf-8",
    )
    write_pcm16_wav(output_dir / f"{name}-output.wav", raw, sample_rate_hz=SAMPLE_RATE_HZ)


async def run(args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "sessions", None) is not None:
        return await run_load(args)
    input_path = Path(args.input_wav)
    input_identity = _input_identity(
        input_path,
        expected_sha256=args.expected_input_sha256,
    )
    pcm = _read_wav_as_float32(input_path)
    if args.tail_s > 0:
        pcm = np.concatenate([pcm, np.zeros(round(args.tail_s * SAMPLE_RATE_HZ), dtype=np.float32)])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = {name: f"personaplex-{name}-{uuid.uuid4().hex}" for name in ("primary", "secondary", "replacement")}
    primary, created = await _open_session(args, session_id=ids["primary"], persona=args.persona)
    secondary, secondary_created = await _open_session(
        args,
        session_id=ids["secondary"],
        persona=args.secondary_persona,
    )
    capabilities = _capabilities(created)
    if _capabilities(secondary_created) != capabilities:
        raise AssertionError("concurrent sessions returned different capabilities")
    expected_capabilities = {
        "implementation_level": "model_native_duplex",
        "chunk_period_ms": 80,
        "supports_multi_session": True,
        "supports_multi_session_same_replica": True,
        "supports_barge_in": False,
    }
    if any(capabilities.get(key) != value for key, value in expected_capabilities.items()):
        raise AssertionError(f"unexpected PersonaPlex capabilities: {capabilities}")

    overflow, error = await _open_session(
        args,
        session_id=f"personaplex-overflow-{uuid.uuid4().hex}",
        persona=args.persona,
        expect_error=True,
    )
    error_body = error.get("error")
    overflow_code = error_body.get("code") if isinstance(error_body, dict) else error.get("code")
    await overflow.__aexit__(None, None, None)
    if overflow_code != "resource_exhausted":
        raise AssertionError(f"overflow session was not rejected: {error}")

    primary_frames, secondary_frames = await asyncio.gather(
        _stream_frames(primary, pcm),
        _stream_frames(secondary, pcm),
    )
    await asyncio.gather(
        wait_for(
            lambda: len(primary.events.audio_bytes()) // (2 * FRAME_SAMPLES) >= primary_frames - args.max_frame_deficit,
            timeout_s=args.timeout_s,
            label="primary frame coverage",
        ),
        wait_for(
            lambda: (
                len(secondary.events.audio_bytes()) // (2 * FRAME_SAMPLES) >= secondary_frames - args.max_frame_deficit
            ),
            timeout_s=args.timeout_s,
            label="secondary frame coverage",
        ),
    )
    await asyncio.sleep(args.drain_s)
    primary_audio, primary_response_ids, primary_stats = _session_result(
        primary,
        input_frames=primary_frames,
        args=args,
        minimum_chunks=args.minimum_audio_chunks,
    )
    secondary_audio, secondary_response_ids, _ = _session_result(
        secondary,
        input_frames=secondary_frames,
        args=args,
        minimum_chunks=args.minimum_audio_chunks,
    )
    if primary_response_ids & secondary_response_ids:
        raise AssertionError("concurrent sessions shared a response id")
    _save(output_dir, "primary", primary, primary_audio)
    await _close_session(primary, timeout_s=args.timeout_s)

    replacement, replacement_created = await _open_session(
        args,
        session_id=ids["replacement"],
        persona=args.replacement_persona,
    )
    if _capabilities(replacement_created) != capabilities:
        raise AssertionError("replacement session returned different capabilities")
    continuation_frames, replacement_frames = await asyncio.gather(
        _stream_frames(secondary, pcm, max_frames=args.replacement_frames),
        _stream_frames(replacement, pcm, max_frames=args.replacement_frames),
    )
    secondary_total_frames = secondary_frames + continuation_frames
    await asyncio.gather(
        wait_for(
            lambda: (
                len(secondary.events.audio_bytes()) // (2 * FRAME_SAMPLES)
                >= secondary_total_frames - args.max_frame_deficit
            ),
            timeout_s=args.timeout_s,
            label="survivor frame coverage after slot recycle",
        ),
        wait_for(
            lambda: (
                len(replacement.events.audio_bytes()) // (2 * FRAME_SAMPLES)
                >= replacement_frames - args.max_frame_deficit
            ),
            timeout_s=args.timeout_s,
            label="replacement frame coverage",
        ),
    )
    await asyncio.sleep(args.drain_s)
    secondary_audio, secondary_after_ids, secondary_stats = _session_result(
        secondary,
        input_frames=secondary_total_frames,
        args=args,
        minimum_chunks=args.minimum_audio_chunks + 1,
    )
    replacement_audio, replacement_response_ids, replacement_stats = _session_result(
        replacement,
        input_frames=replacement_frames,
        args=args,
        minimum_chunks=1,
    )
    if secondary_after_ids != secondary_response_ids:
        raise AssertionError("survivor response changed while another session slot was recycled")
    if replacement_response_ids & (primary_response_ids | secondary_response_ids):
        raise AssertionError("replacement session reused another session's response id")
    _save(output_dir, "secondary", secondary, secondary_audio)
    _save(output_dir, "replacement", replacement, replacement_audio)
    await asyncio.gather(
        _close_session(secondary, timeout_s=args.timeout_s),
        _close_session(replacement, timeout_s=args.timeout_s),
    )

    errors = primary.events.errors() + secondary.events.errors() + replacement.events.errors()
    result = {
        "ok": not errors,
        "model": args.model,
        "input": {**input_identity, "tail_s": args.tail_s},
        "capabilities": capabilities,
        "overflow_error_code": overflow_code,
        "primary": {"session_id": ids["primary"], "input_frames": primary_frames, **primary_stats},
        "secondary": {
            "session_id": ids["secondary"],
            "input_frames": secondary_total_frames,
            "continuation_frames_after_recycle": continuation_frames,
            **secondary_stats,
        },
        "replacement": {
            "session_id": ids["replacement"],
            "input_frames": replacement_frames,
            **replacement_stats,
        },
        "errors": errors,
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise AssertionError(f"Realtime sessions emitted errors: {errors}")
    return result


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "median": None, "p99": None, "max": None, "argmax_index": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": max(values),
        "argmax_index": int(np.argmax(values)),
    }


def _load_metrics(client: RawRealtimeProbe, sends: list[tuple[float, float, float]]) -> dict[str, object]:
    # Use the collector's local clock, never a timestamp supplied by the server.
    packets: list[dict[str, object]] = []
    response_ids: set[str] = set()
    packet_times: list[float] = []
    samples = 0
    for raw, received in zip(client.events.events, client.events.event_received_at_s):
        event = wrap_event(raw)
        chunk = event.audio if isinstance(event, AudioDelta) else None
        if chunk:
            samples += len(chunk) // 2
            packet_times.append(received)
            packets.append({"received_at_s": received, "samples": len(chunk) // 2, "response_id": event.response_id})
            if event.response_id:
                response_ids.add(event.response_id)
    intervals = [(b - a) * 1000 for a, b in zip(packet_times, packet_times[1:])]
    first_send = sends[0][1] if sends else None
    audio_duration = samples / SAMPLE_RATE_HZ
    first_audio_ms = (packet_times[0] - first_send) * 1000 if packet_times and first_send is not None else None
    client_rtf = (packet_times[-1] - first_send) / audio_duration if audio_duration and first_send is not None else None
    return {
        "input_frames": len(sends),
        "output_samples": samples,
        "frame_deficit": len(sends) - samples / FRAME_SAMPLES,
        "response_ids": sorted(response_ids),
        "client_first_audio_after_stream_start_ms": first_audio_ms,
        "client_stream_rtf": client_rtf,
        "client_audio_packet_interval_ms": _distribution(intervals),
        "client_send_lateness_ms": _distribution([max(0.0, sent - planned) * 1000 for planned, sent, _ in sends]),
        "input_timeline": [
            {"frame_index": i, "planned_at_s": planned, "send_started_at_s": sent, "send_completed_at_s": completed}
            for i, (planned, sent, completed) in enumerate(sends)
        ],
        "audio_packet_timeline": packets,
    }


def _check_load_connection(client: RawRealtimeProbe) -> None:
    if client.events.errors():
        raise RuntimeError("server emitted a Realtime error (see the session error count)")
    if client._reader_task is not None and client._reader_task.done():
        client._reader_task.result()  # Propagate malformed JSON/reader failures.
        raise RuntimeError("connection closed before session completion")


async def _wait_load_event(client: RawRealtimeProbe, event_type: str, timeout_s: float) -> None:
    def received() -> bool:
        if client.events.count(event_type):
            return True
        _check_load_connection(client)
        return False

    await wait_for(received, timeout_s=timeout_s, label=event_type)


async def _paced_load_frames(
    client: RawRealtimeProbe,
    pcm: np.ndarray,
    *,
    epoch: float,
    timeout_s: float,
    sends: list[tuple[float, float, float]],
) -> None:
    next_send = epoch
    for seq in range(math.ceil(pcm.size / FRAME_SAMPLES)):
        await asyncio.sleep(max(0.0, next_send - time.monotonic()))
        _check_load_connection(client)
        event = _frame_event(pcm[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES], seq)
        started = time.monotonic()
        await asyncio.wait_for(client.send(event), timeout=timeout_s)
        completed = time.monotonic()
        sends.append((epoch + seq * FRAME_PERIOD_S, started, completed))
        # Do not burst to catch up after a stalled send; report accumulated lag.
        next_send = max(epoch + (seq + 1) * FRAME_PERIOD_S, completed + FRAME_PERIOD_S)


async def _run_load_session(
    args: argparse.Namespace,
    pcm: np.ndarray,
    index: int,
    ready: asyncio.Future[None],
    start: asyncio.Future[float],
) -> dict[str, object]:
    session_id = f"personaplex-load-{index}-{uuid.uuid4().hex}"
    client = RawRealtimeProbe(_realtime_url(args.url, args.model, session_id))
    sends: list[tuple[float, float, float]] = []
    error: str | None = None
    cleanup_error: str | None = None
    metrics: dict[str, object] | None = None
    phase = "admission"
    try:
        async with client:
            try:
                await asyncio.wait_for(
                    client.send(_session_update(args, session_id=session_id, persona=args.persona)), args.timeout_s
                )
                await _wait_load_event(client, "session.created", args.timeout_s)
                capabilities = _capabilities(_events(client, "session.created")[-1])
                if (
                    capabilities.get("chunk_period_ms") != 80
                    or capabilities.get("supports_multi_session_same_replica") is not True
                ):
                    raise AssertionError("server did not advertise PersonaPlex multi-session capabilities")
                ready.set_result(None)
                phase = "stream"
                epoch = await asyncio.shield(start)
                await _paced_load_frames(client, pcm, epoch=epoch, timeout_s=args.timeout_s, sends=sends)
                phase = "drain"
                # Fixed, reported observation window. Closing must not extend
                # it and let a slow session pass by flushing after the cutoff.
                await asyncio.sleep(args.drain_s)
                _check_load_connection(client)
                _session_result(client, input_frames=len(sends), args=args, minimum_chunks=args.minimum_audio_chunks)
            finally:
                metrics = _load_metrics(client, sends)
                if client.events.count("session.created") and not client.events.count("session.closed"):
                    try:
                        await asyncio.wait_for(client.send({"type": "session.close"}), args.timeout_s)
                        await _wait_load_event(client, "session.closed", args.timeout_s)
                    except (OSError, asyncio.TimeoutError, ValueError, RuntimeError, WebSocketException) as exc:
                        cleanup_error = f"cleanup: {type(exc).__name__}"
    except (OSError, asyncio.TimeoutError, ValueError, RuntimeError, AssertionError, WebSocketException) as exc:
        # Exception messages may include handshake credentials or server data.
        # Export the phase/type only; retain structured server codes below.
        error = f"{phase}: {type(exc).__name__}"
    finally:
        # Admission failure must not leave the other sessions at the barrier.
        if not ready.done():
            ready.set_result(None)
    if metrics is None:
        metrics = _load_metrics(client, sends)
    if cleanup_error:
        error = f"{error}; {cleanup_error}" if error else cleanup_error
    if client.events.errors():
        error = error or "server emitted a Realtime error"
    first_audio = metrics["client_first_audio_after_stream_start_ms"]
    if isinstance(first_audio, float) and first_audio < 0:
        error = error or "audio arrived before the first input frame"
    rtf = metrics["client_stream_rtf"]
    if args.max_client_rtf is not None and isinstance(rtf, float) and rtf > args.max_client_rtf:
        error = error or f"client_stream_rtf exceeds {args.max_client_rtf}"
    return {
        "index": index,
        "session_id": session_id,
        "ok": error is None,
        "error": error,
        "server_error_count": len(client.events.errors()),
        "server_error_codes": [
            str(body.get("code", "unknown")) if isinstance(body := event.get("error"), dict) else "unknown"
            for event in client.events.errors()
        ],
        **metrics,
    }


async def run_load(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input_wav)
    identity = _input_identity(input_path, expected_sha256=args.expected_input_sha256)
    pcm = _read_wav_as_float32(input_path)
    if args.tail_s:
        pcm = np.concatenate([pcm, np.zeros(round(args.tail_s * SAMPLE_RATE_HZ), dtype=np.float32)])
    pcm = pcm[: args.load_frames * FRAME_SAMPLES]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite evidence from a previous sweep.
    result_path = output_dir / "load-result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    loop = asyncio.get_running_loop()
    start: asyncio.Future[float] = loop.create_future()
    ready: list[asyncio.Future[None]] = [loop.create_future() for _ in range(args.sessions)]
    tasks = [asyncio.create_task(_run_load_session(args, pcm, i, r, start)) for i, r in enumerate(ready)]
    try:
        await asyncio.gather(*ready)
        start.set_result(time.monotonic())
        rows = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    seen: set[str] = set()
    shared: set[str] = set()
    for row in rows:
        assert isinstance(row["response_ids"], list)
        response_ids = set(row["response_ids"])
        shared.update(seen & response_ids)
        seen.update(response_ids)
    for row in rows:
        assert isinstance(row["response_ids"], list)
        if shared.intersection(row["response_ids"]):
            row.update(ok=False, error="concurrent sessions shared a response id")
    result = {
        "schema_version": 1,
        "mode": "paced_multi_session",
        "model": args.model,
        "requested_sessions": args.sessions,
        "passed_sessions": sum(row["ok"] is True for row in rows),
        "ok": all(row["ok"] is True for row in rows),
        "input": {**identity, "tail_s": args.tail_s, "load_frames": args.load_frames},
        "frame_period_s": FRAME_PERIOD_S,
        "drain_s": args.drain_s,
        "max_client_rtf": args.max_client_rtf,
        "acceptance": {
            "max_frame_deficit": args.max_frame_deficit,
            "min_voiced_frames": args.min_voiced_frames,
            "minimum_audio_chunks": args.minimum_audio_chunks,
            "voiced_frame_rms_threshold": args.voiced_frame_rms_threshold,
        },
        "shared_response_ids": sorted(shared),
        "sessions": rows,
        "reported_server": {"revision": args.server_revision, "hardware": args.server_hardware},
        "measurement_origin": "client monotonic clock; admission excluded; streaming and drain retained",
        "interval_definition": "nonempty audio packet arrivals, not server ticks or per-frame inference latency",
    }
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument(
        "--expected-input-sha256",
        help="Fail before opening sessions when the input WAV does not match this SHA-256.",
    )
    parser.add_argument("--output-dir", default="/tmp/personaplex-realtime-duplex")
    parser.add_argument("--voice", default="NATF2.pt")
    parser.add_argument("--persona", default="You are a concise and helpful assistant.")
    parser.add_argument("--secondary-persona", default="You are a calm and precise assistant.")
    parser.add_argument("--replacement-persona", default="You are a cheerful travel assistant.")
    parser.add_argument("--tail-s", type=float, default=2.0)
    parser.add_argument("--drain-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--max-frame-deficit",
        type=int,
        default=4,
        help=(
            "Maximum unflushed tail frames. PersonaPlex Code2Wav emits five-frame "
            "chunks, so a session cutoff can leave at most four pending frames."
        ),
    )
    parser.add_argument("--voiced-frame-rms-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--min-voiced-frames",
        type=int,
        default=5,
        help=(
            "Require at least one normal five-frame Code2Wav chunk (400 ms) above "
            "the per-frame RMS threshold; this detects inaudible output, not speech quality."
        ),
    )
    parser.add_argument("--minimum-audio-chunks", type=int, default=2)
    parser.add_argument(
        "--replacement-frames",
        type=int,
        default=0,
        help="Frames sent after slot reuse; 0 replays the full input so the replacement must produce audible output.",
    )
    parser.add_argument(
        "--sessions", type=int, help="Opt into paced load mode with N sessions; omit for the lifecycle test."
    )
    parser.add_argument(
        "--load-frames", type=int, default=1000, help="Maximum input frames per load session (no repetition)."
    )
    parser.add_argument("--max-client-rtf", type=float, help="Optional client stream-window RTF acceptance ceiling.")
    parser.add_argument("--server-revision", help="User-reported server commit/version; not verified by this client.")
    parser.add_argument(
        "--server-hardware", help="User-reported server hardware; not inferred from the client machine."
    )
    args = parser.parse_args(argv)
    if args.sessions is not None:
        if not 1 <= args.sessions <= 64 or not 1 <= args.load_frames <= 10000:
            parser.error("load mode requires sessions in [1,64] and load-frames in [1,10000]")
        for name in ("timeout_s", "drain_s", "tail_s"):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0 or (name == "timeout_s" and value == 0):
                parser.error(f"{name} must be finite and non-negative (timeout must be positive)")
        if args.max_client_rtf is not None and (not math.isfinite(args.max_client_rtf) or args.max_client_rtf <= 0):
            parser.error("max-client-rtf must be finite and positive")
        if args.max_frame_deficit < 0 or args.min_voiced_frames < 1 or args.minimum_audio_chunks < 1:
            parser.error("frame-deficit must be non-negative; voiced frames and audio chunks must be positive")
        if not math.isfinite(args.voiced_frame_rms_threshold) or args.voiced_frame_rms_threshold < 0:
            parser.error("voiced-frame-rms-threshold must be finite and non-negative")
    return args


def main() -> None:
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("mode") == "paced_multi_session" and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
