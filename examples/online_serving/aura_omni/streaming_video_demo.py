#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA streaming video demo — real-time frames, timed audio, verbose WS log.

Two send modes:
  1. Steady (``--burst-interval 0``): ``--send-fps`` frames per wall-clock second.
  2. Burst (``--burst-interval 5 --burst-frames 2``): every 5s send 2 frames.

Sampling vs sending (decoupled):
  ``--sample-fps`` controls how many frames are *extracted* per second of source video.
  ``--send-fps`` / ``--fps`` controls how fast frames are *sent* over the WebSocket.
  When omitted, ``--sample-fps`` defaults to the send rate (legacy behaviour).

Usage:
    # Sample 8 frames/s from video, send 2 frames/s (49s clip -> ~392 frames, ~196s wall)
    python examples/online_serving/aura_omni/streaming_video_demo.py \\
        --video /public/wtk/aura_prompts/aura_test.mp4 \\
        --burst-interval 0 --sample-fps 8 --send-fps 2 \\
        --audio-schedule 4:/public/wtk/aura_prompts/01_frame_what.wav \\
        --no-evs
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import websockets
except ImportError:
    print("pip install websockets", file=sys.stderr)
    sys.exit(1)


SILENT_TEXT = "<|silent|>"
# Keep in sync with vllm_omni.model_executor.stage_input_processors.aura_session_history.
_AURA_PUNCT_CHARS = frozenset(".,!?;:，。！？；：、'\"()[]{}''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》﹑·")


def is_effectively_silent(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped == SILENT_TEXT:
        return True
    if not stripped:
        return True
    return all(ch.isspace() or ch in _AURA_PUNCT_CHARS for ch in stripped)


@dataclass
class SessionLog:
    turn: int = 0
    frames_sent: int = 0
    audio_sent: bool = False
    events: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        self.events.append(line)
        print(line, flush=True)


def _load_pcm16_16k(path: str) -> bytes:
    """Load PCM16 16 kHz mono from raw PCM or WAV (incl. float32 IEEE WAV)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"RIFF":
        return raw

    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if sr != 16000:
        n_out = int(round(len(data) * 16000 / sr))
        if n_out > 0 and len(data) > 0:
            x_old = np.arange(len(data), dtype=np.float64)
            x_new = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = 16000
    pcm_i16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()


def _summarize_outbound(msg: dict[str, Any]) -> str:
    t = msg.get("type", "?")
    if t == "session.config":
        return (
            f"session.config modalities={msg.get('modalities')} auto_trigger_min={msg.get('auto_trigger_min_frames')}"
        )
    if t == "video.frame":
        data = msg.get("data", "")
        return f"video.frame b64_len={len(data)}"
    if t == "audio.chunk":
        data = msg.get("data", "")
        return f"audio.chunk pcm_bytes={len(base64.b64decode(data)) if data else 0}"
    if t == "video.done":
        return "video.done"
    return json.dumps({k: v for k, v in msg.items() if k != "data"}, ensure_ascii=False)


def _summarize_inbound(msg: dict[str, Any]) -> str:
    t = msg.get("type", "?")
    if t == "response.text.delta":
        d = msg.get("delta", "")
        return f"response.text.delta {json.dumps(d, ensure_ascii=False)}"
    if t == "response.text.done":
        text = msg.get("text", "")
        silent = is_effectively_silent(text)
        label = " (silent)" if silent else ""
        return f"response.text.done {json.dumps(text, ensure_ascii=False)}{label}"
    if t == "response.audio.delta":
        data = msg.get("data", "")
        return f"response.audio.delta wav_b64_len={len(data)}"
    if t == "response.audio.done":
        return "response.audio.done"
    if t == "response.start":
        return "response.start"
    if t == "session.done":
        return "session.done"
    if t == "error":
        return f"error {msg.get('message')}"
    return json.dumps(msg, ensure_ascii=False)[:500]


async def _receiver(ws: Any, log: SessionLog, done: asyncio.Event) -> None:
    turn_text: list[str] = []
    while not done.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            break
        data = json.loads(raw)
        msg_type = data.get("type")
        ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"

        if msg_type == "response.start":
            log.turn += 1
            turn_text = []
            log.note(f"\n{'=' * 60}\n[{ts}] <<< TURN {log.turn} response.start")
        elif msg_type == "response.text.delta":
            delta = data.get("delta", "")
            turn_text.append(delta)
            log.note(f"[{ts}] <<< {_summarize_inbound(data)}")
        elif msg_type == "response.text.done":
            full = data.get("text", "")
            log.note(f"[{ts}] <<< {_summarize_inbound(data)}")
            log.note(f"         turn {log.turn} full_text={json.dumps(full, ensure_ascii=False)}")
        elif msg_type == "response.audio.delta":
            log.note(f"[{ts}] <<< {_summarize_inbound(data)}")
        elif msg_type == "response.audio.done":
            log.note(f"[{ts}] <<< response.audio.done")
        elif msg_type == "session.done":
            log.note(f"\n[{ts}] <<< session.done")
            log.note(
                f"Session summary: {log.turn} turn(s), {log.frames_sent} frame(s) sent, audio_sent={log.audio_sent}"
            )
            done.set()
            break
        elif msg_type == "error":
            log.note(f"[{ts}] <<< ERROR: {data.get('message')}")
            done.set()
            break
        else:
            log.note(f"[{ts}] <<< {_summarize_inbound(data)}")


async def _send_json(ws: Any, msg: dict[str, Any], log: SessionLog) -> None:
    ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    await ws.send(json.dumps(msg))
    log.note(f"[{ts}] >>> {_summarize_outbound(msg)}")


@dataclass
class AudioScheduleItem:
    at_sec: float
    pcm: bytes
    label: str
    injected: bool = False


def _resolve_sample_send_fps(args: argparse.Namespace) -> tuple[float, float]:
    """Return (sample_fps, send_fps). Sample = extract from file; send = wall-clock wire rate."""
    send_fps = args.send_fps if args.send_fps is not None else args.fps
    sample_fps = args.sample_fps if args.sample_fps is not None else send_fps
    return sample_fps, send_fps


def _parse_audio_schedule(
    audio: str | None,
    audio_at_sec: float,
    schedule_args: list[str],
) -> list[AudioScheduleItem]:
    """Build sorted injection list from --audio-schedule and legacy --audio."""
    items: list[AudioScheduleItem] = []
    for spec in schedule_args:
        if ":" not in spec:
            raise ValueError(f"Invalid --audio-schedule {spec!r}; use SEC:PATH")
        sec_s, path = spec.split(":", 1)
        items.append(
            AudioScheduleItem(
                at_sec=float(sec_s),
                pcm=_load_pcm16_16k(path),
                label=path,
            )
        )
    if audio and not schedule_args:
        items.append(
            AudioScheduleItem(
                at_sec=audio_at_sec,
                pcm=_load_pcm16_16k(audio),
                label=audio,
            )
        )
    items.sort(key=lambda x: x.at_sec)
    return items


async def _inject_due_audio(
    ws: Any,
    log: SessionLog,
    schedule: list[AudioScheduleItem],
    t0: float,
) -> None:
    elapsed = time.monotonic() - t0
    chunk_size = 16000 * 2
    for item in schedule:
        if item.injected or elapsed < item.at_sec:
            continue
        n_chunks = 0
        for offset in range(0, len(item.pcm), chunk_size):
            chunk = item.pcm[offset : offset + chunk_size]
            await _send_json(
                ws,
                {"type": "audio.chunk", "data": base64.b64encode(chunk).decode()},
                log,
            )
            n_chunks += 1
        item.injected = True
        log.audio_sent = True
        log.note(
            f"*** Injected audio at t={elapsed:.2f}s "
            f"({n_chunks} chunk(s), {len(item.pcm)} pcm bytes) from {item.label} ***"
        )


async def _read_encoded_frame(cap: Any, src_idx: int, step: int) -> tuple[int, bytes | None]:
    import cv2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return src_idx, None
        if src_idx % step == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                return src_idx + 1, buf.tobytes()
        src_idx += 1
    return src_idx, None


async def _stream_video_burst(
    ws: Any,
    log: SessionLog,
    args: argparse.Namespace,
    cap: Any,
    audio_schedule: list[AudioScheduleItem],
    done: asyncio.Event,
    step: int,
) -> None:
    audio_desc = ", ".join(f"{x.at_sec}s:{x.label}" for x in audio_schedule) or "none"
    log.note(
        f"Send mode: burst — {args.burst_frames} frame(s) every {args.burst_interval}s "
        f"(audio: {audio_desc}, auto_trigger_min={args.auto_trigger_min_frames})"
    )
    t0 = time.monotonic()
    src_idx = 0
    sent_idx = 0
    burst_idx = 0

    while cap.isOpened() and not done.is_set():
        elapsed = time.monotonic() - t0
        if args.max_duration and elapsed >= args.max_duration:
            log.note(f"Reached --max-duration {args.max_duration}s, stopping")
            break
        if args.max_bursts and burst_idx >= args.max_bursts:
            log.note(f"Reached --max-bursts {args.max_bursts}, stopping")
            break

        await _inject_due_audio(ws, log, audio_schedule, t0)

        target = burst_idx * args.burst_interval
        if elapsed < target:
            await asyncio.sleep(min(0.2, target - elapsed))
            continue

        log.note(f"--- burst #{burst_idx + 1} at t={elapsed:.2f}s ---")
        for i in range(args.burst_frames):
            src_idx, jpeg = await _read_encoded_frame(cap, src_idx, step)
            if jpeg is None:
                log.note(f"End of video after {sent_idx} sent frame(s)")
                cap.release()
                await _send_json(ws, {"type": "video.done"}, log)
                return
            await _send_json(
                ws,
                {"type": "video.frame", "data": base64.b64encode(jpeg).decode()},
                log,
            )
            sent_idx += 1
            log.frames_sent = sent_idx
            if i + 1 < args.burst_frames:
                await asyncio.sleep(args.in_burst_gap_ms / 1000.0)

        burst_idx += 1

    cap.release()
    await _send_json(ws, {"type": "video.done"}, log)


async def _stream_video_steady(
    ws: Any,
    log: SessionLog,
    args: argparse.Namespace,
    cap: Any,
    audio_schedule: list[AudioScheduleItem],
    done: asyncio.Event,
    step: int,
    interval_s: float,
    duration: float,
    src_fps: float,
) -> None:
    audio_desc = ", ".join(f"{x.at_sec}s:{x.label}" for x in audio_schedule) or "none"
    sample_fps, send_fps = _resolve_sample_send_fps(args)
    log.note(
        f"Send mode: steady — sample {sample_fps} fps, send {send_fps} fps "
        f"(every {interval_s:.3f}s), audio: {audio_desc}"
    )
    t0 = time.monotonic()
    src_idx = 0
    sent_idx = 0

    while cap.isOpened() and not done.is_set():
        elapsed = time.monotonic() - t0
        if args.max_duration and elapsed >= args.max_duration:
            log.note(f"Reached --max-duration {args.max_duration}s, stopping frames")
            break

        await _inject_due_audio(ws, log, audio_schedule, t0)

        src_idx, jpeg = await _read_encoded_frame(cap, src_idx, step)
        if jpeg is None:
            log.note(f"End of video after {sent_idx} sent frame(s)")
            break

        await _send_json(
            ws,
            {"type": "video.frame", "data": base64.b64encode(jpeg).decode()},
            log,
        )
        sent_idx += 1
        log.frames_sent = sent_idx
        await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - t0 - sent_idx * interval_s)))

    cap.release()
    await _send_json(ws, {"type": "video.done"}, log)


async def _stream_video(
    ws: Any,
    log: SessionLog,
    args: argparse.Namespace,
    audio_schedule: list[AudioScheduleItem],
    done: asyncio.Event,
) -> None:
    try:
        import cv2
    except ImportError:
        print("pip install opencv-python", file=sys.stderr)
        done.set()
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.note(f"Cannot open video: {args.video}")
        done.set()
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / src_fps if total_frames > 0 else 0.0
    sample_fps, send_fps = _resolve_sample_send_fps(args)
    step = max(1, int(round(src_fps / max(sample_fps, 0.1))))
    interval_s = 1.0 / max(send_fps, 0.1)
    est_frames = int(total_frames / step) if total_frames > 0 else 0
    est_wall_s = est_frames / send_fps if send_fps > 0 else 0.0

    log.note(
        f"Video: {args.video} src_fps={src_fps:.2f} duration≈{duration:.1f}s "
        f"sample_fps={sample_fps} send_fps={send_fps} step={step} "
        f"est_frames≈{est_frames} est_wall≈{est_wall_s:.1f}s"
    )

    modalities = ["text"] if args.text_only else ["text", "audio"]
    config: dict[str, Any] = {
        "type": "session.config",
        "model": args.model,
        "modalities": modalities,
        "max_frames": args.max_frames,
        "auto_trigger": True,
        "auto_trigger_min_frames": args.auto_trigger_min_frames,
        "max_frames_per_round": args.max_frames_per_round,
        "enable_frame_filter": not args.no_evs,
        "frame_filter_threshold": args.evs_threshold,
    }
    if args.max_rounds is not None:
        config["max_rounds"] = args.max_rounds
    if args.num_rounds_keep is not None:
        config["num_rounds_keep"] = args.num_rounds_keep
    if args.max_context_qas is not None:
        config["max_context_qas"] = args.max_context_qas
    if args.no_pruning:
        config["pruning_enabled"] = False
    await _send_json(ws, config, log)

    if args.burst_interval > 0:
        await _stream_video_burst(ws, log, args, cap, audio_schedule, done, step)
    else:
        await _stream_video_steady(ws, log, args, cap, audio_schedule, done, step, interval_s, duration, src_fps)


async def run(args: argparse.Namespace) -> None:
    if not args.video:
        print("--video is required for this demo", file=sys.stderr)
        sys.exit(1)

    uri = args.url or f"ws://{args.host}:{args.port}/v1/video/chat/stream"
    print(f"Connecting to {uri} ...", flush=True)

    try:
        audio_schedule = _parse_audio_schedule(args.audio, args.audio_at_sec, args.audio_schedule)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    for item in audio_schedule:
        print(f"Audio schedule: t={item.at_sec}s -> {item.label} ({len(item.pcm)} pcm bytes)")

    log = SessionLog()

    async with websockets.connect(uri, max_size=32 * 1024 * 1024) as ws:
        done = asyncio.Event()
        recv_task = asyncio.create_task(_receiver(ws, log, done))
        try:
            print("Connected. Streaming video/audio ...", flush=True)
            await _stream_video(ws, log, args, audio_schedule, done)
            await asyncio.wait_for(done.wait(), timeout=args.recv_timeout)
        except asyncio.TimeoutError:
            log.note(f"Timed out after {args.recv_timeout}s waiting for session.done")
        finally:
            if not done.is_set():
                done.set()
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task


def main() -> None:
    p = argparse.ArgumentParser(description="AURA streaming video demo (verbose, real-time)")
    p.add_argument("--url")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="aurateam/AURA")
    p.add_argument("--video", required=True, help="Long video file (mp4, etc.)")
    p.add_argument("--audio", help="WAV or PCM16 16kHz mono; injected at --audio-at-sec")
    p.add_argument("--audio-at-sec", type=float, default=8.0, help="Wall-clock sec to send full audio")
    p.add_argument(
        "--audio-schedule",
        action="append",
        default=[],
        metavar="SEC:PATH",
        help="Repeatable timed audio, e.g. 3:/path/a.wav 10:/path/b.wav",
    )
    p.add_argument(
        "--burst-interval",
        type=float,
        default=5.0,
        help="If >0: send --burst-frames every N seconds (easier to read logs). Set 0 for --fps mode.",
    )
    p.add_argument("--burst-frames", type=int, default=2, help="Frames per burst when --burst-interval > 0")
    p.add_argument(
        "--in-burst-gap-ms",
        type=int,
        default=100,
        help="Gap between frames inside one burst (ms)",
    )
    p.add_argument("--max-bursts", type=int, default=0, help="Stop after N bursts (0=unlimited)")
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Steady mode send rate (frames/s on wire). Alias for --send-fps when that is unset.",
    )
    p.add_argument(
        "--send-fps",
        type=float,
        default=None,
        help="Wall-clock send rate in steady mode. Overrides --fps when set.",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Frames extracted per second of source video (decoupled from send rate). Default: same as send fps.",
    )
    p.add_argument("--max-duration", type=float, default=0, help="Stop sending after N seconds (0=all)")
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--max-frames-per-round", type=int, default=16)
    p.add_argument("--auto-trigger-min-frames", type=int, default=2)
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="SessionHistory sliding-window limit before prune (server default 45). "
        "aura_test.mp4 steady 2fps yields ~43 turns — use e.g. 30 to exercise prune.",
    )
    p.add_argument(
        "--num-rounds-keep",
        type=int,
        default=None,
        help="Rounds kept in sliding window after prune (server default 30).",
    )
    p.add_argument(
        "--max-context-qas",
        type=int,
        default=None,
        help="Max QAs in compressed context history after prune (server default 10).",
    )
    p.add_argument("--no-pruning", action="store_true", help="Disable SessionHistory pruning.")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--no-evs", action="store_true")
    p.add_argument("--evs-threshold", type=float, default=0.95)
    p.add_argument("--recv-timeout", type=float, default=600.0)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
