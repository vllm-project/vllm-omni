# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Minimal single-input MiniCPM-o 4.5 Realtime duplex demo.

Run this after starting the duplex server. Strict lifecycle, overlap, and
multi-session validation lives under ``tests/e2e/online_serving``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
    wait_for,
    write_pcm16_wav,
)
from vllm_omni.experimental.fullduplex.client import chunk_period_ms as _chunk_period_ms  # noqa: E402
from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    has_residual_model_unit as _has_residual_model_unit,
)
from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    reference_audio_data_url as _ref_audio_data_url,
)
from vllm_omni.experimental.fullduplex.video_stacking import (  # noqa: E402
    concat_frames_b64,
    unit_subframe_offsets,
)


class _StreamingOutputWriter:
    """Persist and report output deltas as the client receives them."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audio_chunk_dir = output_dir / "audio_chunks"
        self.audio_chunk_paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_chunk_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "output.pcm").write_bytes(b"")

    def handle(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type in {
            "response.audio_transcript.delta",
            "response.output_text.delta",
        }:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                print(delta, end="", file=sys.stderr, flush=True)
            return
        if event_type != "response.audio.delta":
            return

        delta = event.get("delta") or event.get("audio")
        if not isinstance(delta, str) or not delta:
            return
        try:
            pcm16 = base64.b64decode(delta)
        except ValueError:
            return
        if not pcm16:
            return

        chunk_index = len(self.audio_chunk_paths) + 1
        chunk_path = self.audio_chunk_dir / f"chunk_{chunk_index:04d}.wav"
        sample_rate_hz = event.get("sample_rate_hz")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            sample_rate_hz = 24_000
        with (self.output_dir / "output.pcm").open("ab") as output_pcm:
            output_pcm.write(pcm16)
        write_pcm16_wav(chunk_path, pcm16, sample_rate_hz=sample_rate_hz)
        self.audio_chunk_paths.append(chunk_path)
        print(
            f"\n[audio chunk {chunk_index}: {len(pcm16)} bytes -> {chunk_path}]",
            file=sys.stderr,
            flush=True,
        )


class _StreamingEventCollector(RealtimeEventCollector):
    def __init__(self, writer: _StreamingOutputWriter) -> None:
        super().__init__()
        self._writer = writer

    def add(self, event: dict[str, object], *, received_at_s: float | None = None) -> None:
        super().add(event, received_at_s=received_at_s)
        self._writer.handle(self.events[-1])


def _input_committed_index(
    events: list[dict[str, object]],
    after_index: int,
) -> int | None:
    for index, event in enumerate(events[max(after_index, 0) :], start=max(after_index, 0)):
        if event.get("type") == "input_audio_buffer.committed":
            return index
    return None


def _post_commit_model_decision(
    events: list[dict[str, object]],
    committed_index: int | None,
) -> str | None:
    if committed_index is None:
        return None
    for event in events[committed_index + 1 :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            return "listen"
        if event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                return "speak"
    return None


def _latest_model_decision(
    events: list[dict[str, object]],
    after_index: int,
) -> str | None:
    decision: str | None = None
    for event in events[max(after_index, 0) :]:
        event_type = event.get("type")
        if event_type == "response.listen":
            decision = "listen"
        elif event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                decision = "speak"
    return decision


def _response_in_progress(events: list[dict[str, object]]) -> bool:
    return sum(event.get("type") == "response.created" for event in events) > sum(
        event.get("type") == "response.done" for event in events
    )


async def _commit_input_after_playback_checkpoint(client: RealtimeDuplexClient) -> float:
    """Commit input only after preserving already-played assistant history.

    Native duplex can finish several responses while the user is still
    streaming.  Under the ``ack_only`` playback policy, acknowledging those
    responses after the final input commit is intentionally rejected: it could
    append an old assistant message after the newly committed user message.
    Checkpoint playback first so the server commits each assistant item at its
    original history position.  A second acknowledgement after the commit can
    then safely advance response-local playback that drained in the meantime.
    """
    await client.acknowledge_playback()
    commit_sent_at_s = time.monotonic()
    await client.commit()
    return commit_sent_at_s


def _event_count_after(
    events: list[dict[str, object]],
    event_type: str,
    index: int | None,
) -> int:
    if index is None:
        return 0
    return sum(event.get("type") == event_type for event in events[index + 1 :])


def _probe_video_frames_and_fps(video_path: Path) -> tuple[int, float]:
    """Frame count and average FPS via PyAV (same approach as Daily-Omni)."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        avg_fps = float(stream.average_rate) if stream.average_rate else 0.0
        num_frames = stream.frames or 0
        if num_frames <= 0:
            duration: float | None = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            if duration and avg_fps:
                num_frames = int(math.floor(duration * avg_fps))
        if num_frames <= 0:
            stream.thread_type = "AUTO"
            num_frames = sum(1 for _ in container.decode(stream))
    if num_frames <= 0:
        raise ValueError(f"No decodable video frames in {video_path}")
    return num_frames, avg_fps or 1.0


def _upright(image, rotation: int):
    """Apply a container display-matrix rotation, in degrees, to ``image``.

    PyAV hands back the stored raster and leaves the display matrix for the
    caller, unlike the ffmpeg CLI which autorotates. Phone clips routinely
    carry +/-90, so skipping this feeds the model a sideways frame: arrows and
    digits stop being readable and the model mirrors the motion it reports.
    """
    from PIL import Image

    quarter_turns = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    turn = quarter_turns.get(int(rotation) % 360)
    return image.transpose(turn) if turn is not None else image


def _decode_video_frames_rgb(video_path: Path, frame_idx: list[int]) -> list:
    """Decode presentation-order ``frame_idx`` into upright RGB ``PIL.Image``s."""
    import av

    wanted = sorted({int(i) for i in frame_idx})
    if not wanted:
        return []

    decoded: dict[int, object] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        cursor = 0
        for position, frame in enumerate(container.decode(stream)):
            if wanted[cursor] > position:
                continue
            image = _upright(frame.to_image(), getattr(frame, "rotation", 0) or 0)
            while cursor < len(wanted) and wanted[cursor] <= position:
                decoded[wanted[cursor]] = image
                cursor += 1
            if cursor >= len(wanted):
                break

    if not decoded:
        raise ValueError(f"Decoded no frames from {video_path} for indices {wanted[:8]}")
    last_image = decoded[max(decoded)]
    return [decoded.get(int(i), last_image) for i in frame_idx]


def _resize_max_side(image, max_side: int):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side or max_side <= 0:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size)


def _jpeg_bytes(image, *, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _to_pcm16_bytes(audio_np) -> bytes:
    import numpy as np

    arr = np.asarray(audio_np).reshape(-1)
    if arr.dtype != np.int16:
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return arr.tobytes()


def _pcm16_wav_duration_s(path: Path) -> float:
    pcm16 = read_pcm16_wav(path)
    return len(pcm16) / float(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)


def _extract_video_frames(
    video_path: Path,
    work_dir: Path,
    *,
    fps: float,
    max_side: int,
    duration_s: float,
    stack_frames: int = 1,
) -> tuple[list[str], list[str | None]]:
    """Demux JPEG frames only. Frame count follows ``duration_s``, not soundtrack.

    Returns the base frames (one per model unit) and the parallel stacked track.
    With ``stack_frames=N`` each unit also samples its ``N-1`` interior
    sub-frames and tiles them into one composite, which is how official duplex
    raises the visual refresh rate while keeping one second of audio per unit.
    A unit whose interior falls past the end of the clip stacks nothing.
    """
    num_frames, avg_fps = _probe_video_frames_and_fps(video_path)
    sample_fps = max(float(fps), 1e-6)
    unit_s = 1.0 / sample_fps
    timeline_s = max(float(duration_s), unit_s)
    num_samples = max(1, int(math.ceil(timeline_s * sample_fps - 1e-9)))
    offsets = unit_subframe_offsets(stack_frames, unit_s=unit_s)

    def frame_at(seconds: float) -> int:
        # Official get_video_frame_audio_segments uses truncating int(), not
        # round(), so 29.97 fps lands on the same presentation index.
        return min(int(seconds * avg_fps), num_frames - 1)

    base_idx = [frame_at(unit * unit_s) for unit in range(num_samples)]
    sub_times = [
        [unit * unit_s + offset for offset in offsets if unit * unit_s + offset < timeline_s]
        for unit in range(num_samples)
    ]
    sub_idx = [frame_at(seconds) for times in sub_times for seconds in times]
    images = _decode_video_frames_rgb(video_path, base_idx + sub_idx)
    sub_images = iter(images[len(base_idx) :])

    frame_dir = work_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("*.jpg"):
        stale.unlink()
    frames_b64: list[str] = []
    stacked_b64: list[str | None] = []
    for index, (image, times) in enumerate(zip(images[: len(base_idx)], sub_times), start=1):
        jpeg = _jpeg_bytes(_resize_max_side(image, max_side))
        (frame_dir / f"frame_{index:05d}.jpg").write_bytes(jpeg)
        frames_b64.append(base64.b64encode(jpeg).decode("ascii"))
        interior = [_resize_max_side(next(sub_images), max_side) for _ in times]
        if not interior:
            stacked_b64.append(None)
            continue
        composite = concat_frames_b64(interior)
        (frame_dir / f"stack_{index:05d}.jpg").write_bytes(base64.b64decode(composite))
        stacked_b64.append(composite)
    return frames_b64, stacked_b64


def _extract_video_soundtrack(video_path: Path, work_dir: Path) -> Path:
    """Write a 16 kHz mono WAV from the video soundtrack."""
    try:
        from vllm.multimodal.media.audio import load_audio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--input-video needs vLLM audio helpers (load_audio)") from exc

    audio_np, sr = load_audio(str(video_path), sr=PCM16_SAMPLE_RATE, mono=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "input_16k.wav"
    write_pcm16_wav(audio_path, _to_pcm16_bytes(audio_np), sample_rate_hz=int(sr))
    return audio_path


def _extract_video_input(
    video_path: Path,
    work_dir: Path,
    *,
    fps: float,
    max_side: int,
    stack_frames: int,
) -> tuple[Path, list[str], list[str | None]]:
    """Demux a video into duplex inputs: 16 kHz mono WAV plus JPEG frames.

    Use this only when the video soundtrack is the audio source. When the
    caller already has ``--input-wav``, extract frames against that WAV
    duration instead so a silent clip cannot fail ``load_audio``.
    """
    audio_path = _extract_video_soundtrack(video_path, work_dir)
    frames_b64, stacked_b64 = _extract_video_frames(
        video_path,
        work_dir,
        fps=fps,
        max_side=max_side,
        duration_s=_pcm16_wav_duration_s(audio_path),
        stack_frames=stack_frames,
    )
    return audio_path, frames_b64, stacked_b64


def _resolve_duplex_av_inputs(
    *,
    input_wav: str | None,
    input_video: str | None,
    work_dir: Path,
    fps: float,
    max_side: int,
    stack_frames: int = 1,
) -> tuple[str, list[str], list[str | None]]:
    """Return ``(wav_path, frames_b64, stacked_b64)``. External WAV wins over video audio."""
    if input_video:
        video_path = Path(input_video)
        if input_wav:
            frames, stacked = _extract_video_frames(
                video_path,
                work_dir,
                fps=fps,
                max_side=max_side,
                duration_s=_pcm16_wav_duration_s(Path(input_wav)),
                stack_frames=stack_frames,
            )
            return input_wav, frames, stacked
        extracted_wav, frames, stacked = _extract_video_input(
            video_path,
            work_dir,
            fps=fps,
            max_side=max_side,
            stack_frames=stack_frames,
        )
        return str(extracted_wav), frames, stacked
    if not input_wav:
        raise SystemExit("provide --input-wav, --input-video, or both")
    return input_wav, [], []


async def run_demo(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    stream_writer = _StreamingOutputWriter(output_dir)

    input_wav, video_frames, stacked_video_frames = _resolve_duplex_av_inputs(
        input_wav=args.input_wav,
        input_video=args.input_video,
        work_dir=output_dir / "video_input",
        fps=args.video_fps,
        max_side=args.frame_max_side,
        stack_frames=args.stack_frames,
    )

    input_pcm16 = read_pcm16_wav(Path(input_wav))
    if not input_pcm16:
        raise ValueError("input WAV has no audio")

    url = build_realtime_url(
        args.url,
        args.model,
        autostart=False if args.ref_audio else None,
        session_id=args.session_id,
    )
    client = RealtimeDuplexClient(url)
    client.events = _StreamingEventCollector(stream_writer)
    async with client:
        await client.configure(
            args.model,
            ref_audio=_ref_audio_data_url(args.ref_audio),
            session_id=args.session_id,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        stream_event_cursor = len(client.events.events)
        frames_sent = await client.stream_pcm16(
            input_pcm16,
            chunk_ms=args.chunk_ms,
            realtime=not args.no_realtime_pacing,
            video_frames=video_frames,
            stacked_video_frames=stacked_video_frames,
        )
        commit_event_cursor = len(client.events.events)
        stream_decision = _latest_model_decision(client.events.events, stream_event_cursor)
        input_has_residual_model_unit = _has_residual_model_unit(
            input_pcm16,
            chunk_period_ms=_chunk_period_ms(client.events.events),
        )
        wait_for_post_commit_decision = False
        commit_sent_at_s = await _commit_input_after_playback_checkpoint(client)
        wait_error: str | None = None
        committed_index: int | None = None
        post_commit_decision: str | None = None
        try:
            await wait_for(
                lambda: _input_committed_index(client.events.events, commit_event_cursor) is not None,
                timeout_s=args.timeout_s,
                label="input_audio_buffer.committed",
            )
            committed_index = _input_committed_index(client.events.events, commit_event_cursor)
            stream_decision = _latest_model_decision(client.events.events[: committed_index + 1], stream_event_cursor)
            wait_for_post_commit_decision = input_has_residual_model_unit or _response_in_progress(
                client.events.events[: committed_index + 1]
            )
            if wait_for_post_commit_decision:
                await wait_for(
                    lambda: _post_commit_model_decision(client.events.events, committed_index) is not None,
                    timeout_s=args.timeout_s,
                    label="post-commit model decision or response drain",
                )
                post_commit_decision = _post_commit_model_decision(client.events.events, committed_index)
        except TimeoutError as exc:
            wait_error = str(exc)
        await client.acknowledge_playback()
        close_error: str | None = None
        try:
            await client.close_session(timeout_s=args.timeout_s)
        except TimeoutError as exc:
            close_error = str(exc)

        audio = client.events.audio_bytes()
        first_text_at_s = client.events.first_received_at(
            "response.audio_transcript.delta",
            "response.output_text.delta",
            after_s=commit_sent_at_s,
        )
        first_audio_at_s = client.events.first_received_at(
            "response.audio.delta",
            after_s=commit_sent_at_s,
        )
        response_created_at_s = client.events.first_received_at(
            "response.created",
            after_s=commit_sent_at_s,
        )
        response_done_at_s = client.events.first_received_at(
            "response.done",
            after_s=commit_sent_at_s,
        )
        audio_duration_s = len(audio) / (client.events.output_sample_rate_hz * 2)
        response_generation_s = (
            response_done_at_s - response_created_at_s
            if response_done_at_s is not None and response_created_at_s is not None
            else None
        )
        transcript_deltas = [
            str(event.get("delta", ""))
            for event in client.events.events
            if event.get("type")
            in {
                "response.audio_transcript.delta",
                "response.output_text.delta",
            }
        ]
        response_id = client.events.response_ids[0] if client.events.response_ids else None
        timing = client.events.timing_summary(
            after_s=commit_sent_at_s,
            input_committed_at_s=commit_sent_at_s,
            response_id=response_id,
        )
        errors = client.events.errors()
        if wait_error:
            errors.append({"type": "client.timeout", "message": wait_error})
        if close_error:
            errors.append({"type": "client.timeout", "message": close_error})
        model_decision = post_commit_decision or stream_decision
        (output_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in client.events.events),
            encoding="utf-8",
        )
        (output_dir / "output.pcm").write_bytes(audio)
        if audio:
            write_pcm16_wav(
                output_dir / "output.wav",
                audio,
                sample_rate_hz=client.events.output_sample_rate_hz,
            )

        result = {
            "ok": (
                client.events.count("session.created") > 0
                and client.events.count("session.closed") > 0
                and not errors
                and model_decision is not None
                and (bool(audio) or not args.require_audio)
            ),
            "model_decision": model_decision,
            "post_commit": {
                "input_committed_event_index": committed_index,
                "decision": post_commit_decision,
                "decision_required": wait_for_post_commit_decision,
                "input_had_residual_model_unit": input_has_residual_model_unit,
                "response_listen_count": _event_count_after(
                    client.events.events,
                    "response.listen",
                    committed_index,
                ),
                "response_done_count": _event_count_after(
                    client.events.events,
                    "response.done",
                    committed_index,
                ),
            },
            "input_video": args.input_video,
            "video_frames_extracted": len(video_frames),
            "video_frames_sent": frames_sent,
            "video_stacked_frames_extracted": sum(frame is not None for frame in stacked_video_frames),
            "audio_bytes": len(audio),
            "audio_chunk_count": len(stream_writer.audio_chunk_paths),
            "audio_chunk_files": [str(path) for path in stream_writer.audio_chunk_paths],
            "output_sample_rate_hz": client.events.output_sample_rate_hz,
            "latency": {
                "ttft_ms": (
                    round((first_text_at_s - commit_sent_at_s) * 1000, 2) if first_text_at_s is not None else None
                ),
                "ttfp_ms": (
                    round((first_audio_at_s - commit_sent_at_s) * 1000, 2) if first_audio_at_s is not None else None
                ),
                "rtf": (
                    round(response_generation_s / audio_duration_s, 4)
                    if response_generation_s is not None and audio_duration_s > 0
                    else None
                ),
                "response_generation_ms": (
                    round(response_generation_s * 1000, 2) if response_generation_s is not None else None
                ),
                "text_stream_ms": (
                    round((response_done_at_s - first_text_at_s) * 1000, 2)
                    if response_done_at_s is not None and first_text_at_s is not None
                    else None
                ),
                "transcript_delta_count": len(transcript_deltas),
                "audio_duration_s": round(audio_duration_s, 3),
                "measurement_origin": "input_audio_buffer.commit send",
            },
            "timing": timing,
            "response_ids": client.events.response_ids,
            "transcript": "".join(transcript_deltas),
            "errors": errors,
            "output_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--input-wav",
        help="Mono 16 kHz PCM16 clip to stream. Optional when --input-video carries the audio track.",
    )
    parser.add_argument(
        "--input-video",
        help=(
            "Video to stream as omni duplex input. PyAV demuxes frames and "
            "vLLM load_audio extracts a 16 kHz mono WAV (unless --input-wav overrides the audio)."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Frame extraction rate. The wire contract carries ~1 frame per second of audio.",
    )
    parser.add_argument(
        "--frame-max-side",
        type=int,
        default=0,
        help=(
            "Optional client-side downscale of the longest side, in pixels. "
            "0 (default) keeps the capture resolution so the server "
            "process_image can normalize at scale_resolution=448 from a "
            "higher-quality source. The web demo never resizes on the client."
        ),
    )
    parser.add_argument(
        "--stack-frames",
        type=int,
        default=1,
        help=(
            "Frames sampled per 1 s audio unit. 1 (default) sends the unit's "
            "frame alone. N > 1 also samples the N-1 sub-frames inside the unit "
            "and tiles them into one composite sent next to the base frame, so "
            "motion (arrows, doors, digits changing) is visible at N fps while "
            "the audio stays 1 s per unit. Official uses 5 for high refresh "
            "rate mode; the wire carries 2 images per unit at any N because the "
            "sub-frames share one composite."
        ),
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help=(
            "Reference WAV for the MiniCPM-o duplex assistant voice. "
            "This demo matches the official flow by always providing a reference audio clip."
        ),
    )
    parser.add_argument("--output-dir", default="/tmp/minicpmo_realtime_duplex_demo")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Stage 0 sampling temperature; omit to preserve the model default.",
    )
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--require-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run_demo(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
