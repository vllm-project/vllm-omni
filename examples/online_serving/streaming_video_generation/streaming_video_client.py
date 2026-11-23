#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WebSocket client for `/v1/realtime/video`.

The client prints a line for every binary video chunk it receives and writes
the received bytes to disk when the session finishes or is interrupted.
``--image-reference`` sends a first frame for image-conditioned models such as
LingBot-World 2.0; text-only models need no image.

Requirements:
    pip install av websockets
"""

import argparse
import asyncio
import base64
import io
import json
import mimetypes
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
from camera_utils import wasd_to_camera_payload
from websockets.exceptions import ConnectionClosed

try:
    from websockets.asyncio.client import connect  # pyright: ignore[reportMissingImports]
except ImportError:
    try:
        from websockets import connect  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("Please install websockets: pip install websockets")
        raise SystemExit(1)


HELIOS_DISTILLED_EXTRA_PARAMS = {
    "is_enable_stage2": True,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [1, 1, 1],
    "is_amplify_first_chunk": True,
}

DEFAULT_TRANSITION_CHUNKS = 3


@dataclass(frozen=True)
class ScheduledPromptUpdate:
    """Client-side schedule entry for a midway text ``session.interaction``."""

    at: float
    prompt: str
    transition_chunks: int


@dataclass(frozen=True)
class ScheduledCameraUpdate:
    """Client-side schedule entry for a midway camera ``session.interaction``."""

    at: float
    camera: dict[str, Any]
    transition_chunks: int | None


def _image_reference(value: str) -> dict[str, Any]:
    """Accept an http(s) URL as-is, or inline a local image as a data URL."""
    if value.startswith(("http://", "https://", "data:")):
        return {"image_url": value}
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"--image-reference is neither a URL nor a readable file: {value}")
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"image_url": f"data:{media_type};base64,{encoded}"}


def _parse_extra_params(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--extra-params must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--extra-params must be a JSON object")
    return parsed


def _parse_prompt_updates(value: str) -> list[ScheduledPromptUpdate]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--prompt-updates must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("--prompt-updates must be a JSON array")

    updates: list[ScheduledPromptUpdate] = []
    for index, item in enumerate(parsed):
        try:
            at = float(item["at"])
            prompt = item["prompt"]
        except (KeyError, TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"--prompt-updates[{index}] must have valid 'at' and 'prompt' fields"
            ) from exc
        if at < 0:
            raise argparse.ArgumentTypeError(f"--prompt-updates[{index}].at must be >= 0")
        if not isinstance(prompt, str) or not prompt.strip():
            raise argparse.ArgumentTypeError(f"--prompt-updates[{index}].prompt must be a non-empty string")

        try:
            transition_chunks = item.get("transition_chunks", DEFAULT_TRANSITION_CHUNKS)
            transition_chunks = int(transition_chunks)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"--prompt-updates[{index}].transition_chunks must be an integer") from exc
        if transition_chunks < 0:
            raise argparse.ArgumentTypeError(f"--prompt-updates[{index}].transition_chunks must be >= 0")

        updates.append(
            ScheduledPromptUpdate(
                at=at,
                prompt=prompt,
                transition_chunks=transition_chunks,
            )
        )

    return sorted(updates, key=lambda update: update.at)


def _parse_camera_updates(value: str) -> list[ScheduledCameraUpdate]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--camera-updates must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("--camera-updates must be a JSON array")

    updates: list[ScheduledCameraUpdate] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise argparse.ArgumentTypeError(f"--camera-updates[{index}] must be a JSON object")
        try:
            at = float(item["at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"--camera-updates[{index}] must have a valid 'at' field") from exc
        if at < 0:
            raise argparse.ArgumentTypeError(f"--camera-updates[{index}].at must be >= 0")

        if "camera" in item:
            camera = item["camera"]
            if not isinstance(camera, dict):
                raise argparse.ArgumentTypeError(f"--camera-updates[{index}].camera must be an object")
        elif "actions" in item:
            actions = item["actions"]
            if not isinstance(actions, list):
                raise argparse.ArgumentTypeError(f"--camera-updates[{index}].actions must be a list of WASD/IJKL keys")
            mode = item.get("mode", "velocity")
            if mode not in ("target", "velocity"):
                raise argparse.ArgumentTypeError(f"--camera-updates[{index}].mode must be target or velocity")
            camera = wasd_to_camera_payload(actions, mode=mode)
        else:
            raise argparse.ArgumentTypeError(
                f"--camera-updates[{index}] must provide 'camera' (structural SE3) "
                "or 'actions' (client-side WASD helper)"
            )

        transition_chunks: int | None
        if "transition_chunks" not in item:
            transition_chunks = None if camera.get("mode") == "velocity" else DEFAULT_TRANSITION_CHUNKS
        else:
            try:
                transition_chunks = int(item["transition_chunks"])
            except (TypeError, ValueError) as exc:
                raise argparse.ArgumentTypeError(
                    f"--camera-updates[{index}].transition_chunks must be an integer"
                ) from exc
            if transition_chunks < 0:
                raise argparse.ArgumentTypeError(f"--camera-updates[{index}].transition_chunks must be >= 0")

        updates.append(ScheduledCameraUpdate(at=at, camera=camera, transition_chunks=transition_chunks))

    return sorted(updates, key=lambda update: update.at)


def _maybe_set(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def build_session_start(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.start",
        "model": args.model,
        "prompt": args.prompt,
    }

    _maybe_set(payload, "image_reference", args.image_reference)
    _maybe_set(payload, "negative_prompt", args.negative_prompt)
    _maybe_set(payload, "width", args.width)
    _maybe_set(payload, "height", args.height)
    _maybe_set(payload, "size", args.size)
    _maybe_set(payload, "seconds", args.seconds)
    _maybe_set(payload, "fps", args.fps)
    _maybe_set(payload, "num_frames", args.num_frames)
    _maybe_set(payload, "num_inference_steps", args.num_inference_steps)
    _maybe_set(payload, "guidance_scale", args.guidance_scale)
    _maybe_set(payload, "guidance_scale_2", args.guidance_scale_2)
    _maybe_set(payload, "boundary_ratio", args.boundary_ratio)
    _maybe_set(payload, "flow_shift", args.flow_shift)
    _maybe_set(payload, "true_cfg_scale", args.true_cfg_scale)
    _maybe_set(payload, "seed", args.seed)

    extra_params: dict[str, Any] = {}
    if "Helios-Distilled" in args.model and args.helios_distilled_preset:
        extra_params.update(HELIOS_DISTILLED_EXTRA_PARAMS)
    if args.extra_params:
        extra_params.update(args.extra_params)
    if extra_params:
        payload["extra_params"] = extra_params

    return payload


async def _run_prompt_update_scheduler(
    websocket: Any,
    updates: list[ScheduledPromptUpdate],
    *,
    video_started_at: float,
    send_lock: asyncio.Lock,
) -> None:
    """Send one session's prompt schedule with IDs independent of timestamps."""
    for index, update in enumerate(updates):
        delay = (video_started_at + update.at) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        payload = {
            "type": "session.interaction",
            "interaction": {
                "event_id": f"scheduled-prompt-{index}",
                "event": {"prompt": update.prompt},
                "transition_chunks": update.transition_chunks,
            },
        }
        async with send_lock:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        print(f"Sent session.interaction at t={update.at:.2f}s: {json.dumps(payload, ensure_ascii=False)}")


async def _run_camera_update_scheduler(
    websocket: Any,
    updates: list[ScheduledCameraUpdate],
    *,
    video_started_at: float,
    send_lock: asyncio.Lock,
) -> None:
    """Send one session's camera schedule with IDs independent of timestamps."""
    for index, update in enumerate(updates):
        delay = (video_started_at + update.at) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        interaction: dict[str, Any] = {
            "event_id": f"scheduled-camera-{index}",
            "event": {"multi_modal_data": {"camera": update.camera}},
        }
        if update.transition_chunks is not None:
            interaction["transition_chunks"] = update.transition_chunks
        payload = {"type": "session.interaction", "interaction": interaction}
        async with send_lock:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        print(f"Sent session.interaction at t={update.at:.2f}s: {json.dumps(payload, ensure_ascii=False)}")


class _StreamingFrameCounter:
    """Count frames once on a worker, retaining fMP4 context between messages.

    PyAV reads from this non-seekable input on a daemon thread. Reads wait
    for bytes until finish() or a failure closes the input. Feed and snapshots
    never wait for decoding; accounting failures make the count unknown.
    """

    _MAX_PENDING_BYTES = 64 * 1024 * 1024
    _JOIN_TIMEOUT = 5.0

    def __init__(self, *, stream_format: str) -> None:
        """Initialize an empty counter; start decoding on the first nonempty feed.

        ``stream_format='m4s'`` selects the MP4 demuxer; other values let PyAV
        detect the container format from the input bytes.
        """
        self._format = "mp4" if stream_format == "m4s" else None
        self._condition = threading.Condition()
        self._pending: deque[memoryview] = deque()
        self._pending_bytes = 0
        self._eof = False
        self._count = 0
        self._error: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def count(self) -> int | None:
        """Return a decoded-frame snapshot, or None after any accounting failure.

        The snapshot can lag received input. Use finish() to drain the stream
        before reading its final count, including any decodable trailer frames.
        """
        with self._condition:
            return self._count if self._error is None else None

    @property
    def error(self) -> str | None:
        """Return the first accounting failure code, or None if none was recorded."""
        with self._condition:
            return self._error

    def _fail(self, reason: str) -> None:
        """Latch the first failure reason, discard queued bytes, and wake readers at EOF."""
        with self._condition:
            self._error = self._error or reason
            self._pending.clear()
            self._pending_bytes = 0
            self._eof = True
            self._condition.notify_all()

    def feed(self, data: bytes) -> None:
        """Queue immutable media bytes without waiting for them to be decoded.

        The first nonempty input starts the worker. Empty input and input after
        a failure are ignored. Exceeding the pending-byte limit or feeding a
        closed input disables accounting, clears the queue, and makes count
        return None; it does not raise an accounting error in the receive loop.
        """
        with self._condition:
            if not data or self._error is not None:
                return
            if self._eof:
                self._fail("decoder_stopped")
                return
            if self._pending_bytes + len(data) > self._MAX_PENDING_BYTES:
                self._fail("buffer_limit_exceeded")
                return
            self._pending.append(memoryview(data))
            self._pending_bytes += len(data)
            if self._thread is None:
                self._thread = threading.Thread(target=self._decode, name="video-frame-counter", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def read(self, size: int) -> bytes:
        """Block the decoder worker until size bytes are available or input closes.

        Positive reads coalesce message boundaries so a short queued piece is
        not interpreted as EOF inside an MP4 box. At EOF, return remaining
        bytes, then b"". Failures discard queued bytes and also signal EOF.
        A nonpositive size returns b"" immediately.
        """
        if size <= 0:
            return b""
        with self._condition:
            # Short reads inside an MP4 box can be treated as EOF by FFmpeg.
            # Transport messages need not coincide with either boxes or reads.
            self._condition.wait_for(lambda: self._pending_bytes >= size or self._eof)
            remaining = min(size, self._pending_bytes)
            self._pending_bytes -= remaining
            pieces = []
            while remaining:
                data = self._pending.popleft()
                pieces.append(data[:remaining])
                if len(data) > remaining:
                    self._pending.appendleft(data[remaining:])
                remaining -= len(pieces[-1])
            return b"".join(pieces)

    def _decode(self) -> None:
        """Decode one stream on the worker and publish frame counts under the lock.

        Decoder exceptions latch decode_failed instead of exposing a partial
        count as a successful total. Worker exit closes the input to new feeds.
        """
        try:
            with av.open(
                self,
                format=self._format,
                mode="r",
                # Keep the full-read requirement small enough for live progress.
                buffer_size=1024,
                options={"probesize": "32", "analyzeduration": "0"},
            ) as container:
                for _ in container.decode(container.streams.video[0]):
                    with self._condition:
                        if self._error is not None:
                            break
                        self._count += 1
        except Exception:
            # Accounting is optional: expose failure rather than losing the
            # receive loop or presenting a partial count as a successful total.
            self._fail("decode_failed")
        finally:
            with self._condition:
                self._eof = True

    async def finish(self) -> int | None:
        """Signal EOF and drain the worker without blocking the event loop.

        Return the decoded count (zero for no input), or None after a failure.
        Interrupted input counts only decodable received frames. The worker
        join is bounded by _JOIN_TIMEOUT; expiry latches shutdown_timeout.

        Cancellation latches shutdown_cancelled, waits for the same bounded
        join, then re-raises CancelledError. Further cancellation may interrupt
        that wait. A native codec call cannot be forcibly stopped, so the daemon
        worker may outlive this method after timeout or repeated cancellation.
        """
        with self._condition:
            self._eof = True
            self._condition.notify_all()
        if self._thread is not None:
            join_task = asyncio.create_task(asyncio.to_thread(self._thread.join, self._JOIN_TIMEOUT))
            try:
                await asyncio.shield(join_task)
            except asyncio.CancelledError:
                self._fail("shutdown_cancelled")
                await asyncio.shield(join_task)
                raise
            finally:
                if self._thread.is_alive():
                    self._fail("shutdown_timeout")
        return self.count


def _fresh_rgb_frame(frame: av.VideoFrame) -> av.VideoFrame:
    """Copy RGB pixels without carrying decoder color or picture-type metadata."""
    rgb = frame.reformat(format="rgb24")
    fresh = av.VideoFrame(rgb.width, rgb.height, "rgb24")
    source = rgb.planes[0]
    target = fresh.planes[0]
    pixels = memoryview(source)
    if source.line_size == target.line_size:
        target.update(pixels[: target.buffer_size])
    else:
        packed = bytearray(target.buffer_size)
        row_bytes = rgb.width * 3
        for row in range(rgb.height):
            src = row * source.line_size
            dst = row * target.line_size
            packed[dst : dst + row_bytes] = pixels[src : src + row_bytes]
        target.update(packed)
    return fresh


def _finalize_video_bytes(video_bytes: bytes, *, stream_format: str, fps: float) -> bytes:
    """Build a playback MP4 using only PyAV, retaining input on decode errors.

    Video-only streams are re-encoded at the requested FPS. Streams with audio
    are remuxed to preserve the compressed tracks and their original timing.
    """
    if not video_bytes:
        return video_bytes
    if stream_format.lower() != "m4s":
        raise ValueError(f"Unsupported streaming video format: {stream_format}")

    try:
        with av.open(io.BytesIO(video_bytes), format="mp4") as source:
            if source.streams.audio:
                buffer = io.BytesIO()
                media_streams = [stream for stream in source.streams if stream.type in ("video", "audio")]
                with av.open(buffer, mode="w", format="mp4") as output:
                    streams = {stream.index: output.add_stream_from_template(stream) for stream in media_streams}
                    for packet in source.demux(*media_streams):
                        if packet.dts is not None:
                            packet.stream = streams[packet.stream.index]
                            output.mux(packet)
                return buffer.getvalue()
            if not source.streams.video:
                return video_bytes
            frames = [_fresh_rgb_frame(frame) for frame in source.decode(video=0)]
    except av.FFmpegError:
        return video_bytes

    if not frames:
        return video_bytes

    buffer = io.BytesIO()
    rate = Fraction(fps).limit_denominator(10000)
    with av.open(buffer, mode="w", format="mp4") as output:
        video = cast(av.VideoStream, output.add_stream("h264", rate=rate))
        video.width, video.height = frames[0].width, frames[0].height
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18"}
        for index, frame in enumerate(frames):
            frame.pts = index
            frame.time_base = 1 / rate
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
    return buffer.getvalue()


def save_video(
    chunks: list[bytes],
    output_path: Path,
    *,
    fps: float,
    stream_format: str,
) -> None:
    """Finalize received chunks and atomically publish the local video file."""
    if not chunks:
        print("No video chunks received; nothing to save.")
        return

    streamed_bytes = b"".join(chunks)
    playback_bytes = _finalize_video_bytes(streamed_bytes, stream_format=stream_format, fps=fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as f:
            tmp_path = Path(f.name)
            f.write(playback_bytes)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    print(
        f"Saved {len(chunks)} streamed chunk(s), {len(streamed_bytes)} streamed bytes "
        f"-> {len(playback_bytes)} playback bytes at {output_path}"
    )


async def _send_heartbeats(websocket: Any, send_lock: asyncio.Lock) -> None:
    while True:
        await asyncio.sleep(10)
        try:
            async with send_lock:
                await websocket.send(json.dumps({"type": "session.ping"}))
        except ConnectionClosed:
            return


async def stream_video(args: argparse.Namespace) -> None:
    payload = build_session_start(args)
    output_path = Path(args.output)
    url = f"ws://{args.host}:{args.port}/v1/realtime/video"
    chunks: list[bytes] = []
    total_bytes = 0
    frame_counter: _StreamingFrameCounter | None = None
    stream_format = "m4s"
    done = False
    ws = None
    send_lock = asyncio.Lock()
    prompt_update_task: asyncio.Task[None] | None = None
    camera_update_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    pending_chunk_metadata: dict[str, Any] | None = None

    try:
        # MP4 is already compressed; deflate adds CPU work before output credit
        # can be returned to the generator, for little reduction in bytes.
        async with connect(url, max_size=None, compression=None) as websocket:
            ws = websocket
            heartbeat_task = asyncio.create_task(_send_heartbeats(websocket, send_lock))
            async with send_lock:
                await websocket.send(json.dumps(payload))
            print(f"Connected: {url}")
            print(f"Sent session.start: {json.dumps(payload, ensure_ascii=False)}")
            started_at = time.perf_counter()
            last_chunk_at = started_at

            while True:
                message = await websocket.recv()
                now = time.perf_counter()

                if isinstance(message, bytes):
                    # Use chunk metadata just received before video chunks
                    metadata = pending_chunk_metadata or {}
                    pending_chunk_metadata = None
                    chunks.append(message)
                    chunk_bytes = len(message)
                    chunk_elapsed = now - last_chunk_at
                    last_chunk_at = now

                    total_bytes += chunk_bytes
                    if frame_counter is None:
                        frame_counter = _StreamingFrameCounter(stream_format=stream_format)
                    frame_counter.feed(message)
                    decoded_total = frame_counter.count

                    total_elapsed = now - started_at
                    print(
                        f"[chunk {len(chunks):03d}] "
                        f"kind={metadata.get('kind', 'media')} "
                        f"bytes={chunk_bytes} "
                        f"frames={metadata.get('num_frames', 'unknown')} "
                        f"generation_chunk={metadata.get('generation_chunk_index')} "
                        f"active_interactions={metadata.get('active_event_ids', [])} "
                        f"elapsed={chunk_elapsed:.2f}s "
                        f"total_bytes={total_bytes} "
                        f"total_frames={decoded_total if decoded_total is not None else 'unknown'} "
                        f"total_elapsed={total_elapsed:.2f}s"
                    )
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "video.start":
                    stream_format = msg.get("format") or stream_format
                    print(f"Video session started: request_id={msg.get('request_id')} format={msg.get('format')}")
                    video_started_at = time.perf_counter()
                    if args.prompt_updates and prompt_update_task is None:
                        prompt_update_task = asyncio.create_task(
                            _run_prompt_update_scheduler(
                                websocket,
                                args.prompt_updates,
                                video_started_at=video_started_at,
                                send_lock=send_lock,
                            )
                        )
                    if args.camera_updates and camera_update_task is None:
                        camera_update_task = asyncio.create_task(
                            _run_camera_update_scheduler(
                                websocket,
                                args.camera_updates,
                                video_started_at=video_started_at,
                                send_lock=send_lock,
                            )
                        )
                elif msg_type == "video.chunk_metadata":
                    # Save the metadata for the incoming video chunk
                    pending_chunk_metadata = msg
                elif msg_type == "session.interaction.queued":
                    print(f"Interaction queued by server: event_id={msg.get('event_id')}")
                elif msg_type == "session.done":
                    print(f"Session complete: {json.dumps(msg, ensure_ascii=False)}")
                    done = True
                    if prompt_update_task is not None:
                        prompt_update_task.cancel()
                    if camera_update_task is not None:
                        camera_update_task.cancel()
                    break
                elif msg_type == "error":
                    print(f"ERROR: {msg.get('message', msg)}")
                else:
                    print(f"Received control message: {msg}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted. Asking server to stop the active session...")
        if ws is not None:
            try:
                async with send_lock:
                    await ws.send(json.dumps({"type": "session.stop"}))
                await ws.close(code=1000, reason="client interrupted")
            except Exception:
                pass
        raise
    finally:
        tasks = [task for task in (heartbeat_task, prompt_update_task, camera_update_task) if task is not None]
        for task in tasks:
            task.cancel()
        try:
            try:
                if frame_counter is not None:
                    await frame_counter.finish()
            finally:
                if frame_counter is not None:
                    decoded_total = frame_counter.count
                    print(
                        f"[decode] total_frames={decoded_total if decoded_total is not None else 'unknown'} "
                        f"status={frame_counter.error or 'complete'} "
                        f"total_elapsed={time.perf_counter() - started_at:.2f}s"
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        raise result
        finally:
            print(
                "Saving video... (May take a while. Remuxing concatenated chunks into one progressive MP4 file with total duration metadata in file header)"
            )
            save_video(
                chunks,
                output_path,
                fps=float(payload.get("fps") or args.fps),
                stream_format=stream_format,
            )
            if not done:
                print("Saved partial output because the stream did not finish normally.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming video generation WebSocket client")
    parser.add_argument("--host", default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument(
        "--model",
        default="BestWishYsh/Helios-Distilled",
        help="Model name served by the API server",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. "
            "The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. "
            "The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. "
            "The water is crystal clear, allowing for a view of the sandy ocean floor below. "
            "The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. "
            "The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. "
            "A close-up shot with dynamic movement."
        ),
        help="Text prompt for video generation",
    )
    parser.add_argument("--output", default="streaming_video_output.mp4", help="Path for the received video bytes")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=384, help="Video height")
    parser.add_argument("--size", default=None, help="Video size as WIDTHxHEIGHT; alternative to width/height")
    parser.add_argument("--seconds", default=None, help="Clip duration in seconds")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--num-frames", type=int, default=429, help="Number of generated frames")

    parser.add_argument(
        "--image-reference",
        type=_image_reference,
        default=None,
        help="First frame for image-conditioned models: a local image path or an http(s) URL.",
    )
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--guidance-scale-2", type=float, default=None, help="High-noise CFG scale for video models")
    parser.add_argument("--boundary-ratio", type=float, default=None, help="Boundary split ratio for video models")
    parser.add_argument("--flow-shift", type=float, default=None, help="Scheduler flow_shift for video models")
    parser.add_argument("--true-cfg-scale", type=float, default=None, help="True CFG scale for models that support it")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--helios-distilled-preset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send the Helios-Distilled extra_params used by the bundled example.",
    )
    parser.add_argument(
        "--extra-params",
        type=_parse_extra_params,
        default=None,
        help="JSON object merged into extra_params; overrides preset keys on conflict.",
    )
    # Midway interactions are pipeline-specific: unsupported modalities reject
    # the update (and may leave the session running). Nothing is scheduled unless
    # the caller asks for it.
    parser.add_argument(
        "--prompt-updates",
        type=_parse_prompt_updates,
        default=[],
        help=(
            "JSON array of scheduled prompt updates; none are sent by default. "
            "Each object requires "
            "'at' (seconds after video.start) and 'prompt'. Optional "
            f"'transition_chunks' defaults to {DEFAULT_TRANSITION_CHUNKS}."
        ),
    )
    parser.add_argument(
        "--camera-updates",
        type=_parse_camera_updates,
        default=[],
        help=(
            "JSON array of scheduled camera updates for models that register the "
            "camera modality (e.g. LingBot-World). Each object needs 'at' and either "
            "'camera' (structural SE3: mode + translation/rotation) or "
            "'actions' (client-side WASD/IJKL helper converted before send). "
            "Optional 'transition_chunks' defaults to None for velocity and "
            f"{DEFAULT_TRANSITION_CHUNKS} for target."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(stream_video(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
