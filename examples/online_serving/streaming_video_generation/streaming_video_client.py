#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only WebSocket client for `/v1/realtime/video`.

The client prints a line for every binary video chunk it receives and writes
the received bytes to disk when the session finishes or is interrupted.
Image/reference input is intentionally not supported in this example yet.

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import contextlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av

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


def _maybe_set(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def build_session_start(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "session.start",
        "model": args.model,
        "prompt": args.prompt,
    }

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
    for update in updates:
        delay = (video_started_at + update.at) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        payload = {
            "type": "session.interaction",
            "interaction": {
                "event_id": f"scheduled-{update.at:.3f}",
                "event": {"prompt": update.prompt},
                "transition_chunks": update.transition_chunks,
            },
        }
        async with send_lock:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        print(f"Sent session.interaction at t={update.at:.2f}s: {json.dumps(payload, ensure_ascii=False)}")


def _count_decoded_frames(video_bytes: bytes, *, stream_format: str) -> int | None:
    """Return the number of decodable video frames in a streamed byte sequence, if known."""
    if not video_bytes:
        return 0

    av_format = "mp4" if stream_format == "m4s" else None
    try:
        with av.open(io.BytesIO(video_bytes), format=av_format, mode="r") as container:
            stream = container.streams.video[0]
            return sum(1 for _ in container.decode(stream))
    except Exception:
        return None


def save_video(
    chunks: list[bytes],
    output_path: Path,
    *,
    fps: float,
    stream_format: str,
) -> None:
    if not chunks:
        print("No video chunks received; nothing to save.")
        return

    from vllm_omni.diffusion.utils.media_utils import finalize_streaming_video_bytes

    streamed_bytes = b"".join(chunks)
    playback_bytes = finalize_streaming_video_bytes(
        streamed_bytes,
        input_format="m4s",
        fps=fps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(playback_bytes)
    os.replace(tmp_path, output_path)
    print(
        f"Saved {len(chunks)} streamed chunk(s), {len(streamed_bytes)} streamed bytes "
        f"-> {len(playback_bytes)} playback bytes at {output_path}"
    )


async def stream_video(args: argparse.Namespace) -> None:
    payload = build_session_start(args)
    output_path = Path(args.output)
    url = f"ws://{args.host}:{args.port}/v1/realtime/video"
    chunks: list[bytes] = []
    total_bytes = 0
    total_frames = 0
    accumulated_stream = bytearray()
    stream_format = "m4s"
    done = False
    ws = None
    send_lock = asyncio.Lock()
    prompt_update_task: asyncio.Task[None] | None = None
    pending_chunk_metadata: dict[str, Any] | None = None

    try:
        async with connect(url, max_size=None) as websocket:
            ws = websocket
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
                    accumulated_stream.extend(message)
                    decoded_total = _count_decoded_frames(bytes(accumulated_stream), stream_format=stream_format)
                    if decoded_total is not None:
                        chunk_frames = max(decoded_total - total_frames, 0)
                        total_frames = decoded_total
                    else:
                        chunk_frames = 0

                    total_elapsed = now - started_at
                    print(
                        f"[chunk {len(chunks):03d}] "
                        f"kind={metadata.get('kind', 'media')} "
                        f"bytes={chunk_bytes} "
                        f"frames={metadata.get('num_frames', chunk_frames)} "
                        f"generation_chunk={metadata.get('generation_chunk_index')} "
                        f"active_interactions={metadata.get('active_event_ids', [])} "
                        f"elapsed={chunk_elapsed:.2f}s "
                        f"total_bytes={total_bytes} "
                        f"total_frames={total_frames} "
                        f"total_elapsed={total_elapsed:.2f}s"
                    )
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "video.start":
                    stream_format = msg.get("format") or stream_format
                    print(f"Video session started: request_id={msg.get('request_id')} format={msg.get('format')}")
                    if args.prompt_updates and prompt_update_task is None:
                        video_started_at = time.perf_counter()
                        prompt_update_task = asyncio.create_task(
                            _run_prompt_update_scheduler(
                                websocket,
                                args.prompt_updates,
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
        if prompt_update_task is not None:
            prompt_update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await prompt_update_task
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
    parser.add_argument(
        "--prompt-updates",
        type=_parse_prompt_updates,
        default=json.dumps(
            [
                {
                    "at": 4,
                    "prompt": (
                        "An underwater tornado appears and affects the ocean floor in a dramatic and chaotic scene. "
                        "The water is murky, swirling violently, carrying debris and marine life into the vortex. "
                        "The tropical fish on the scene all swim in panic, trying to avoid the powerful currents. "
                        "The camera remains stationary, capturing the intensity of the underwater tornado as it disrupts the serene ocean floor. "
                        "Close-up shot emphasizing the turbulent motion and destruction."
                    ),
                },
                {
                    "at": 11,
                    "prompt": (
                        "The swirling underwater vortex now seizes a heavy, encrusted treasure chest, its lid flapping open as it is smashed onto the ocean floor. "
                        "Gold coins and silver trinkets spill out, glittering briefly in the murky water before being swept instantly into the violent funnel. "
                        "The heavy wooden box tumbles end over end, colliding with floating rocks and adding to the debris field. "
                        "Swirling sediment and bubbles surround the spilling fortune, highlighting the chaotic power of the storm as it ravages the seabed. "
                        "Close-up shot emphasizing the turbulent motion and destruction."
                    ),
                    # "transition_chunks": 3,
                },
            ]
        ),
        help=(
            "JSON array of scheduled prompt updates. Each object requires "
            "'at' (seconds after video.start) and 'prompt'. Optional "
            f"'transition_chunks' defaults to {DEFAULT_TRANSITION_CHUNKS}."
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
