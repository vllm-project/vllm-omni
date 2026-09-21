#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime streaming benchmark for LingBot-World 2.0 over ``WS /v1/realtime/video``.

Every other diffusion benchmark in this repository measures "one request, one
finished video". LingBot-World is not that: one request is a whole rollout, and
the thing a deployment lives or dies by is whether chunk *N + 1* keeps arriving
before a viewer has finished watching chunk *N*. This benchmark therefore
measures a **cadence**, not a throughput:

* ``TTFC`` -- session.start to the first chunk's bytes, which pays for prompt
  encode, the first-frame VAE encode, and any compile or capture the deploy
  config asks for;
* **inter-arrival latency** over all chunks and, separately, over the steady
  state after the attention window saturates;
* ``VIDEO_RTF`` -- wall seconds spent per second of video generated, matching how
  vLLM-Omni computes RTF everywhere else (``generation_s / video_duration``).
  **Lower is better and below 1.0 is real time**; ``VIDEO_RTFX`` is reported
  alongside it as the reciprocal, under the repo's own name for that;
* **playback continuity** -- a simulated viewer with a start buffer, because a
  p99 that is harmless behind a three-chunk buffer is a visible stall behind
  none.

The server is not started here. Bring one up first (see the README), then point
this script at it. Requires ``websockets``; ``--save-video`` additionally needs
a vLLM-Omni installation for the remux helper.

Example:

    python benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py \\
      --host 127.0.0.1 --port 8000 \\
      --num-chunks 16 \\
      --output-json /tmp/lingbot_world_realtime.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_world.workload import (  # noqa: E402
    CAMERA_PATTERNS,
    DEFAULT_FLOW_SHIFT,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_SEED,
    DEFAULT_WARMUP_CHUNKS,
    DEFAULT_WIDTH,
    ChunkRecord,
    Workload,
    aggregate_metrics,
    compute_metrics,
    default_workload,
    load_workload,
    steady_chunk_deadline_ms,
)

# The shared vLLM asset keeps the default workload runnable with no setup and
# adds no binary to this repository. Any real photograph conditions a rollout.
_DEFAULT_IMAGE_ASSET = "2560px-Gfp-wisconsin-madison-the-nature-boardwalk"
_WS_PATH = "/v1/realtime/video"


class BenchmarkError(RuntimeError):
    """A session failed in a way that invalidates its numbers."""


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #


def resolve_image_reference(value: str | None) -> str:
    """Return an ``http(s)`` or ``data:`` URL for the session's first frame."""
    if value is None:
        from vllm.assets.image import ImageAsset

        value = str(ImageAsset(_DEFAULT_IMAGE_ASSET).get_path("jpg"))
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise BenchmarkError(f"--image is neither a URL nor a readable file: {value}")
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_workload(args: argparse.Namespace) -> Workload:
    if args.workload is not None:
        # The file names its own first frame, so resolving happens inside
        # load_workload -- never eagerly, which would fetch the default asset a
        # --workload run has no use for.
        return load_workload(
            args.workload,
            image_resolver=lambda value: resolve_image_reference(value if args.image is None else args.image),
            overrides={
                "width": args.width_override,
                "height": args.height_override,
                "fps": args.fps_override,
                "seed": args.seed_override,
                "prompt": args.prompt_override,
                "flow_shift": args.flow_shift,
                "negative_prompt": args.negative_prompt,
            },
        )
    return default_workload(
        image_reference=resolve_image_reference(args.image),
        num_chunks=args.num_chunks,
        camera_pattern=args.camera_pattern,
        prompt=args.prompt_override if args.prompt_override is not None else DEFAULT_PROMPT,
        width=args.width_override if args.width_override is not None else DEFAULT_WIDTH,
        height=args.height_override if args.height_override is not None else DEFAULT_HEIGHT,
        fps=args.fps_override if args.fps_override is not None else DEFAULT_FPS,
        seed=args.seed_override if args.seed_override is not None else DEFAULT_SEED,
        flow_shift=args.flow_shift if args.flow_shift is not None else DEFAULT_FLOW_SHIFT,
        negative_prompt=args.negative_prompt,
    )


# --------------------------------------------------------------------------- #
# One streaming session
# --------------------------------------------------------------------------- #


@dataclass
class SessionResult:
    request_id: str | None
    records: list[ChunkRecord]
    media: bytes
    ttfc_s: float | None
    video_start_s: float | None
    wall_s: float
    done_event: dict[str, Any]


async def _keepalive(websocket: Any, send_lock: asyncio.Lock, interval: float) -> None:
    """Refresh the server's stall clock while a slow first chunk is generating.

    A compiled deploy config records CUDA graphs on the first block, which can
    outlast the server's ~60 s stall timeout. ``session.ping`` is the documented
    way to say the client is still there.
    """
    while True:
        await asyncio.sleep(interval)
        async with send_lock:
            await websocket.send(json.dumps({"type": "session.ping"}))


async def run_session(
    url: str,
    model: str,
    workload: Workload,
    *,
    first_chunk_timeout: float,
    chunk_timeout: float,
    ping_interval: float,
    collect_media: bool,
    print_chunks: bool,
) -> SessionResult:
    """Drive one rollout and timestamp every chunk as its bytes land."""
    try:
        from websockets.asyncio.client import connect
    except ImportError:  # pragma: no cover - exercised only without websockets
        try:
            from websockets import connect  # type: ignore[attr-defined]
        except ImportError as exc:
            raise BenchmarkError("This benchmark needs the websockets package: pip install websockets") from exc

    payload = workload.session_start_payload(model)
    records: list[ChunkRecord] = []
    media = bytearray()
    pending: dict[str, Any] | None = None
    request_id: str | None = None
    ttfc_s: float | None = None
    video_start_s: float | None = None
    last_arrival: float | None = None
    done_event: dict[str, Any] = {}
    send_lock = asyncio.Lock()
    keepalive: asyncio.Task[None] | None = None
    # Fire remaining prompt updates in chunk order as their boundary passes.
    updates = sorted(workload.prompt_updates, key=lambda update: update.after_chunk)
    update_index = 0

    async with connect(url, max_size=None, ping_interval=None) as websocket:
        async with send_lock:
            await websocket.send(json.dumps(payload, ensure_ascii=False))
        started = time.perf_counter()
        if ping_interval > 0:
            keepalive = asyncio.create_task(_keepalive(websocket, send_lock, ping_interval))

        try:
            while True:
                timeout = first_chunk_timeout if not records else chunk_timeout
                try:
                    deadline = started + (last_arrival if last_arrival is not None else 0.0) + timeout
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise BenchmarkError(
                        f"No media chunk for {timeout:g} s after {len(records)} chunk(s); "
                        "the server may still be loading, compiling, or capturing graphs."
                    ) from exc
                now = time.perf_counter()

                if isinstance(message, (bytes, bytearray)):
                    if pending is None:
                        raise BenchmarkError("Received a binary frame with no preceding video.chunk_metadata event.")
                    if collect_media:
                        media.extend(message)
                    if pending.get("kind") == "media":
                        index = int(pending["generation_chunk_index"])
                        arrival = now - started
                        if ttfc_s is None:
                            ttfc_s = arrival
                        records.append(
                            ChunkRecord(
                                index=index,
                                arrival_s=arrival,
                                inter_arrival_s=arrival - (last_arrival if last_arrival is not None else 0.0),
                                byte_length=len(message),
                                num_frames=int(pending["num_frames"]),
                            )
                        )
                        last_arrival = arrival
                        if print_chunks:
                            print(
                                f"  chunk {index:3d}  t={arrival:8.3f}s  "
                                f"interval={records[-1].inter_arrival_s * 1000:8.1f}ms  "
                                f"frames={records[-1].num_frames:3d}  bytes={len(message)}",
                                flush=True,
                            )
                        while update_index < len(updates) and updates[update_index].after_chunk <= index:
                            async with send_lock:
                                await websocket.send(json.dumps(updates[update_index].to_payload(), ensure_ascii=False))
                            update_index += 1
                    pending = None
                    continue

                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "video.chunk_metadata":
                    pending = event
                elif event_type == "video.start":
                    request_id = event.get("request_id")
                    video_start_s = now - started
                elif event_type == "error":
                    # Say what was asked for alongside what the server refused:
                    # some builds cap a rollout's num_frames, and the operator
                    # needs to know which --num-chunks would fit.
                    raise BenchmarkError(
                        f"Server rejected the session after {len(records)} chunk(s): "
                        f"{event.get('message', event)} "
                        f"(requested {workload.num_chunks} chunks = num_frames {workload.num_frames})"
                    )
                elif event_type == "session.done":
                    done_event = event
                    break
        finally:
            if keepalive is not None:
                keepalive.cancel()
                try:
                    await keepalive
                except asyncio.CancelledError:
                    pass

        wall_s = time.perf_counter() - started

    if not records:
        raise BenchmarkError("The session finished without delivering a single media chunk.")
    if done_event.get("stopped"):
        raise BenchmarkError(f"The session was stopped before it finished: {done_event}")
    observed = [record.index for record in records]
    if observed != list(range(len(records))):
        raise BenchmarkError(f"Chunk indices are not contiguous from zero: {observed}")
    if len(records) != workload.num_chunks:
        raise BenchmarkError(
            f"Expected {workload.num_chunks} chunks for num_frames={workload.num_frames}, received {len(records)}."
        )

    return SessionResult(
        request_id=request_id,
        records=records,
        media=bytes(media),
        ttfc_s=ttfc_s,
        video_start_s=video_start_s,
        wall_s=wall_s,
        done_event=done_event,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _print_row(label: str, value: str) -> None:
    print(f"{label:<44}{value:>16}")


def print_report(
    metrics: dict[str, Any],
    *,
    workload: Workload,
    endpoint: str,
    sessions: int,
    target_fps: float,
) -> None:
    header = " LingBot-World realtime streaming benchmark "
    print("\n{s:{c}^{n}}".format(s=header, n=60, c="="))
    _print_row("Endpoint:", endpoint)
    _print_row("Geometry:", f"{workload.width}x{workload.height}")
    _print_row("Chunks per session:", str(workload.num_chunks))
    _print_row("Sessions:", str(sessions))
    frames = workload.num_frames
    _print_row("Real-time basis:", f"{target_fps:g} fps")
    _print_row("Video seconds per session:", f"{frames / target_fps:.2f} ({frames} frames)")
    if abs(target_fps - workload.fps) > 1e-9:
        _print_row("Mux label sent to server:", f"{workload.fps} fps")

    if sessions == 1:
        print("{s:{c}^{n}}".format(s=" Cadence ", n=60, c="-"))
        if metrics.get("ttfc_ms") is not None:
            _print_row("TTFC (session.start -> first chunk):", f"{metrics['ttfc_ms']:.1f} ms")
        interval_all = metrics.get("interval_all") or {}
        interval_steady = metrics.get("interval_steady") or {}
        for title, summary in (("All chunks", interval_all), ("Steady state", interval_steady)):
            if not summary:
                continue
            print(f"  {title} (n={int(summary['count'])}):")
            _print_row("    mean / median:", f"{summary['mean_ms']:.1f} / {summary['median_ms']:.1f} ms")
            _print_row("    std / max:", f"{summary['std_ms']:.1f} / {summary['max_ms']:.1f} ms")
            tail = f"{summary['p90_ms']:.0f} / {summary['p95_ms']:.0f} / {summary['p99_ms']:.0f} ms"
            _print_row("    p90 / p95 / p99:", tail)
            if summary["count"] < 100:
                print(
                    "    NOTE: fewer than 100 intervals; tail percentiles are descriptive, not reliable tail estimates."
                )

        print("{s:{c}^{n}}".format(s=" Real time ", n=60, c="-"))
        _print_row("VIDEO_RTF (wall s / video s, <1 real time):", f"{metrics['video_rtf']:.3f}")
        _print_row("VIDEO_RTFX (video s / wall s):", f"{metrics['video_rtfx']:.3f}")
        if metrics.get("steady_video_rtf") is not None:
            _print_row("VIDEO_RTF steady state:", f"{metrics['steady_video_rtf']:.3f}")
        _print_row("Frames per second:", f"{metrics['frames_per_second']:.2f}")
        _print_row("Chunk deadline:", f"{metrics['chunk_deadline_ms']:.1f} ms")
        if metrics.get("slo_attainment") is not None:
            _print_row(
                "Steady chunks within deadline:",
                f"{metrics['slo_attainment'] * 100:.1f}% ({metrics['slo_violations']} late)",
            )
        if not metrics.get("steady_state_reached", False):
            print(
                f"  NOTE: no chunk reached the steady state; the attention window needs "
                f"{metrics['warmup_chunks']} chunks. Raise --num-chunks."
            )

        playback = metrics.get("playback") or {}
        print("{s:{c}^{n}}".format(s=" Playback continuity ", n=60, c="-"))
        _print_row("Start buffer:", f"{playback.get('buffer_chunks')} chunk(s)")
        _print_row("Underruns:", str(playback.get("underrun_count")))
        _print_row("Total stall:", f"{playback.get('total_stall_ms', 0.0):.1f} ms")
        _print_row("Stall ratio:", f"{playback.get('stall_ratio', 0.0) * 100:.2f}%")
        if playback.get("first_underrun_chunk") is not None:
            _print_row("First underrun at chunk:", str(playback["first_underrun_chunk"]))
    else:
        print("{s:{c}^{n}}".format(s=" Across sessions ", n=60, c="-"))
        if metrics.get("ttfc_ms_mean") is not None:
            _print_row("TTFC mean / max:", f"{metrics['ttfc_ms_mean']:.1f} / {metrics['ttfc_ms_max']:.1f} ms")
        if metrics.get("steady_interval_ms_mean") is not None:
            _print_row("Steady interval mean:", f"{metrics['steady_interval_ms_mean']:.1f} ms")
            _print_row("Steady interval spread:", f"{metrics['steady_interval_ms_spread']:.1f} ms")
        _print_row("VIDEO_RTF mean (lower is better):", f"{metrics['video_rtf_mean']:.3f}")
        _print_row(
            "VIDEO_RTF best / worst:",
            f"{metrics['video_rtf_best']:.3f} / {metrics['video_rtf_worst']:.3f}",
        )
        _print_row("Chunk deadline:", f"{metrics['chunk_deadline_ms']:.1f} ms")
        _print_row("Underruns (all sessions):", str(metrics["underrun_count"]))
    print("=" * 60)


def save_video(media: bytes, path: Path, *, fps: int) -> None:
    from vllm_omni.diffusion.utils.media_utils import finalize_streaming_video_bytes

    playback_bytes = finalize_streaming_video_bytes(media, input_format="m4s", fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(playback_bytes)
    os.replace(tmp_path, path)
    print(f"Saved {len(media)} streamed bytes -> {len(playback_bytes)} playback bytes at {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LingBot-World 2.0 realtime streaming over WS /v1/realtime/video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    endpoint = parser.add_argument_group("endpoint")
    endpoint.add_argument("--host", default="127.0.0.1")
    endpoint.add_argument("--port", type=int, default=8000)
    endpoint.add_argument("--base-url", default=None, help="Full ws:// URL; overrides --host/--port.")
    endpoint.add_argument("--model", default=DEFAULT_MODEL)

    workload = parser.add_argument_group("workload")
    workload.add_argument("--workload", default=None, help="JSON rollout spec; overrides the built-in workload.")
    workload.add_argument("--image", default=None, help="First frame: path or URL. Defaults to the shared vLLM asset.")
    workload.add_argument("--prompt", dest="prompt_override", default=None, help=f"Default: {DEFAULT_PROMPT!r}")
    workload.add_argument("--negative-prompt", default=None)
    workload.add_argument("--num-chunks", type=int, default=16, help="AR blocks per session; ignored with --workload.")
    workload.add_argument(
        "--camera-pattern",
        choices=CAMERA_PATTERNS,
        default="forward",
        help="Camera pattern for the built-in workload only; ignored with --workload.",
    )
    workload.add_argument("--width", dest="width_override", type=int, default=None, help=f"Default: {DEFAULT_WIDTH}")
    workload.add_argument("--height", dest="height_override", type=int, default=None, help=f"Default: {DEFAULT_HEIGHT}")
    workload.add_argument("--fps", dest="fps_override", type=int, default=None, help=f"Default: {DEFAULT_FPS}")
    workload.add_argument("--seed", dest="seed_override", type=int, default=None, help=f"Default: {DEFAULT_SEED}")
    workload.add_argument("--flow-shift", type=float, default=None, help=f"Default: {DEFAULT_FLOW_SHIFT}")

    run = parser.add_argument_group("run")
    run.add_argument("--sessions", type=int, default=1, help="Sequential measured rollouts.")
    run.add_argument("--warmup-sessions", type=int, default=0, help="Unmeasured rollouts first (pays compile cost).")
    run.add_argument(
        "--warmup-chunks",
        type=int,
        default=DEFAULT_WARMUP_CHUNKS,
        help="Chunks excluded from steady state; the attention window saturates after 6.",
    )
    run.add_argument(
        "--chunk-slo-ms",
        type=float,
        default=None,
        help="Deadline a steady chunk must meet. Default: one chunk of video at --target-fps.",
    )
    run.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Playback rate used for real-time factor, the chunk deadline, and the playback "
            "simulation. The checkpoint declares no frame rate, so this is a consumption "
            "choice, not a model property. Default: the --fps sent to the server."
        ),
    )
    run.add_argument("--playback-buffer-chunks", type=int, default=1, help="Chunks buffered before playback starts.")
    run.add_argument("--first-chunk-timeout", type=float, default=900.0)
    run.add_argument("--chunk-timeout", type=float, default=300.0)
    run.add_argument("--ping-interval", type=float, default=20.0, help="session.ping cadence; 0 disables.")

    output = parser.add_argument_group("output")
    output.add_argument("--output-json", default=None)
    output.add_argument("--save-video", default=None, help="Write the last measured session's video here.")
    output.add_argument("--print-chunks", action="store_true")

    args = parser.parse_args(argv)
    if args.sessions < 1:
        parser.error("--sessions must be at least 1.")
    if args.warmup_sessions < 0:
        parser.error("--warmup-sessions must be non-negative.")
    if args.warmup_chunks < 0:
        parser.error("--warmup-chunks must be non-negative.")
    if args.playback_buffer_chunks < 1:
        parser.error("--playback-buffer-chunks must be at least 1.")
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be positive.")
    if args.workload is None and args.num_chunks < 1:
        parser.error("--num-chunks must be at least 1.")
    return args


def endpoint_url(args: argparse.Namespace) -> str:
    if args.base_url is not None:
        base = args.base_url.rstrip("/")
        for prefix, replacement in (("http://", "ws://"), ("https://", "wss://")):
            if base.startswith(prefix):
                base = replacement + base[len(prefix) :]
                break
        return base if base.endswith(_WS_PATH) else base + _WS_PATH
    return f"ws://{args.host}:{args.port}{_WS_PATH}"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    workload = build_workload(args)
    url = endpoint_url(args)
    # The mux label and the real-time basis are different questions; only the
    # latter decides whether a run passes.
    target_fps = args.target_fps if args.target_fps is not None else float(workload.fps)
    if max(1, args.warmup_chunks) >= workload.num_chunks - 1:
        print(
            f"WARNING: --warmup-chunks={args.warmup_chunks} leaves no nonterminal steady-state chunk in a "
            f"{workload.num_chunks}-chunk rollout; steady-state metrics will be empty.",
            file=sys.stderr,
        )

    print(f"Connecting to {url}")
    print(
        f"Rollout: {workload.num_chunks} chunks, num_frames={workload.num_frames}, "
        f"{workload.video_seconds:.2f}s of video"
    )
    print(
        f"Real-time basis: {target_fps:g} fps -> a steady 12-frame chunk is "
        f"{steady_chunk_deadline_ms(target_fps):.1f} ms of video"
        + ("" if args.target_fps is None else f" (mux label stays {workload.fps} fps)")
    )

    for index in range(args.warmup_sessions):
        print(f"Warmup session {index + 1}/{args.warmup_sessions} (not measured)...")
        await run_session(
            url,
            args.model,
            workload,
            first_chunk_timeout=args.first_chunk_timeout,
            chunk_timeout=args.chunk_timeout,
            ping_interval=args.ping_interval,
            collect_media=False,
            print_chunks=False,
        )

    session_metrics: list[dict[str, Any]] = []
    session_payloads: list[dict[str, Any]] = []
    last_result: SessionResult | None = None
    for index in range(args.sessions):
        if args.sessions > 1:
            print(f"Session {index + 1}/{args.sessions}...")
        result = await run_session(
            url,
            args.model,
            workload,
            first_chunk_timeout=args.first_chunk_timeout,
            chunk_timeout=args.chunk_timeout,
            ping_interval=args.ping_interval,
            collect_media=args.save_video is not None,
            print_chunks=args.print_chunks,
        )
        last_result = result
        metrics = compute_metrics(
            result.records,
            fps=target_fps,
            warmup_chunks=args.warmup_chunks,
            slo_ms=args.chunk_slo_ms,
            playback_buffer_chunks=args.playback_buffer_chunks,
            ttfc_s=result.ttfc_s,
            session_wall_s=result.wall_s,
        )
        session_metrics.append(metrics)
        session_payloads.append(
            {
                "session_index": index,
                "request_id": result.request_id,
                "video_start_ms": (result.video_start_s * 1000.0) if result.video_start_s is not None else None,
                "chunks": [record.to_dict() for record in result.records],
                "metrics": metrics,
            }
        )

    aggregate = aggregate_metrics(session_metrics, fps=target_fps)
    print_report(aggregate, workload=workload, endpoint=url, sessions=args.sessions, target_fps=target_fps)

    if args.save_video is not None and last_result is not None:
        save_video(last_result.media, Path(args.save_video), fps=workload.fps)

    document = {
        "benchmark": "lingbot_world_realtime",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": url,
        "model": args.model,
        "git_revision": _git_revision(),
        "hostname": platform.node(),
        "workload": workload.describe(),
        "config": {
            "sessions": args.sessions,
            "warmup_sessions": args.warmup_sessions,
            "warmup_chunks": args.warmup_chunks,
            "camera_pattern": args.camera_pattern,
            "chunk_slo_ms": args.chunk_slo_ms,
            "target_fps": target_fps,
            "mux_fps": workload.fps,
            "playback_buffer_chunks": args.playback_buffer_chunks,
            "workload_file": args.workload,
        },
        "sessions": session_payloads,
        "metrics": aggregate,
    }
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {path}")
    return document


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(run_benchmark(args))
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
