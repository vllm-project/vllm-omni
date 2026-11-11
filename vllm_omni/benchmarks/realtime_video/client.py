# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One realtime-video rollout; request scheduling belongs to bench serve."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import ssl
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybase64 as base64

from .workload import ChunkRecord, Workload

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE_ASSET = "2560px-Gfp-wisconsin-madison-the-nature-boardwalk"
_WS_PATH = "/v1/realtime/video"


def realtime_video_base_url(value: str) -> str:
    """Normalize a base or full endpoint URL for the shared HTTP CLI."""
    base = value.rstrip("/").removesuffix(_WS_PATH)
    for scheme, replacement in (("wss://", "https://"), ("ws://", "http://")):
        if base.startswith(scheme):
            return replacement + base[len(scheme) :]
    return base


def realtime_video_url(value: str) -> str:
    """Return the WS endpoint used by both the adapter and legacy CLI."""
    base = realtime_video_base_url(value)
    for scheme, replacement in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(scheme):
            base = replacement + base[len(scheme) :]
            break
    return base + _WS_PATH


class BenchmarkError(RuntimeError):
    """A session failed in a way that invalidates its numbers."""


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


@dataclass
class SessionResult:
    request_id: str | None
    records: list[ChunkRecord]
    media: bytes
    ttfc_s: float | None
    video_start_s: float | None
    wall_s: float
    done_event: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)


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
    failure_output: Path | None = None,
    on_failure: Callable[[SessionResult, BaseException | None], None] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    insecure: bool = False,
) -> SessionResult:
    """Drive one rollout and timestamp every chunk as its bytes land."""
    try:
        from websockets.asyncio.client import connect

        header_option = "additional_headers"
    except ImportError:  # pragma: no cover - exercised only without websockets
        try:
            from websockets import connect  # type: ignore[attr-defined]

            header_option = "extra_headers"
        except ImportError as exc:
            raise BenchmarkError("This benchmark needs the websockets package: pip install websockets") from exc

    connect_options: dict[str, Any] = {"max_size": None, "ping_interval": None}
    if extra_headers:
        connect_options[header_option] = dict(extra_headers)
    if insecure and url.startswith("wss://"):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connect_options["ssl"] = context

    payload = workload.session_start_payload(model)
    records: list[ChunkRecord] = []
    media = bytearray()
    pending: dict[str, Any] | None = None
    request_id: str | None = None
    ttfc_s: float | None = None
    video_start_s: float | None = None
    last_arrival: float | None = None
    done_event: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    send_lock = asyncio.Lock()
    keepalive: asyncio.Task[None] | None = None
    # Fire remaining prompt updates in chunk order as their boundary passes.
    updates = sorted(workload.prompt_updates, key=lambda update: update.after_chunk)
    update_index = 0

    started: float | None = None
    complete = False
    try:
        async with connect(url, **connect_options) as websocket:
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
                            raise BenchmarkError(
                                "Received a binary frame with no preceding video.chunk_metadata event."
                            )
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
                                    await websocket.send(
                                        json.dumps(updates[update_index].to_payload(), ensure_ascii=False)
                                    )
                                update_index += 1
                        pending = None
                        continue

                    event = json.loads(message)
                    events.append(event)
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

        result = SessionResult(
            request_id=request_id,
            records=records,
            media=bytes(media),
            ttfc_s=ttfc_s,
            video_start_s=video_start_s,
            wall_s=wall_s,
            done_event=done_event,
            events=events,
        )
        complete = True
        return result
    finally:
        if not complete and on_failure is not None:
            # Retain failed raw observations after the transport has unwound.
            # This callback never participates in a successful measurement.
            partial = SessionResult(
                request_id=request_id,
                records=records,
                media=bytes(media),
                ttfc_s=ttfc_s,
                video_start_s=video_start_s,
                wall_s=time.perf_counter() - started if started is not None else 0.0,
                done_event=done_event,
                events=events,
            )
            try:
                on_failure(partial, sys.exc_info()[1])
            except Exception as callback_error:
                # Diagnostics must not replace the original transport error.
                print(f"Could not retain failed session: {type(callback_error).__name__}", file=sys.stderr)
        if not complete and failure_output is not None:
            # A diagnostic trace is never a benchmark result. Write only after
            # the failed session has unwound; successful timing is unchanged.
            error = sys.exc_info()[1]
            trace = {
                "complete": False,
                "error": {"type": type(error).__name__, "message": str(error)},
                "request_id": request_id,
                "expected_chunks": workload.num_chunks,
                "video_start_ms": video_start_s * 1000.0 if video_start_s is not None else None,
                "ttfc_ms": ttfc_s * 1000.0 if ttfc_s is not None else None,
                "wall_seconds": time.perf_counter() - started if started is not None else None,
                "chunks": [record.to_dict() for record in records],
                "events": events,
                "done_event": done_event,
            }
            try:
                failure_output.parent.mkdir(parents=True, exist_ok=True)
                temporary = failure_output.with_suffix(failure_output.suffix + ".tmp")
                temporary.write_text(json.dumps(trace, indent=2) + "\n")
                os.replace(temporary, failure_output)
            except OSError as write_error:
                # A diagnostic filesystem error must not replace the original
                # timeout, protocol error, disconnect or cancellation.
                print(f"Could not retain failed session trace at {failure_output}: {write_error}", file=sys.stderr)


def save_video(media: bytes, path: Path, *, fps: int) -> None:
    from .media import finalize_streaming_video_bytes

    playback_bytes = finalize_streaming_video_bytes(media, input_format="m4s", fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(playback_bytes)
    os.replace(tmp_path, path)
    print(f"Saved {len(media)} streamed bytes -> {len(playback_bytes)} playback bytes at {path}")


def git_revision() -> str | None:
    # An installed wheel has no source checkout. Never report an unrelated
    # ancestor repository or the caller's working directory as benchmark code.
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
