# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Paced visual observations followed by one MCQ in a persistent session."""

from __future__ import annotations

import asyncio
import hashlib
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pybase64 as base64

from vllm_omni.benchmarks.data_modules.long_video_dataset import VideoQuestion
from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig, build_realtime_url


@dataclass(frozen=True)
class PreparedVideo:
    frames: tuple[Path, ...]
    duration_s: float
    sha256: str


def prepare_video(case: VideoQuestion, directory: Path, *, fps: float, max_edge: int) -> PreparedVideo:
    """Decode chronologically to bounded-size PNGs on disk, never past the query time."""
    import av

    digest = hashlib.sha256()
    with case.video.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    paths = []
    with av.open(str(case.video)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            raise ValueError("Video has no duration")
        end = case.end_s if case.end_s is not None else duration
        if not math.isfinite(end) or not 0 < end <= 3600:
            raise ValueError("Visual-session videos must be between 0 and 3600 seconds")
        if end > duration + 0.1:
            raise ValueError(f"Video ends at {duration}s before query time {end}s")
        start = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no timestamp")
            timestamp = float(frame.pts * stream.time_base) - start
            if timestamp >= end:
                break
            if timestamp + 1e-6 < len(paths) / fps:
                continue
            image = frame.to_image()
            image.thumbnail((max_edge, max_edge))
            path = directory / f"{len(paths):06d}.png"
            image.save(path)
            paths.append(path)
    if not paths or len(paths) < math.floor(end * fps):
        raise ValueError(f"Incomplete video: decoded {len(paths)} frames for {end}s at {fps} FPS")
    return PreparedVideo(tuple(paths), end, digest.hexdigest())


async def replay_video(
    case: VideoQuestion,
    prepared: PreparedVideo,
    *,
    url: str,
    model: str,
    fps: float,
    timeout_s: float,
    extra_body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """At most one observation response is in flight; record any pacing lag.

    The question is submitted after the prefix/full-video clock expires,
    together with its final sampled frame. It never sees the answer key.
    Reconnection and retries are deliberately explicit test scenarios, rather
    than hidden retries inside an accuracy measurement.
    """
    if extra_body.get("auto_response", False) is not False:
        raise ValueError("Visual-session QA requires explicit commits: auto_response must be false")
    collector = EventCollector()
    condition = asyncio.Condition()
    completed: list[str] = []
    config = SessionConfig(
        modalities=("text",),
        auto_response=False,
        temperature=0.0,
        max_output_tokens=256,
        idle_timeout_s=timeout_s,
        extra_body=extra_body,
    )
    url = build_realtime_url(url, model, autostart=False)
    async with DuplexClient(url, model=model, config=config, extra_headers=headers, reconnect=None) as client:
        capabilities = client.session_info.get("capabilities", {})
        required = capabilities.get("required_input_modalities", [])
        optional = capabilities.get("optional_input_modalities", [])
        if "audio" in required or "video" not in (*required, *optional):
            raise ValueError("Visual-session QA requires a session accepting video without mandatory audio")
        client.attach_event_collector(collector)

        async def consume() -> None:
            async for event in client.events():
                async with condition:
                    collector.add(event)
                    if event.type == "response.done" and event.response_id:
                        if event.raw["response"]["status"] != "completed":
                            raise RuntimeError(f"Observation failed: {event.raw}")
                        completed.append(event.response_id)
                    if event.type == "error":
                        raise RuntimeError(str(event.raw))
                    condition.notify_all()

        reader = asyncio.create_task(consume())

        async def step(frame: Path, query: str | None = None) -> str:
            count = len(completed)
            encoded = await asyncio.to_thread(lambda: base64.b64encode(frame.read_bytes()).decode("ascii"))
            if query is not None:
                await client.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": query}],
                        },
                    }
                )
            await client.append_audio(b"", video_frames=[encoded])
            await client.commit(create_response=True)

            async def wait_done() -> None:
                async with condition:
                    await condition.wait_for(lambda: len(completed) > count)

            waiter = asyncio.create_task(wait_done())
            try:
                done, _ = await asyncio.wait({reader, waiter}, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    reader.result()
                    raise RuntimeError("Session reader ended before response.done")
                if waiter not in done:
                    raise TimeoutError("Observation did not reach response.done")
                return completed[-1]
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

        started = time.monotonic()
        lags: list[float] = []
        try:
            for index, frame in enumerate(prepared.frames):
                target = started + index / fps
                await asyncio.sleep(max(0, target - time.monotonic()))
                lags.append(max(0, time.monotonic() - target))
                await step(frame)
            await asyncio.sleep(max(0, started + prepared.duration_s - time.monotonic()))
            question_sent = time.monotonic()
            response_id = await step(prepared.frames[-1], case.prompt)
            question_done = time.monotonic()
            first_text = next(
                (
                    received
                    for event, received in zip(collector.events, collector.event_received_at_s, strict=True)
                    if collector.response_id(event) == response_id
                    and event.get("delta")
                    and event.get("type")
                    in {"response.output_text.delta", "response.text.delta", "response.output_audio_transcript.delta"}
                ),
                None,
            )
            stats = await client.get_session_stats() if capabilities.get("supports_session_stats") else None
            metrics = collect_duplex_session_metrics(
                collector,
                stream_start=started,
                session_id=client.session_id,
                session_stats=stats,
            )
            return {
                "text": collector.response_text(response_id),
                "response_id": response_id,
                "session_id": client.session_id,
                "input_frames": len(prepared.frames),
                "query_frame_repeated": True,
                "input_duration_s": prepared.duration_s,
                "video_sha256": prepared.sha256,
                "elapsed_s": question_done - started,
                "query_completion_ms": (question_done - question_sent) * 1000,
                "query_ttft_ms": (first_text - question_sent) * 1000 if first_text is not None else None,
                "pacing_max_lag_s": max(lags),
                "pacing_mean_lag_s": sum(lags) / len(lags),
                "request_metrics": metrics.request_metrics,
                "session_metrics": metrics.session_metrics,
                "output_tokens": metrics.output_tokens,
            }
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)


async def run_video_question(
    case: VideoQuestion,
    *,
    fps: float,
    max_edge: int,
    url: str,
    model: str,
    timeout_s: float,
    extra_body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    # This temporary directory lives until the session closes, including on
    # cancellation. Decode/resize and file IO stay off the websocket event loop.
    with tempfile.TemporaryDirectory(prefix="omni-long-video-") as directory:
        preparation = asyncio.create_task(
            asyncio.to_thread(prepare_video, case, Path(directory), fps=fps, max_edge=max_edge)
        )
        try:
            prepared = await asyncio.shield(preparation)
        finally:
            # A thread cannot be cancelled: join it before removing its files.
            if not preparation.done():
                await preparation
        return await asyncio.wait_for(
            replay_video(
                case,
                prepared,
                fps=fps,
                url=url,
                model=model,
                timeout_s=timeout_s,
                extra_body=extra_body,
                headers=headers,
            ),
            timeout=prepared.duration_s + timeout_s,
        )
