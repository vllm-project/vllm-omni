# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MP4 muxing must not run on the event loop.

Encoding a finished video is CPU-bound and scales with resolution and duration.
Called inline from an async handler it holds the loop for its whole duration,
so the server answers nothing while it runs, liveness probes included, and a
long render is indistinguishable from a dead server to anything watching it.

These tests pin the encoders to a worker thread rather than the loop thread,
and pin that worker to one mux at a time, which is the memory profile inline
encoding had.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai import serving_video
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _no_frame_metadata(monkeypatch):
    # The fake videos below are opaque objects with no frames to count.
    monkeypatch.setattr(serving_video, "_video_metadata_from_artifacts", lambda artifacts: {})


def _artifacts(count: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        videos=[object() for _ in range(count)],
        audios=[None] * count,
        actions=[None] * count,
        output_fps=24,
        audio_sample_rate=None,
        stage_durations={},
        peak_memory_mb=0.0,
    )


@pytest.fixture
def make_handler():
    executors: list[ThreadPoolExecutor] = []

    def _make(artifacts: SimpleNamespace) -> OmniOpenAIServingVideo:
        handler = object.__new__(OmniOpenAIServingVideo)
        handler._video_frame_converter = None
        handler._video_mux_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-mux")
        handler._video_encoding_options = lambda: {}
        executors.append(handler._video_mux_executor)

        async def _run_and_extract(*args, **kwargs):
            return artifacts

        handler._run_and_extract = _run_and_extract
        return handler

    yield _make
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)


def _request() -> SimpleNamespace:
    return SimpleNamespace(extra_params=None)


@pytest.mark.asyncio
async def test_mp4_bytes_encoding_leaves_the_event_loop_free(monkeypatch, make_handler):
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_encode(video, **kwargs):
        seen["thread"] = threading.get_ident()
        return b"\x00\x00\x00\x18ftyp"

    monkeypatch.setattr(serving_video, "_encode_video_bytes", fake_encode)

    handler = make_handler(_artifacts())
    video_bytes, *_ = await handler.generate_video_bytes(_request(), "req-1")

    assert video_bytes.startswith(b"\x00\x00\x00")
    assert seen["thread"] != loop_thread, "encoding ran on the event loop thread"


@pytest.mark.asyncio
async def test_base64_encoding_leaves_the_event_loop_free(monkeypatch, make_handler):
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_encode(video, **kwargs):
        seen.append(threading.get_ident())
        return "AAAA"

    monkeypatch.setattr(serving_video, "encode_video_base64", fake_encode)

    handler = make_handler(_artifacts(count=2))
    response = await handler.generate_videos(_request(), "req-2")

    assert len(response.data) == 2
    assert seen and all(t != loop_thread for t in seen), "encoding ran on the event loop thread"


@pytest.mark.asyncio
async def test_other_requests_are_served_while_a_video_is_muxed(monkeypatch, make_handler):
    """The point of the offload, stated as behaviour rather than as a thread id.

    The encoder blocks until the loop tells it to stop. If the loop is free it
    says so in milliseconds; if the encoder is holding the loop, nothing can
    reach the release and the only way out is the timeout. So the elapsed time
    is the assertion, and it separates the two cases by two orders of magnitude.
    """
    release = threading.Event()
    stuck_for = 10.0

    def fake_encode(video, **kwargs):
        # Stands in for a mux long enough to matter. It blocks its own thread,
        # which is what a real encode does.
        release.wait(timeout=stuck_for)
        return b"\x00\x00\x00\x18ftyp"

    monkeypatch.setattr(serving_video, "_encode_video_bytes", fake_encode)

    async def other_traffic():
        for _ in range(5):
            await asyncio.sleep(0.01)
        release.set()

    handler = make_handler(_artifacts())
    started = time.perf_counter()
    await asyncio.gather(
        handler.generate_video_bytes(_request(), "req-3"),
        other_traffic(),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < stuck_for / 2, (
        f"the loop was held for {elapsed:.1f}s; other requests could not be served while the video was encoded"
    )


@pytest.mark.asyncio
async def test_overlapping_requests_mux_one_at_a_time(monkeypatch, make_handler):
    """Moving the mux off the loop must not let every finished request mux at once.

    Inline encoding admitted one mux at a time, so only one frame buffer was
    being encoded. Two requests that finish together must keep that: the second
    mux starts only after the first one returns.
    """
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_encode(video, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return b"\x00\x00\x00\x18ftyp"

    monkeypatch.setattr(serving_video, "_encode_video_bytes", fake_encode)

    handler = make_handler(_artifacts())
    await asyncio.gather(
        handler.generate_video_bytes(_request(), "req-4"),
        handler.generate_video_bytes(_request(), "req-5"),
    )

    assert peak == 1, f"{peak} muxes ran at once"


@pytest.mark.asyncio
async def test_cancelled_request_does_not_mux_after_it_left(monkeypatch, make_handler):
    """A mux queued behind another one is dropped when its request is cancelled."""
    release = threading.Event()
    started: list[str] = []

    def fake_encode(video, **kwargs):
        started.append(video)
        release.wait(timeout=10.0)
        return b"\x00\x00\x00\x18ftyp"

    monkeypatch.setattr(serving_video, "_encode_video_bytes", fake_encode)

    first = _artifacts()
    first.videos = ["first"]
    second = _artifacts()
    second.videos = ["second"]

    running = make_handler(first)
    queued = make_handler(second)
    queued._video_mux_executor.shutdown()
    queued._video_mux_executor = running._video_mux_executor

    first_task = asyncio.create_task(running.generate_video_bytes(_request(), "req-6"))
    while not started:
        await asyncio.sleep(0.01)
    second_task = asyncio.create_task(queued.generate_video_bytes(_request(), "req-7"))
    await asyncio.sleep(0.05)
    second_task.cancel()
    release.set()
    await first_task
    with pytest.raises(asyncio.CancelledError):
        await second_task
    await asyncio.sleep(0.05)

    assert started == ["first"], f"muxed for a cancelled request: {started}"
