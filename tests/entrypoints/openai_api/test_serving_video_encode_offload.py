# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MP4 muxing must not run on the event loop.

Encoding a finished video is CPU-bound and scales with resolution and duration.
Called inline from an async handler it holds the loop for its whole duration,
so the server answers nothing while it runs, liveness probes included, and a
long render is indistinguishable from a dead server to anything watching it.

These tests pin the encoders to a worker thread rather than the loop thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai import serving_video
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def _handler(artifacts: SimpleNamespace) -> OmniOpenAIServingVideo:
    handler = object.__new__(OmniOpenAIServingVideo)
    handler._video_frame_converter = None

    async def _run_and_extract(*args, **kwargs):
        return artifacts

    handler._run_and_extract = _run_and_extract
    return handler


def _request() -> SimpleNamespace:
    return SimpleNamespace(extra_params=None)


@pytest.mark.asyncio
async def test_mp4_bytes_encoding_leaves_the_event_loop_free(monkeypatch):
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_encode(video, **kwargs):
        seen["thread"] = threading.get_ident()
        return b"\x00\x00\x00\x18ftyp"

    monkeypatch.setattr(serving_video, "_encode_video_bytes", fake_encode)

    handler = _handler(_artifacts())
    video_bytes, _, _, _ = await handler.generate_video_bytes(_request(), "req-1")

    assert video_bytes.startswith(b"\x00\x00\x00")
    assert seen["thread"] != loop_thread, "encoding ran on the event loop thread"


@pytest.mark.asyncio
async def test_base64_encoding_leaves_the_event_loop_free(monkeypatch):
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_encode(video, **kwargs):
        seen.append(threading.get_ident())
        return "AAAA"

    monkeypatch.setattr(serving_video, "encode_video_base64", fake_encode)

    handler = _handler(_artifacts(count=2))
    response = await handler.generate_videos(_request(), "req-2")

    assert len(response.data) == 2
    assert seen and all(t != loop_thread for t in seen), (
        "encoding ran on the event loop thread"
    )


@pytest.mark.asyncio
async def test_other_requests_are_served_while_a_video_is_muxed(monkeypatch):
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

    handler = _handler(_artifacts())
    started = time.perf_counter()
    await asyncio.gather(
        handler.generate_video_bytes(_request(), "req-3"),
        other_traffic(),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < stuck_for / 2, (
        f"the loop was held for {elapsed:.1f}s; other requests could not be "
        "served while the video was encoded"
    )
