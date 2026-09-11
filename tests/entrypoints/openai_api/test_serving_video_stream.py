# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the serving-layer streaming video WebSocket handler."""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import threading
import weakref
from typing import Any

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai import video_stream_base, video_stream_envs
from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_base import OmniStreamingVideoHandler
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (64, 64), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _text_result(text: str) -> OmniRequestOutput:
    class Output:
        text: str

    class RequestOutput:
        outputs: list[Output]

    output = Output()
    output.text = text
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput.from_stage_output(request_output, final_output_type="text")


def _audio_result(audio_data: Any) -> OmniRequestOutput:
    class Output:
        multimodal_output: dict[str, Any]

    class RequestOutput:
        outputs: list[Output]

    output = Output()
    output.multimodal_output = {"audio": audio_data}
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput.from_stage_output(request_output, final_output_type="audio")


class MockWebSocket:
    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            await asyncio.sleep(999)
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)


class TimedWebSocket:
    def __init__(self):
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        return await self._q.get()

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)

    def put(self, msg: dict[str, Any]):
        self._q.put_nowait(json.dumps(msg))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


def test_api_server_registers_video_stream_route():
    from vllm_omni.entrypoints.openai.api_server import router

    assert any(getattr(route, "path", None) == "/v1/video/chat/stream" for route in router.routes)


@pytest.mark.asyncio
async def test_receive_config_accepts_client_legacy_aliases():
    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "num_sample_frames": 7,
                    "evs_enabled": False,
                    "evs_threshold": 0.87,
                }
            )
        ]
    )
    handler = OmniStreamingVideoHandler(chat_service=object())

    config = await handler._receive_config(ws)

    assert config is not None
    assert config.num_frames == 7
    assert config.enable_frame_filter is False
    assert config.frame_filter_threshold == 0.87


@pytest.mark.asyncio
async def test_video_frame_ack_reports_receiver_buffer_state():
    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "enable_frame_filter": False,
                }
            ),
            json.dumps(
                {
                    "type": "video.frame",
                    "data": _b64(_make_jpeg()),
                    "frame_id": "frame-7",
                    "pts_ms": 700,
                    "capture_ts_ms": 1234.5,
                }
            ),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = QwenOmniStreamingVideoHandler(chat_service=object())

    await handler.handle_session(ws)

    ack = next(message for message in ws.sent if message.get("type") == "video.frame.ack")
    assert ack["frame_id"] == "frame-7"
    assert ack["pts_ms"] == 700
    assert ack["capture_ts_ms"] == 1234.5
    assert ack["accepted"] is True
    assert ack["buffered_frames"] == 1
    assert ack["server_receive_ts_ms"] > 0


@pytest.mark.asyncio
async def test_video_frames_consumed_is_emitted_after_engine_uses_frame_prompt():
    class OneOutputEngine:
        def generate(self, **_kwargs):
            async def _gen():
                await asyncio.sleep(0.05)
                yield _text_result("visible")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "with-video"}

    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "modalities": ["text"],
                    "num_frames": 1,
                    "enable_frame_filter": False,
                }
            ),
            json.dumps(
                {
                    "type": "video.frame",
                    "data": _b64(_make_jpeg()),
                    "frame_id": "frame-9",
                    "pts_ms": 900,
                    "source_pts_ms": 880,
                    "quality_profile": "balanced",
                }
            ),
            json.dumps({"type": "video.query", "text": "describe"}),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = CapturingHandler(
        chat_service=object(),
        engine_client=OneOutputEngine(),
        idle_timeout=2.0,
    )

    await handler.handle_session(ws)

    consumed = next(message for message in ws.sent if message.get("type") == "video.frames.consumed")
    assert consumed["frame_ids"] == ["frame-9"]
    assert consumed["latest_pts_ms"] == 900
    assert consumed["request_id"].startswith("video-")
    assert consumed["model_selected_ts_ms"] > 0
    assert consumed["frames"] == [
        {
            "frame_id": "frame-9",
            "pts_ms": 900,
            "source_pts_ms": 880,
            "quality_profile": "balanced",
            "receiver_received_ts_ms": consumed["frames"][0]["receiver_received_ts_ms"],
            "decoded_ready_ts_ms": consumed["frames"][0]["decoded_ready_ts_ms"],
        }
    ]
    assert consumed["frames"][0]["receiver_received_ts_ms"] > 0
    assert consumed["frames"][0]["decoded_ready_ts_ms"] >= consumed["frames"][0]["receiver_received_ts_ms"]
    assert consumed["model_selected_ts_ms"] >= consumed["frames"][0]["decoded_ready_ts_ms"]
    assert ws.sent.index(consumed) < next(
        index for index, message in enumerate(ws.sent) if message.get("type") == "response.text.delta"
    )


@pytest.mark.asyncio
async def test_audio_in_video_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=True)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}


@pytest.mark.asyncio
async def test_audio_in_video_disabled_omits_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=False)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs is None


@pytest.mark.asyncio
async def test_query_inline_audio_data_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket(
        [
            json.dumps({"type": "session.config", "model": "test"}),
            json.dumps({"type": "video.frame", "data": _b64(_make_jpeg())}),
            json.dumps(
                {
                    "type": "video.query",
                    "text": "describe",
                    "audio_data": _b64(b"\x00\x00"),
                }
            ),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=2.0)

    await handler.handle_session(ws)

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}
    assert "session.done" in [m.get("type") for m in ws.sent]


def test_audio_delta_mode_is_read_by_serving_code_at_runtime(monkeypatch):
    handler = OmniStreamingVideoHandler(chat_service=object())
    result = _audio_result([object()])

    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_fast",
        classmethod(lambda cls, audio_data, chunks_drained: ("fast-path", chunks_drained)),
    )
    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_slow",
        classmethod(lambda cls, audio_data, chunks_drained: ("slow-path", chunks_drained)),
    )

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "fast")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "fast-path"

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "slow")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "slow-path"


def test_video_stream_envs_strip_and_warn_once_per_invalid_value(monkeypatch):
    warnings = []

    video_stream_envs._warned_invalid_envs.clear()
    try:
        monkeypatch.setattr(
            video_stream_envs.logger,
            "warning",
            lambda message, *args, **_kwargs: warnings.append((message, args)),
        )

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", " off ")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "off"
        assert not warnings

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 1

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "still_bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 2
    finally:
        video_stream_envs._warned_invalid_envs.clear()


@pytest.mark.asyncio
async def test_async_chunk_mode_is_read_by_engine_path_at_runtime(monkeypatch):
    class TextEngine:
        def generate(self, **_kwargs):
            async def _gen():
                yield _text_result("hello")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "x"}

    handler = CapturingHandler(chat_service=object(), engine_client=TextEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text"])

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    ws_on = MockWebSocket()
    await handler._process_query_engine(
        ws_on,
        config,
        [_b64(_make_jpeg())],
        bytearray(),
        [],
        "describe",
        "req-on",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.delta", "delta": "hello"} in ws_on.sent

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "off")
    ws_off = MockWebSocket()
    await handler._process_query_engine(
        ws_off,
        config,
        [_b64(_make_jpeg())],
        bytearray(),
        [],
        "describe",
        "req-off",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.done", "text": "hello"} in ws_off.sent
    assert not any(m.get("type") == "response.text.delta" for m in ws_off.sent)


@pytest.mark.asyncio
async def test_query_without_engine_client_sends_error():
    ws = MockWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=None)

    await handler._process_query(
        ws,
        StreamingVideoSessionConfig(model="test"),
        [],
        bytearray(),
        [],
        "describe",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert {"type": "error", "message": "Streaming video requires an engine client"} in ws.sent


@pytest.mark.asyncio
async def test_new_query_cancels_in_flight_query():
    query_started = asyncio.Event()
    query_cancelled = asyncio.Event()
    calls = 0

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                return
            query_started.set()
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                query_cancelled.set()
                raise

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.query", "text": "interrupt"})
    await asyncio.wait_for(query_cancelled.wait(), timeout=2.0)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_abort", [False, True], ids=["immediate-ack", "delayed-ack"])
async def test_interrupted_queries_wait_for_abort_without_fixed_delay(monkeypatch, delay_abort):
    started: asyncio.Queue[str] = asyncio.Queue()
    abort_started: asyncio.Queue[str] = asyncio.Queue()
    allow_abort = asyncio.Event()
    if not delay_abort:
        allow_abort.set()
    request_ids: list[str] = []
    closed: list[str] = []
    aborted: list[str] = []
    delays: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay, result=None):
        # Observe requested delays without a machine-dependent latency limit.
        delays.append(delay)
        return await original_sleep(0, result)

    monkeypatch.setattr(video_stream_base.asyncio, "sleep", record_sleep)

    class BlockingEngine:
        async def generate(self, *, request_id, **kwargs):
            if request_ids:
                assert aborted[-1] == request_ids[-1]
            request_ids.append(request_id)
            started.put_nowait(request_id)
            try:
                if len(request_ids) <= 3:
                    yield _text_result("partial")
                    await asyncio.Event().wait()
                else:
                    yield _text_result("final")
            finally:
                closed.append(request_id)

        async def abort(self, request_id):
            assert request_id in closed
            abort_started.put_nowait(request_id)
            await allow_abort.wait()
            aborted.append(request_id)

    class PreprocessedHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt_token_ids": [1]}

    ws = TimedWebSocket()
    handler = PreprocessedHandler(chat_service=object(), engine_client=BlockingEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))
    try:
        ws.put({"type": "session.config", "modalities": ["text"], "enable_frame_filter": False})
        ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
        ws.put({"type": "video.query", "text": "first"})
        previous_id = await asyncio.wait_for(started.get(), timeout=2.0)

        for _ in range(3):
            ws.put({"type": "video.query", "text": "interrupt"})
            assert await asyncio.wait_for(abort_started.get(), timeout=2.0) == previous_id
            if delay_abort:
                assert previous_id not in aborted
                assert started.empty()
                allow_abort.set()
            previous_id = await asyncio.wait_for(started.get(), timeout=2.0)
            if delay_abort:
                allow_abort.clear()

        ws.put({"type": "video.done"})
        await asyncio.wait_for(task, timeout=2.0)

        assert aborted == request_ids[:-1]
        assert len(set(request_ids)) == 4
        assert [msg["text"] for msg in ws.sent if msg["type"] == "response.text.done"] == ["final"]
        assert "error" not in ws.sent_types()
        assert "session.done" in ws.sent_types()
        assert not [delay for delay in delays if delay > 0], "Restart must not add a timed grace period after abort"
    finally:
        allow_abort.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_video_done_waits_for_in_flight_query():
    query_started = asyncio.Event()
    allow_finish = asyncio.Event()
    query_finished = asyncio.Event()

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()
            await allow_finish.wait()
            query_finished.set()

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.done"})
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not query_finished.is_set()

    allow_finish.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert query_finished.is_set()
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_frame_prewarm_does_not_block_following_query(monkeypatch):
    decode_started = threading.Event()
    release_decode = threading.Event()
    query_started = asyncio.Event()

    def blocked_decode(raw_bytes: bytes):
        decode_started.set()
        release_decode.wait(timeout=2.0)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()

    monkeypatch.setattr(video_stream_base, "_decode_frame_bytes", blocked_decode)

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})

    for _ in range(100):
        if decode_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert decode_started.is_set()

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    release_decode.set()
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_frames", [1, 2], ids=["evicted", "retained"])
async def test_frame_prewarm_only_keeps_retained_images(monkeypatch, max_frames):
    frame_a = _make_jpeg(255, 0, 0)
    frame_b = _make_jpeg(0, 255, 0)
    decode_started = asyncio.Event()
    release_decode = asyncio.Event()
    frame_b_accepted = asyncio.Event()
    decoded_images: dict[bytes, weakref.ReferenceType[Image.Image]] = {}
    prewarm_task: asyncio.Task | None = None
    original_to_thread = asyncio.to_thread

    async def controlled_to_thread(function, *args, **kwargs):
        nonlocal prewarm_task
        if function is video_stream_base._decode_frame_bytes:
            if args[0] == frame_a:
                prewarm_task = asyncio.current_task()
                decode_started.set()
                await release_decode.wait()
            image = await original_to_thread(function, *args, **kwargs)
            decoded_images[args[0]] = weakref.ref(image)
            return image
        return await original_to_thread(function, *args, **kwargs)

    class AckWebSocket(TimedWebSocket):
        async def send_json(self, data):
            await super().send_json(data)
            if data.get("type") == "video.frame.ack" and data.get("frame_id") == "B":
                frame_b_accepted.set()

    monkeypatch.setattr(video_stream_base.asyncio, "to_thread", controlled_to_thread)
    ws = AckWebSocket()
    handler = QwenOmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    session_task = asyncio.create_task(handler.handle_session(ws))
    ws.put({"type": "session.config", "max_frames": max_frames, "enable_frame_filter": False})
    try:
        ws.put({"type": "video.frame", "frame_id": "A", "data": _b64(frame_a)})
        await asyncio.wait_for(decode_started.wait(), timeout=5.0)
        ws.put({"type": "video.frame", "frame_id": "B", "data": _b64(frame_b)})
        await asyncio.wait_for(frame_b_accepted.wait(), timeout=5.0)
        ack = next(message for message in ws.sent if message.get("frame_id") == "B")
        assert ack["accepted"] is True
        assert ack.get("dropped_frame_id") == ("A" if max_frames == 1 else None)

        # Finish A's real decode only after B has either evicted A or joined it.
        release_decode.set()
        assert prewarm_task is not None
        await asyncio.wait_for(asyncio.shield(prewarm_task), timeout=5.0)
        gc.collect()
        assert not session_task.done()
        # A finished task must not keep an evicted PIL image alive for the session.
        assert (decoded_images[frame_a]() is not None) == (max_frames == 2)
    finally:
        release_decode.set()
        ws.put({"type": "video.done"})
        await asyncio.wait_for(session_task, timeout=5.0)

    gc.collect()
    assert decoded_images[frame_a]() is None
    assert not any(message.get("type") == "error" for message in ws.sent)


@pytest.mark.asyncio
async def test_client_cannot_send_internal_frame_decode_failed_message():
    captured_frames: list[list[str]] = []
    frame = _b64(_make_jpeg())

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(
            self,
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
        ):
            captured_frames.append(list(frame_buffer))

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame})
    await asyncio.sleep(0)
    ws.put({"type": "_internal.frame_decode_failed", "b64": frame})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Unknown type: _internal.frame_decode_failed"} in ws.sent
    assert captured_frames == [[frame]]


@pytest.mark.asyncio
async def test_failed_frame_prewarm_removes_frame_before_query():
    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test", "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(b"not-a-jpeg")})

    for _ in range(100):
        if any(m.get("message") == "Frame decode failed" for m in ws.sent):
            break
        await asyncio.sleep(0.01)

    assert {"type": "error", "message": "Frame decode failed"} in ws.sent

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "No frames buffered"} in ws.sent


@pytest.mark.asyncio
async def test_frame_filter_error_sends_invalid_image(monkeypatch):
    def fail_should_retain(self, frame_jpeg):
        raise ValueError("decode failed")

    monkeypatch.setattr(video_stream_base.FrameSimilarityFilter, "should_retain", fail_should_retain)

    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Invalid image data"} in ws.sent
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_audio_buffer_overflow_clears_buffer_before_query(monkeypatch):
    captured_audio_lengths: list[int] = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query_engine(
            self,
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
        ):
            captured_audio_lengths.append(len(audio_buffer))

    monkeypatch.setattr(video_stream_base, "_MAX_AUDIO_BUFFER_BYTES", 4)

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"1234")})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"5")})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Audio buffer overflow"} in ws.sent
    assert captured_audio_lengths == [0]


def test_build_messages_keeps_recent_history_text_only():
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    old_frame = _b64(_make_jpeg(1, 2, 3))
    current_frame = _b64(_make_jpeg(4, 5, 6))
    history = [
        {"role": "user", "content": [{"type": "text", "text": "old question"}]},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{old_frame}"}},
                {"type": "input_audio", "input_audio": {"data": "ignored", "format": "wav"}},
                {"type": "text", "text": "recent question"},
            ],
        },
        {"role": "assistant", "content": "recent answer"},
    ]

    messages, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test", num_frames=1),
        [current_frame],
        bytearray(),
        history,
        "current question",
        {},
    )

    assert messages[0] == {"role": "user", "content": "recent question"}
    assert messages[1] == {"role": "assistant", "content": "recent answer"}
    assert messages[2] == user_message
    assert user_message["content"][-1] == {"type": "text", "text": "current question"}
