# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the serving-layer streaming video WebSocket handler."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from typing import Any

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai import serving_video_stream, video_stream_envs
from vllm_omni.entrypoints.openai.serving_video_stream import (
    OmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
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
        pass

    class RequestOutput:
        pass

    output = Output()
    output.text = text
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="text", request_output=request_output)


def _audio_result(audio_data: Any) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.multimodal_output = {"audio": audio_data}
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="audio", request_output=request_output)


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
async def test_audio_in_video_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(OmniStreamingVideoHandler):
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

    class CapturingHandler(OmniStreamingVideoHandler):
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

    class CapturingHandler(OmniStreamingVideoHandler):
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

    class CapturingHandler(OmniStreamingVideoHandler):
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

    class BlockingHandler(OmniStreamingVideoHandler):
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
async def test_video_done_waits_for_in_flight_query():
    query_started = asyncio.Event()
    allow_finish = asyncio.Event()
    query_finished = asyncio.Event()

    class BlockingHandler(OmniStreamingVideoHandler):
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

    class BlockingHandler(OmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()

    monkeypatch.setattr(serving_video_stream, "_decode_frame_bytes", blocked_decode)

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
async def test_client_cannot_send_internal_frame_decode_failed_message():
    captured_frames: list[list[str]] = []
    frame = _b64(_make_jpeg())

    class CapturingHandler(OmniStreamingVideoHandler):
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

    monkeypatch.setattr(serving_video_stream.FrameSimilarityFilter, "should_retain", fail_should_retain)

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

    class CapturingHandler(OmniStreamingVideoHandler):
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

    monkeypatch.setattr(serving_video_stream, "_MAX_AUDIO_BUFFER_BYTES", 4)

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
    handler = OmniStreamingVideoHandler(chat_service=object())
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
