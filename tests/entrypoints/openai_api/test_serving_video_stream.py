# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the serving-layer streaming video WebSocket handler."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai import video_stream_base, video_stream_envs
from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler,
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
    request_output = SimpleNamespace(outputs=[SimpleNamespace(text=text)])
    return OmniRequestOutput.from_stage_output(request_output, final_output_type="text")


def _audio_result(audio_data: Any) -> OmniRequestOutput:
    request_output = SimpleNamespace(outputs=[SimpleNamespace(multimodal_output={"audio": audio_data})])
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
async def test_query_waits_for_frame_prewarm(monkeypatch):
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
    await asyncio.sleep(0.05)
    assert not query_started.is_set()

    release_decode.set()
    await asyncio.wait_for(query_started.wait(), timeout=2.0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


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


# ---------------------------------------------------------------------------
# Incremental prefill (warmup) lifecycle
# ---------------------------------------------------------------------------


class RecordingReuseEngine:
    """Engine stub with stage-0 prefix caching on; records generate() and abort()."""

    def __init__(self, warmup_delay: float = 0.0, query_delay: float = 0.0):
        self.requests: list[dict[str, Any]] = []
        self.aborted: list[str] = []
        self._warmup_delay = warmup_delay
        self._query_delay = query_delay
        self.engine = SimpleNamespace(
            stage_vllm_configs=[SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))]
        )

    def warmup_ids(self) -> list[str]:
        return [r["request_id"] for r in self.requests if r["request_id"].startswith("video-warmup-")]

    def query_ids(self) -> list[str]:
        return [r["request_id"] for r in self.requests if not r["request_id"].startswith("video-warmup-")]

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        is_warmup = kwargs["request_id"].startswith("video-warmup-")
        delay = self._warmup_delay if is_warmup else self._query_delay

        async def _gen():
            if delay:
                await asyncio.sleep(delay)
            yield _text_result("ok")

        return _gen()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class PromptRecordingHandler(QwenOmniStreamingVideoHandler):
    """Records each query turn's user-message content and processor kwargs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_contents: list[list[dict[str, Any]]] = []
        self.warmup_contents: list[list[dict[str, Any]]] = []
        self.mm_processor_kwargs: list[Any] = []

    async def _preprocess_to_engine_prompt(self, request):
        self.mm_processor_kwargs.append(getattr(request, "mm_processor_kwargs", None))
        return {"prompt": "p"}

    def build_engine_prompt(self, config, frame_buffer, audio_buffer, message_history, query_text, prewarmed_frames):
        messages, user_message = super().build_engine_prompt(
            config, frame_buffer, audio_buffer, message_history, query_text, prewarmed_frames
        )
        self.query_contents.append(list(user_message["content"]))
        return messages, user_message

    def build_engine_prompt_prefix(self, config, frame_buffer, message_history, prewarmed_frames):
        messages = super().build_engine_prompt_prefix(config, frame_buffer, message_history, prewarmed_frames)
        if messages:
            self.warmup_contents.append(list(messages[-1]["content"]))
        return messages


async def _poll(predicate, timeout: float = 3.0) -> bool:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


@pytest.mark.asyncio
async def test_warmup_fires_on_frame_and_query_cancels_it():
    engine = RecordingReuseEngine(warmup_delay=10.0, query_delay=0.3)
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)

    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put({"type": "session.config", "model": "test", "modalities": ["text"], "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})

    assert await _poll(lambda: engine.warmup_ids())
    warmup_id = engine.warmup_ids()[0]
    assert engine.requests[0]["sampling_params"].max_tokens == 1

    ws.put({"type": "video.query", "text": "describe"})
    assert await _poll(lambda: engine.query_ids())
    # in-flight warmup was cancelled + aborted, and not restarted mid-query
    assert warmup_id in engine.aborted
    assert len(engine.warmup_ids()) == 1
    assert handler.mm_processor_kwargs[:2] == [
        {"use_audio_in_video": True},
        {"use_audio_in_video": True},
    ]

    assert await _poll(lambda: "response.text.done" in ws.sent_types())
    # the committed turn changed the context -> a fresh warmup fires
    assert await _poll(lambda: len(engine.warmup_ids()) == 2)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_audio_session_never_starts_warmup():
    engine = RecordingReuseEngine()
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)

    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put({"type": "session.config", "model": "test", "modalities": ["text", "audio"], "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(9, 9, 9))})
    await asyncio.sleep(0.3)

    assert engine.warmup_ids() == []
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_warmup_starts_only_after_frames_ready(monkeypatch):
    decode_started = threading.Event()
    release_decode = threading.Event()

    def gated_decode(raw_bytes: bytes):
        decode_started.set()
        release_decode.wait(timeout=5.0)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    monkeypatch.setattr(video_stream_base, "_decode_frame_bytes", gated_decode)
    engine = RecordingReuseEngine()
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)

    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put({"type": "session.config", "model": "test", "modalities": ["text"], "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    assert await _poll(decode_started.is_set)
    await asyncio.sleep(0.1)
    assert engine.warmup_ids() == []

    release_decode.set()
    assert await _poll(lambda: engine.warmup_ids())
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_query_cancelling_warmup_keeps_prewarm_alive(monkeypatch):
    """Cancel warmup during prefill must not drop a later frame still decoding."""
    frame1 = _b64(_make_jpeg(1, 1, 1))
    frame2 = _b64(_make_jpeg(2, 2, 2))
    raw2 = base64.b64decode(frame2)
    decode_started = threading.Event()
    release_decode = threading.Event()

    def gated_decode(raw_bytes: bytes):
        if raw_bytes == raw2:
            decode_started.set()
            release_decode.wait(timeout=5.0)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    monkeypatch.setattr(video_stream_base, "_decode_frame_bytes", gated_decode)
    engine = RecordingReuseEngine(warmup_delay=10.0)
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)

    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put({"type": "session.config", "model": "test", "modalities": ["text"], "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame1})
    assert await _poll(lambda: engine.warmup_ids())

    ws.put({"type": "video.frame", "data": frame2})
    assert await _poll(decode_started.is_set)

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0.1)
    release_decode.set()

    assert await _poll(lambda: "response.text.done" in ws.sent_types())
    assert len(handler.query_contents) == 1
    image_parts = [p for p in handler.query_contents[0] if p.get("type") == "image_pil"]
    assert len(image_parts) == 2
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_query_keeps_frames_evicted_while_pinned(monkeypatch):
    """Regression: frames FIFO-evicted while an in-flight query awaits prewarm
    must keep their PIL entries and stay in the query prompt."""
    frame_a = _b64(_make_jpeg(1, 1, 1))
    frame_a2 = _b64(_make_jpeg(2, 2, 2))
    frame_b = _b64(_make_jpeg(3, 3, 3))
    raw_a = base64.b64decode(frame_a)
    decoded: list[bytes] = []
    release_others = threading.Event()

    def gated_decode(raw_bytes: bytes):
        if raw_bytes != raw_a:
            release_others.wait(timeout=5.0)
        decoded.append(raw_bytes)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    monkeypatch.setattr(video_stream_base, "_decode_frame_bytes", gated_decode)
    engine = RecordingReuseEngine()
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)

    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text"],
            "max_frames": 2,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame_a})
    assert await _poll(lambda: raw_a in decoded)
    ws.put({"type": "video.frame", "data": frame_a2})  # decode gated -> query will wait

    ws.put({"type": "video.query", "text": "describe"})  # pins [a, a2]
    await asyncio.sleep(0.1)
    ws.put({"type": "video.frame", "data": frame_b})  # buffer full -> evicts pinned a
    await asyncio.sleep(0.1)
    release_others.set()

    assert await _poll(lambda: "response.text.done" in ws.sent_types())
    assert len(handler.query_contents) == 1
    uuids = [p["uuid"] for p in handler.query_contents[0] if p.get("type") == "image_pil"]
    expected = [hashlib.md5(base64.b64decode(f), usedforsecurity=False).hexdigest() for f in (frame_a, frame_a2)]
    assert uuids == expected
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


async def _run_frame_audio_query(modalities: list[str], frame: str, pcm: bytes) -> PromptRecordingHandler:
    engine = RecordingReuseEngine()
    handler = PromptRecordingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)
    ws = TimedWebSocket()
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": modalities,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame})
    if "audio" not in modalities:
        assert await _poll(lambda: engine.warmup_ids())
    else:
        await asyncio.sleep(0.05)
        assert engine.warmup_ids() == []
    ws.put({"type": "audio.chunk", "data": _b64(pcm)})
    ws.put({"type": "video.query", "text": "describe"})
    assert await _poll(lambda: engine.query_ids())
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    return handler


@pytest.mark.asyncio
async def test_reuse_query_with_audio_matches_legacy_prompt():
    """Incremental and legacy query prompts match; both set the processor kwarg."""
    frame = _b64(_make_jpeg())
    pcm = b"\x01\x00\x02\x00"
    reuse = await _run_frame_audio_query(["text"], frame, pcm)
    legacy = await _run_frame_audio_query(["text", "audio"], frame, pcm)

    assert reuse.warmup_contents
    assert all(part.get("type") != "input_audio" for part in reuse.warmup_contents[0])
    assert [part.get("type") for part in reuse.warmup_contents[0]] == ["image_pil"]

    assert reuse.query_contents and legacy.query_contents
    reuse_q = reuse.query_contents[0]
    legacy_q = legacy.query_contents[0]
    assert [part.get("type") for part in reuse_q] == ["image_pil", "input_audio", "text"]
    assert [part.get("type") for part in legacy_q] == ["image_pil", "input_audio", "text"]
    assert reuse_q[0]["uuid"] == legacy_q[0]["uuid"]
    assert reuse_q[1] == legacy_q[1]
    assert reuse_q[2] == legacy_q[2] == {"type": "text", "text": "describe"}

    assert reuse.mm_processor_kwargs[0] == {"use_audio_in_video": True}
    assert reuse.mm_processor_kwargs[1] == {"use_audio_in_video": True}
    assert legacy.mm_processor_kwargs[0] == {"use_audio_in_video": True}
