# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime behavior tests for VideoStreamHandler.

Every test creates a mock WebSocket, drives handle_session through a
specific code path, and asserts on the JSON messages actually sent back.
No inspect.getsource() tricks — these test real async execution.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.video_stream_session import (
    VideoStreamHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (64, 64), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _config_msg(**overrides) -> str:
    msg = {"type": "session.config", "model": "test", "evs_enabled": False}
    msg.update(overrides)
    return json.dumps(msg)


def _frame_msg(jpeg: bytes | None = None) -> str:
    if jpeg is None:
        jpeg = _make_jpeg()
    return json.dumps({"type": "video.frame", "data": _b64(jpeg)})


def _query_msg(text: str = "What do you see?") -> str:
    return json.dumps({"type": "video.query", "text": text})


def _done_msg() -> str:
    return json.dumps({"type": "video.done"})


def _audio_msg(pcm: bytes = b"\x00" * 320) -> str:
    return json.dumps({"type": "audio.chunk", "data": _b64(pcm)})


def _make_sse_line(content: str) -> str:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}\n\n"


class MockWebSocket:
    """Async-compatible mock WebSocket that feeds messages from a list."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._idx = 0
        self._sent: list[dict | bytes] = []
        self._accepted = False

    async def accept(self):
        self._accepted = True

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            # Simulate connection close by hanging forever (will be timed out)
            await asyncio.sleep(999)
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_json(self, data: dict):
        self._sent.append(data)

    async def send_bytes(self, data: bytes):
        self._sent.append(data)

    @property
    def sent_messages(self) -> list[dict]:
        return [m for m in self._sent if isinstance(m, dict)]

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent_messages]


class MockChatHandler:
    """Mock OmniOpenAIServingChat for testing _handle_query paths."""

    def __init__(self, response=None):
        self._response = response

    async def create_chat_completion(self, request, raw_request=None):
        if self._response is not None:
            return self._response

        # Default: return a simple streaming generator
        async def _gen():
            yield _make_sse_line("Hello")
            yield _make_sse_line("World")
            yield "data: [DONE]\n\n"

        return _gen()


# ---------------------------------------------------------------------------
# _receive_config path tests
# ---------------------------------------------------------------------------


class TestReceiveConfig:
    @pytest.mark.asyncio
    async def test_config_timeout(self):
        """No message within config_timeout → error sent, session ends."""
        ws = MockWebSocket([])  # no messages — will hang
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), config_timeout=0.05)
        await handler.handle_session(ws)

        assert ws._accepted
        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert len(errors) == 1
        assert "Timeout" in errors[0]["message"]

    @pytest.mark.asyncio
    async def test_config_invalid_json(self):
        """Non-JSON config message → error, session ends."""
        ws = MockWebSocket(["not json"])
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), config_timeout=1.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Invalid JSON" in e["message"] for e in errors)

    @pytest.mark.asyncio
    async def test_config_wrong_type(self):
        """Message with wrong type → error, session ends."""
        ws = MockWebSocket([json.dumps({"type": "wrong"})])
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), config_timeout=1.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Expected session.config" in e["message"] for e in errors)

    @pytest.mark.asyncio
    async def test_config_invalid_field_type(self):
        """Config with wrong field type → error, session ends."""
        ws = MockWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.config",
                        "model": "test",
                        "max_frames": "potato",
                    }
                )
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), config_timeout=1.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("max_frames" in e["message"] for e in errors)


# ---------------------------------------------------------------------------
# Normal session flow
# ---------------------------------------------------------------------------


class TestNormalFlow:
    @pytest.mark.asyncio
    async def test_frame_query_done(self):
        """Happy path: config → frame → query → done → session.done."""
        ws = MockWebSocket(
            [
                _config_msg(),
                _frame_msg(),
                _query_msg("Describe."),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        types = ws.sent_types()
        assert "response.start" in types
        assert "response.text.delta" in types
        assert "response.text.done" in types
        assert "session.done" in types

        # Check text content was streamed
        deltas = [m["delta"] for m in ws.sent_messages if m.get("type") == "response.text.delta"]
        assert "Hello" in deltas
        assert "World" in deltas

        # response.text.done has full text
        done_msg = next(m for m in ws.sent_messages if m.get("type") == "response.text.done")
        assert done_msg["text"] == "HelloWorld"

    @pytest.mark.asyncio
    async def test_multiple_frames_before_query(self):
        """Multiple frames accumulate, query uses sampled frames."""
        frames = [_frame_msg(_make_jpeg(r=i * 50)) for i in range(5)]
        ws = MockWebSocket(
            [
                _config_msg(num_sample_frames=3),
                *frames,
                _query_msg(),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)
        assert "session.done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_evs_stats_in_done(self):
        """EVS stats sent before session.done when evs_enabled=True."""
        frame = _frame_msg(_make_jpeg(100, 100, 100))
        ws = MockWebSocket(
            [
                _config_msg(evs_enabled=True, evs_threshold=0.90),
                frame,
                frame,  # second should be dropped
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        evs_msgs = [m for m in ws.sent_messages if m.get("type") == "response.evs_stats"]
        assert len(evs_msgs) == 1
        assert evs_msgs[0]["retained_count"] == 1
        assert evs_msgs[0]["dropped_count"] == 1


# ---------------------------------------------------------------------------
# _handle_query error paths
# ---------------------------------------------------------------------------


class TestHandleQueryErrors:
    @pytest.mark.asyncio
    async def test_query_with_no_frames(self):
        """Query before any frames → error, session continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                _query_msg("Hello?"),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("No frames" in e["message"] for e in errors)
        # Session still ends cleanly
        assert "session.done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_query_returns_error_response(self):
        """create_chat_completion returns ErrorResponse → error forwarded."""
        from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

        err = ErrorResponse(
            error=ErrorInfo(
                message="model overloaded",
                type="server_error",
                code=503,
            ),
        )

        chat = AsyncMock()
        chat.create_chat_completion = AsyncMock(return_value=err)

        ws = MockWebSocket(
            [
                _config_msg(),
                _frame_msg(),
                _query_msg(),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=chat, idle_timeout=2.0)
        await handler.handle_session(ws)

        # Should see response.start, then error, then response.text.done
        assert "response.start" in ws.sent_types()
        assert "response.text.done" in ws.sent_types()
        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("model overloaded" in e["message"] for e in errors)

    @pytest.mark.asyncio
    async def test_query_generator_raises(self):
        """Exception during streaming → error sent, session continues."""

        async def _exploding_gen():
            yield _make_sse_line("partial")
            raise RuntimeError("CUDA OOM")

        chat = MockChatHandler()
        chat._response = None  # override below

        class BoomChat:
            async def create_chat_completion(self, req, raw_request=None):
                return _exploding_gen()

        ws = MockWebSocket(
            [
                _config_msg(),
                _frame_msg(),
                _query_msg(),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=BoomChat(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Query processing failed" in e["message"] for e in errors)
        # Crucially: no raw exception message leaked
        assert not any("CUDA OOM" in e["message"] for e in errors)
        # Protocol: response.text.done always sent after response.start
        assert "response.text.done" in ws.sent_types()
        # Session still ends cleanly
        assert "session.done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_empty_query_text_rejected(self):
        """video.query with empty text → error, session continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                _frame_msg(),
                json.dumps({"type": "video.query", "text": ""}),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("non-empty" in e["message"] for e in errors)
        assert "session.done" in ws.sent_types()


# ---------------------------------------------------------------------------
# Timeout paths
# ---------------------------------------------------------------------------


class TestTimeouts:
    @pytest.mark.asyncio
    async def test_idle_timeout(self):
        """No message after config within idle_timeout → error, session ends."""
        ws = MockWebSocket([_config_msg()])  # config then nothing
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=0.05)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Idle timeout" in e["message"] for e in errors)


# ---------------------------------------------------------------------------
# Reader error paths
# ---------------------------------------------------------------------------


class TestReaderErrors:
    @pytest.mark.asyncio
    async def test_invalid_json_mid_session(self):
        """Invalid JSON mid-session → error sent, session continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                "{{bad json",
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Invalid JSON" in e["message"] for e in errors)
        assert "session.done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_non_dict_message(self):
        """Non-object JSON → error sent, continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                json.dumps([1, 2, 3]),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("JSON objects" in e["message"] for e in errors)

    @pytest.mark.asyncio
    async def test_unknown_message_type(self):
        """Unknown type → error, session continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                json.dumps({"type": "teleport.now"}),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Unknown message type" in e["message"] for e in errors)
        assert "session.done" in ws.sent_types()

    @pytest.mark.asyncio
    async def test_invalid_base64_frame(self):
        """Invalid base64 in video.frame → error, continues."""
        ws = MockWebSocket(
            [
                _config_msg(),
                json.dumps({"type": "video.frame", "data": "!!!not-base64!!!"}),
                _done_msg(),
            ]
        )
        handler = VideoStreamHandler(chat_handler=MockChatHandler(), idle_timeout=2.0)
        await handler.handle_session(ws)

        errors = [m for m in ws.sent_messages if m["type"] == "error"]
        assert any("Invalid base64" in e["message"] for e in errors)
        assert "session.done" in ws.sent_types()


# ---------------------------------------------------------------------------
# Concurrency: frames arrive during query processing
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_frames_during_query(self):
        """Frames sent while a query is processing must be buffered."""
        query_started = asyncio.Event()
        query_can_finish = asyncio.Event()
        captured_frame_count: list[int] = []

        class SlowChat:
            async def create_chat_completion(self, req, raw_request=None):
                query_started.set()
                await query_can_finish.wait()

                async def _gen():
                    yield _make_sse_line("answer")
                    yield "data: [DONE]\n\n"

                return _gen()

        class InspectingSlowChat(SlowChat):
            """On the second query, record how many frames are in the request."""

            _call_count = 0

            async def create_chat_completion(self, req, raw_request=None):
                self._call_count += 1
                if self._call_count == 2:
                    # Count image_url blocks in the request to verify buffering
                    for msg in req.messages:
                        if isinstance(msg.get("content", msg), list):
                            content = msg.get("content", msg)
                        else:
                            content = getattr(msg, "content", [])
                        if isinstance(content, list):
                            n = sum(
                                1
                                for p in content
                                if (isinstance(p, dict) and p.get("type") == "image_url")
                                or (hasattr(p, "type") and p.type == "image_url")
                            )
                            captured_frame_count.append(n)
                return await super().create_chat_completion(req, raw_request)

        # Custom WebSocket that can inject frames after query starts
        class TimedWebSocket(MockWebSocket):
            def __init__(self):
                self._q: asyncio.Queue[str] = asyncio.Queue()
                self._sent: list[dict | bytes] = []
                self._accepted = False

            async def accept(self):
                self._accepted = True

            async def receive_text(self) -> str:
                return await self._q.get()

            def put(self, msg: str):
                self._q.put_nowait(msg)

        ws = TimedWebSocket()
        chat = InspectingSlowChat()
        handler = VideoStreamHandler(chat_handler=chat, idle_timeout=5.0)

        session_task = asyncio.create_task(handler.handle_session(ws))

        # 1. Config
        ws.put(_config_msg())
        await asyncio.sleep(0.01)

        # 2. Initial frame + query
        ws.put(_frame_msg(_make_jpeg(10, 10, 10)))
        await asyncio.sleep(0.01)
        ws.put(_query_msg("first"))

        # 3. Wait for query to start processing
        await asyncio.wait_for(query_started.wait(), timeout=2.0)

        # 4. Send more frames WHILE query is running
        ws.put(_frame_msg(_make_jpeg(200, 200, 200)))
        ws.put(_frame_msg(_make_jpeg(50, 50, 50)))
        await asyncio.sleep(0.05)

        # 5. Let query finish
        query_can_finish.set()
        await asyncio.sleep(0.1)

        # 6. Second query — should see all 3 frames (1 initial + 2 during)
        query_started.clear()
        query_can_finish.clear()
        ws.put(_query_msg("second"))
        await asyncio.wait_for(query_started.wait(), timeout=2.0)
        query_can_finish.set()
        await asyncio.sleep(0.1)

        # 7. Done
        ws.put(_done_msg())

        await asyncio.wait_for(session_task, timeout=5.0)

        assert "session.done" in ws.sent_types()
        # Verify: the 2 frames sent during query 1 were buffered and
        # included in query 2 (total 3 frames).
        assert captured_frame_count == [3]


# ---------------------------------------------------------------------------
# SSE parsing (kept from original, these are valid unit tests)
# ---------------------------------------------------------------------------


class TestParseSSEDeltas:
    def test_single_delta(self):
        assert VideoStreamHandler._parse_sse_deltas(_make_sse_line("hello")) == ["hello"]

    def test_multiple_deltas(self):
        chunk = _make_sse_line("a") + _make_sse_line("b")
        assert VideoStreamHandler._parse_sse_deltas(chunk) == ["a", "b"]

    def test_done_skipped(self):
        chunk = _make_sse_line("x") + "data: [DONE]\n\n"
        assert VideoStreamHandler._parse_sse_deltas(chunk) == ["x"]

    def test_empty_content_skipped(self):
        p = json.dumps({"choices": [{"delta": {"content": ""}}]})
        assert VideoStreamHandler._parse_sse_deltas(f"data: {p}\n\n") == []

    def test_malformed_json_skipped(self):
        chunk = "data: {bad}\n\n" + _make_sse_line("ok")
        assert VideoStreamHandler._parse_sse_deltas(chunk) == ["ok"]

    def test_empty_string(self):
        assert VideoStreamHandler._parse_sse_deltas("") == []

    def test_unicode(self):
        assert VideoStreamHandler._parse_sse_deltas(_make_sse_line("你好🌍")) == ["你好🌍"]
