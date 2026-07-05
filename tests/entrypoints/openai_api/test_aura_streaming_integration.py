# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket smoke tests for AURA streaming video."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from vllm_omni.model_executor.stage_input_processors.aura_session_history import SessionHistory
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    AuraSessionState,
    SessionHistory,
)
from vllm_omni.entrypoints.openai.serving_video_stream import (
    AuraStreamingVideoHandler,
    AuraStreamingVideoSessionConfig,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (32, 32), (r, g, b))
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


def _audio_result_with_b64(b64_audio: str) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.multimodal_output = {"audio": [b64_audio]}
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="audio", request_output=request_output)


class TimedWebSocket:
    def __init__(self) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        return await self._q.get()

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent.append(data)

    def put(self, payload: dict[str, Any]) -> None:
        self._q.put_nowait(json.dumps(payload))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


@pytest.mark.asyncio
async def test_aura_auto_trigger_fires_after_min_frames():
    query_started = asyncio.Event()

    class CapturingAuraHandler(AuraStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()

    ws = TimedWebSocket()
    handler = CapturingAuraHandler(chat_service=object(), engine_client=MagicMock(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "auto_trigger": True,
            "auto_trigger_min_frames": 2,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10, 10, 10))})
    await asyncio.sleep(0.05)
    assert not query_started.is_set()

    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(20, 20, 20))})
    await asyncio.sleep(0.05)
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_aura_ignores_video_query_while_generating():
    query_count = 0
    gen_started = asyncio.Event()
    gen_release = asyncio.Event()

    class SlowAuraHandler(AuraStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal query_count
            query_count += 1
            gen_started.set()
            await gen_release.wait()

    ws = TimedWebSocket()
    handler = SlowAuraHandler(chat_service=object(), engine_client=MagicMock(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "auto_trigger_min_frames": 2,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10, 10, 10))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(20, 20, 20))})
    await asyncio.wait_for(gen_started.wait(), timeout=2.0)
    assert query_count == 1

    ws.put({"type": "video.query", "text": "ignored"})
    await asyncio.sleep(0.1)
    assert query_count == 1

    gen_release.set()
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_aura_releases_turn_after_text_allows_next_trigger_during_tts(monkeypatch):
    """After assistant text, frame trigger may fire while TTS audio still streams."""
    query_count = 0
    tts_release = asyncio.Event()
    second_turn_started = asyncio.Event()

    class SlowTtsEngine:
        def generate(self, **_kwargs):
            async def _gen():
                yield _text_result("第一輪")
                yield _audio_result_with_b64("unused")
                await tts_release.wait()
                yield _audio_result_with_b64("unused2")

            return _gen()

    class CapturingAuraHandler(AuraStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal query_count
            query_count += 1
            if query_count == 2:
                second_turn_started.set()
            await super()._process_query(*args, **kwargs)

        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "engine"}

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    monkeypatch.setattr(
        CapturingAuraHandler,
        "_extract_audio_delta_b64",
        classmethod(lambda cls, result, chunks_drained: ("UklGRiQAAABXQVZFZm10IBAAAAABAAEA", chunks_drained + 1)),
    )

    ws = TimedWebSocket()
    handler = CapturingAuraHandler(chat_service=object(), engine_client=SlowTtsEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text", "audio"],
            "auto_trigger_min_frames": 2,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10, 10, 10))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(20, 20, 20))})
    await asyncio.sleep(0.1)
    assert query_count == 1

    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(30, 30, 30))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(40, 40, 40))})
    await asyncio.wait_for(second_turn_started.wait(), timeout=2.0)
    assert query_count == 2

    tts_release.set()
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_aura_releases_turn_lock_when_generation_raises(monkeypatch):
    """A failed generation must not leave the turn lock held for the next trigger."""
    query_count = 0
    second_turn_started = asyncio.Event()

    class FailingThenOkEngine:
        def __init__(self) -> None:
            self._calls = 0

        def generate(self, **_kwargs):
            self._calls += 1
            call = self._calls

            async def _gen():
                if call == 1:
                    raise RuntimeError("boom")
                yield _text_result("第二輪")
                yield _audio_result_with_b64("unused")

            return _gen()

    class CapturingAuraHandler(AuraStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal query_count
            query_count += 1
            if query_count == 2:
                second_turn_started.set()
            await super()._process_query(*args, **kwargs)

        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "engine"}

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    monkeypatch.setattr(
        CapturingAuraHandler,
        "_extract_audio_delta_b64",
        classmethod(lambda cls, result, chunks_drained: ("UklGRiQAAABXQVZFZm10IBAAAAABAAEA", chunks_drained + 1)),
    )

    ws = TimedWebSocket()
    handler = CapturingAuraHandler(
        chat_service=object(),
        engine_client=FailingThenOkEngine(),
        idle_timeout=5.0,
    )
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text", "audio"],
            "auto_trigger_min_frames": 2,
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(10, 10, 10))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(20, 20, 20))})
    await asyncio.sleep(0.2)
    assert query_count == 1

    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(30, 30, 30))})
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(40, 40, 40))})
    await asyncio.wait_for(second_turn_started.wait(), timeout=2.0)
    assert query_count == 2

    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_aura_websocket_streams_text_and_audio(monkeypatch):
    class TextAudioEngine:
        def generate(self, **_kwargs):
            async def _gen():
                yield _text_result("你好")
                yield _audio_result_with_b64("unused")

            return _gen()

    class EngineAuraHandler(AuraStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "engine"}

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    monkeypatch.setattr(
        EngineAuraHandler,
        "_extract_audio_delta_b64",
        classmethod(lambda cls, result, chunks_drained: ("UklGRiQAAABXQVZFZm10IBAAAAABAAEA", chunks_drained + 1)),
    )

    handler = EngineAuraHandler(chat_service=object(), engine_client=TextAudioEngine())
    config = AuraStreamingVideoSessionConfig(model="test", modalities=["text", "audio"])
    state = AuraSessionState(history=SessionHistory(pruning_enabled=False), turn_frame_arrays=[])
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]

    ws = TimedWebSocket()
    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        bytearray(),
        state,
        "",
        "req-aura-audio",
        asyncio.Event(),
        {},
    )

    types = ws.sent_types()
    assert "response.start" in types
    assert "response.text.delta" not in types
    assert "response.text.done" in types
    assert "response.audio.delta" in types
    assert "response.audio.done" in types
    audio_msgs = [m for m in ws.sent if m.get("type") == "response.audio.delta"]
    assert audio_msgs and audio_msgs[0].get("format") == "wav"


@pytest.mark.asyncio
async def test_aura_multi_turn_accumulates_session_history():
    turns_completed = 0
    turn_done = asyncio.Event()

    class CountingAuraHandler(AuraStreamingVideoHandler):
        async def _process_query(self, *args, release_turn_lock=None, **kwargs):
            nonlocal turns_completed
            message_history = kwargs.get("message_history", args[4] if len(args) > 4 else None)
            request_id = kwargs.get("request_id", args[6] if len(args) > 6 else "")
            websocket = kwargs.get("websocket", args[0])
            response_text = f"reply-{turns_completed}"
            if release_turn_lock is not None:
                await release_turn_lock(
                    message_history=message_history,
                    user_message={"role": "user", "content": f"user-{turns_completed}"},
                    response_text=response_text,
                    request_id=request_id,
                )
            elif isinstance(message_history, AuraSessionState):
                message_history.history.add_user_message(f"user-{turns_completed}")
                message_history.history.add_assistant_message(response_text)
                message_history.turn_frame_arrays.clear()
            turns_completed += 1
            await websocket.send_json({"type": "response.text.done", "text": response_text})
            turn_done.set()

    ws = TimedWebSocket()
    handler = CountingAuraHandler(chat_service=object(), engine_client=MagicMock(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test", "auto_trigger_min_frames": 2, "enable_frame_filter": False})
    await asyncio.sleep(0.05)

    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(1, 1, 1))})
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(2, 2, 2))})
    await asyncio.wait_for(turn_done.wait(), timeout=2.0)
    turn_done.clear()

    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(3, 3, 3))})
    await asyncio.sleep(0.05)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(4, 4, 4))})
    await asyncio.sleep(0.05)
    await asyncio.wait_for(turn_done.wait(), timeout=2.0)

    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=3.0)

    assert turns_completed == 2
