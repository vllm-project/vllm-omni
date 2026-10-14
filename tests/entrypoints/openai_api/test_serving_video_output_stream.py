# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for `/v1/realtime/video` WebSocket video output streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, WebSocket
from pytest_mock import MockerFixture
from starlette.testclient import TestClient

from tests.helpers.media import generate_synthetic_image
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_video_frames(num_frames: int = 2) -> list[np.ndarray]:
    return [np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(num_frames)]


def _receive_video_chunk(ws) -> tuple[dict, bytes]:
    metadata = ws.receive_json()
    assert metadata["type"] == "video.chunk_metadata"
    return metadata, ws.receive_bytes()


def _build_test_app(
    *,
    mocker: MockerFixture,
    streaming_chunks: list[tuple[bytes, bool]],
    mock_generate: Callable[..., AsyncGenerator[OmniRequestOutput, None]],
    final_chunk: bytes = b"",
    stall_timeout: float = 5.0,
    stage_configs: list[object] | None = None,
) -> tuple[FastAPI, OmniStreamingVideoOutputHandler, MagicMock]:
    engine_client = mocker.MagicMock()
    engine_client.abort = mocker.AsyncMock()
    engine_client.submit_interaction_async = mocker.AsyncMock()
    engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine_client.stage_configs = stage_configs or [
        SimpleNamespace(
            stage_type="diffusion",
            final_output=True,
            final_output_type="video",
        )
    ]
    engine_client.generate = mock_generate

    class FakeStreamingVideoEncoder:
        encode_calls: list[int] = []

        def encode(self, video):
            del video
            idx = len(self.encode_calls)
            self.encode_calls.append(idx)
            return streaming_chunks[idx][0]

        def close(self):
            return final_chunk

    def _make_encoder(*, output_format, fps, video_codec_options=None):
        del output_format, fps, video_codec_options
        return FakeStreamingVideoEncoder()

    encoder_factory = mocker.MagicMock(side_effect=_make_encoder)
    engine_client.streaming_encoder_factory = encoder_factory
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.create_streaming_video_encoder",
        encoder_factory,
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.build_stage_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_default_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )

    handler = OmniStreamingVideoOutputHandler(
        engine_client=engine_client,
        model_name="test-model",
        stage_configs=engine_client.stage_configs,
        stall_timeout=stall_timeout,
        start_timeout=5.0,
    )
    app = FastAPI()

    @app.websocket("/v1/realtime/video")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, handler, engine_client


def test_playback_feedback_reaches_engine_and_releases_next_chunk(mocker: MockerFixture):
    feedback = asyncio.Event()
    observed = []

    async def generate(*args, **kwargs):
        observed.append(kwargs["sampling_params_list"][0].streaming_buffer_seconds)
        yield OmniRequestOutput.from_diffusion(
            request_id="video",
            images=[_fake_video_frames(2)],
            final_output_type="image",
            finished=False,
        )
        await feedback.wait()
        yield OmniRequestOutput.from_diffusion(
            request_id="video",
            images=[_fake_video_frames(2)],
            final_output_type="image",
            finished=True,
        )

    app, _handler, engine = _build_test_app(
        mocker=mocker,
        streaming_chunks=[(b"first", False), (b"last", True)],
        mock_generate=generate,
    )

    # Exercise the real stage-parameter propagation instead of this helper's
    # canned default sampling params.
    from vllm_omni.entrypoints.openai.stage_params import build_stage_sampling_params_list

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.build_stage_sampling_params_list",
        side_effect=build_stage_sampling_params_list,
    )

    async def update(request_id, position):
        assert position == 0.25
        feedback.set()

    engine.update_streaming_playback = mocker.AsyncMock(side_effect=update)
    with TestClient(app) as client, client.websocket_connect("/v1/realtime/video") as ws:
        ws.send_json({"type": "session.start", "prompt": "video", "streaming_buffer_seconds": 0.75})
        start = ws.receive_json()
        assert start["type"] == "video.start"
        assert _receive_video_chunk(ws)[1] == b"first"
        ws.send_json({"type": "session.playback", "position_seconds": -1})
        assert ws.receive_json()["type"] == "error"
        engine.update_streaming_playback.assert_not_called()
        ws.send_json({"type": "session.playback", "position_seconds": 10**401})
        assert ws.receive_json()["type"] == "error"
        engine.update_streaming_playback.assert_not_called()
        ws.send_json({"type": "session.playback", "position_seconds": 0.25})
        assert _receive_video_chunk(ws)[1] == b"last"
        assert ws.receive_json()["type"] == "session.done"
    assert observed == [0.75]
    engine.update_streaming_playback.assert_awaited_once_with(start["request_id"], 0.25)


def test_paced_session_rejects_multi_stage_topology_at_start(mocker: MockerFixture):
    generate_called = False

    async def generate(*_args, **_kwargs):
        nonlocal generate_called
        generate_called = True
        raise AssertionError("generation must not start for an unsupported topology")
        yield  # pragma: no cover

    app, _handler, engine = _build_test_app(
        mocker=mocker,
        streaming_chunks=[],
        mock_generate=generate,
        stage_configs=[
            SimpleNamespace(stage_type="diffusion", final_output=False, final_output_type="video"),
            SimpleNamespace(stage_type="diffusion", final_output=True, final_output_type="video"),
        ],
    )
    engine.validate_streaming_playback_request = mocker.Mock(
        side_effect=ValueError("playback feedback requires single-stage diffusion")
    )
    with TestClient(app) as client, client.websocket_connect("/v1/realtime/video") as ws:
        ws.send_json({"type": "session.start", "prompt": "video", "streaming_buffer_seconds": 0.75})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "single-stage diffusion" in error["message"]
    assert not generate_called


def test_disconnect_aborts_while_generation_waits_for_playback(mocker: MockerFixture):
    aborted = asyncio.Event()

    async def generate(*args, **kwargs):
        yield OmniRequestOutput.from_diffusion(
            request_id="video",
            images=[_fake_video_frames(2)],
            final_output_type="image",
            finished=False,
        )
        await aborted.wait()

    app, _handler, engine = _build_test_app(
        mocker=mocker,
        streaming_chunks=[(b"first", False)],
        mock_generate=generate,
    )

    async def abort(request_id):
        aborted.set()

    engine.abort = mocker.AsyncMock(side_effect=abort)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime/video") as ws:
            ws.send_json({"type": "session.start", "prompt": "video", "streaming_buffer_seconds": 0.75})
            assert ws.receive_json()["type"] == "video.start"
            assert _receive_video_chunk(ws)[1] == b"first"
    engine.abort.assert_awaited()


class TestStreamingVideoOutputWebSocket:
    """WebSocket protocol tests for streaming generated video output."""

    def test_sampling_params_quality_reaches_h3_pipeline_boundary(
        self,
        mocker: MockerFixture,
    ) -> None:
        async def mock_generate(*_args, **_kwargs):
            if False:
                yield None

        _app, handler, _engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[],
            mock_generate=mock_generate,
        )
        request_data = handler._coerce_request_data(
            {
                "type": "session.start",
                "prompt": "H3 streaming quality",
                "sampling_params": {"quality": "high"},
            }
        )
        request = VideoGenerationRequest(**request_data)

        _prompt, sampling_params, _video_params = asyncio.run(handler._build_prompt_and_sampling_params(request))

        assert sampling_params.quality == "high"

    def test_omitted_quality_preserves_engine_default(self, mocker: MockerFixture) -> None:
        async def mock_generate(*_args, **_kwargs):
            if False:
                yield None

        _app, handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[],
            mock_generate=mock_generate,
        )
        engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams(quality="high")]
        request_data = handler._coerce_request_data(
            {
                "type": "session.start",
                "prompt": "H3 streaming default quality",
            }
        )
        request = VideoGenerationRequest(**request_data)

        _prompt, sampling_params, _video_params = asyncio.run(handler._build_prompt_and_sampling_params(request))

        assert sampling_params.quality == "high"

    def test_streaming_session_emits_video_start_binary_chunks_and_done(self, mocker: MockerFixture):
        """A full session delivers video.start, binary chunks, then session.done."""

        async def mock_generate(*_args, **_kwargs):
            for frames, finished in [
                (_fake_video_frames(2), False),
                (_fake_video_frames(3), True),
            ]:
                yield OmniRequestOutput.from_diffusion(
                    request_id="req-ws",
                    images=[frames],
                    final_output_type="image",
                    finished=finished,
                )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "A cat walking in the rain"})

                start = ws.receive_json()
                assert start["type"] == "video.start"
                assert start["format"] == "m4s"
                assert "format" not in start["config"]
                assert "request_id" in start

                metadata, chunk = _receive_video_chunk(ws)
                assert metadata["transport_chunk_index"] == 0
                assert metadata["kind"] == "media"
                assert chunk == b"mp4-chunk-0"
                metadata, chunk = _receive_video_chunk(ws)
                assert metadata["transport_chunk_index"] == 1
                assert chunk == b"mp4-chunk-1"

                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["chunks"] == 2
                assert done["stopped"] is False

        engine_client.abort.assert_not_called()
        engine_client.streaming_encoder_factory.assert_called_once()
        assert engine_client.streaming_encoder_factory.call_args.kwargs["output_format"] == "m4s"

    def test_streaming_session_accepts_initial_image_reference(self, mocker: MockerFixture):
        """session.start forwards image_reference as prompt multi_modal_data.image."""
        captured_prompts: list[dict[str, object]] = []

        async def mock_generate(*_args, **kwargs):
            captured_prompts.append(kwargs["prompt"])
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", True)],
            mock_generate=mock_generate,
        )

        synth_image = generate_synthetic_image(4, 4)
        image_url = f"data:image/jpeg;base64,{synth_image['base64']}"

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json(
                    {
                        "type": "session.start",
                        "prompt": "animate this image",
                        "width": 4,
                        "height": 4,
                        "image_reference": {"image_url": image_url},
                    }
                )
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"
                assert ws.receive_json()["type"] == "session.done"

        prompt = captured_prompts[0]
        image = prompt["multi_modal_data"]["image"]  # type: ignore[index]
        assert image.size == (4, 4)
        engine_client.abort.assert_not_called()

    def test_streaming_session_rejects_unknown_format(self, mocker: MockerFixture):
        async def mock_generate(*_args, **_kwargs):
            if False:
                yield None

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "hello", "format": "webm"})
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "format" in err["message"]

        engine_client.abort.assert_not_called()

    def test_first_message_must_be_session_start(self, mocker: MockerFixture):
        """Control messages sent before session.start are rejected at the handshake.

        This test can practically ensure ``session.interaction`` before ``session.start`` is rejected,
        but the implementation is more generic to also guard against other control messages.
        """

        async def mock_generate(*_args, **_kwargs):
            if False:
                yield None

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.interaction", "interaction": {"event": {"prompt": "too early"}}})
                err = ws.receive_json()
                assert err["type"] == "error"
                assert err["message"] == "Expected session.start, got: session.interaction"

        engine_client.abort.assert_not_called()
        engine_client.submit_interaction_async.assert_not_called()

    def test_streaming_session_emits_final_encoder_delta_before_done(self, mocker: MockerFixture):
        """A non-empty encoder close delta is delivered as the final binary chunk."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", True)],
            mock_generate=mock_generate,
            final_chunk=b"mp4-trailer",
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "A cat walking in the rain"})
                assert ws.receive_json()["type"] == "video.start"
                metadata, chunk = _receive_video_chunk(ws)
                assert metadata["kind"] == "media"
                assert chunk == b"mp4-chunk-0"
                metadata, chunk = _receive_video_chunk(ws)
                assert metadata["kind"] == "trailer"
                assert metadata["generation_chunk_index"] is None
                assert chunk == b"mp4-trailer"

                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["chunks"] == 2
                assert done["stopped"] is False

        engine_client.abort.assert_not_called()

    def test_generation_error_output_returns_websocket_error(self, mocker: MockerFixture):
        """An erroneous OmniRequestOutput from generate is sent to the client as a WebSocket error."""
        error_message = "diffusion engine exploded"

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            yield OmniRequestOutput.from_error("req-ws", error_message)

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "hello"})
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"
                err = ws.receive_json()
                assert err["type"] == "error"
                assert error_message in err["message"]

        engine_client.abort.assert_awaited_once()


class _MockVideoOutputWebSocket:
    """Feeds scripted client messages and records server sends."""

    def __init__(self, messages: list[str]) -> None:
        self._messages: asyncio.Queue[str] = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)
        self.sent: list[dict | bytes] = []

    def enqueue(self, message: str) -> None:
        self._messages.put_nowait(message)

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        return await self._messages.get()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    @property
    def sent_json(self) -> list[dict]:
        return [m for m in self.sent if isinstance(m, dict)]


def _build_handler_for_async_tests(
    mocker: MockerFixture,
    *,
    mock_generate: Callable[..., AsyncGenerator[OmniRequestOutput, None]],
    streaming_chunks: list[tuple[bytes, bool]] | None = None,
    stall_timeout: float = 5.0,
    streaming_buffer_seconds: float | None = None,
) -> tuple[OmniStreamingVideoOutputHandler, MagicMock, _MockVideoOutputWebSocket]:
    engine_client = mocker.MagicMock()
    engine_client.abort = mocker.AsyncMock()
    engine_client.submit_interaction_async = mocker.AsyncMock()
    engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine_client.stage_configs = [
        SimpleNamespace(
            stage_type="diffusion",
            final_output=True,
            final_output_type="video",
        )
    ]
    engine_client.generate = mock_generate

    chunks = streaming_chunks or [(b"mp4-chunk-0", True)]

    class FakeStreamingVideoEncoder:
        def __init__(self) -> None:
            self.encode_calls = 0

        def encode(self, video):
            del video
            idx = self.encode_calls
            self.encode_calls += 1
            return chunks[idx][0]

        def close(self):
            return b""

    encoder_factory = mocker.MagicMock(side_effect=lambda **_: FakeStreamingVideoEncoder())
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.create_streaming_video_encoder",
        encoder_factory,
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.build_stage_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_default_sampling_params_list",
        return_value=[OmniDiffusionSamplingParams()],
    )

    handler = OmniStreamingVideoOutputHandler(
        engine_client=engine_client,
        model_name="test-model",
        stage_configs=engine_client.stage_configs,
        stall_timeout=stall_timeout,
        start_timeout=5.0,
    )
    start_data: dict[str, object] = {"type": "session.start", "prompt": "async test"}
    if streaming_buffer_seconds is not None:
        start_data["streaming_buffer_seconds"] = streaming_buffer_seconds
    start_msg = json.dumps(start_data)
    ws = _MockVideoOutputWebSocket([start_msg])
    return handler, engine_client, ws


class TestStreamingVideoOutputStallTimeout:
    """Stall-timeout and session.ping behavior."""

    def test_long_generation_without_client_messages_succeeds(self, mocker: MockerFixture):
        """Engine progress between yields keeps the session alive without client pings."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
            stall_timeout=0.5,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "slow generation"})
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["stopped"] is False

        engine_client.abort.assert_not_called()

    def test_stall_timeout_aborts_when_engine_silent(self, mocker: MockerFixture):
        """No engine output for longer than stall_timeout aborts the request."""

        async def mock_generate(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", True)],
            mock_generate=mock_generate,
            stall_timeout=0.2,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "stalled"})
                assert ws.receive_json()["type"] == "video.start"
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "Stall timeout" in err["message"]

        engine_client.abort.assert_called()

    @pytest.mark.asyncio
    async def test_session_ping_returns_pong(self, mocker: MockerFixture):
        """session.ping during generation receives session.pong."""

        async def mock_generate(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        handler, _engine_client, ws = _build_handler_for_async_tests(
            mocker,
            mock_generate=mock_generate,
            stall_timeout=5.0,
        )

        async def _send_ping_after_delay() -> None:
            await asyncio.sleep(0.05)
            ws.enqueue(json.dumps({"type": "session.ping"}))

        ping_task = asyncio.create_task(_send_ping_after_delay())
        await handler.handle_session(ws)  # type: ignore[arg-type]
        await ping_task

        assert any(m.get("type") == "session.pong" for m in ws.sent_json)

    @pytest.mark.asyncio
    async def test_ping_prevents_stall_during_silent_engine(self, mocker: MockerFixture):
        """Client pings refresh the stall clock while the engine produces no output."""

        release = asyncio.Event()

        async def mock_generate(*_args, **_kwargs):
            await release.wait()
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        handler, engine_client, ws = _build_handler_for_async_tests(
            mocker,
            mock_generate=mock_generate,
            stall_timeout=0.3,
        )

        async def _pinger() -> None:
            while not release.is_set():
                ws.enqueue(json.dumps({"type": "session.ping"}))
                await asyncio.sleep(0.1)

        pinger_task = asyncio.create_task(_pinger())
        session_task = asyncio.create_task(handler.handle_session(ws))  # type: ignore[arg-type]
        await asyncio.sleep(0.8)
        release.set()
        await session_task
        await pinger_task

        errors = [m for m in ws.sent_json if m.get("type") == "error"]
        assert not any("Stall timeout" in m.get("message", "") for m in errors)
        assert any(m.get("type") == "session.pong" for m in ws.sent_json)
        assert any(m.get("type") == "session.done" for m in ws.sent_json)
        engine_client.abort.assert_not_called()


class TestStreamingVideoOutputPromptUpdate:
    """session.interaction control messages during streaming."""

    def test_prompt_update_accepted_during_streaming(self, mocker: MockerFixture):
        """A valid interaction during generation is forwarded to the engine and acknowledged."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial scene"})
                start = ws.receive_json()
                assert start["type"] == "video.start"
                request_id = start["request_id"]

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event_id": "ui-update-42",
                            "event": {"prompt": "new scene"},
                        },
                    }
                )
                queued = ws.receive_json()
                assert queued["type"] == "session.interaction.queued"
                assert queued["request_id"] == request_id
                assert queued["event_id"] == "ui-update-42"

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                done = ws.receive_json()
                assert done["type"] == "session.done"
                assert done["stopped"] is False

        engine_client.submit_interaction_async.assert_awaited_once_with(
            request_id,
            interaction={"event_id": "ui-update-42", "event": {"prompt": "new scene"}},
        )

    def test_prompt_update_rejects_duplicate_event_id(self, mocker: MockerFixture):
        """Any repeated event_id is rejected at the session API boundary."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        interaction = {
            "type": "session.interaction",
            "interaction": {
                "event_id": "ui-update-42",
                "event": {"prompt": "new scene"},
            },
        }

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial scene"})
                request_id = ws.receive_json()["request_id"]
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(interaction)
                queued = ws.receive_json()
                assert queued["type"] == "session.interaction.queued"

                ws.send_json(interaction)
                err = ws.receive_json()
                assert err["type"] == "error"
                assert err["code"] == "event_id_conflict"
                assert err["request_id"] == request_id
                assert err["event_id"] == "ui-update-42"

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_awaited_once_with(
            request_id,
            interaction={"event_id": "ui-update-42", "event": {"prompt": "new scene"}},
        )

    def test_prompt_update_forwards_transition_chunks(self, mocker: MockerFixture):
        """transition_chunks is passed through to the engine client."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial"})
                request_id = ws.receive_json()["request_id"]
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event_id": "ui-update-42",
                            "event": {"prompt": "fade to sunset"},
                            "transition_chunks": 5,
                        },
                    }
                )
                assert ws.receive_json()["type"] == "session.interaction.queued"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_awaited_once_with(
            request_id,
            interaction={
                "event_id": "ui-update-42",
                "event": {"prompt": "fade to sunset"},
                "transition_chunks": 5,
            },
        )

    def test_prompt_update_rejects_empty_prompt(self, mocker: MockerFixture):
        """An empty prompt is rejected before calling the engine."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial"})
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json({"type": "session.interaction", "interaction": {"event": {"prompt": "   "}}})
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "non-empty event.prompt" in err["message"]

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_not_called()

    @pytest.mark.parametrize(
        ("transition_chunks"),
        ["not-an-int", 2.9, True, -1],
    )
    def test_prompt_update_rejects_invalid_transition_chunks(
        self,
        mocker: MockerFixture,
        transition_chunks: object,
    ):
        """Non-integral or negative transition_chunks returns an error."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial"})
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event": {"prompt": "new scene"},
                            "transition_chunks": transition_chunks,
                        },
                    }
                )
                err = ws.receive_json()
                assert err["type"] == "error"

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_not_called()

    def test_prompt_update_engine_failure_returns_error(self, mocker: MockerFixture):
        """Engine failures during interaction are surfaced to the client."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )
        engine_client.submit_interaction_async.side_effect = RuntimeError("engine unavailable")

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial"})
                assert ws.receive_json()["type"] == "video.start"
                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event_id": "ui-update-42",
                            "event": {"prompt": "new scene"},
                        },
                    }
                )
                queued = ws.receive_json()
                assert queued["type"] == "session.interaction.queued"
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "Failed to apply interaction" in err["message"]
                assert err["code"] == "interaction_failed"
                assert err["event_id"] == "ui-update-42"

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prompt_update_accepted_via_async_handler(self, mocker: MockerFixture):
        """interaction acknowledgement is sent while generation is in progress."""

        release = asyncio.Event()

        async def mock_generate(*_args, **_kwargs):
            await release.wait()
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        handler, engine_client, ws = _build_handler_for_async_tests(
            mocker,
            mock_generate=mock_generate,
        )

        async def _send_interaction_after_delay() -> None:
            await asyncio.sleep(0.05)
            ws.enqueue(
                json.dumps(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event_id": "ui-update-42",
                            "event": {"prompt": "mid-stream update"},
                            "transition_chunks": 2,
                        },
                    }
                )
            )

        prompt_task = asyncio.create_task(_send_interaction_after_delay())
        session_task = asyncio.create_task(handler.handle_session(ws))  # type: ignore[arg-type]
        await asyncio.sleep(0.15)
        release.set()
        await session_task
        await prompt_task

        start = next(m for m in ws.sent_json if m.get("type") == "video.start")
        request_id = start["request_id"]
        assert any(m.get("type") == "session.interaction.queued" for m in ws.sent_json)
        engine_client.submit_interaction_async.assert_awaited_once_with(
            request_id,
            interaction={
                "event_id": "ui-update-42",
                "event": {"prompt": "mid-stream update"},
                "transition_chunks": 2,
            },
        )

    @pytest.mark.asyncio
    async def test_interaction_ingress_and_ack_do_not_wait_for_delivery(self, mocker: MockerFixture):
        delivery_started = asyncio.Event()
        release_delivery = asyncio.Event()
        generation_release = asyncio.Event()
        delivery_finished = asyncio.Event()
        delivered_ids: list[str] = []

        async def mock_generate(*_args, **_kwargs):
            await generation_release.wait()
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        handler, engine_client, ws = _build_handler_for_async_tests(
            mocker,
            mock_generate=mock_generate,
            streaming_buffer_seconds=0.75,
        )

        engine_client.reserve_streaming_interaction = mocker.AsyncMock(return_value=12.5)
        engine_client.release_streaming_interaction = mocker.AsyncMock()

        async def submit(request_id, *, interaction):
            del request_id
            delivered_ids.append(interaction["event_id"])
            delivery_started.set()
            await release_delivery.wait()
            if interaction["event_id"] == "second":
                delivery_finished.set()

        engine_client.submit_interaction_async = mocker.AsyncMock(side_effect=submit)
        session_task = asyncio.create_task(handler.handle_session(ws))  # type: ignore[arg-type]
        ws.enqueue(
            json.dumps(
                {"type": "session.interaction", "interaction": {"event_id": "first", "event": {"prompt": "one"}}}
            )
        )
        await asyncio.wait_for(delivery_started.wait(), 1)
        ws.enqueue(
            json.dumps(
                {"type": "session.interaction", "interaction": {"event_id": "second", "event": {"prompt": "two"}}}
            )
        )
        ws.enqueue(json.dumps({"type": "session.ping"}))

        async def wait_for_ingress_responses() -> None:
            while True:
                queued_ids = [m["event_id"] for m in ws.sent_json if m.get("type") == "session.interaction.queued"]
                if queued_ids == ["first", "second"] and any(m.get("type") == "session.pong" for m in ws.sent_json):
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_ingress_responses(), 1)

        queued = [m["event_id"] for m in ws.sent_json if m.get("type") == "session.interaction.queued"]
        assert queued == ["first", "second"]
        assert any(m.get("type") == "session.pong" for m in ws.sent_json)
        assert delivered_ids == ["first"]

        release_delivery.set()
        await asyncio.wait_for(delivery_finished.wait(), 1)
        generation_release.set()
        await asyncio.wait_for(session_task, 2)
        assert delivered_ids == ["first", "second"]

    def test_composite_prompt_and_camera_interaction_forwarded(self, mocker: MockerFixture):
        """Prompt + camera multi_modal_data is admitted and forwarded to the engine."""

        async def mock_generate(*_args, **_kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=False,
            )
            await asyncio.sleep(0.3)
            yield OmniRequestOutput.from_diffusion(
                request_id="req-ws",
                images=[_fake_video_frames(2)],
                final_output_type="image",
                finished=True,
            )

        app, _handler, engine_client = _build_test_app(
            mocker=mocker,
            streaming_chunks=[(b"mp4-chunk-0", False), (b"mp4-chunk-1", True)],
            mock_generate=mock_generate,
        )

        camera_payload = {
            "mode": "target",
            "data": {"translation": [0.0, 0.0, 1.0]},
        }

        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/video") as ws:
                ws.send_json({"type": "session.start", "prompt": "initial scene"})
                start = ws.receive_json()
                assert start["type"] == "video.start"
                request_id = start["request_id"]

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-0"

                ws.send_json(
                    {
                        "type": "session.interaction",
                        "interaction": {
                            "event_id": "combo-1",
                            "event": {
                                "prompt": "new scene",
                                "multi_modal_data": {"camera": camera_payload},
                            },
                            "transition_chunks": 2,
                        },
                    }
                )
                queued = ws.receive_json()
                assert queued["type"] == "session.interaction.queued"
                assert queued["event_id"] == "combo-1"

                assert _receive_video_chunk(ws)[1] == b"mp4-chunk-1"
                assert ws.receive_json()["type"] == "session.done"

        engine_client.submit_interaction_async.assert_awaited_once_with(
            request_id,
            interaction={
                "event_id": "combo-1",
                "event": {
                    "prompt": "new scene",
                    "multi_modal_data": {"camera": camera_payload},
                },
                "transition_chunks": 2,
            },
        )


class TestStreamingVideoOutputExtraParams:
    """extra_params handling when building streaming generation inputs."""

    @staticmethod
    def _handler() -> OmniStreamingVideoOutputHandler:
        handler = object.__new__(OmniStreamingVideoOutputHandler)
        handler._engine_client = SimpleNamespace(default_sampling_params_list=[])
        return handler

    async def test_preencode_mp4_is_rejected_for_streaming_sessions(self):
        """Worker-side pre-encoding emits one progressive MP4 the fMP4 encoder cannot consume."""
        request = VideoGenerationRequest(prompt="p", extra_params={"preencode_mp4": True})

        with pytest.raises(HTTPException) as excinfo:
            await self._handler()._build_prompt_and_sampling_params(request)

        assert excinfo.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "preencode_mp4" in excinfo.value.detail

    async def test_other_extra_params_still_reach_the_engine(self):
        request = VideoGenerationRequest(prompt="p", extra_params={"flow_shift": 3.0})

        _, gen_params, _ = await self._handler()._build_prompt_and_sampling_params(request)

        assert gen_params.extra_args["flow_shift"] == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize("first_control", ["session.ping", "session.interaction"])
async def test_blocked_media_send_does_not_block_playback_or_stop(mocker: MockerFixture, first_control):
    sending = asyncio.Event()
    release_send = asyncio.Event()
    aborted = asyncio.Event()

    async def generate(*_args, **_kwargs):
        yield OmniRequestOutput.from_diffusion(
            request_id="req-ws", images=[_fake_video_frames(2)], final_output_type="image", finished=True
        )

    handler, engine, ws = _build_handler_for_async_tests(mocker, mock_generate=generate)
    engine.update_streaming_playback = mocker.AsyncMock()
    engine.abort = mocker.AsyncMock(side_effect=lambda *_: aborted.set())

    async def send_bytes(data):
        sending.set()
        await release_send.wait()
        ws.sent.append(data)

    mocker.patch.object(ws, "send_bytes", send_bytes)
    tasks_before = asyncio.all_tasks()
    session = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(sending.wait(), 1)
        ws.enqueue(
            json.dumps({"type": first_control, "interaction": {"event_id": "first", "event": {"prompt": "one"}}})
        )
        ws.enqueue(json.dumps({"type": "session.playback", "position_seconds": 0.25}))
        ws.enqueue(json.dumps({"type": "session.stop"}))
        await asyncio.wait_for(aborted.wait(), 1)
        engine.update_streaming_playback.assert_awaited_once()
        assert engine.update_streaming_playback.call_args.args[1] == 0.25
        assert not release_send.is_set()
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)
    assert not (asyncio.all_tasks() - tasks_before)


@pytest.mark.asyncio
async def test_control_ack_outbox_keeps_fifo_and_media_pair(mocker: MockerFixture):
    sending = asyncio.Event()
    release_send = asyncio.Event()
    delivered = asyncio.Event()
    ids = []

    async def generate(*_args, **_kwargs):
        yield OmniRequestOutput.from_diffusion(
            request_id="req-ws", images=[_fake_video_frames(2)], final_output_type="image", finished=True
        )

    handler, engine, ws = _build_handler_for_async_tests(mocker, mock_generate=generate)

    async def submit(_request_id, *, interaction):
        ids.append(interaction["event_id"])
        if len(ids) == 2:
            delivered.set()

    engine.submit_interaction_async = mocker.AsyncMock(side_effect=submit)

    async def send_bytes(data):
        sending.set()
        await release_send.wait()
        ws.sent.append(data)

    mocker.patch.object(ws, "send_bytes", send_bytes)
    session = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(sending.wait(), 1)
        for event_id in ("first", "second"):
            ws.enqueue(
                json.dumps(
                    {
                        "type": "session.interaction",
                        "interaction": {"event_id": event_id, "event": {"prompt": event_id}},
                    }
                )
            )
        await asyncio.wait_for(delivered.wait(), 1)
        assert not any(item.get("type") == "session.interaction.queued" for item in ws.sent_json)
        release_send.set()
        await asyncio.wait_for(session, 2)
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)
    assert [item["event_id"] for item in ws.sent_json if item.get("type") == "session.interaction.queued"] == ids
    metadata_index = next(
        index
        for index, item in enumerate(ws.sent)
        if isinstance(item, dict) and item.get("type") == "video.chunk_metadata"
    )
    assert ws.sent[metadata_index + 1] == b"mp4-chunk-0"
    assert ws.sent_json[-1]["type"] == "session.done"


@pytest.mark.asyncio
async def test_control_ack_outbox_overflow_aborts_before_notification(mocker: MockerFixture):
    sending = asyncio.Event()
    aborted = asyncio.Event()

    async def generate(*_args, **_kwargs):
        yield OmniRequestOutput.from_diffusion(
            request_id="req-ws", images=[_fake_video_frames(2)], final_output_type="image", finished=True
        )

    handler, engine, ws = _build_handler_for_async_tests(mocker, mock_generate=generate)
    engine.abort = mocker.AsyncMock(side_effect=lambda *_: aborted.set())

    async def send_bytes(_data):
        sending.set()
        await asyncio.Event().wait()

    mocker.patch.object(ws, "send_bytes", send_bytes)
    tasks_before = asyncio.all_tasks()
    session = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(sending.wait(), 1)
        for _ in range(66):
            ws.enqueue(json.dumps({"type": "session.ping"}))
        await asyncio.wait_for(aborted.wait(), 1)
        assert not any(item.get("type") == "error" for item in ws.sent_json)
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)
    assert not (asyncio.all_tasks() - tasks_before)


def test_chunk_metadata_preserves_engine_pacing_fields():
    pacing = {"generated_seconds": 1.5, "played_seconds": 0.75, "inflight_seconds": 0.0}
    metadata = OmniStreamingVideoOutputHandler._build_chunk_metadata(
        request_id="req",
        kind="media",
        byte_length=12,
        num_frames=12,
        stream_metadata={"generation_chunk_index": 3, "output_chunk_index": 4, "pacing": pacing},
    )
    assert metadata["output_chunk_index"] == 4
    assert metadata["pacing"] == pacing


@pytest.mark.asyncio
async def test_interaction_event_id_is_bounded_before_queueing(mocker: MockerFixture):
    from vllm_omni.entrypoints.openai.serving_video_output_stream import _SessionProgress

    async def generate(*_args, **_kwargs):
        if False:
            yield

    handler, engine, ws = _build_handler_for_async_tests(mocker, mock_generate=generate)
    responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
    await handler._handle_interaction(
        ws,
        {"interaction": {"event_id": "x" * 129, "event": {"prompt": "one"}}},
        request_id="req",
        send_lock=asyncio.Lock(),
        interaction_event_ids=set(),
        interaction_queue=asyncio.Queue(maxsize=64),
        stop_event=asyncio.Event(),
        progress=_SessionProgress(),
        response_queue=responses,
    )
    assert "at most 128" in responses.get_nowait()["message"]
    engine.reserve_streaming_interaction.assert_not_called()
    engine.submit_interaction_async.assert_not_called()


@pytest.mark.asyncio
async def test_stop_after_final_media_cancels_pending_interaction(mocker: MockerFixture):
    delivery_started = asyncio.Event()
    media_sent = asyncio.Event()

    async def generate(*_args, **_kwargs):
        await delivery_started.wait()
        yield OmniRequestOutput.from_diffusion(
            request_id="req-ws", images=[_fake_video_frames(2)], final_output_type="image", finished=True
        )

    handler, engine, ws = _build_handler_for_async_tests(mocker, mock_generate=generate)

    async def submit(*_args, **_kwargs):
        delivery_started.set()
        await asyncio.Event().wait()

    async def send_bytes(data):
        ws.sent.append(data)
        media_sent.set()

    engine.submit_interaction_async = mocker.AsyncMock(side_effect=submit)
    mocker.patch.object(ws, "send_bytes", send_bytes)
    tasks_before = asyncio.all_tasks()
    session = asyncio.create_task(handler.handle_session(ws))
    try:
        ws.enqueue(
            json.dumps(
                {
                    "type": "session.interaction",
                    "interaction": {"event_id": "last", "event": {"prompt": "last update"}},
                }
            )
        )
        await asyncio.wait_for(media_sent.wait(), 1)
        ws.enqueue(json.dumps({"type": "session.stop"}))
        await asyncio.wait_for(session, 1)
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)
    engine.abort.assert_awaited()
    assert ws.sent_json[-1]["type"] == "session.done"
    assert ws.sent_json[-1]["stopped"] is True
    assert not (asyncio.all_tasks() - tasks_before)
