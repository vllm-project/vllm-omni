# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `/v1/realtime/video` WebSocket video output streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from pytest_mock import MockerFixture
from starlette.testclient import TestClient

from tests.helpers.media import generate_synthetic_image
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
) -> tuple[FastAPI, OmniStreamingVideoOutputHandler, MagicMock]:
    engine_client = mocker.MagicMock()
    engine_client.abort = mocker.AsyncMock()
    engine_client.submit_interaction_async = mocker.AsyncMock()
    engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine_client.stage_configs = [SimpleNamespace(stage_type="diffusion")]
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
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_stage_type",
        return_value="diffusion",
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


class TestStreamingVideoOutputWebSocket:
    """WebSocket protocol tests for streaming generated video output."""

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

        engine_client.abort.assert_not_called()


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
) -> tuple[OmniStreamingVideoOutputHandler, MagicMock, _MockVideoOutputWebSocket]:
    engine_client = mocker.MagicMock()
    engine_client.abort = mocker.AsyncMock()
    engine_client.submit_interaction_async = mocker.AsyncMock()
    engine_client.default_sampling_params_list = [OmniDiffusionSamplingParams()]
    engine_client.stage_configs = [SimpleNamespace(stage_type="diffusion")]
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
        "vllm_omni.entrypoints.openai.serving_video_output_stream.get_stage_type",
        return_value="diffusion",
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
    start_msg = json.dumps({"type": "session.start", "prompt": "async test"})
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
