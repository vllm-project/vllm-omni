# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.streaming_input import validate_streaming_input_sampling_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class _FakeModel:
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        assert modality == "audio"
        assert i == 0
        return "<audio-test>"


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.messages.append(payload)


class _FakeAsyncChunkEngine:
    async_chunk = True

    def __init__(self) -> None:
        self.default_sampling_params_list = [
            SamplingParams(
                n=1,
                output_kind=RequestOutputKind.CUMULATIVE,
            )
        ]
        self.generate_kwargs: dict[str, Any] | None = None
        self.abort_calls: list[str] = []

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs

        async def _results():
            yield SimpleNamespace(
                prompt_token_ids=[11, 12],
                outputs=[
                    SimpleNamespace(
                        token_ids=[21, 22],
                        text="hello",
                    )
                ],
                multimodal_output={
                    "audio": np.array([0.0, 0.5], dtype=np.float32),
                    "sample_rate": 24000,
                },
            )

        return _results()

    async def abort(self, request_id: str) -> None:
        self.abort_calls.append(request_id)


class _FakeServing:
    model_cls = _FakeModel
    model_config = None


class _FakePrivateEngineServing:
    model_cls = _FakeModel
    model_config = None

    def __init__(self, engine: _FakeAsyncChunkEngine) -> None:
        self._engine_client = engine


class TestRealtimeConnectionAsyncChunkBridge:
    def _connection(self) -> tuple[RealtimeConnection, _FakeAsyncChunkEngine, _FakeWebSocket]:
        conn = RealtimeConnection.__new__(RealtimeConnection)
        engine = _FakeAsyncChunkEngine()
        websocket = _FakeWebSocket()
        conn.engine = engine
        conn.serving = _FakeServing()
        conn.websocket = websocket
        conn.audio_queue = asyncio.Queue()
        conn.connection_id = "test-conn"
        conn.generation_task = None
        conn._is_connected = True
        conn._is_model_validated = True
        conn._max_audio_filesize_mb = 64
        conn._realtime_audio_ref = None
        return conn, engine, websocket

    def test_init_binds_engine_without_runtime_async_omni_import(self) -> None:
        engine = _FakeAsyncChunkEngine()
        serving = SimpleNamespace(engine_client=engine)

        conn = RealtimeConnection(_FakeWebSocket(), serving)

        assert conn.engine is engine
        assert conn._realtime_audio_ref is None

    def test_init_accepts_private_upstream_engine_client_attr(self) -> None:
        engine = _FakeAsyncChunkEngine()
        serving = _FakePrivateEngineServing(engine)

        conn = RealtimeConnection(_FakeWebSocket(), serving)

        assert conn.engine is engine

    def test_build_realtime_audio_prompt_uses_audio_tuple(self) -> None:
        audio = np.array([[0.1], [-0.2]], dtype=np.float64)
        prompt = RealtimeConnection._build_realtime_audio_prompt(audio, 16000, _FakeModel)

        assert prompt["prompt"] == "<|im_start|>user\n<audio-test><|im_end|>\n<|im_start|>assistant\n"
        prompt_audio, sample_rate = prompt["multi_modal_data"]["audio"]
        assert sample_rate == 16000
        assert prompt_audio.dtype == np.float32
        assert prompt_audio.shape == (2,)
        np.testing.assert_allclose(prompt_audio, [0.1, -0.2], rtol=1e-6)

    @pytest.mark.asyncio
    async def test_collect_committed_audio_concatenates_until_sentinel(self) -> None:
        conn, _, _ = self._connection()
        conn.audio_queue.put_nowait(np.array([0.1, 0.2], dtype=np.float32))
        conn.audio_queue.put_nowait(np.array([[0.3]], dtype=np.float32))
        conn.audio_queue.put_nowait(None)

        audio = await conn._collect_committed_audio()

        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, [0.1, 0.2, 0.3], rtol=1e-6)

    @pytest.mark.asyncio
    async def test_collect_committed_audio_rejects_empty_commit(self) -> None:
        conn, _, _ = self._connection()
        conn.audio_queue.put_nowait(None)

        with pytest.raises(ValueError, match="No audio data"):
            await conn._collect_committed_audio()

    @pytest.mark.asyncio
    async def test_non_final_commit_does_not_start_async_chunk_generation(self) -> None:
        conn, _, _ = self._connection()
        called = False

        async def _fake_start() -> None:
            nonlocal called
            called = True

        conn._start_async_chunk_bridge_generation = _fake_start

        await conn.handle_event({"type": "input_audio_buffer.commit", "final": False})

        assert called is False
        assert conn.audio_queue.empty()

    @pytest.mark.asyncio
    async def test_final_commit_starts_async_chunk_generation(self) -> None:
        conn, _, _ = self._connection()
        called = False

        async def _fake_start() -> None:
            nonlocal called
            called = True

        conn._start_async_chunk_bridge_generation = _fake_start

        await conn.handle_event({"type": "input_audio_buffer.commit", "final": True})

        assert called is True

    @pytest.mark.asyncio
    async def test_async_chunk_generation_passes_single_prompt_to_engine(self) -> None:
        conn, engine, websocket = self._connection()
        conn.audio_queue.put_nowait(np.array([0.25, -0.25], dtype=np.float32))
        conn.audio_queue.put_nowait(None)

        await conn._run_async_chunk_bridge_generation(asyncio.Queue())

        assert engine.generate_kwargs is not None
        prompt = engine.generate_kwargs["prompt"]
        assert not hasattr(prompt, "__aiter__")
        assert prompt["prompt"] == "<|im_start|>user\n<audio-test><|im_end|>\n<|im_start|>assistant\n"
        prompt_audio, sample_rate = prompt["multi_modal_data"]["audio"]
        assert sample_rate == 16000
        np.testing.assert_allclose(prompt_audio, [0.25, -0.25], rtol=1e-6)

        sampling_params_list = engine.generate_kwargs["sampling_params_list"]
        assert sampling_params_list[0].output_kind == RequestOutputKind.DELTA

        events = [json.loads(message) for message in websocket.messages]
        event_types = [event["type"] for event in events]
        assert event_types == [
            "transcription.delta",
            "response.audio.delta",
            "transcription.done",
            "response.audio.done",
        ]
        assert events[0]["delta"] == "hello"
        assert events[1]["sample_rate_hz"] == 24000
        assert events[-1]["has_audio"] is True

    def test_realtime_sampling_params_do_not_mutate_engine_defaults(self) -> None:
        conn, engine, _ = self._connection()
        default_params = engine.default_sampling_params_list[0]
        default_params.skip_clone = True
        default_params.output_kind = RequestOutputKind.CUMULATIVE

        sampling_params_list = conn._realtime_sampling_params_list()

        assert sampling_params_list[0] is not default_params
        assert sampling_params_list[0].output_kind == RequestOutputKind.DELTA
        assert default_params.output_kind == RequestOutputKind.CUMULATIVE


class TestOpenAIEntrypointExports:
    def test_serving_chat_export_does_not_eager_import_api_server(self) -> None:
        sys.modules.pop("vllm_omni.entrypoints.openai.api_server", None)

        openai_entrypoints = importlib.import_module("vllm_omni.entrypoints.openai")

        assert "vllm_omni.entrypoints.openai.api_server" not in sys.modules
        assert openai_entrypoints.OmniOpenAIServingChat.__name__ == "OmniOpenAIServingChat"
        assert "vllm_omni.entrypoints.openai.api_server" not in sys.modules


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            validate_streaming_input_sampling_params(p)
