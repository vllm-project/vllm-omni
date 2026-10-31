# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text-intermediate request isolation, failure and stream ownership contracts."""

import asyncio
import copy
import json
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import anyio
import numpy as np
import pytest
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_omni.entrypoints.openai import drvexo_transcription as module
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

AUDIO = {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Renderer:
    get_tokenizer: Callable


@dataclass
class _Engine:
    default_sampling_params_list: list
    output_modalities: list
    abort: AsyncMock
    calls: list
    generate: Callable | None = None


@dataclass
class _ServingModelConfig:
    hf_config: object


@dataclass
class _MediaConnector:
    fetch_audio_async: Callable


@dataclass
class _TextOutput:
    error: object
    outputs: list


@dataclass
class _HFConfig:
    model_type: str
    drvexo_text_intermediate: bool = False


@dataclass
class _Completion:
    text: str
    finish_reason: str
    token_ids: list[int]


class TranscriptionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fetch = AsyncMock(return_value=(np.ones(16), 16000))
        patcher = patch.object(module, "MediaConnector", return_value=_MediaConnector(fetch_audio_async=self.fetch))
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, **kwargs):
        messages = [
            {"role": "system", "content": "Product facts"},
            {"role": "user", "content": "之前调整三档"},
            {"role": "assistant", "content": "已记录"},
            {"role": "user", "content": [{"type": "text", "text": "按录音要求回答"}, copy.deepcopy(AUDIO)]},
        ]
        req = ChatCompletionRequest(model="test", messages=messages, **kwargs)
        handler = object.__new__(OmniOpenAIServingChat)
        handler.chat_template = None
        handler.chat_template_content_format = "auto"
        handler.renderer = _Renderer(get_tokenizer=lambda: object())
        handler._preprocess_chat = AsyncMock(
            return_value=([], [{"prompt_token_ids": [1, 2], "multi_modal_data": {"audio": [(np.ones(16), 16000)]}}])
        )
        engine = _Engine(
            default_sampling_params_list=[SamplingParams(max_tokens=2)],
            output_modalities=["text"],
            abort=AsyncMock(),
            calls=[],
        )

        async def generate(**kw):
            engine.calls.append(kw)
            yield _TextOutput(error=None, outputs=[_Completion(text="你好。", finish_reason="stop", token_ids=[3, 4])])

        engine.generate = generate
        handler.engine_client = engine
        return handler, engine, req

    async def test_text_skips_transcription(self):
        handler, engine, req = self.make()
        req.messages[-1]["content"] = "当前文本"
        converted, metadata = await module.current_audio_to_text(handler, req)
        self.assertIs(converted, req)
        self.assertIsNone(metadata)
        handler._preprocess_chat.assert_not_awaited()
        self.assertEqual(engine.calls, [])

    async def test_audio_sees_no_history_and_preserves_committed_text(self):
        handler, engine, req = self.make(decision_head_enabled=True, temperature=1.2, max_tokens=7)
        original = copy.deepcopy(req.model_dump())
        converted, metadata = await module.current_audio_to_text(handler, req)
        internal, messages = handler._preprocess_chat.call_args.args
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": module.TRANSCRIBE})
        self.assertEqual(messages[0]["content"][1]["type"], "input_audio")
        self.assertEqual(messages[0]["content"][1]["input_audio"]["format"], "wav")
        self.fetch.assert_awaited_once_with("data:audio/wav;base64,AA==")
        self.assertFalse(internal.decision_head_enabled)
        self.assertEqual(internal.max_tokens, 192)
        self.assertEqual(converted.messages[:-1], req.messages[:-1])
        self.assertEqual(converted.messages[-1]["content"], "按录音要求回答\n你好。")
        self.assertTrue(converted.decision_head_enabled)
        self.assertEqual(req.model_dump(), original)
        self.assertEqual(metadata["text"], "你好。")
        self.assertEqual(engine.calls[0]["output_modalities"], ["text"])
        self.assertEqual(engine.calls[0]["sampling_params_list"][0].temperature, 0)
        self.assertEqual(engine.default_sampling_params_list[0].max_tokens, 2)
        engine.abort.assert_not_awaited()

    async def test_silence_stops_before_generation(self):
        handler, engine, req = self.make()
        self.fetch.return_value = (np.zeros(50), 16000)
        with self.assertRaisesRegex(ValueError, "silence"):
            await module.current_audio_to_text(handler, req)
        self.assertEqual(engine.calls, [])
        handler._preprocess_chat.assert_not_awaited()

    async def test_empty_incomplete_engine_error_rejected(self):
        for text, reason, error, exception in [
            ("", "stop", None, ValueError),
            ("半句", "length", None, RuntimeError),
            ("内容", "stop", "failed", RuntimeError),
        ]:
            handler, engine, req = self.make()

            async def generate(**kw):
                yield _TextOutput(error=error, outputs=[_Completion(text=text, finish_reason=reason, token_ids=[])])

            engine.generate = generate
            with self.assertRaises(exception):
                await module.current_audio_to_text(handler, req)

    async def test_cancel_aborts_only_owned_recognition(self):
        handler, engine, req = self.make()
        started = asyncio.Event()
        closed = asyncio.Event()

        async def generate(**kw):
            engine.calls.append(kw)
            started.set()
            try:
                await asyncio.Event().wait()
                yield None
            finally:
                closed.set()

        engine.generate = generate
        task = asyncio.create_task(module.current_audio_to_text(handler, req))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        engine.abort.assert_awaited_once_with(engine.calls[0]["request_id"])
        self.assertTrue(engine.calls[0]["request_id"].startswith("drvexo-transcription-"))

    async def test_chat_rejects_legacy_interception_before_generation(self):
        handler, engine, req = self.make(decision_head_enabled=True, modalities=["text"])
        handler.model_config = _ServingModelConfig(hf_config=_HFConfig(model_type="drvexo_speech_8b"))
        response = await handler._create_chat_completion(req)
        self.assertEqual(response.error.code, 400)
        self.assertEqual(len(engine.calls), 0)

    async def test_sse_metadata_and_close_propagation(self):
        closed = asyncio.Event()

        async def source():
            try:
                yield 'data: {"choices": []}\n\n'
                await asyncio.Event().wait()
            finally:
                closed.set()

        wrapper = module.attach_transcription(source(), {"text": "你好"})
        self.assertEqual(json.loads((await anext(wrapper))[6:])["input_transcription"]["text"], "你好")
        await wrapper.aclose()
        self.assertTrue(closed.is_set())

    def test_invalid_audio_layouts(self):
        cases = [
            [],
            ["not-an-object"],
            [{"role": "user", "content": [AUDIO, AUDIO]}],
            [{"role": "assistant", "content": [AUDIO]}],
            [{"role": "user", "content": [AUDIO]}, {"role": "user", "content": "current"}],
            [{"role": "user", "content": [AUDIO, {"type": "text"}]}],
            [{"role": "user", "content": [AUDIO, {"type": "image_url"}]}],
        ]
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                module.split_current(messages)

    def test_exact_zero_is_not_general_vad(self):
        for value in [np.zeros(10), (np.zeros(10), 16000), [(np.zeros(10), 16000)], [0.0, 0.0]]:
            self.assertTrue(module.exact_silence(value))
        self.assertFalse(module.exact_silence(np.full(10, 1e-12)))
        self.assertFalse(module.exact_silence(None))
        for value in [np.array([]), [], np.array([float("nan")])]:
            with self.assertRaises(ValueError):
                module.exact_silence(value)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.asyncio
async def test_transcription_disconnect_shields_inner_chat_abort():
    from vllm_omni.entrypoints.openai.stream_lifecycle import owned_chat_stream

    closed = []

    async def response():
        try:
            yield 'data: {"choices": []}\n\n'
        finally:
            await anyio.sleep(0)
            closed.append("response")

    async def results():
        yield None

    engine = AsyncMock()
    inner = owned_chat_stream(response(), results(), engine, "only-this-chat")

    async def adapter():
        try:
            async for chunk in inner:
                yield chunk
        finally:
            # Recursive response adapters may have cleanup checkpoints before
            # reaching the owned generator's own shield.
            await anyio.sleep(0)
            await inner.aclose()

    outer = module.attach_transcription(adapter(), {"text": "hello"})
    await anext(outer)
    with anyio.CancelScope() as scope:
        scope.cancel()
        await outer.aclose()
    assert closed == ["response"]
    engine.abort.assert_awaited_once_with("only-this-chat")


@pytest.mark.asyncio
async def test_internal_transcription_typeerror_is_not_a_client_error(mocker):
    handler = object.__new__(OmniOpenAIServingChat)
    handler.model_config = _ServingModelConfig(hf_config=_HFConfig("drvexo_speech_8b", True))
    handler._check_model = AsyncMock(return_value=None)
    mocker.patch.object(module, "current_audio_to_text", side_effect=TypeError("internal defect"))
    req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(TypeError, match="internal defect"):
        await handler._create_chat_completion(req)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError, TypeError])
async def test_recursive_serving_failure_keeps_server_error_identity(mocker, error):
    handler = object.__new__(OmniOpenAIServingChat)
    handler.model_config = _ServingModelConfig(hf_config=_HFConfig("drvexo_speech_8b", True))
    handler._check_model = AsyncMock(return_value=None)
    req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
    mocker.patch.object(module, "current_audio_to_text", return_value=(req, {"text": "hi"}))
    handler._create_chat_completion = AsyncMock(side_effect=error("internal recursive error"))
    with pytest.raises(error, match="internal recursive error"):
        await OmniOpenAIServingChat._create_chat_completion(handler, req)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError, TypeError])
async def test_chat_validation_errors_remain_client_errors(mocker, error):
    from vllm_omni.entrypoints.openai import drvexo_decision

    handler = object.__new__(OmniOpenAIServingChat)
    handler.model_config = _ServingModelConfig(hf_config=_HFConfig("drvexo_speech_8b", True))
    mocker.patch.object(drvexo_decision, "validate_chat_request", side_effect=error("bad request"))
    response = mocker.patch.object(handler, "_create_error_response", return_value="client-error")
    req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
    assert await handler._create_chat_completion(req) == "client-error"
    response.assert_called_once_with("bad request", status_code=400)
