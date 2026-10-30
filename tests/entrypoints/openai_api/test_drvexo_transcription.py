# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text-intermediate request isolation, failure and stream ownership contracts."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_omni.entrypoints.openai import drvexo_transcription as module
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

AUDIO = {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TranscriptionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fetch = AsyncMock(return_value=(np.ones(16), 16000))
        patcher = patch.object(module, "MediaConnector", return_value=SimpleNamespace(fetch_audio_async=self.fetch))
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
        handler.renderer = SimpleNamespace(get_tokenizer=lambda: object())
        handler._preprocess_chat = AsyncMock(
            return_value=([], [{"prompt_token_ids": [1, 2], "multi_modal_data": {"audio": [(np.ones(16), 16000)]}}])
        )
        engine = SimpleNamespace(
            default_sampling_params_list=[SamplingParams(max_tokens=2)],
            output_modalities=["text"],
            abort=AsyncMock(),
            calls=[],
        )

        async def generate(**kw):
            engine.calls.append(kw)
            yield SimpleNamespace(
                error=None, outputs=[SimpleNamespace(text="你好。", finish_reason="stop", token_ids=[3, 4])]
            )

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
                yield SimpleNamespace(
                    error=error, outputs=[SimpleNamespace(text=text, finish_reason=reason, token_ids=[])]
                )

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
        handler.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="drvexo_speech_8b"))
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
