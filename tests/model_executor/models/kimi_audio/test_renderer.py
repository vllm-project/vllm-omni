# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio renderer contracts with the installed vLLM renderer API."""

import asyncio

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers import BaseRenderer

from vllm_omni.model_executor.models.kimi_audio.renderer import KimiAudioRenderer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]

MESSAGES = [{"role": "user", "content": "Hi"}]


def _chat_params(**request):
    """Build chat params as the server does, including its template defaults."""
    params = ChatCompletionRequest(messages=MESSAGES, **request).build_chat_params(None, "auto")
    return params.with_defaults({"cohere_format": "cmd4"})


@pytest.fixture
def renderer():
    # Validation and engine forwarding need no tokenizer or model config.
    return KimiAudioRenderer.__new__(KimiAudioRenderer)


@pytest.mark.parametrize("output_type", ["text", "both"])
def test_server_template_defaults_are_accepted(renderer, output_type):
    params = _chat_params(chat_template_kwargs={"output_type": output_type})
    assert "cohere_format" in params.chat_template_kwargs
    assert renderer._validate_messages(MESSAGES, params) == output_type


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"chat_template_kwargs": {"output_type": "audio"}},
        {"reasoning_effort": "low"},
        {"continue_final_message": True, "add_generation_prompt": False},
        {"chat_template": "{{ messages }}"},
    ],
)
def test_conversation_changing_options_are_rejected(renderer, request_kwargs):
    with pytest.raises(ValueError, match="Kimi-Audio"):
        renderer._validate_messages(MESSAGES, _chat_params(**request_kwargs))


def test_engine_processing_forwards_base_options(renderer, monkeypatch):
    calls = []
    tok_params = object()

    def process(self, prompt, arrival_time, **kwargs):
        calls.append(kwargs)
        return {"prompt_token_ids": prompt["prompt_token_ids"]}

    async def process_async(self, prompt, arrival_time, **kwargs):
        return process(self, prompt, arrival_time, **kwargs)

    monkeypatch.setattr(BaseRenderer, "process_for_engine", process)
    monkeypatch.setattr(BaseRenderer, "process_for_engine_async", process_async)
    prompt = {"prompt_token_ids": [1, 2], "model_intermediate_buffer": {"kimi_audio_input": {}}}

    results = [
        renderer.process_for_engine(prompt, 0.0, skip_mm_cache=True, tok_params=tok_params),
        asyncio.run(renderer.process_for_engine_async(prompt, 0.0, skip_mm_cache=True, tok_params=tok_params)),
    ]

    assert calls == [{"skip_mm_cache": True, "tok_params": tok_params}] * 2
    for result in results:
        assert result["model_intermediate_buffer"] is prompt["model_intermediate_buffer"]
