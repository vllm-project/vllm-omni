# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure-diffusion chat completions must honor ``stream=true`` (issue #4049)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeDiffusionEngine:
    stage_configs: list = []

    def __init__(self, result):
        self.result = result
        self.prompts = []

    async def generate(self, prompt, sampling_params_list, request_id, **kwargs):
        self.prompts.append(prompt)
        yield self.result


def _text_result(text: str):
    return SimpleNamespace(
        final_output_type="text",
        multimodal_output={"text": text, "metadata": {"text": {"think_text": "hmm"}}},
    )


def _image_result():
    return SimpleNamespace(
        final_output_type="image",
        images=[Image.new("RGB", (4, 4))],
        multimodal_output={},
        stage_durations={},
        peak_memory_mb=0.0,
    )


def _serving_chat(result, **kwargs) -> tuple[OmniOpenAIServingChat, _FakeDiffusionEngine]:
    engine = _FakeDiffusionEngine(result)
    serving_chat = OmniOpenAIServingChat.for_diffusion(diffusion_engine=engine, model_name="sensenova", **kwargs)
    serving_chat._diffusion_extra_body_params = frozenset()
    serving_chat._diffusion_extra_output_params = frozenset({"think_text"})
    return serving_chat, engine


def _request(modalities: list[str], **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="sensenova",
        messages=[{"role": "user", "content": "Describe this image"}],
        modalities=modalities,
        **kwargs,
    )


async def _collect(generator) -> list:
    events = [event async for event in generator]
    assert events[-1] == "data: [DONE]\n\n"
    assert events.count("data: [DONE]\n\n") == 1
    return [json.loads(event.removeprefix("data: ")) for event in events[:-1]]


def test_text_output_streams_sse_chunks():
    serving_chat, engine = _serving_chat(_text_result("A cat on a mat."))
    request = _request(["text"], stream=True, stream_options={"include_usage": True})

    generator = asyncio.run(serving_chat._create_chat_completion(request))
    chunks = asyncio.run(_collect(generator))

    assert engine.prompts[0]["modalities"] == ["text"]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1
    content_chunk, finish_chunk, usage_chunk = chunks
    assert content_chunk["choices"][0]["delta"] == {"role": "assistant", "content": "A cat on a mat."}
    assert content_chunk["choices"][0]["finish_reason"] is None
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"
    assert finish_chunk["metrics"] == {"think_text": "hmm"}
    assert "usage" not in content_chunk and "usage" not in finish_chunk
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] == 5


def test_continuous_usage_is_attached_to_every_chunk():
    serving_chat, _ = _serving_chat(_text_result("A cat on a mat."))
    request = _request(["text"], stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})

    generator = asyncio.run(serving_chat._create_chat_completion(request))
    chunks = asyncio.run(_collect(generator))

    content_chunk, finish_chunk, usage_chunk = chunks
    expected_usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    for chunk in (content_chunk, finish_chunk):
        assert {k: chunk["usage"][k] for k in expected_usage} == expected_usage
    assert [chunk for chunk in chunks if chunk["choices"] == []] == [usage_chunk]
    assert {k: usage_chunk["usage"][k] for k in expected_usage} == expected_usage


def test_server_forced_usage_without_stream_options():
    serving_chat, _ = _serving_chat(_text_result("A cat on a mat."), enable_force_include_usage=True)

    generator = asyncio.run(serving_chat._create_chat_completion(_request(["text"], stream=True)))
    chunks = asyncio.run(_collect(generator))

    assert len(chunks) == 3
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 5


def test_usage_chunk_omitted_without_include_usage():
    serving_chat, _ = _serving_chat(_text_result("ok"))

    generator = asyncio.run(serving_chat._create_chat_completion(_request(["text"], stream=True)))
    chunks = asyncio.run(_collect(generator))

    assert len(chunks) == 2
    assert all("usage" not in chunk for chunk in chunks)


def test_image_output_streams_image_content():
    serving_chat, _ = _serving_chat(_image_result())

    generator = asyncio.run(serving_chat._create_chat_completion(_request(["image"], stream=True)))
    content_chunk, finish_chunk = asyncio.run(_collect(generator))

    assert content_chunk["modality"] == "image"
    image_content = content_chunk["choices"][0]["delta"]["content"]
    assert image_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"


def test_non_stream_request_still_returns_full_response():
    serving_chat, _ = _serving_chat(_text_result("A cat on a mat."))

    response = asyncio.run(serving_chat._create_chat_completion(_request(["text"])))

    assert response.choices[0].message.content == "A cat on a mat."
