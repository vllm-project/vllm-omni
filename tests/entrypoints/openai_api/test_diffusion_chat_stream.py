# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from argparse import Namespace
from collections.abc import AsyncGenerator, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DiffusionEngine(AsyncOmni):
    def __init__(self, output: SimpleNamespace) -> None:
        self.engine = SimpleNamespace(
            stage_configs=[SimpleNamespace(stage_type="diffusion")],
            default_sampling_params_list=[OmniDiffusionSamplingParams()],
        )
        self.default_sampling_params_list = self.engine.default_sampling_params_list
        self.output = output

    async def generate(self, **kwargs: Any) -> AsyncGenerator[SimpleNamespace, None]:
        yield self.output


@pytest.fixture(params=["image", "text", "audio"])
def diffusion_client(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    modality = request.param
    output = SimpleNamespace(
        final_output_type=modality,
        images=[Image.new("RGB", (16, 16), "blue")] if modality == "image" else [],
        stage_durations={},
        peak_memory_mb=0.0,
        multimodal_output={
            "text": "a blue square",
            "audio": torch.zeros(1, 480),
            "sample_rate": 24000,
        },
        request_output=None,
    )
    engine = _DiffusionEngine(output=output)
    app = FastAPI()
    app.include_router(router)
    init_exception_handler(app)
    app.state.engine_client = engine
    app.state.args = Namespace(log_error_stack=False)
    app.state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
        diffusion_engine=engine, model_name="m"
    )
    app.state.test_modality = modality
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _request_body(client: TestClient, *, stream: bool) -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "a blue square"}],
        "modalities": [client.app.state.test_modality],
        "audio": {"format": "wav"},
        "stream": stream,
    }


def test_diffusion_chat_nonstream_remains_json(diffusion_client: TestClient) -> None:
    response = diffusion_client.post(
        url="/v1/chat/completions", json=_request_body(client=diffusion_client, stream=False)
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("include_usage", [False, True])
def test_diffusion_chat_stream_emits_sse(
    diffusion_client: TestClient, include_usage: bool
) -> None:
    body = _request_body(client=diffusion_client, stream=True)
    body["stream_options"] = {"include_usage": include_usage}
    response = diffusion_client.post(url="/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert len(chunks) == 1 + int(include_usage)
    chunk = chunks[0]
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["modality"] == diffusion_client.app.state.test_modality
    choice = chunk["choices"][0]
    assert choice["delta"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    content = choice["delta"]["content"]
    modality = diffusion_client.app.state.test_modality
    if modality == "image":
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    elif modality == "text":
        assert content == "a blue square"
    else:
        assert content.startswith("UklGR")
        assert choice["audio_metadata"]["sample_rate_hz"] == 24000
    if include_usage:
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] > 0
        assert chunks[-1]["id"] == chunk["id"]
    else:
        assert chunk.get("usage") is None


def test_openai_sdk_receives_diffusion_chunks(diffusion_client: TestClient) -> None:
    def forward(request: httpx.Request) -> httpx.Response:
        response = diffusion_client.post(
            url=request.url.path, content=request.content, headers={"content-type": "application/json"}
        )
        return httpx.Response(
            status_code=response.status_code, headers=response.headers, content=response.content
        )

    with httpx.Client(transport=httpx.MockTransport(handler=forward)) as http_client:
        with openai.OpenAI(api_key="x", base_url="http://omni.local/v1", http_client=http_client) as client:
            body = _request_body(client=diffusion_client, stream=True)
            with client.chat.completions.create(**body) as stream:
                chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].object == "chat.completion.chunk"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content
    assert chunks[0].choices[0].finish_reason == "stop"
