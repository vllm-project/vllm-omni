# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HTTP error mapping tests for /v1/chat/completions and /v1/chat/completions/batch."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from vllm import SamplingParams
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
    VLLMValidationError,
)

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_client(mocker: MockerFixture):
    stage = mocker.MagicMock()
    stage.stage_id = 0
    stage.stage_type = "llm"
    stage.is_comprehension = True
    stage.model_stage = "comprehension"
    stage.model_config = mocker.MagicMock(default_sampling_params={})

    engine_client = mocker.MagicMock()
    engine_client.errored = False
    engine_client.stage_configs = [stage]
    engine_client.output_modalities = ["text"]
    engine_client.default_sampling_params_list = [SamplingParams(max_tokens=16)]
    engine_client.resolve_entry_stage_id.return_value = 0
    engine_client.lora_requests = []
    engine_client.model_config = mocker.MagicMock(
        multimodal_config=None,
        max_model_len=4096,
        skip_tokenizer_init=False,
    )
    return engine_client


def _make_renderer(mocker: MockerFixture, *, fail_on_empty: bool = False):
    renderer = mocker.MagicMock()
    renderer.get_tokenizer.return_value = mocker.MagicMock()

    async def fake_render_chat_async(conversations, chat_params, *_args, **_kwargs):
        if fail_on_empty and not conversations[0]:
            raise VLLMValidationError("Cannot apply chat template to an empty conversation.")
        if getattr(chat_params, "chat_template_kwargs", None) and chat_params.chat_template_kwargs.get("bad_kwarg"):
            raise VLLMValidationError("Unsupported chat_template_kwargs", parameter="chat_template_kwargs")
        return conversations, [{"prompt": "hi"} for _ in conversations]

    renderer.render_chat_async = mocker.AsyncMock(side_effect=fake_render_chat_async)
    return renderer


def _build_serving_instance(
    cls: type[OmniOpenAIServingChat] | type[OmniOpenAIServingChatBatch],
    mocker: MockerFixture,
    *,
    fail_on_empty: bool = False,
):
    engine_client = _make_engine_client(mocker)
    renderer = _make_renderer(mocker, fail_on_empty=fail_on_empty)
    models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=[BaseModelPath(name="demo-model", model_path="demo-model")],
    )
    online_renderer = mocker.MagicMock()
    online_renderer.validate_chat_template.return_value = None

    instance = cls(
        engine_client=engine_client,
        models=models,
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    instance.renderer = renderer
    mocker.patch.object(instance, "_check_model", new=mocker.AsyncMock(return_value=None))
    return instance


def _make_chat_app(chat_handler=None, batch_handler=None) -> FastAPI:
    app = FastAPI()
    app.state.openai_serving_chat = chat_handler
    app.state.openai_serving_chat_batch = batch_handler
    app.state.serving_tokenization = None
    app.add_api_route("/v1/chat/completions", api_server.create_chat_completion, methods=["POST"])
    app.add_api_route("/v1/chat/completions/batch", api_server.create_batch_chat_completion, methods=["POST"])
    return app


@pytest.mark.parametrize(
    ("overrides", "expected_param"),
    [
        ({"max_tokens": -1}, "max_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"messages": []}, None),
    ],
    ids=["negative-max-tokens", "negative-temperature", "empty-messages"],
)
def test_chat_completion_validation_errors_return_400(mocker: MockerFixture, overrides: dict, expected_param) -> None:
    """Invalid chat requests raising VLLMValidationError must return HTTP 400 BadRequestError."""
    serving_chat = _build_serving_instance(OmniOpenAIServingChat, mocker, fail_on_empty=True)
    app = _make_chat_app(chat_handler=serving_chat)

    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        **overrides,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert error["code"] == 400
    assert error["param"] == expected_param
    assert error["message"]


@pytest.mark.parametrize(
    ("overrides", "expected_param"),
    [
        ({"max_tokens": -1}, "max_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"chat_template_kwargs": {"bad_kwarg": True}}, "chat_template_kwargs"),
    ],
    ids=["negative-max-tokens", "negative-temperature", "bad-chat-template-kwargs"],
)
def test_batch_chat_completion_validation_errors_return_400(
    mocker: MockerFixture, overrides: dict, expected_param
) -> None:
    """Invalid batch chat requests raising VLLMValidationError must return HTTP 400 BadRequestError."""
    serving_batch = _build_serving_instance(OmniOpenAIServingChatBatch, mocker, fail_on_empty=True)
    app = _make_chat_app(batch_handler=serving_batch)

    payload = {
        "model": "demo-model",
        "messages": [[{"role": "user", "content": "hello"}]],
        **overrides,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions/batch", json=payload)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert error["code"] == 400
    assert error["param"] == expected_param
    assert error["message"]


def test_chat_completion_streaming_validation_error_returns_400(mocker: MockerFixture) -> None:
    """Streaming chat request failing pre-stream validation returns HTTP 400 JSON."""
    serving_chat = _build_serving_instance(OmniOpenAIServingChat, mocker)
    app = _make_chat_app(chat_handler=serving_chat)

    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "max_tokens": -1,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert error["code"] == 400
    assert error["param"] == "max_tokens"


def test_batch_chat_to_chat_completion_request_vllm_validation_error_returns_400(mocker: MockerFixture) -> None:
    """VLLMValidationError during batch subrequest conversion returns HTTP 400 with param."""
    serving_batch = _build_serving_instance(OmniOpenAIServingChatBatch, mocker)
    app = _make_chat_app(batch_handler=serving_batch)
    mocker.patch(
        "vllm.entrypoints.openai.chat_completion.protocol.BatchChatCompletionRequest.to_chat_completion_request",
        side_effect=VLLMValidationError("Invalid per-item option", parameter="tools"),
    )

    payload = {
        "model": "demo-model",
        "messages": [[{"role": "user", "content": "hello"}]],
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions/batch", json=payload)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert error["code"] == 400
    assert error["param"] == "tools"


@pytest.mark.parametrize(
    ("route", "payload_fn"),
    [
        (
            "/v1/chat/completions",
            lambda: {"model": "demo-model", "messages": [{"role": "user", "content": "hello"}], "stream": False},
        ),
        (
            "/v1/chat/completions/batch",
            lambda: {"model": "demo-model", "messages": [[{"role": "user", "content": "hello"}]]},
        ),
    ],
    ids=["chat", "batch-chat"],
)
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_type", "expected_param"),
    [
        (VLLMValidationError("bad param", parameter="top_p", value=-1), 400, "BadRequestError", "top_p"),
        (VLLMNotFoundError("adapter missing"), 404, "NotFoundError", None),
        (
            VLLMUnprocessableEntityError("bad image url", parameter="image_url"),
            422,
            "UnprocessableEntityError",
            "image_url",
        ),
        (VLLMClientError("client error"), 400, "BadRequestError", None),
        (
            OmniClientError("blocked by guardrail", status_code=422, error_type="GuardrailViolationError"),
            422,
            "GuardrailViolationError",
            None,
        ),
    ],
    ids=["vllm-validation", "vllm-not-found", "vllm-unprocessable", "vllm-client", "omni-client"],
)
def test_chat_routes_map_escaped_client_errors_to_4xx(
    mocker: MockerFixture,
    route: str,
    payload_fn,
    exc: Exception,
    expected_status: int,
    expected_type: str,
    expected_param: str | None,
) -> None:
    """VLLMClientError and OmniClientError escaping the serving handler map to 4xx OpenAI errors."""
    serving_chat = _build_serving_instance(OmniOpenAIServingChat, mocker)
    serving_batch = _build_serving_instance(OmniOpenAIServingChatBatch, mocker)
    mocker.patch.object(serving_chat, "create_chat_completion", new=mocker.AsyncMock(side_effect=exc))
    mocker.patch.object(serving_batch, "create_batch_chat_completion", new=mocker.AsyncMock(side_effect=exc))
    app = _make_chat_app(chat_handler=serving_chat, batch_handler=serving_batch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(route, json=payload_fn())

    assert response.status_code == expected_status
    error = response.json()["error"]
    assert error["code"] == expected_status
    assert error["type"] == expected_type
    assert error["param"] == expected_param


@pytest.mark.parametrize(
    ("route", "payload_fn"),
    [
        (
            "/v1/chat/completions",
            lambda: {"model": "demo-model", "messages": [{"role": "user", "content": "hello"}], "stream": False},
        ),
        (
            "/v1/chat/completions/batch",
            lambda: {"model": "demo-model", "messages": [[{"role": "user", "content": "hello"}]]},
        ),
    ],
    ids=["chat", "batch-chat"],
)
@pytest.mark.parametrize(
    "exc",
    [ValueError("unmigrated internal value error"), RuntimeError("unexpected failure")],
    ids=["generic-value-error", "generic-runtime-error"],
)
def test_chat_routes_keep_unmigrated_server_errors_at_500(
    mocker: MockerFixture,
    route: str,
    payload_fn,
    exc: Exception,
) -> None:
    """Generic ValueError and RuntimeError escaping the serving handler stay HTTP 500."""
    serving_chat = _build_serving_instance(OmniOpenAIServingChat, mocker)
    serving_batch = _build_serving_instance(OmniOpenAIServingChatBatch, mocker)
    mocker.patch.object(serving_chat, "create_chat_completion", new=mocker.AsyncMock(side_effect=exc))
    mocker.patch.object(serving_batch, "create_batch_chat_completion", new=mocker.AsyncMock(side_effect=exc))
    app = _make_chat_app(chat_handler=serving_chat, batch_handler=serving_batch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(route, json=payload_fn())

    assert response.status_code == 500
