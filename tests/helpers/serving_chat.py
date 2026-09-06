# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for OmniOpenAIServingChat unit tests.

Provides factories for a minimal serving-chat instance, chat requests, and
OmniRequestOutput objects, plus SSE stream collection/parsing utilities.
"""

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput


def make_text_omni_output(
    request_id: str = "test-req",
    text: str = "hello",
    token_ids: list[int] | None = None,
    finish_reason: str | None = None,
    index: int = 0,
    num_prompt_tokens: int = 3,
    num_cached_tokens: int | None = None,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput wrapping a text RequestOutput."""
    if token_ids is None:
        token_ids = [10, 11, 12]
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=finish_reason is not None,
        num_cached_tokens=num_cached_tokens,
    )
    return OmniRequestOutput.from_stage_output(
        res,
        request_id=request_id,
        final_output_type="text",
        finished=finish_reason is not None,
    )


def mock_audio_choices(index: int = 0, role: str = "assistant"):
    return [
        ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(role=role, content="dGVzdA=="),
            logprobs=None,
            finish_reason="stop",
        )
    ]


def build_serving_chat(
    *,
    engine_client=None,
    models=None,
    online_renderer=None,
    response_role: str = "assistant",
    request_logger=None,
    chat_template=None,
    chat_template_content_format="auto",
    **init_overrides,
) -> OmniOpenAIServingChat:
    """Create a real OmniOpenAIServingChat via its actual __init__.

    engine_client/models/online_renderer default to minimal mocks; pass your
    own to wire in test-specific behavior (e.g. engine_client.stage_configs).
    Any other OmniOpenAIServingChat.__init__ keyword — enable_auto_tools,
    enable_prompt_tokens_details, trust_request_chat_template, etc. — can be
    passed through **init_overrides instead of set on the instance after
    construction.
    """
    if engine_client is None:
        engine_client = MagicMock()
        engine_client.errored = False
    if models is None:
        models = OpenAIServingModels(
            engine_client=engine_client,
            base_model_paths=[],
        )
    if online_renderer is None:
        online_renderer = MagicMock()

    instance = OmniOpenAIServingChat(
        engine_client=engine_client,
        models=models,
        response_role=response_role,
        online_renderer=online_renderer,
        request_logger=request_logger,
        chat_template=chat_template,
        chat_template_content_format=chat_template_content_format,
        **init_overrides,
    )
    instance._create_audio_choice = MagicMock(
        side_effect=lambda omni_res, role, request, stream=False: mock_audio_choices(
            index=omni_res.outputs[0].index,
            role=role,
        )
    )
    return instance


def make_request(
    modalities: list[str],
    n: int = 1,
    *,
    stream: bool = True,
    include_usage: bool = False,
) -> ChatCompletionRequest:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        n=n,
        stream=stream,
        stream_options={"include_usage": True} if include_usage else None,
    )
    req.modalities = modalities  # type: ignore[attr-defined]
    return req


def parse_sse_chunks(lines: list[str]) -> list[dict]:
    """Parse SSE lines into JSON dicts."""
    prefix = "data: "
    chunks = []
    for line in lines:
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


async def collect_stream(gen):
    result = []
    async for item in gen:
        result.append(item)
    return result
