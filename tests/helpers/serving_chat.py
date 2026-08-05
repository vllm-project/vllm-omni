# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for OmniOpenAIServingChat unit tests.

Provides factories for a minimal serving-chat instance, chat requests, and
OmniRequestOutput objects, plus SSE stream collection/parsing utilities.
"""

import enum
import json
from unittest.mock import MagicMock

# Python 3.10 compat: StrEnum was added in 3.11
if not hasattr(enum, "StrEnum"):

    class _StrEnum(str, enum.Enum):
        """Minimal StrEnum backport for Python 3.10."""

    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

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
    return OmniRequestOutput(
        request_id=request_id,
        final_output_type="text",
        request_output=res,
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


def build_serving_chat():
    """Create a minimal OmniOpenAIServingChat for testing."""
    mock_engine = MagicMock()
    mock_engine.errored = False

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=[],
    )
    mock_render = MagicMock()

    instance = OmniOpenAIServingChat(
        engine_client=mock_engine,
        models=models,
        response_role="assistant",
        online_renderer=mock_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    instance._create_audio_choice = MagicMock(
        side_effect=lambda omni_res, role, request, stream=False: mock_audio_choices(
            index=omni_res.request_output.outputs[0].index,
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
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return chunks


async def collect_stream(gen):
    result = []
    async for item in gen:
        result.append(item)
    return result
