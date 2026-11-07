# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from dataclasses import dataclass
from dataclasses import field as dc_field
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_audio import ChatCompletionAudio
from starlette.requests import Request
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    UsageInfo,
)
from vllm.outputs import CompletionOutput
from vllm.renderers.chat_utils import ChatCompletionMessageParam

from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MSG_BATCH: list[list[ChatCompletionMessageParam]] = [
    [ChatCompletionUserMessageParam(role="user", content="What color is the sky?")],
    [ChatCompletionUserMessageParam(role="user", content="What is 2+2?")],
]
collapse = OmniOpenAIServingChatBatch._maybe_collapse_choices


# Helpers for testing creating a __new__ serving instance and ensuring IDs are correct
def _make_raw_request(headers: list[tuple[bytes, bytes]]):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions/batch",
        "headers": headers,
    }
    return Request(scope)


def _make_completion(text):
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
            )
        ],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def _make_handler(responses):
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler.create_chat_completion = AsyncMock(side_effect=responses)
    return handler


# Helpers for creating choices of different modality types
def _text_choice(text="hello"):
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text),
    )


def _audio_choice():
    audio = ChatCompletionAudio(id="a1", data="base64audio", expires_at=0, transcript="")
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=None, audio=audio),
    )


def _image_choice():
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    msg = ChatMessage.model_construct(role="assistant")
    object.__setattr__(msg, "content", content)
    return ChatCompletionResponseChoice.model_construct(
        index=0,
        message=msg,
    )


def _get_subrequest_ids(handler):
    return [call.args[0].request_id for call in handler.create_chat_completion.call_args_list]


@pytest.mark.parametrize("has_header", [True, False])
def test_subrequest_ids_are_unique(has_header: bool):
    """Ensure that a submitted batch request creates unique subrequest IDs"""
    header = None if not has_header else b"user-123"
    headers = [] if not header else [(b"x-request-id", header)]

    handler = _make_handler([_make_completion("hello"), _make_completion("world")])
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=MSG_BATCH,
        seed=42,
    )
    result = asyncio.run(handler._create_batch_chat_completion_diffusion(request, _make_raw_request(headers)))
    assert not isinstance(result, ErrorResponse)
    assert result.id.startswith("chatcmpl-batch")

    # Ensure request IDs are unique and that if we had a header, it's in all reqs
    req_ids = _get_subrequest_ids(handler)
    assert len(set(req_ids)) == len(req_ids) == 2

    if header is not None:
        assert all(header.decode("utf-8") in req_id for req_id in req_ids)


### Tests for ensuring that the (currently batch specific) choice collapsing is correct
def test_single_text_passthrough():
    result = collapse([_text_choice()])
    assert result.message.content == "hello"


def test_single_image_passthrough():
    result = collapse([_image_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"


def test_text_plus_audio():
    result = collapse([_text_choice(), _audio_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_audio_plus_text_order_independent():
    result = collapse([_audio_choice(), _text_choice()])
    assert result.message.content == "hello"
    assert result.message.audio.data == "base64audio"


def test_image_plus_audio():
    result = collapse([_image_choice(), _audio_choice()])
    assert isinstance(result.message.content, list)
    assert result.message.content[0]["type"] == "image_url"
    assert result.message.audio.data == "base64audio"


def test_two_content_choices_raises():
    with pytest.raises(ValueError, match="Multiple content choices cannot be set"):
        collapse([_text_choice(), _text_choice()])


def test_three_choices_raises():
    with pytest.raises(ValueError, match="got 3"):
        collapse([_text_choice(), _audio_choice(), _text_choice()])


# ---------------------------------------------------------------------------
# Tests for render_batch_chat_request (PR B: validate-once)
# ---------------------------------------------------------------------------


def _make_render_handler():
    """Build an OmniOpenAIServingChatBatch with mocked internals for render tests."""
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler._check_model = AsyncMock(return_value=None)
    handler.engine_client = MagicMock()
    handler.engine_client.errored = False
    handler.engine_client.output_modalities = ["text"]
    handler.engine_client.resolve_entry_stage_id = MagicMock(return_value=0)
    handler._maybe_get_adapters = MagicMock(return_value=None)
    handler.models = MagicMock()
    handler.models.model_name.return_value = "test-model"
    handler.renderer = MagicMock()
    handler.renderer.get_tokenizer.return_value = MagicMock()
    handler.parser_cls = None
    handler.use_harmony = False
    handler.online_renderer = MagicMock()
    handler.online_renderer.validate_chat_template.return_value = None
    handler.enable_auto_tools = False
    handler.exclude_tools_when_tool_choice_none = False
    handler.chat_template = None
    handler.chat_template_content_format = "string"
    handler.trust_request_chat_template = False
    handler._preprocess_chat = AsyncMock(
        return_value=(
            [{"role": "user", "content": "hi"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )
    handler._effective_chat_template_kwargs = MagicMock(return_value={})
    handler.create_error_response = MagicMock(
        side_effect=lambda msg, **kw: ErrorResponse(
            error=ErrorInfo(message=msg, type="BadRequestError", code=400),
        )
    )
    handler._resolve_audio_format = MagicMock(return_value="wav")
    return handler


def _make_batch_request(n: int = 2) -> BatchChatCompletionRequest:
    messages = [[ChatCompletionUserMessageParam(role="user", content=f"Question {i}")] for i in range(n)]
    return BatchChatCompletionRequest(model="test-model", messages=messages)


def test_render_happy_path():
    handler = _make_render_handler()
    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(4)))
    assert not isinstance(result, ErrorResponse)
    conversations, prompts, single_requests, entry_stage_ids = result
    assert len(conversations) == 4
    assert len(prompts) == 4
    assert len(single_requests) == 4
    assert entry_stage_ids == [0, 0, 0, 0]
    handler._check_model.assert_called_once()
    handler.online_renderer.validate_chat_template.assert_called_once()
    assert handler._preprocess_chat.call_count == 4


def test_render_resolves_entry_stage_per_item():
    """Items that bypass stage 0 must use the entry stage's renderer."""
    handler = _make_render_handler()
    handler.engine_client.resolve_entry_stage_id = MagicMock(side_effect=[1, 0])
    stage_renderer = MagicMock()
    handler.engine_client.get_stage_renderer = MagicMock(return_value=stage_renderer)

    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(2)))
    assert not isinstance(result, ErrorResponse)
    _, _, _, entry_stage_ids = result
    assert entry_stage_ids == [1, 0]

    # Only the bypassing item resolves a stage renderer; that item's
    # preprocess call carries the stage renderer and entry_stage_id.
    handler.engine_client.get_stage_renderer.assert_called_once_with(1)
    first_call = handler._preprocess_chat.call_args_list[0]
    assert first_call.kwargs["renderer"] is stage_renderer
    assert first_call.kwargs["entry_stage_id"] == 1
    second_call = handler._preprocess_chat.call_args_list[1]
    assert second_call.kwargs["entry_stage_id"] == 0


def test_render_error_short_circuits():
    handler = _make_render_handler()
    handler._check_model.return_value = ErrorResponse(
        error=ErrorInfo(message="not found", type="NotFoundError", code=404),
    )
    result = asyncio.run(handler.render_batch_chat_request(_make_batch_request(5)))
    assert isinstance(result, ErrorResponse)
    handler._preprocess_chat.assert_not_called()


def test_render_engine_dead_raises():
    handler = _make_render_handler()
    handler.engine_client.errored = True
    handler.engine_client.dead_error = RuntimeError("Engine dead")
    with pytest.raises(RuntimeError, match="Engine dead"):
        asyncio.run(handler.render_batch_chat_request(_make_batch_request(3)))


# ---------------------------------------------------------------------------
# Tests for chat_completion_full_generator_batch (PR C: response builder)
# ---------------------------------------------------------------------------


@dataclass
class _FakeOmniOutput:
    """Minimal stand-in for OmniRequestOutput for testing."""

    request_id: str = "req-1"
    finished: bool = True
    stage_id: int | None = 0
    final_output_type: str = "text"
    outputs: list = dc_field(default_factory=list)
    prompt_token_ids: list = dc_field(default_factory=lambda: [1, 2, 3])
    prompt_logprobs: object = None
    encoder_prompt_token_ids: list | None = None
    kv_transfer_params: dict | None = None
    metrics: dict = dc_field(default_factory=dict)
    _multimodal_output: dict = dc_field(default_factory=dict)


def _fake_completion_output(text="hello", n_tokens=3):
    out = CompletionOutput.__new__(CompletionOutput)
    out.index = 0
    out.text = text
    out.token_ids = list(range(n_tokens))
    out.cumulative_logprob = None
    out.logprobs = None
    out.finish_reason = "stop"
    out.stop_reason = None
    out.lora_request = None
    return out


def _make_text_omni_output(text="hello", prompt_tokens=5, completion_tokens=3):
    return _FakeOmniOutput(
        final_output_type="text",
        stage_id=0,
        prompt_token_ids=list(range(prompt_tokens)),
        outputs=[_fake_completion_output(text, completion_tokens)],
    )


def _make_audio_omni_output(prompt_tokens=3, completion_tokens=4):
    return _FakeOmniOutput(
        final_output_type="audio",
        stage_id=1,
        prompt_token_ids=list(range(prompt_tokens)),
        outputs=[_fake_completion_output("audio", completion_tokens)],
    )


async def _async_gen(*items):
    for item in items:
        yield item


def _make_generator_handler():
    """Build handler with mocked internals for generator batch tests."""
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler.create_error_response = MagicMock(
        side_effect=lambda msg, **kw: ErrorResponse(
            error=ErrorInfo(message=msg, type="InternalServerError", code=500),
        )
    )
    handler.get_chat_request_role = MagicMock(return_value="assistant")
    handler.response_role = "assistant"

    def fake_create_text_choice(request, omni_output, tokenizer, conversation, role, reasoning_parser=None):
        n_prompt = len(omni_output.prompt_token_ids or [])
        n_completion = sum(len(o.token_ids) for o in omni_output.outputs)
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=omni_output.outputs[0].text),
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ]
        usage = UsageInfo(
            prompt_tokens=n_prompt,
            completion_tokens=n_completion,
            total_tokens=n_prompt + n_completion,
        )
        return choices, usage, None, None, None

    handler._create_text_choice = MagicMock(side_effect=fake_create_text_choice)

    def fake_create_audio_choice(omni_output, role, request, stream=False):
        audio = ChatCompletionAudio(id="a1", data="base64audio", expires_at=0, transcript="")
        return [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=None, audio=audio),
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ]

    handler._create_audio_choice = MagicMock(side_effect=fake_create_audio_choice)
    handler._create_image_choice = MagicMock(return_value=[])
    return handler


def test_generator_batch_happy_path():
    handler = _make_generator_handler()
    generators = [
        _async_gen(_make_text_omni_output("a", prompt_tokens=10, completion_tokens=5)),
        _async_gen(_make_text_omni_output("b", prompt_tokens=20, completion_tokens=15)),
    ]
    request = _make_batch_request(2)
    request.modalities = ["text"]
    conversations = [[{"role": "user", "content": "q"}]] * 2
    metadata = RequestResponseMetadata(request_id="batch-1")

    result = asyncio.run(
        handler.chat_completion_full_generator_batch(
            request,
            generators,
            "batch-1",
            "test-model",
            conversations,
            MagicMock(),
            metadata,
        )
    )
    assert not isinstance(result, ErrorResponse)
    assert len(result.choices) == 2
    for i, choice in enumerate(result.choices):
        assert choice.index == i
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 50


def test_generator_batch_text_plus_audio():
    handler = _make_generator_handler()
    generators = [
        _async_gen(
            _make_text_omni_output("hello", prompt_tokens=5, completion_tokens=2),
            _make_audio_omni_output(),
        )
        for _ in range(2)
    ]
    request = _make_batch_request(2)
    request.modalities = ["text", "audio"]
    conversations = [[{"role": "user", "content": "hi"}]] * 2
    metadata = RequestResponseMetadata(request_id="batch-2")

    result = asyncio.run(
        handler.chat_completion_full_generator_batch(
            request,
            generators,
            "batch-2",
            "test-model",
            conversations,
            MagicMock(),
            metadata,
        )
    )
    assert not isinstance(result, ErrorResponse)
    assert len(result.choices) == 2
    for choice in result.choices:
        assert choice.message.content == "hello"
        assert choice.message.audio is not None
        assert choice.message.audio.data == "base64audio"
    # text (5 prompt + 2 completion via _create_text_choice) +
    # audio (3 prompt + 4 completion counted from the output) per item
    assert result.usage.prompt_tokens == 16
    assert result.usage.completion_tokens == 12
    assert result.usage.total_tokens == 28


def test_generator_batch_audio_only_usage_counted():
    """Audio-only items must contribute usage, not report usage: 0."""
    handler = _make_generator_handler()
    generators = [_async_gen(_make_audio_omni_output(prompt_tokens=7, completion_tokens=4)) for _ in range(2)]
    request = _make_batch_request(2)
    request.modalities = ["audio"]
    conversations = [[{"role": "user", "content": "hi"}]] * 2
    metadata = RequestResponseMetadata(request_id="batch-audio")

    result = asyncio.run(
        handler.chat_completion_full_generator_batch(
            request,
            generators,
            "batch-audio",
            "test-model",
            conversations,
            MagicMock(),
            metadata,
        )
    )
    assert not isinstance(result, ErrorResponse)
    assert result.usage.prompt_tokens == 14
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 22


def test_generator_batch_uses_per_item_request_for_choices():
    """Choice builders must get the per-item ChatCompletionRequest, not the
    batch request (which lacks chat-only fields like ``tools``)."""
    handler = _make_generator_handler()
    generators = [
        _async_gen(_make_text_omni_output("a")),
        _async_gen(_make_text_omni_output("b")),
    ]
    request = _make_batch_request(2)
    request.modalities = ["text"]
    item_requests = [MagicMock() for _ in range(2)]
    metadata = RequestResponseMetadata(request_id="batch-item-req")

    asyncio.run(
        handler.chat_completion_full_generator_batch(
            request,
            generators,
            "batch-item-req",
            "test-model",
            [[{"role": "user", "content": "q"}]] * 2,
            MagicMock(),
            metadata,
            single_requests=item_requests,
        )
    )
    passed = [call.args[0] for call in handler._create_text_choice.call_args_list]
    assert passed == item_requests


def test_generator_batch_empty_generator_returns_error():
    handler = _make_generator_handler()
    generators = [
        _async_gen(_make_text_omni_output("ok")),
        _async_gen(),  # empty
    ]
    request = _make_batch_request(2)
    request.modalities = ["text"]
    conversations = [[{"role": "user", "content": "q"}]] * 2
    metadata = RequestResponseMetadata(request_id="batch-4")

    result = asyncio.run(
        handler.chat_completion_full_generator_batch(
            request,
            generators,
            "batch-4",
            "test-model",
            conversations,
            MagicMock(),
            metadata,
        )
    )
    assert isinstance(result, ErrorResponse)


# ---------------------------------------------------------------------------
# Tests for create_batch_chat_completion wiring (PR D: optimized path)
# ---------------------------------------------------------------------------


def _make_wiring_handler(n_items=2):
    """Build handler with mocked render + engine for optimized-path tests."""
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)

    conversations = [[{"role": "user", "content": f"Q{i}"}] for i in range(n_items)]
    engine_prompts = [{"prompt_token_ids": [1, 2, 3]} for _ in range(n_items)]
    single_reqs = [MagicMock() for _ in range(n_items)]

    async def _render_side_effect(req):
        req.modalities = ["text"]
        return (conversations, engine_prompts, single_reqs, [0] * n_items)

    handler.render_batch_chat_request = AsyncMock(
        side_effect=_render_side_effect,
    )
    handler._maybe_get_adapters = MagicMock(return_value=None)
    handler.models = MagicMock()
    handler.models.model_name.return_value = "test-model"
    handler.renderer = MagicMock()
    handler.renderer.get_tokenizer.return_value = MagicMock()
    handler.parser_cls = None
    handler._log_inputs = MagicMock()
    handler._build_sampling_params_list_from_request = MagicMock(
        return_value=[MagicMock()],
    )

    handler.engine_client = MagicMock()
    handler.engine_client.stage_configs = [{"stage_type": "llm"}]
    handler.engine_client.generate = MagicMock(
        side_effect=lambda **kw: _async_gen(
            _make_text_omni_output("ok", prompt_tokens=5, completion_tokens=3),
        ),
    )

    handler.chat_completion_full_generator_batch = AsyncMock(
        return_value=ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content="ok"),
                )
                for i in range(n_items)
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=6, total_tokens=16),
        ),
    )
    handler.create_error_response = MagicMock(
        side_effect=lambda msg, **kw: ErrorResponse(
            error=ErrorInfo(message=msg, type="InternalServerError", code=500),
        ),
    )
    return handler


def test_optimized_path_happy_path():
    handler = _make_wiring_handler(3)
    request = _make_batch_request(3)
    raw_request = _make_raw_request([])

    result = asyncio.run(handler.create_batch_chat_completion(request, raw_request))
    assert not isinstance(result, ErrorResponse)
    handler.render_batch_chat_request.assert_called_once_with(request)
    assert handler.engine_client.generate.call_count == 3
    ids = [call.kwargs["request_id"] for call in handler.engine_client.generate.call_args_list]
    assert len(set(ids)) == 3
    call_args = handler.chat_completion_full_generator_batch.call_args
    passed_request_id = call_args[0][2]
    assert passed_request_id.startswith("chatcmpl-batch")


def test_optimized_path_passes_entry_stage_to_generate():
    """A homogeneous batch that bypasses stage 0 must submit at its entry stage."""
    handler = _make_wiring_handler(2)
    conversations = [[{"role": "user", "content": f"Q{i}"}] for i in range(2)]
    single_reqs = [MagicMock() for _ in range(2)]

    async def _render_side_effect(req):
        req.modalities = ["text"]
        return (conversations, [{"prompt_token_ids": [1]}] * 2, single_reqs, [1, 1])

    handler.render_batch_chat_request = AsyncMock(side_effect=_render_side_effect)
    stage_renderer = MagicMock()
    stage_renderer.get_tokenizer.return_value = MagicMock()
    handler.engine_client.get_stage_renderer = MagicMock(return_value=stage_renderer)

    result = asyncio.run(handler.create_batch_chat_completion(_make_batch_request(2), _make_raw_request([])))
    assert not isinstance(result, ErrorResponse)
    for call in handler.engine_client.generate.call_args_list:
        assert call.kwargs["entry_stage_id"] == 1
    # The entry stage's tokenizer is used for response building.
    handler.engine_client.get_stage_renderer.assert_called_with(1)


# ---------------------------------------------------------------------------
# Tests for diffusion dispatch in create_batch_chat_completion
# ---------------------------------------------------------------------------


def _make_dispatch_handler(stage_configs, diffusion_mode: bool = False):
    """Build a handler whose only live parts are the dispatch targets."""
    handler = OmniOpenAIServingChatBatch.__new__(OmniOpenAIServingChatBatch)
    handler._diffusion_mode = diffusion_mode
    handler.engine_client = MagicMock()
    handler.engine_client.stage_configs = list(stage_configs)
    handler.render_batch_chat_request = AsyncMock()
    handler._create_batch_chat_completion_diffusion = AsyncMock(
        return_value=ChatCompletionResponse(
            model="test-model",
            choices=[],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        ),
    )
    return handler


@pytest.mark.parametrize(
    "stage_configs",
    [
        ({"stage_type": "diffusion"},),  # single-stage diffusion
        ({"stage_type": "llm"}, {"stage_type": "diffusion"}),  # hybrid AR + diffusion
    ],
)
def test_diffusion_stages_route_to_fanout(stage_configs):
    """Engines with any diffusion stage must use the per-item fan-out path."""
    handler = _make_dispatch_handler(stage_configs)
    asyncio.run(handler.create_batch_chat_completion(_make_batch_request(2), _make_raw_request([])))
    handler._create_batch_chat_completion_diffusion.assert_called_once()
    handler.render_batch_chat_request.assert_not_called()


def test_diffusion_mode_routes_to_fanout():
    """Diffusion mode has no AR engine client and must use the fan-out path."""
    handler = _make_dispatch_handler((), diffusion_mode=True)
    handler.engine_client = None
    asyncio.run(handler.create_batch_chat_completion(_make_batch_request(2), _make_raw_request([])))
    handler._create_batch_chat_completion_diffusion.assert_called_once()
    handler.render_batch_chat_request.assert_not_called()


def test_pure_ar_stages_route_to_optimized():
    """Engines without diffusion stages must use the optimized path."""
    handler = _make_dispatch_handler(({"stage_type": "llm"}, {"stage_type": "llm"}))
    # Reuse the full wiring mocks so the optimized path can run to completion.
    wired = _make_wiring_handler(2)
    handler.render_batch_chat_request = wired.render_batch_chat_request
    handler._maybe_get_adapters = wired._maybe_get_adapters
    handler.models = wired.models
    handler.renderer = wired.renderer
    handler.parser_cls = wired.parser_cls
    handler._log_inputs = wired._log_inputs
    handler._build_sampling_params_list_from_request = wired._build_sampling_params_list_from_request
    handler.engine_client = wired.engine_client
    handler.engine_client.stage_configs = [{"stage_type": "llm"}, {"stage_type": "llm"}]
    handler.chat_completion_full_generator_batch = wired.chat_completion_full_generator_batch

    result = asyncio.run(handler.create_batch_chat_completion(_make_batch_request(2), _make_raw_request([])))
    assert not isinstance(result, ErrorResponse)
    handler.render_batch_chat_request.assert_called_once()
