# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for output modality validation in chat completions.

Covers:
- #4719: unsupported modality must return an error, not empty choices
- Single-stage diffusion text-output path must remain accessible
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.sampling_params import RequestOutputKind, SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_serving_chat(output_modalities: list[str], stage_configs=None):
    """Build a minimal OmniOpenAIServingChat with mocked internals."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = SimpleNamespace(
        output_modalities=output_modalities,
        errored=False,
        stage_configs=stage_configs or [],
        resolve_entry_stage_id=lambda input_modalities: 0,
    )
    instance._diffusion_mode = False
    instance.models = MagicMock()
    instance.models.model_name.return_value = "test-model"
    instance.enable_prompt_tokens_details = False

    renderer = MagicMock()
    renderer.get_tokenizer.return_value = MagicMock()
    instance.renderer = renderer
    instance.online_renderer = MagicMock()
    instance.online_renderer.validate_chat_template.return_value = None

    instance.parser_cls = None
    instance.use_harmony = False
    instance.trust_request_chat_template = True
    instance.chat_template = None
    instance.chat_template_content_format = "auto"
    instance.default_chat_template_kwargs = {}
    instance.enable_auto_tools = False
    instance.exclude_tools_when_tool_choice_none = False
    instance._check_model = AsyncMock(return_value=None)
    return instance


def _make_request(modalities):
    """Build a ChatCompletionRequest with the given modalities."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    req.modalities = modalities
    return req


@pytest.mark.asyncio
@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize(
    "modalities", [{}, {"modalities": None}, {"modalities": ["text", "audio"]}, {"modalities": ["text"]}]
)
async def test_ming3_resolved_modalities_reach_output_kind_correction(mocker, async_chunk, modalities):
    serving = _make_serving_chat(
        ["text", "audio"],
        stage_configs=[
            {
                "engine_args": {"model_arch": "BailingMoeV3AudioForConditionalGeneration", "model_stage": "thinker"},
                "connector_config": {"async_chunk": async_chunk},
            },
            {"engine_args": {"model_arch": "BailingTalker2", "model_stage": "ming3_tts"}},
        ],
    )
    mocker.patch.object(serving, "_maybe_get_adapters", return_value=None)
    mocker.patch.object(serving, "_preprocess_chat", new=mocker.AsyncMock(return_value=([], [{}])))
    mocker.patch.object(
        serving, "_build_sampling_params_list_from_request", return_value=[SamplingParams(), SamplingParams()]
    )
    serving.request_logger = None
    mocker.patch.object(serving, "chat_completion_stream_generator", return_value="stream")
    serving.engine_client.generate = mocker.Mock()
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hello"}], stream=True, **modalities
    )

    assert await serving._create_chat_completion(request) == "stream"

    params = serving.engine_client.generate.call_args.kwargs["sampling_params_list"]
    assert request.modalities == (modalities.get("modalities") or ["text", "audio"])
    assert params[0].output_kind is (
        RequestOutputKind.FINAL_ONLY if "audio" in request.modalities and not async_chunk else RequestOutputKind.DELTA
    )
    assert params[1].output_kind is RequestOutputKind.DELTA


@pytest.mark.asyncio
async def test_unsupported_modality_returns_error():
    """#4719: requesting 'video' on a text/audio model must return an error."""
    serving = _make_serving_chat(["text", "audio"])
    request = _make_request(["video"])

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        result = await serving._create_chat_completion(request)

    assert isinstance(result, ErrorResponse)
    assert "video" in result.error.message
    assert "Unsupported" in result.error.message


@pytest.mark.asyncio
async def test_supported_modality_passes_validation():
    """Requesting a supported modality must not trigger an error."""
    serving = _make_serving_chat(["text", "audio"])
    request = _make_request(["text"])

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        try:
            result = await serving._create_chat_completion(request)
        except (AttributeError, TypeError):
            return

    assert not isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_invalid_modalities_type_returns_error():
    """Modalities must be a list of strings."""
    serving = _make_serving_chat(["text", "audio"])
    request = _make_request("audio")  # bare string, not a list

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        result = await serving._create_chat_completion(request)

    assert isinstance(result, ErrorResponse)
    assert "list of strings" in result.error.message


@pytest.mark.asyncio
async def test_modalities_with_non_string_element_returns_error():
    """Modalities list with non-string elements must be rejected."""
    serving = _make_serving_chat(["text", "audio"])
    request = _make_request(["text", 123])

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        result = await serving._create_chat_completion(request)

    assert isinstance(result, ErrorResponse)
    assert "list of strings" in result.error.message


@pytest.mark.asyncio
async def test_single_stage_diffusion_allows_text_modality():
    """Single-stage diffusion engines that advertise 'image' must also accept 'text'."""
    stage_configs = [SimpleNamespace(stage_type="diffusion", is_comprehension=False)]
    serving = _make_serving_chat(["image"], stage_configs=stage_configs)
    request = _make_request(["text"])

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        try:
            result = await serving._create_chat_completion(request)
        except (AttributeError, TypeError):
            return

    assert not isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_single_stage_diffusion_rejects_unsupported_modality():
    """Single-stage diffusion must still reject truly unsupported modalities."""
    stage_configs = [SimpleNamespace(stage_type="diffusion", is_comprehension=False)]
    serving = _make_serving_chat(["image"], stage_configs=stage_configs)
    request = _make_request(["video"])

    with patch.object(serving, "_preprocess_chat", new_callable=AsyncMock, return_value=([], [{}])):
        result = await serving._create_chat_completion(request)

    assert isinstance(result, ErrorResponse)
    assert "video" in result.error.message
