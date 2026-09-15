# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniCPM-o 4.5 chat reference audio transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs import TokensPrompt

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO45_REFERENCE_AUDIO_KEY
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    llm2tts,
    tts2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _EngineClientStub:
    stage_configs: list[dict[str, object]]


@dataclass
class _ModelConfigStub:
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None


@dataclass
class _ThinkerCompletionStub:
    token_ids: list[int]
    text: str
    hidden_states: torch.Tensor
    multimodal_output: dict[str, object] = field(default_factory=dict)


@dataclass
class _ThinkerRequestOutputStub:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[_ThinkerCompletionStub]
    multimodal_output: dict[str, object] | None = None


@dataclass
class _Code2WavRequestStub:
    external_req_id: str
    request_id: str
    model_intermediate_buffer: dict[str, object]


@dataclass
class _TransferManagerStub:
    request_payload: dict[str, object] = field(default_factory=dict)


@pytest.fixture
def serving_chat() -> OmniOpenAIServingChat:
    return object.__new__(OmniOpenAIServingChat)


def _stage_config(model_arch: str) -> dict[str, object]:
    return {"engine_args": {"model_arch": model_arch}}


def _chat_request(reference_audio_url: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "target text"}],
        ref_audio=reference_audio_url,
    )


@pytest.mark.asyncio
async def test_chat_reference_audio_reaches_code2wav_transport(
    serving_chat: OmniOpenAIServingChat,
    mocker: MockerFixture,
) -> None:
    reference_audio_url = "data:audio/wav;base64,AAAA"
    reference_waveform = np.array([0.25, -0.5], dtype=np.float32)
    serving_chat.engine_client = _EngineClientStub(
        [_stage_config("MiniCPMO45OmniForConditionalGeneration")],
    )
    serving_chat.model_config = _ModelConfigStub()
    engine_prompt: TokensPrompt = {"type": "tokens", "prompt_token_ids": [1]}
    media_connector_class = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_chat.MediaConnector",
    )
    media_connector = media_connector_class.return_value
    media_connector.fetch_audio_async = mocker.AsyncMock(return_value=(reference_waveform, 16000))

    await serving_chat._attach_minicpmo45_reference_audio(
        engine_prompt,
        _chat_request(reference_audio_url),
    )

    media_connector_class.assert_called_once_with(
        media_io_kwargs=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
    )
    media_connector.fetch_audio_async.assert_awaited_once_with(reference_audio_url)
    preserved_reference = cast(dict[str, object], engine_prompt)[MINICPMO45_REFERENCE_AUDIO_KEY]
    assert isinstance(preserved_reference, tuple)
    preserved_waveform, preserved_sample_rate = preserved_reference
    assert isinstance(preserved_waveform, np.ndarray)
    assert isinstance(preserved_sample_rate, int)
    np.testing.assert_array_equal(preserved_waveform, reference_waveform)
    assert preserved_sample_rate == 16000

    thinker_output = _ThinkerRequestOutputStub(
        request_id="request-0",
        prompt_token_ids=[10],
        outputs=[
            _ThinkerCompletionStub(
                token_ids=[20],
                text="target text",
                hidden_states=torch.zeros((2, 4)),
            )
        ],
    )
    talker_prompt = llm2tts([thinker_output], prompt=engine_prompt)[0]
    talker_request = _Code2WavRequestStub(
        external_req_id="request-0",
        request_id="request-0",
        model_intermediate_buffer=talker_prompt["model_intermediate_buffer"],
    )
    transfer_manager = _TransferManagerStub()

    code2wav_payload = tts2code2wav_async_chunk(
        transfer_manager,
        {"codes": {"audio": torch.arange(7).reshape(-1, 1)}},
        talker_request,
        True,
    )

    assert code2wav_payload is not None
    torch.testing.assert_close(code2wav_payload.codes.ref, torch.from_numpy(reference_waveform))
    assert code2wav_payload.meta.ref_audio_sr == 16000


@pytest.mark.parametrize("request_shape", ["extra_body", "model_extra"])
@pytest.mark.asyncio
async def test_reference_audio_accepts_request_extra_mappings(
    serving_chat: OmniOpenAIServingChat,
    mocker: MockerFixture,
    request_shape: str,
) -> None:
    reference_audio_url = "data:audio/wav;base64,AAAA"
    reference_waveform = np.array([0.25, -0.5], dtype=np.float32)
    serving_chat.engine_client = _EngineClientStub(
        [_stage_config("MiniCPMO45OmniForConditionalGeneration")],
    )
    serving_chat.model_config = _ModelConfigStub()
    engine_prompt: TokensPrompt = {"type": "tokens", "prompt_token_ids": [1]}
    if request_shape == "extra_body":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "target text"}],
            extra_body={"ref_audio": reference_audio_url},
        )
    else:
        request = SimpleNamespace(model_extra={"ref_audio": reference_audio_url})
    assert not hasattr(request, "ref_audio")
    media_connector = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_chat.MediaConnector",
    ).return_value
    media_connector.fetch_audio_async = mocker.AsyncMock(return_value=(reference_waveform, 16000))

    await serving_chat._attach_minicpmo45_reference_audio(engine_prompt, request)

    media_connector.fetch_audio_async.assert_awaited_once_with(reference_audio_url)
    preserved_reference = cast(dict[str, object], engine_prompt)[MINICPMO45_REFERENCE_AUDIO_KEY]
    assert isinstance(preserved_reference, tuple)
    np.testing.assert_array_equal(preserved_reference[0], reference_waveform)
    assert preserved_reference[1] == 16000


@pytest.mark.asyncio
async def test_reference_audio_is_ignored_for_other_models(
    serving_chat: OmniOpenAIServingChat,
    mocker: MockerFixture,
) -> None:
    reference_audio_url = "data:audio/wav;base64,AAAA"
    serving_chat.engine_client = _EngineClientStub([_stage_config("SomeOtherOmniForConditionalGeneration")])
    engine_prompt: TokensPrompt = {"type": "tokens", "prompt_token_ids": [1]}
    media_connector_class = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_chat.MediaConnector",
    )

    await serving_chat._attach_minicpmo45_reference_audio(
        engine_prompt,
        _chat_request(reference_audio_url),
    )

    media_connector_class.assert_not_called()
    assert MINICPMO45_REFERENCE_AUDIO_KEY not in cast(dict[str, object], engine_prompt)
