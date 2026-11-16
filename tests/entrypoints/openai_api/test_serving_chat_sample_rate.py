# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chat serving resolves audio rates via the shared reader, with no defaults."""

from unittest.mock import MagicMock

import pytest
import torch

from tests.helpers.serving_chat import build_serving_chat, make_request, make_text_omni_output
from vllm_omni.entrypoints.openai.protocol.audio import AudioChunkMetadata
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.sample_rate import MissingSampleRateError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNUSUAL_RATE = 22050


def _build_serving() -> OmniOpenAIServingChat:
    # build_serving_chat mocks _create_audio_choice by default; rebind the
    # real method (the test_stream_finish_reason.py pattern).
    serving = build_serving_chat()
    serving._create_audio_choice = lambda omni_res, role, request, stream=False: (
        OmniOpenAIServingChat._create_audio_choice(serving, omni_res, role, request, stream=stream)
    )
    return serving


def _make_audio_omni_output(mm_output: dict) -> object:
    output = make_text_omni_output(text="", token_ids=[10], finish_reason="stop")
    output.final_output_type = "audio"
    output.outputs[0].multimodal_output = mm_output
    return output


def _capture_create_audio(serving) -> MagicMock:
    response = MagicMock(audio_data=b"audio", media_type="audio/wav")
    response.audio_metadata = AudioChunkMetadata(format="wav", sample_rate_hz=UNUSUAL_RATE, frame_count=320, channels=1)
    create_audio = MagicMock(return_value=response)
    serving.create_audio = create_audio
    return create_audio


@pytest.mark.asyncio
async def test_audio_choice_labels_output_with_model_rate() -> None:
    # The talker's advertised rate (an unusual value proves no default is
    # involved) must reach the serialized audio's metadata.
    serving = _build_serving()
    create_audio = _capture_create_audio(serving)
    request = make_request(modalities=["audio"], stream=False)

    choices = serving._create_audio_choice(
        _make_audio_omni_output({"audio": torch.zeros(320), "sr": torch.tensor(UNUSUAL_RATE, dtype=torch.int32)}),
        "assistant",
        request,
        stream=False,
    )

    assert not hasattr(choices, "error")
    assert create_audio.call_args[0][0].sample_rate == UNUSUAL_RATE


@pytest.mark.asyncio
async def test_audio_choice_without_rate_fails_loudly() -> None:
    # Every audio-emitting model advertises its rate; a payload without one
    # is a producer bug that must surface, not fall back to 24000.
    serving = _build_serving()
    _capture_create_audio(serving)
    request = make_request(modalities=["audio"], stream=False)

    with pytest.raises(MissingSampleRateError):
        serving._create_audio_choice(
            _make_audio_omni_output({"audio": torch.zeros(320)}),
            "assistant",
            request,
            stream=False,
        )
