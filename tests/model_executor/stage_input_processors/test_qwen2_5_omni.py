# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
    TALKER_CODEC_END_TOKEN_ID,
    TALKER_CODEC_PAD_TOKEN_ID,
    TALKER_CODEC_START_TOKEN_ID,
    talker2code2wav,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skip(
        reason="qwen2_5_omni stage input processor tests need alignment with model changes that are not yet implemented"
    ),
]


def _source_outputs(output):
    return [SimpleNamespace(outputs=[output])]


def test_talker2code2wav_accepts_completion_output_token_ids():
    output = SimpleNamespace(token_ids=[TALKER_CODEC_START_TOKEN_ID, 11, 22, TALKER_CODEC_END_TOKEN_ID])

    prompts = talker2code2wav(_source_outputs(output))

    assert [prompt["prompt_token_ids"] for prompt in prompts] == [[11, 22]]


def test_talker2code2wav_prefers_cumulative_token_ids_when_present():
    output = SimpleNamespace(
        token_ids=[999],
        cumulative_token_ids=[TALKER_CODEC_START_TOKEN_ID, 33, 44, TALKER_CODEC_END_TOKEN_ID],
    )

    prompts = talker2code2wav(_source_outputs(output))

    assert [prompt["prompt_token_ids"] for prompt in prompts] == [[33, 44]]


def test_talker2code2wav_strips_terminal_codec_pad():
    output = SimpleNamespace(cumulative_token_ids=[TALKER_CODEC_START_TOKEN_ID, 55, 66, TALKER_CODEC_PAD_TOKEN_ID])

    prompts = talker2code2wav(_source_outputs(output))

    assert [prompt["prompt_token_ids"] for prompt in prompts] == [[55, 66]]


def test_talker2code2wav_keeps_empty_codec_terminal_input():
    output = SimpleNamespace(cumulative_token_ids=[TALKER_CODEC_START_TOKEN_ID, TALKER_CODEC_END_TOKEN_ID])

    prompts = talker2code2wav(_source_outputs(output))

    assert [prompt["prompt_token_ids"] for prompt in prompts] == [[]]
