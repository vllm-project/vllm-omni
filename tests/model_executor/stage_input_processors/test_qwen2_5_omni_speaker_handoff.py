# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The requested speaker reaches both the Qwen2.5-Omni talker and Code2Wav."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
    TALKER_CODEC_END_TOKEN_ID,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
    thinker2talker_full_payload,
    thinker2talker_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPTS = [
    pytest.param({"prompt_token_ids": [1, 2, 3], "additional_information": {"speaker": ["ethan"]}}, ["ethan"]),
    pytest.param({"prompt_token_ids": [1, 2, 3]}, None, id="no-speaker"),
]


def _request(speaker: str | None, **fields) -> SimpleNamespace:
    info = serialize_additional_information({"speaker": [speaker]}) if speaker else None
    return SimpleNamespace(request_id="req-0", additional_information=info, **fields)


@pytest.mark.parametrize(("prompt", "speaker"), PROMPTS)
def test_thinker2talker_token_only_copies_prompt_speaker(prompt, speaker) -> None:
    thinker_output = SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2, 3])

    [talker_prompt] = thinker2talker_token_only([thinker_output], prompt)

    expected = {"speaker": speaker} if speaker else None
    assert talker_prompt["additional_information"] == expected


@pytest.mark.parametrize(("prompt", "speaker"), PROMPTS)
def test_talker2code2wav_token_only_copies_prompt_speaker(prompt, speaker) -> None:
    talker_output = SimpleNamespace(
        outputs=[SimpleNamespace(cumulative_token_ids=[10, 11, TALKER_CODEC_END_TOKEN_ID])],
    )

    [code2wav_prompt] = talker2code2wav_token_only([talker_output], prompt)

    expected = {"speaker": speaker} if speaker else None
    assert code2wav_prompt["additional_information"] == expected
    assert code2wav_prompt["prompt_token_ids"] == [0, 0]


@pytest.mark.parametrize("speaker", ["ethan", None])
def test_full_payloads_forward_request_speaker(speaker) -> None:
    thinker_request = _request(
        speaker,
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4],
        all_token_ids=[1, 2, 3, 4],
        status=None,
        sampling_params=None,
    )
    talker_request = _request(speaker, output_token_ids=[10, 11, TALKER_CODEC_END_TOKEN_ID])

    talker_payload = thinker2talker_full_payload(None, {"hidden": torch.zeros(4, 2)}, thinker_request)
    code2wav_payload = talker2code2wav_full_payload(None, {}, talker_request)

    assert talker_payload.get("speaker") == speaker
    assert code2wav_payload.get("speaker") == speaker
    assert code2wav_payload["codes"]["audio"] == [10, 11]
