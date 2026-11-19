# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech sampling override precedence and request validation."""

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_top_level_sampling_overrides_win_without_mutating_extra_params():
    extras: dict[str, object] = {"temperature": 0.9, "repetition_penalty": 1.05, "subtalker_sampling_params": {}}
    request = OpenAICreateSpeechRequest(input="hello", temperature=0, repetition_penalty=1, extra_params=extras)
    assert request.extra_params == {"temperature": 0.0, "repetition_penalty": 1.0, "subtalker_sampling_params": {}}
    assert extras == {"temperature": 0.9, "repetition_penalty": 1.05, "subtalker_sampling_params": {}}


@pytest.mark.parametrize("overrides", [{"temperature": float("nan")}, {"repetition_penalty": 0}])
def test_extra_params_reject_invalid_sampling_numbers(overrides):
    with pytest.raises(ValidationError, match="extra_params"):
        OpenAICreateSpeechRequest(input="hello", extra_params=overrides)


def test_null_overrides_preserve_nested_values_and_empty_subtalker_policy():
    extras: dict[str, object] = {"temperature": 0.3, "repetition_penalty": None, "subtalker_sampling_params": {}}
    request = OpenAICreateSpeechRequest(input="hello", temperature=None, repetition_penalty=None, extra_params=extras)
    assert request.extra_params == extras
    assert OpenAICreateSpeechRequest(input="hello").extra_params is None
