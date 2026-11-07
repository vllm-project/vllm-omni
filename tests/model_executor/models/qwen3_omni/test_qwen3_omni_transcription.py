# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_wrapper_delegates_transcription_to_thinker(mocker):
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration
    wrapper = Qwen3OmniMoeForConditionalGeneration
    config = object()
    params = object()
    speech_config = mocker.patch.object(thinker, "get_speech_to_text_config", return_value=object())
    prompt = mocker.patch.object(thinker, "get_generation_prompt", return_value=object())
    postprocess = mocker.patch.object(thinker, "post_process_output", return_value="recognized")

    assert wrapper.get_speech_to_text_config(config, "transcribe") is speech_config.return_value
    speech_config.assert_called_once_with(config, "transcribe")
    assert wrapper.get_generation_prompt(params) is prompt.return_value
    prompt.assert_called_once_with(params)
    assert wrapper.post_process_output("raw") == "recognized"
    postprocess.assert_called_once_with("raw")
    assert wrapper.supported_languages is thinker.supported_languages
    assert "de" in wrapper.supported_languages
    assert "en" in wrapper.supported_languages
