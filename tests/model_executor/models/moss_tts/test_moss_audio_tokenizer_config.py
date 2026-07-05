# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_moss_audio_tokenizer_config_imports_with_current_transformers():
    config = MossAudioTokenizerConfig()

    assert config.model_type == "moss-audio-tokenizer"
