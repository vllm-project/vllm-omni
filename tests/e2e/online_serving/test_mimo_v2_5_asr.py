# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for MiMo-V2.5-ASR (speech-to-text via OpenAI-compatible API).

Shares deploy config and chat template with MiMo-Audio; tests focus on
audio-in / text-out transcription rather than TTS or audio generation.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

MIMO_AUDIO_TOKENIZER_REPO = "XiaomiMiMo/MiMo-Audio-Tokenizer"
CHAT_TEMPLATE_PATH = str(
    Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "mimo_audio" / "chat_template.jinja"
)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "XiaomiMiMo/MiMo-V2.5-ASR"

# TTS synthetic clip defaults to the word "test" (see tests.helpers.media.generate_synthetic_audio).
ASR_REFERENCE_WORD = "test"
TRANSCRIBE_PROMPT = "Please transcribe this audio."
# Aligns with examples/offline_inference/mimo_audio/end2end.py audio_trancribing_sft.
TRANSCRIBE_REPEAT_PROMPT = "Please transcribe this audio and repeat it once."


def download_tokenizer():
    tokenizer_path = os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", MIMO_AUDIO_TOKENIZER_REPO)
    if os.path.exists(tokenizer_path):
        return tokenizer_path
    return download_weights_from_hf_specific(
        model_name_or_path=MIMO_AUDIO_TOKENIZER_REPO,
        cache_dir=None,
        allow_patterns=["*"],
        require_all=True,
    )


try:
    stage_configs = [get_deploy_config_path("mimo_audio.yaml")]
    os.environ["MIMO_AUDIO_TOKENIZER_PATH"] = download_tokenizer()
    test_params = [
        OmniServerParams(
            model=MODEL,
            stage_config_path=stage_config,
            server_args=["--chat-template", CHAT_TEMPLATE_PATH],
        )
        for stage_config in stage_configs
    ]
except Exception as exc:
    pytest.skip(
        f"MiMo-V2.5-ASR online serving tests skipped: module setup failed ({type(exc).__name__}: {exc})",
        allow_module_level=True,
    )


def _asr_messages(content_text: str):
    audio_b64 = generate_synthetic_audio(
        5,
        1,
        sample_rate=24000,
        phrase_text=ASR_REFERENCE_WORD,
    )["base64"]
    audio_data_url = f"data:audio/wav;base64,{audio_b64}"
    return dummy_messages_from_mix_data(
        audio_data_url=audio_data_url,
        content_text=content_text,
    )


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_asr_smoke(omni_server, openai_client) -> None:
    """
    ASR smoke: audio + transcribe instruction -> text only.

    Deploy: mimo_audio.yaml
    Input: synthetic WAV (spoken word "test") + transcribe prompt
    Output: text containing the reference word
    """
    request_config = {
        "model": omni_server.model,
        "messages": _asr_messages(TRANSCRIBE_PROMPT),
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": [ASR_REFERENCE_WORD]},
    }
    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_asr_transcribe_and_repeat(omni_server, openai_client) -> None:
    """
    ASR with offline-example prompt (audio_trancribing_sft style).

    Expects the reference word at least once in the transcription output.
    """
    request_config = {
        "model": omni_server.model,
        "messages": _asr_messages(TRANSCRIBE_REPEAT_PROMPT),
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": [ASR_REFERENCE_WORD]},
    }
    openai_client.send_omni_request(request_config)
