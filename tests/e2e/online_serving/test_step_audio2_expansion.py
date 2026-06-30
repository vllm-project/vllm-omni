# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E expansion tests for Step-Audio2 online serving (weekly CI).

Tests speech-to-speech translation via /v1/chat/completions.
Request shape follows examples/online_serving/step_audio2/openai_chat_completion_client.py.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL = "stepfun-ai/Step-Audio-2-mini"
STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "step_audio2_ci.yaml")
SAMPLE_RATE = 16000
SEED = 42
# pyttsx3 synthetic input phrase; Step-Audio2 text output includes codec tokens
# (<audio_NNNN>) so we validate audio only, not text↔audio similarity.
SYNTH_PHRASE = "how are you"

TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            env_dict={"VLLM_IMAGE_FETCH_TIMEOUT": "60"},
        ),
        id="step_audio2",
    )
]

pytestmark = [
    pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True),
    pytest.mark.slow,
    pytest.mark.tts,
]


def _synthetic_audio_base64(duration_sec: int = 2) -> str:
    return generate_synthetic_audio(
        duration_sec,
        1,
        SAMPLE_RATE,
        phrase_text=SYNTH_PHRASE,
    )["base64"]


def _default_sampling_params_list() -> list[dict]:
    return [
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
            "max_tokens": 256,
            "seed": SEED,
            "detokenize": True,
            "repetition_penalty": 1.05,
        },
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1,
            "seed": SEED,
            "detokenize": False,
        },
    ]


def _build_s2st_request_config(omni_server, audio_base64: str) -> dict:
    """Build audio-to-audio (S2ST) chat completion request config."""
    audio_url = f"data:audio/wav;base64,{audio_base64}"
    return {
        "model": omni_server.model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an audio processing assistant. Listen and repeat the audio content.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": "Please listen to this audio and repeat its content."},
                ],
            },
            {"role": "assistant", "content": "<tts_start>"},
        ],
        # Thinker text includes undetokenized audio codec placeholders; assert audio only.
        "modalities": ["audio"],
        "audio_ref_text": SYNTH_PHRASE,
        "sampling_params_list": _default_sampling_params_list(),
        "extra_body": {
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    }


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_single_s2st_request(omni_server, openai_client) -> None:
    """Test single speech-to-speech request via chat completions API."""
    request_config = _build_s2st_request_config(
        omni_server,
        _synthetic_audio_base64(duration_sec=2),
    )
    openai_client.send_omni_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("request_num", [2, 4])
def test_concurrent_s2st_requests(omni_server, openai_client, request_num: int) -> None:
    """Test concurrent speech-to-speech requests."""
    request_config = _build_s2st_request_config(
        omni_server,
        _synthetic_audio_base64(duration_sec=2),
    )
    responses = openai_client.send_omni_request(request_config, request_num=request_num)
    assert len(responses) == request_num
