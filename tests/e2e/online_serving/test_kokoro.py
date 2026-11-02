# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Kokoro-82M via /v1/audio/speech.

Real model inference (no mocks): built-in voices across languages, native
speed control, chunked streaming and the reject contract for unsupported
fields. Needs the ``vllm-omni[kokoro]`` G2P extra.
"""

import io
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytest.importorskip("misaki")

MODEL = os.environ.get("KOKORO_MODEL_PATH", "hexgrad/Kokoro-82M")
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 120.0
SAMPLE_RATE = 24000
MAX_CONCURRENT = 4

# ~0.5 s of 24 kHz mono PCM_16 WAV.
_MIN_AUDIO_BYTES = 24_000
# The PCM HNR floor (1.0 dB) was calibrated on other models. The reference
# kokoro package measures -1.6 to +1.7 dB on clean speech depending on the
# voice (am_michael -0.9, af_heart +0.7), so keep only the catastrophic check.
_MIN_HNR_DB = -3.0

# Over 510 phonemes, so the request spans several network passes (one per
# streamed chunk).
_LONG_TEXT = " ".join(
    [
        "The quick brown fox jumps over the lazy dog, then takes a long nap in the warm afternoon sun.",
        "Later that evening the fox wakes up, stretches, and wanders back through the quiet forest.",
    ]
    * 4
)


def get_prompt(prompt_type: str = "en") -> str:
    prompts = {
        "en": "The weather is nice today, perfect for a walk in the park.",
        "en_gb": "Colour and flavour are spelt differently in Britain.",
        # No "the X is" phrases: whisper-small contracts them in fast speech.
        "fast": "Please close the window before leaving the office tonight.",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("kokoro.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="kokoro",
    )
]


def _wav_seconds(audio_bytes: bytes) -> float:
    data, sr = sf.read(io.BytesIO(audio_bytes))
    assert sr == SAMPLE_RATE
    return len(data) / sr


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Baseline smoke: default deploy and voice, non-streaming WAV, concurrent.

    Deploy Setting: kokoro.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests (max_num_seqs=4)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_long_text_streaming_002(omni_server, openai_client) -> None:
    """Multi-chunk input streamed as raw PCM, one network pass per chunk.

    Input Setting: stream=True, response_format=pcm
    """
    request_config = {
        "model": omni_server.model,
        "input": _LONG_TEXT,
        "voice": "am_michael",
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm",
        "min_audio_bytes": 20 * SAMPLE_RATE * 2,
        "expected_sample_rate": SAMPLE_RATE,
        "transcript_pcm_sample_rate": SAMPLE_RATE,
        "min_hnr_db": _MIN_HNR_DB,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_british_voice_003(omni_server, openai_client) -> None:
    """A ``b*`` voice selects British English G2P from the voice name."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en_gb"),
        "voice": "bm_george",
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_native_speed_004(omni_server, openai_client) -> None:
    """``speed`` scales predicted durations once (no extra server-side stretch)."""
    base = {
        "model": omni_server.model,
        "input": get_prompt("fast"),
        "voice": "af_heart",
        "seed": 5,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES // 2,
    }
    normal = openai_client.send_audio_speech_request(base)[0]
    fast = openai_client.send_audio_speech_request({**base, "speed": 1.4})[0]
    ratio = _wav_seconds(normal.audio_bytes) / _wav_seconds(fast.audio_bytes)
    # Per-phoneme durations are rounded to whole frames, so not exactly 1.4.
    assert 1.25 < ratio < 1.55, ratio


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.parametrize(
    ("extra", "err_message"),
    [
        ({"voice": "not_a_voice"}, "Invalid voice"),
        ({"ref_audio": "data:audio/wav;base64,AAAA"}, "ref_audio"),
        ({"language": "klingon"}, "Unsupported language"),
        ({"input": "   "}, "Input text cannot be empty"),
    ],
)
def test_rejects_unsupported_requests_005(omni_server, openai_client, extra: dict, err_message: str) -> None:
    request_config = {
        "json": {"model": omni_server.model, "input": get_prompt(), "response_format": "wav", **extra},
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
    }
    openai_client.send_audio_speech_http_request(request_config, err_code=400, err_message=err_message)
