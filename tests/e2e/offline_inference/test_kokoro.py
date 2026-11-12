# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E offline inference tests for Kokoro-82M (single-stage, chunk per step).

Prompts carry a one-token placeholder; text, voice, speed and seed ride
``additional_information`` as the serving adapter sends them. Needs the
``vllm-omni[kokoro]`` G2P extra.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner, OmniRunnerHandler
from tests.helpers.stage_config import get_deploy_config_path

pytest.importorskip("misaki")

MODEL_NAME = os.environ.get("KOKORO_MODEL_PATH", "hexgrad/Kokoro-82M")
STAGE_CONFIG = get_deploy_config_path("kokoro.yaml")
SAMPLE_RATE = 24000

_OMNI_RUNNER_PARAM: tuple[str, str, dict] = (MODEL_NAME, STAGE_CONFIG, {})

pytestmark = [
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

# Three languages, each with its own voice and seed.
_VOICE_REQUESTS = [
    ("Please close the window before leaving the office tonight.", "af_heart", 1),
    ("Colour and flavour are spelt differently in Britain.", "bm_george", 2),
    ("Hola, hoy es un día precioso en Madrid.", "ef_dora", 3),
]
# Over 510 phonemes: synthesized as several chunks, one engine step each.
_LONG_TEXT = " ".join(
    ["The quick brown fox jumps over the lazy dog, then takes a long nap in the warm afternoon sun."] * 8
)

# At ``core_model`` the harness loads dummy weights, so predicted durations
# (and therefore audio length) are arbitrary; bound them only with real weights.
_REAL_WEIGHT_RUN_LEVELS = frozenset({"advanced_model", "full_model"})


def _info(text: str, voice: str, seed: int) -> dict:
    return {"text": [text], "voice": [voice], "speed": [1.0], "seed": [seed]}


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_kokoro_offline_concurrent_voices(omni_runner: OmniRunner, run_level: str) -> None:
    """Three concurrent requests in three languages each return 24 kHz audio."""
    request_config = {
        "input": _VOICE_REQUESTS[0][0],
        "prompt_token_ids": [[0]] * len(_VOICE_REQUESTS),
        "additional_information": [_info(*request) for request in _VOICE_REQUESTS],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        "assert_distinct_outputs": True,
    }
    if run_level in _REAL_WEIGHT_RUN_LEVELS:
        request_config.update(min_duration_s=0.5, max_duration_s=30.0)
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_kokoro_offline_long_text_spans_chunks(omni_runner: OmniRunner) -> None:
    request_config = {
        "input": _LONG_TEXT,
        "prompt_token_ids": [0],
        "additional_information": _info(_LONG_TEXT, "am_michael", 0),
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        # ~45 s of speech: far past what a single 510-phoneme pass can produce.
        "min_duration_s": 30.0,
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)
