# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online tests for Fish Speech S2 Pro with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks. Fish Speech S2 Pro has no built-in
speakers: a plain request uses the model's internal default reference
voice, while ``ref_audio`` + ``ref_text`` enables zero-shot voice
cloning. See the reference client in
``examples/online_serving/text_to_speech/fish_speech/speech_client.py``
and the serving path in ``vllm_omni/entrypoints/openai/serving_speech.py``.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams

# ``core_model`` keeps this smoke collectible by the PR-tier selector
# (``core_model and tts``); ``full_model`` is what makes it actually run on
# GPU, in the nightly ``TTS \u00b7 Function Test`` job
# (``full_model and L4 and B200 and tts and cards_1``, see
# ``.buildkite/cuda/test-nightly.yml``). Fish Speech had no ``tests/e2e``
# coverage and no ``tts`` marker before, so both TTS CI tiers were no-ops for
# it. See https://github.com/vllm-project/vllm-omni/issues/4226.
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.full_model,
    pytest.mark.tts,
]

MODEL = "fishaudio/s2-pro"

# Fish Speech S2 Pro resolves to its default 2-stage slow_ar -> dac_decoder
# deploy config (``vllm_omni/deploy/fish_qwen3_omni.yaml``, picked up from the
# HF ``model_type=fish_qwen3_omni``) when no explicit stage config is given,
# matching ``examples/online_serving/text_to_speech/fish_speech/run_server.sh``.
tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="fish_speech_s2_pro",
    )
]

DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0

# Fish Speech decodes to 44.1 kHz mono WAV (``DAC_SAMPLE_RATE``), so this floor
# is ~0.45 s of PCM_16 payload (44_100 * 0.45 * 2 ~= 40 KiB) plus a WAV header.
# A conservative floor that catches truncated / silence-only outputs without
# flagging short legitimate clips (same rationale as
# tests/e2e/online_serving/test_higgs_audio_v2_expansion.py).
_MIN_AUDIO_BYTES = 40_000

# Reuse the vendored qwen3_tts reference clip (clean ~5 s English speech) and
# its transcript for zero-shot voice cloning. Keeping a single shared reference
# clip across TTS tests avoids duplicating WAVs in the repo; see the asset
# rationale in tests/e2e/online_serving/test_qwen3_tts_base.py.
REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt() -> str:
    """English prompt matching the English reference voice."""
    return "The weather is nice today, perfect for a walk in the park."


@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
class TestFishSpeechTTS:
    """Fish Speech S2 Pro online serving via OpenAI-compatible /v1/audio/speech."""

    @hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
    def test_text_to_audio_001(self, omni_server, online_client) -> None:
        """Basic text-to-speech with the model's default reference voice.

        Fish Speech has no built-in speakers, so ``voice`` is omitted and the
        model falls back to its internal default reference; see
        ``examples/online_serving/text_to_speech/fish_speech/gradio_demo.py``.
        """
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": get_prompt(),
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
    def test_voice_clone_002(self, omni_server, online_client) -> None:
        """Zero-shot voice cloning via ``ref_audio`` + ``ref_text``.

        ``FishSpeechAdapter.validate`` rejects ``ref_audio`` without
        ``ref_text``, and the structured clone prompt is only built when both
        are present (``vllm_omni/entrypoints/openai/tts_adapters/fish_speech.py``).
        """
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": get_prompt(),
                "stream": False,
                "response_format": "wav",
                "ref_audio": REF_AUDIO_URL,
                "ref_text": REF_TEXT,
                "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )
