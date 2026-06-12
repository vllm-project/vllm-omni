# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online tests for higgs-audio v3 against /v1/audio/speech.

Mirrors the higgs_audio_v2 test layout. Covers the plain-text-in / audio-out
happy path, a small concurrent burst (Stage 0 prefix caching has to coexist
with batching), the voice-clone path via ``ref_audio`` + ``ref_text``, the
upstream ``references[]`` cookbook alias added by
``normalize_references_alias`` in ``protocol/audio.py``, and the few
validator rejections that should remain 4xx.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "bosonai/higgs-audio-v3-tts-4b"
STAGE_CONFIG = get_deploy_config_path("higgs_multimodal_qwen3.yaml")
SERVER_ARGS = ["--trust-remote-code", "--disable-log-stats"]
SERVER_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_MOE_USE_DEEP_GEMM": "0"}

TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            server_args=SERVER_ARGS,
            env_dict=SERVER_ENV,
        ),
        id="higgs_audio_v3_plain_text",
    )
]

DEFAULT_SPEECH_TIMEOUT_S = 180.0
# Floor for ~0.5 s of 24 kHz mono PCM_16: 24000 * 0.5 * 2 bytes ~= 24 KiB.
# A WAV header adds 44 bytes; conservative floor catches truncated /
# silence-only outputs without flagging short legitimate clips.
_MIN_AUDIO_BYTES = 20_000

# Reuse the shared TTS reference clip (clean ~5 s 24 kHz mono human speech)
# vendored under tests/assets/qwen3_tts/. Keeps a single WAV across TTS
# tests rather than duplicating asset bytes.
_REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV3OnlineHappyPath:
    """Plain-text -> audio happy paths against the live HTTP server."""

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_plain_text_wav(self, omni_server, openai_client) -> None:
        """Single non-streaming WAV request - canonical TTS happy path."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_plain_text_with_max_new_tokens(self, omni_server, openai_client) -> None:
        """``max_new_tokens`` is one of the few extra fields the v3 validator accepts."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Innovation distinguishes between a leader and a follower.",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "max_new_tokens": 500,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_concurrent_plain_text(self, omni_server, openai_client) -> None:
        """Three concurrent non-streaming requests - guards per-slot audio state and Stage-0 PC under batching."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "It was the night before my birthday.",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            },
            request_num=3,
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_plain_text_pcm_streaming(self, omni_server, openai_client) -> None:
        """Streaming PCM via talker2code2wav_async_chunk + the connector's codec_streaming path.

        The deploy YAML pins ``async_chunk: true`` and ``codec_streaming: true`` so chunks
        flow Stage 0 -> Stage 1 per AR loop. The Stage-1 codec honors
        ``meta.left_context_size`` and ``meta.right_holdback_size`` so per-chunk codec
        windows stitch into a coherent PCM stream. The byte-count gate is the same as
        the sync paths; per-chunk audio content is verified offline against Whisper.

        NOTE: ``min_hnr_db=0.0`` sits below the typical speech-noise floor so the
        check still catches catastrophic codec failure (silence, white noise, sample
        scramble all give HNR << 0) while allowing for the sliding-window codec's
        slightly-noisier-than-sync output. The default 1.0 dB threshold has only
        ~0.16 dB margin over measured single-request output (1.16 dB on L4), so
        flake risk is too high there.
        """
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "The quick brown fox jumps over the lazy dog.",
                "stream": True,
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
                "min_hnr_db": 0.0,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_concurrent_pcm_streaming(self, omni_server, openai_client) -> None:
        """Three concurrent streaming requests - guards per-request frame cursors
        in ``talker2code2wav_async_chunk`` and per-slot delay-pattern state under batched AR.

        NOTE: ``min_hnr_db=-2.0`` is looser than the single-request streaming test
        because batched AR adds per-request codec quality variance (DAC boundary
        artifacts compounded across the 3-way batch); -2.0 dB gives ~1.6 dB margin
        over the measured worst-of-3 on L4 (-0.42 dB) while still well above the
        catastrophic-failure region.
        """
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "She sells seashells by the seashore.",
                "stream": True,
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
                "min_hnr_db": -2.0,
            },
            request_num=3,
        )


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV3OnlineInlineControlTokens:
    """Inline control-token surface from the bosonai cookbook.

    The model exposes four categories of ``<|category:value|>`` tokens
    (emotion, style, prosody, sfx). Delivery tokens (emotion / style /
    prosody speed-pitch-expressive) go at the START of input; positional
    tokens (pause, sfx) go inline. SFX tokens must pair with their written
    onomatopoeia.

    These tests check the *serving surface* - the validator accepts the
    payload, the engine produces audio, and the WAV has non-trivial size.
    They do not assert audio quality (out of scope for CI).
    """

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_inline_emotion_and_expressive(self, omni_server, openai_client) -> None:
        """Delivery tokens (emotion + expressive_high) at the start of input."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": ("<|emotion:amusement|><|prosody:expressive_high|>Wait, that was actually hilarious."),
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_inline_style_whispering(self, omni_server, openai_client) -> None:
        """Style token at the start - ``whispering``."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "<|style:whispering|>It is just between you and me, alright?",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_inline_prosody_speed_and_pitch(self, omni_server, openai_client) -> None:
        """Two prosody tokens at the start - slow speed plus low pitch."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": (
                    "<|prosody:speed_slow|><|prosody:pitch_low|>The radar shows a storm approaching from the east."
                ),
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_inline_pause_mid_text(self, omni_server, openai_client) -> None:
        """Positional pause token placed inline between two clauses."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hold on a moment <|prosody:pause|> let me think about it.",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_inline_sfx_with_onomatopoeia(self, omni_server, openai_client) -> None:
        """SFX token paired with its written onomatopoeia (``<|sfx:laughter|>Hehe``)."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": (
                    "<|emotion:amusement|>I cannot believe that just happened. "
                    "<|sfx:laughter|>Hehe, I am still recovering from it."
                ),
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV3OnlineVoiceClone:
    """Voice clone via the two payload shapes the model serves."""

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_voice_clone_ref_audio_ref_text(self, omni_server, openai_client) -> None:
        """Canonical vllm-omni voice clone via ``ref_audio`` + ``ref_text``."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "ref_audio": _REF_AUDIO_URL,
                "ref_text": _REF_TEXT,
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_voice_clone_references_alias(self, omni_server, openai_client) -> None:
        """BosonAI cookbook payload: ``references=[{audio_path, text}]``.

        ``normalize_references_alias`` (``protocol/audio.py``) is supposed to
        translate the cookbook field into ``ref_audio`` / ``ref_text``; without
        it the request would silently fall through to zero-shot synthesis.
        """
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "references": [{"audio_path": _REF_AUDIO_URL, "text": _REF_TEXT}],
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV3OnlineValidatorRejections:
    """Out-of-scope shapes must come back as 4xx.

    These cases use ``send_audio_speech_http_request`` (raw HTTP POST against
    ``/v1/audio/speech``) rather than the OpenAI SDK helper because:

    - We need the failure-path matcher; the SDK helper asserts success.
    - The cookbook ``references`` field is not in the OpenAI SDK schema, so
      the SDK strips it before the request leaves the client and the server
      sees a plain TTS request that returns 200. Raw HTTP keeps the JSON body
      intact so the validator actually fires.
    """

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_empty_input(self, omni_server, openai_client) -> None:
        """Empty text input - validator should reject before reaching the engine."""
        openai_client.send_audio_speech_http_request(
            {
                "json": {
                    "model": omni_server.model,
                    "input": "",
                    "response_format": "wav",
                },
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "err_code": (400, 422),
                "err_message": ("input", "empty"),
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_multi_reference_payload(self, omni_server, openai_client) -> None:
        """``references[]`` with more than one entry - multi-shot voice clone is not supported."""
        openai_client.send_audio_speech_http_request(
            {
                "json": {
                    "model": omni_server.model,
                    "input": "Hello world.",
                    "references": [
                        {"audio_path": _REF_AUDIO_URL, "text": _REF_TEXT},
                        {"audio_path": _REF_AUDIO_URL, "text": _REF_TEXT},
                    ],
                    "response_format": "wav",
                },
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "err_code": (400, 422),
                "err_message": "references",
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_conflicting_ref_audio_and_references(self, omni_server, openai_client) -> None:
        """``ref_audio`` and ``references`` cannot both be set to different values."""
        openai_client.send_audio_speech_http_request(
            {
                "json": {
                    "model": omni_server.model,
                    "input": "Hello world.",
                    "ref_audio": _REF_AUDIO_URL,
                    "references": [{"audio_path": "https://example.com/other.wav"}],
                    "response_format": "wav",
                },
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "err_code": (400, 422),
                "err_message": "mutually exclusive",
            }
        )
