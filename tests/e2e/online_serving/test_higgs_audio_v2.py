# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end online tests for higgs-audio v2 against /v1/audio/speech.

v1 scope is plain text -> 24 kHz speech only. The model-aware request
validator (vllm_omni/entrypoints/openai/serving_speech.py::_validate_higgs_audio_v2_request)
rejects voice cloning fields, multi-speaker tags, language overrides, and
task_type/voice selection with explicit 4xx — this suite exercises both the
happy path (plain text in, audio bytes out) and the validator rejections,
so a regression that loosens the schema will fail loudly.

Unit-level coverage of the validator + tokenizer + DualFFN internals lives
at tests/model_executor/models/higgs_audio_v2/test_higgs_audio_v2_units.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Match run_server.sh: DeepGEMM FP8 kernels are optional and trip warmup on
# images without the deep_gemm backend, so disable them by default.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
STAGE_CONFIG = get_deploy_config_path("higgs_audio_v2.yaml")
SERVER_ARGS = ["--trust-remote-code", "--disable-log-stats"]
# DeepGEMM warmup is optional; mirror run_server.sh and switch it off in the
# server subprocess env too (parent-process os.environ above only affects this
# test driver; the engine subprocesses inherit through env_dict).
SERVER_ENV = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_MOE_USE_DEEP_GEMM": "0"}

TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            server_args=SERVER_ARGS,
            env_dict=SERVER_ENV,
        ),
        id="higgs_audio_v2_plain_text",
    )
]

DEFAULT_SPEECH_TIMEOUT_S = 180.0
# Floor for ~0.5 s of 24 kHz mono PCM_16: 24000 * 0.5 * 2 bytes ≈ 24 KiB.
# A WAV header adds 44 bytes; pick a conservative floor that catches truncated
# / silence-only outputs without flagging short legitimate clips.
_MIN_AUDIO_BYTES = 20_000


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV2OnlineHappyPath:
    """Plain-text -> audio happy paths against the live HTTP server."""

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_plain_text_wav(self, omni_server, openai_client) -> None:
        """Single non-streaming WAV request — covers the canonical TTS happy path."""
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
    def test_plain_text_pcm_streaming(self, omni_server, openai_client) -> None:
        """Streaming PCM via the shared-memory connector's codec_streaming path.

        higgs_audio_v2.yaml pins ``codec_streaming: true`` + ``async_chunk: false``,
        so the only streaming surface exposed to clients is the WAV/PCM bytes
        served chunk-by-chunk from Stage 1 — exercise it directly.
        """
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Streaming the quick brown fox over the lazy dog.",
                "stream": True,
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_plain_text_with_max_new_tokens(self, omni_server, openai_client) -> None:
        """max_new_tokens is one of the few extra fields the higgs validator accepts."""
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
        """Three concurrent non-streaming requests — guards the per-slot audio state."""
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


# ---------------------------------------------------------------------------
# Validator rejections served over HTTP. Each case targets one out-of-scope
# field; the validator returns 4xx via the OpenAI error path. We keep the
# error-message substring loose so a phrasing tweak in the validator does not
# break CI, but tight enough to catch a regression that silently accepts the
# field.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV2OnlineValidatorRejections:
    """Out-of-scope fields must come back as 4xx with a higgs-named message."""

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_ref_audio_voice_cloning(self, omni_server, openai_client) -> None:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "ref_audio": "data:audio/wav;base64,SUQ=",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "ref_audio",
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_task_type(self, omni_server, openai_client) -> None:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "task_type": "Base",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "task_type",
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_language_override(self, omni_server, openai_client) -> None:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "language": "Chinese",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "language",
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_multi_speaker_tag_in_text(self, omni_server, openai_client) -> None:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "[SPEAKER0] hi",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "multi-speaker",
            }
        )

    @pytest.mark.core_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_rejects_empty_input(self, omni_server, openai_client) -> None:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "   ",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "empty",
            }
        )
