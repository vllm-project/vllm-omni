# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online expansion tests for higgs-audio v2 against /v1/audio/speech.

v1 scope is plain text -> 24 kHz speech plus shallow voice clone via
ref_audio + ref_text (inline) or voice=<name> (after POST /v1/audio/voices).
The model-aware request validator
(vllm_omni/entrypoints/openai/tts_adapters/higgs_audio_v2::validate)
rejects multi-speaker tags, language overrides, task_type, and bare
voice=<name> for names that do not match an uploaded speaker — this suite
exercises both the happy path (plain text in, audio bytes out) and the
validator rejections, so a regression that loosens the schema will fail
loudly.
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
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

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

# Reuse the qwen3_tts vendored reference clip (clean ~5 s 24 kHz mono human
# speech) + its transcript. See tests/e2e/online_serving/test_qwen3_tts_base.py
# for the asset rationale — keeping a single shared reference clip across TTS
# tests avoids duplicating WAVs in the repo.
_REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV2OnlineHappyPath:
    """Plain-text -> audio happy paths against the live HTTP server."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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
                "stream_format": "audio",
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_rejects_ref_audio_without_ref_text(self, omni_server, openai_client) -> None:
        """Voice clone needs the transcript too — half-supplied is a 4xx."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "ref_audio": _REF_AUDIO_URL,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "ref_text",
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_rejects_ref_text_without_ref_audio(self, omni_server, openai_client) -> None:
        """Symmetric guard: ref_text alone is not enough — must come with ref_audio."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Hello world.",
                "ref_text": "some transcript",
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "status_code": (400, 422),
                "err_message": "ref_audio",
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
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


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV2OnlineVoiceClone:
    """Shallow voice clone: ref_audio + ref_text -> speech in the cloned voice.

    The HF processor that the serving layer calls (lazy-loaded at the first
    request) ships with the bundled HiggsAudioV2TokenizerModel and encodes
    the reference clip in-process. The talker substitutes the encoded codes
    at the prompt-side audio placeholders via
    :meth:`HiggsAudioV2TalkerForConditionalGeneration._maybe_apply_ref_audio_substitution`.
    """

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_basic(self, omni_server, openai_client) -> None:
        """Single non-streaming WAV request driven by the qwen3_tts ref clip."""
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

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_pcm_streaming(self, omni_server, openai_client) -> None:
        """Voice clone over the streaming PCM path."""
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Innovation distinguishes a leader from a follower.",
                "ref_audio": _REF_AUDIO_URL,
                "ref_text": _REF_TEXT,
                "stream": True,
                "stream_format": "audio",
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )
