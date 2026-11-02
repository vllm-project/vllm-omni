# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online expansion tests for higgs-audio v2 against /v1/audio/speech.

v1 scope is plain text -> 24 kHz speech plus shallow voice clone via
ref_audio + ref_text (inline) or voice=<name> (after POST /v1/audio/voices).
Model-aware validator rejections live in L1
``tests/entrypoints/openai_api/test_tts_adapter.py``
(``test_higgs_audio_v2_validate_*``).
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Match run_server.sh: DeepGEMM FP8 kernels are optional and trip warmup on
# images without the deep_gemm backend, so disable them by default.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import pytest
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
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

# The mrv2 opt-in profile runs the talker on model_runner v2 with the codec
# stage on v1. Under v2 the talker delegates LM token sampling to the native
# staged sampler, so penalties and logprobs stay in the request surface —
# unlike Higgs v3, whose owned sampler narrows it.
TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=STAGE_CONFIG,
            server_args=SERVER_ARGS,
            env_dict=SERVER_ENV,
        ),
        id="higgs_audio_v2_plain_text",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("higgs_audio_v2_mrv2.yaml"),
            server_args=SERVER_ARGS,
            env_dict=SERVER_ENV,
        ),
        id="higgs_audio_v2_mrv2",
        marks=pytest.mark.skipif(not current_platform.is_cuda(), reason="MRV2 validation is CUDA-only"),
    ),
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
_REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
_REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestHiggsAudioV2OnlineHappyPath:
    """Plain-text -> audio happy paths against the live HTTP server."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_plain_text_wav(self, omni_server, online_client) -> None:
        """Single non-streaming WAV request — covers the canonical TTS happy path."""
        online_client.send_audio_speech_request(
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
    def test_pcm_streaming_client_abort(self, omni_server, online_client) -> None:
        """A client aborting mid-stream must not wedge the server.

        Exiting the streaming context mid-iteration closes the connection
        (client-side cancel). Guards request cleanup on the v1 runner and on
        the MRV2 sampling hook alike. A follow-up request proves the server
        survived.
        """
        long_text = (
            "This sentence is intentionally long so the stream has plenty of chunks to outlive the reader. " * 20
        )
        with online_client.client.audio.speech.with_streaming_response.create(
            model=omni_server.model,
            input=long_text,
            voice=None,
            response_format="pcm",
            extra_body={"stream_format": "audio"},
            timeout=DEFAULT_SPEECH_TIMEOUT_S,
        ) as resp:
            chunks = resp.iter_bytes()
            first = next(chunks)
            assert first, "stream produced no audio before abort"
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Server still healthy after abort.",
                "stream": False,
                "response_format": "wav",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_long_input_text(self, omni_server, online_client) -> None:
        """A long prompt near the stage-0 context budget must synthesize fully.

        Stage 0 caps generation at 1024 audio frames (~41 s at 25 fps) under
        a 4096-token context; the two-sentence pair repeated 40 times lands
        the prompt near that budget while the synthesized speech stays under
        the frame cap.
        """
        long_input = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 40
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": long_input,
                "stream": False,
                "response_format": "wav",
                "timeout": 300.0,
                # The byte floor stays generous-under-dummy weights:
                # audio-eos timing is weight-dependent, so only require
                # ~20 s of synthesized audio to prove the long input
                # completes rather than truncating.
                "min_audio_bytes": 1_000_000,
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_plain_text_pcm_streaming(self, omni_server, online_client) -> None:
        """Streaming PCM via the shared-memory connector's codec_streaming path.

        higgs_audio_v2.yaml pins ``codec_streaming: true`` + ``async_chunk: false``,
        so the only streaming surface exposed to clients is the WAV/PCM bytes
        served chunk-by-chunk from Stage 1 — exercise it directly.

        NOTE: ``min_hnr_db=-3.0`` keeps the check as a catastrophic-codec-failure
        guard (white noise / sample scramble measure around -10 dB) while
        tolerating Higgs sampling variance: legitimate speech for this prompt
        measured -1.03 dB on L4 (weekly CI, issue #5045) and 1.34 dB on H100, so
        the default 1.0 dB threshold flakes on hardware-dependent tail samples.
        Same rationale as the higgs_audio_v3 streaming thresholds (issue #4411);
        content correctness is separately guarded by the full_model transcript
        check.
        """
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Streaming the quick brown fox over the lazy dog.",
                "stream": True,
                "stream_format": "audio",
                "response_format": "pcm",
                "timeout": DEFAULT_SPEECH_TIMEOUT_S,
                "min_audio_bytes": _MIN_AUDIO_BYTES,
                "min_hnr_db": -3.0,
            }
        )

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_plain_text_with_max_new_tokens(self, omni_server, online_client) -> None:
        """max_new_tokens is one of the few extra fields the higgs validator accepts."""
        online_client.send_audio_speech_request(
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
    def test_concurrent_plain_text(self, omni_server, online_client) -> None:
        """Three concurrent non-streaming requests — guards the per-slot audio state."""
        online_client.send_audio_speech_request(
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
    def test_voice_clone_basic(self, omni_server, online_client) -> None:
        """Single non-streaming WAV request driven by the qwen3_tts ref clip."""
        online_client.send_audio_speech_request(
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
    def test_voice_clone_pcm_streaming(self, omni_server, online_client) -> None:
        """Voice clone over the streaming PCM path."""
        online_client.send_audio_speech_request(
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
