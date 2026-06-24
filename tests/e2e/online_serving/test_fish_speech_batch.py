# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E test for Fish Speech S2 Pro concurrent serving with batched decoder.

Verifies that the DAC decoder stage processes multiple requests concurrently
when max_num_seqs > 1, that each request produces valid independent audio,
and that the --enforce-eager rotary embedding fix works correctly.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import concurrent.futures
import struct
import tempfile

import httpx
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text, cosine_similarity_text
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "fishaudio/s2-pro"
STAGE_INIT_TIMEOUT_S = 300
MIN_AUDIO_BYTES = 10000


@pytest.fixture(scope="class")
def omni_server():
    """Start Fish Speech server with default deploy config."""
    stage_config_path = get_deploy_config_path("fish_qwen3_omni.yaml")

    with OmniServer(
        MODEL,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            str(STAGE_INIT_TIMEOUT_S),
            "--trust-remote-code",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


@pytest.fixture(scope="class")
def omni_server_enforce_eager():
    """Start Fish Speech server with --enforce-eager on all stages."""
    stage_config_path = get_deploy_config_path("fish_qwen3_omni.yaml")

    with OmniServer(
        MODEL,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            str(STAGE_INIT_TIMEOUT_S),
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


def _send_tts_request(
    host: str,
    port: int,
    text: str,
    timeout: float = 120.0,
) -> httpx.Response:
    """Send a single /v1/audio/speech request."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "input": text,
        "stream": False,
        "response_format": "wav",
    }
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def _verify_wav(audio_bytes: bytes) -> bool:
    """Check WAV header."""
    if len(audio_bytes) < 44:
        return False
    return audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _assert_not_silence(audio_bytes: bytes):
    """Assert PCM samples are not all-zero."""
    pcm = audio_bytes[44:]
    if len(pcm) < 4:
        return
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])
    unique = set(samples[:1000])
    assert len(unique) > 1, f"Audio is silence: {len(samples)} samples, {len(unique)} unique"


def _assert_whisper_matches(audio_bytes: bytes, expected_text: str, min_similarity: float = 0.7):
    """Assert Whisper transcription of audio matches expected text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        wav_path = f.name
    try:
        transcript = convert_audio_file_to_text(wav_path)
        assert len(transcript.strip()) > 0, "Empty transcript — likely silence"
        similarity = cosine_similarity_text(transcript.lower(), expected_text.lower())
        assert similarity > min_similarity, (
            f"Transcript mismatch: similarity={similarity:.2f}, expected='{expected_text}', got='{transcript}'"
        )
    finally:
        os.unlink(wav_path)


class TestFishSpeechConcurrentServing:
    """E2E tests for Fish Speech concurrent serving with batched decoder."""

    @pytest.mark.full_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_single_request(self, omni_server) -> None:
        """Single request produces valid, non-silent audio."""
        resp = _send_tts_request(
            omni_server.host,
            omni_server.port,
            "Hello, this is a test.",
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        assert _verify_wav(resp.content)
        assert len(resp.content) > MIN_AUDIO_BYTES
        _assert_not_silence(resp.content)

    @pytest.mark.full_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_concurrent_four_requests(self, omni_server) -> None:
        """Four concurrent requests all produce valid, non-silent audio."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    _send_tts_request,
                    omni_server.host,
                    omni_server.port,
                    text,
                )
                for text in texts
            ]
            responses = [f.result() for f in futures]

        for i, resp in enumerate(responses):
            assert resp.status_code == 200, f"req{i} failed: {resp.text}"
            assert _verify_wav(resp.content), f"req{i}: invalid WAV"
            assert len(resp.content) > MIN_AUDIO_BYTES, f"req{i}: audio too small ({len(resp.content)} bytes)"
            _assert_not_silence(resp.content)

    @pytest.mark.full_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_whisper_transcription(self, omni_server) -> None:
        """Whisper transcription of generated audio matches input text."""
        input_text = "Good morning, welcome to the speech synthesis test."
        resp = _send_tts_request(
            omni_server.host,
            omni_server.port,
            input_text,
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        assert _verify_wav(resp.content)
        _assert_whisper_matches(resp.content, input_text)


class TestFishSpeechEnforceEager:
    """E2E test for the --enforce-eager rotary embedding fix."""

    @pytest.mark.full_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_enforce_eager_produces_valid_audio(self, omni_server_enforce_eager) -> None:
        """Server with --enforce-eager starts and produces correct audio.

        Regression test for the Fast AR rotary embedding shape mismatch
        that crashed when --enforce-eager dispatched to the C++ kernel
        with 2D position_ids.
        """
        input_text = "Testing enforce eager mode with Fish Speech."
        resp = _send_tts_request(
            omni_server_enforce_eager.host,
            omni_server_enforce_eager.port,
            input_text,
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        assert _verify_wav(resp.content)
        assert len(resp.content) > MIN_AUDIO_BYTES
        _assert_not_silence(resp.content)
        _assert_whisper_matches(resp.content, input_text)
