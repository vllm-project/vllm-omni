# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for Qwen3-TTS Base voice-clone model.

Regression test for #1663: speech tokenizer loaded in bfloat16 produced
all-silence PCM due to NaN overflow in SnakeBeta activation.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import struct
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.conftest import (
    OmniServer,
    convert_audio_file_to_text,
    cosine_similarity_text,
)
from tests.utils import hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Official Qwen3-TTS reference audio/text from examples/offline_inference/qwen3_tts/end2end.py
REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
SYN_TEXT = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."


def get_stage_config():
    return str(
        Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "qwen3_tts.yaml"
    )


@pytest.fixture(scope="module")
def omni_server():
    """Start vLLM-Omni server with Base voice-clone model."""
    stage_config_path = get_stage_config()

    with OmniServer(
        MODEL,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            "120",
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


def make_base_speech_request(
    host: str,
    port: int,
    text: str = SYN_TEXT,
    ref_text: str = REF_TEXT,
    ref_audio: str = REF_AUDIO_URL,
    response_format: str = "wav",
    timeout: float = 120.0,
) -> httpx.Response:
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {
        "model": MODEL,
        "input": text,
        "task_type": "Base",
        "ref_text": ref_text,
        "ref_audio": ref_audio,
        "response_format": response_format,
    }
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_audio(content: bytes) -> bool:
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


def assert_not_silence(pcm_bytes: bytes):
    """Assert PCM16 samples are not all identical (e.g. all-silence)."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    unique = set(samples)
    assert len(unique) > 1, (
        f"All-silence detected: {len(samples)} samples, unique values: {unique}. "
        "See https://github.com/vllm-project/vllm-omni/issues/1663"
    )


MIN_AUDIO_BYTES = 10000
MIN_HNR_DB = 1.2  # Clean voice clone > 1.2 dB; distorted < 1.0 dB


def compute_hnr_db(pcm_samples: np.ndarray, sr: int = 24000) -> float:
    """Compute mean Harmonic-to-Noise Ratio (dB) for speech quality.

    Clean cloned speech has HNR > 1.2 dB; distorted speech (e.g. lost
    ref_code decoder context) drops below 1.0 dB.
    """
    frame_len = int(0.03 * sr)  # 30ms frames
    hop = frame_len // 2
    hnr_values: list[float] = []

    for start in range(0, len(pcm_samples) - frame_len, hop):
        frame = pcm_samples[start : start + frame_len]
        if np.max(np.abs(frame)) < 0.01:
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        ac = ac / (ac[0] + 1e-10)
        min_lag = int(sr / 400)
        max_lag = min(int(sr / 80), len(ac))
        if min_lag >= max_lag:
            continue
        peak = float(np.max(ac[min_lag:max_lag]))
        if 0 < peak < 1:
            hnr_values.append(10 * np.log10(peak / (1 - peak + 1e-10)))

    return float(np.mean(hnr_values)) if hnr_values else 0.0


class TestQwen3TTSBaseVoiceClone:
    """Regression tests for Base voice-clone (fix #1663)."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_base_voice_clone_not_silence(self, omni_server) -> None:
        """PCM output must contain real audio, not all-silence."""
        response = make_base_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            response_format="pcm",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > MIN_AUDIO_BYTES, f"Audio too small: {len(response.content)} bytes"
        assert_not_silence(response.content)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_base_voice_clone_whisper_transcription(self, omni_server) -> None:
        """Whisper must transcribe the output as intelligible speech."""
        response = make_base_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            response_format="wav",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(response.content)
            wav_path = f.name

        try:
            transcript = convert_audio_file_to_text(wav_path)
            print(f"Whisper transcript: {transcript}")
            assert len(transcript.strip()) > 0, "Whisper returned empty transcript — audio is likely silence"
            similarity = cosine_similarity_text(transcript.lower(), SYN_TEXT.lower())
            print(f"Cosine similarity: {similarity:.3f}")
            assert similarity > 0.9, (
                f"Transcript doesn't match input: similarity={similarity:.2f}, transcript='{transcript}'"
            )
        finally:
            os.unlink(wav_path)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_base_voice_clone_wav_format(self, omni_server) -> None:
        """WAV response must have valid headers and sufficient size."""
        response = make_base_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            response_format="wav",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV"
        assert len(response.content) > MIN_AUDIO_BYTES

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_base_voice_clone_no_distortion(self, omni_server) -> None:
        """Audio must not be distorted (regression for ref_code context loss).

        When the decoder loses ref_code speaker context on later streaming
        chunks, HNR drops below 1.0 dB.  Clean voice clone should be > 1.2 dB.
        See https://github.com/vllm-project/vllm-omni/issues/1944
        """
        response = make_base_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            response_format="pcm",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert len(response.content) > MIN_AUDIO_BYTES

        pcm_samples = np.frombuffer(response.content, dtype=np.int16).astype(np.float32) / 32768.0
        hnr = compute_hnr_db(pcm_samples)
        print(f"Voice clone HNR: {hnr:.2f} dB (threshold: {MIN_HNR_DB} dB)")
        assert hnr >= MIN_HNR_DB, (
            f"Audio distortion detected: HNR={hnr:.2f} dB < {MIN_HNR_DB} dB. "
            "Voice clone decoder may be losing ref_code speaker context on later chunks."
        )
