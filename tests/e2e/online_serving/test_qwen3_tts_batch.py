# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for the /v1/audio/speech/batch endpoint.

Validates bulk synthesis via the batch API with actual model inference.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import base64
import shutil
import struct
import tempfile
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_file_to_text, cosine_similarity_text
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
STAGE_INIT_TIMEOUT_S = 120


def get_stage_config(name: str = "qwen3_tts.yaml") -> str:
    """Resolve a deploy config path under vllm_omni/deploy/."""
    return get_deploy_config_path(name)


@pytest.fixture(scope="module")
def omni_server():
    """Start vLLM-Omni server with CustomVoice model."""
    stage_config_path = get_stage_config()

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


def make_batch_request(
    host: str,
    port: int,
    items: list[dict],
    voice: str | None = "vivian",
    language: str | None = "English",
    response_format: str = "wav",
    timeout: float = 300.0,
    **kwargs,
) -> httpx.Response:
    """Make a request to the /v1/audio/speech/batch endpoint."""
    url = f"http://{host}:{port}/v1/audio/speech/batch"
    payload: dict = {
        "items": items,
        "response_format": response_format,
    }
    if voice is not None:
        payload["voice"] = voice
    if language is not None:
        payload["language"] = language
    payload.update(kwargs)

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_bytes(audio_bytes: bytes) -> bool:
    """Verify decoded audio bytes form a valid WAV."""
    if len(audio_bytes) < 44:
        return False
    return audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def assert_not_silence(pcm_bytes: bytes):
    """Assert PCM16 samples are not all identical."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    unique = set(samples)
    assert len(unique) > 1, f"All-silence: {len(samples)} samples, unique={unique}"


MIN_AUDIO_BYTES = 10000


class TestSpeechBatchE2E:
    """E2E tests for /v1/audio/speech/batch endpoint."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_basic_two_items(self, omni_server) -> None:
        """Batch with two items returns two successful base64-encoded results."""
        items = [
            {"input": "Hello, how are you today?"},
            {"input": "The weather is nice outside."},
        ]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert data["id"].startswith("speech-batch-")
        assert len(data["results"]) == 2

        for i, result in enumerate(data["results"]):
            assert result["index"] == i
            assert result["status"] == "success"
            assert result["audio_data"] is not None
            assert result["error"] is None
            audio_bytes = base64.b64decode(result["audio_data"])
            assert verify_wav_bytes(audio_bytes), f"Item {i}: invalid WAV"
            assert len(audio_bytes) > MIN_AUDIO_BYTES, f"Item {i}: audio too small ({len(audio_bytes)} bytes)"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_single_item(self, omni_server) -> None:
        """Batch with a single item works correctly."""
        items = [{"input": "Single item batch test."}]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "success"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_per_item_voice_override(self, omni_server) -> None:
        """Per-item voice overrides the batch-level default."""
        items = [
            {"input": "First item with default voice."},
            {"input": "Second item with ryan voice.", "voice": "ryan"},
        ]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
            voice="vivian",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        assert data["total"] == 2
        assert data["succeeded"] == 2
        # Both items should produce valid audio
        for result in data["results"]:
            assert result["status"] == "success"
            audio_bytes = base64.b64decode(result["audio_data"])
            assert verify_wav_bytes(audio_bytes)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_multiple_languages(self, omni_server) -> None:
        """Batch items with different languages per item."""
        items = [
            {"input": "Hello, nice to meet you.", "language": "English"},
            {"input": "你好，很高兴认识你。", "language": "Chinese"},
        ]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        assert data["succeeded"] == 2
        for result in data["results"]:
            assert result["status"] == "success"
            audio_bytes = base64.b64decode(result["audio_data"])
            assert len(audio_bytes) > MIN_AUDIO_BYTES

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_whisper_transcription(self, omni_server) -> None:
        """Whisper transcription of batch output matches input text."""
        input_text = "Good morning, welcome to the speech synthesis test."
        items = [{"input": input_text}]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        assert data["succeeded"] == 1

        audio_bytes = base64.b64decode(data["results"][0]["audio_data"])
        assert verify_wav_bytes(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            wav_path = f.name

        try:
            transcript = convert_audio_file_to_text(wav_path)
            print(f"Whisper transcript: {transcript}")
            assert len(transcript.strip()) > 0, "Empty transcript — likely silence"
            similarity = cosine_similarity_text(transcript.lower(), input_text.lower())
            print(f"Cosine similarity: {similarity:.3f}")
            assert similarity > 0.7, f"Transcript mismatch: similarity={similarity:.2f}, transcript='{transcript}'"
        finally:
            os.unlink(wav_path)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_result_indices_ordered(self, omni_server) -> None:
        """Result indices match the input item order."""
        items = [
            {"input": "First sentence."},
            {"input": "Second sentence."},
            {"input": "Third sentence."},
        ]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        indices = [r["index"] for r in data["results"]]
        assert indices == [0, 1, 2]


class TestSpeechBatchValidation:
    """Validation / error-handling tests for the batch endpoint."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_empty_items_rejected(self, omni_server) -> None:
        """Empty items list returns a 4xx error."""
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech/batch"
        payload = {"items": [], "voice": "vivian"}

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

        assert response.status_code in (400, 422)

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch_exceeds_max_items(self, omni_server) -> None:
        """Batch exceeding 32 items returns 400 error."""
        items = [{"input": f"Item {i}"} for i in range(33)]
        response = make_batch_request(
            host=omni_server.host,
            port=omni_server.port,
            items=items,
        )

        assert response.status_code == 400


def _make_batch2_stage_config() -> str:
    """Create a temporary stage config with max_batch_size=2 for both stages."""
    src = Path(get_stage_config("qwen3_tts_batch.yaml"))
    with open(src) as f:
        cfg = yaml.safe_load(f)

    for stage in cfg["stage_args"]:
        stage["runtime"]["max_batch_size"] = 2

    tmp_dir = tempfile.mkdtemp(prefix="tts_batch2_")
    dst = Path(tmp_dir) / "qwen3_tts_batch2.yaml"
    with open(dst, "w") as f:
        yaml.dump(cfg, f)
    return str(dst), tmp_dir


def make_single_request(
    host: str,
    port: int,
    text: str,
    voice: str = "vivian",
    language: str = "English",
    timeout: float = 300.0,
) -> httpx.Response:
    """Make a request to the single /v1/audio/speech endpoint."""
    url = f"http://{host}:{port}/v1/audio/speech"
    payload = {"input": text, "voice": voice, "language": language}
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


@pytest.fixture(scope="module")
def omni_server_batch2():
    """Start vLLM-Omni server with max_batch_size=2 config."""
    config_path, tmp_dir = _make_batch2_stage_config()

    with OmniServer(
        MODEL,
        [
            "--stage-configs-path",
            config_path,
            "--stage-init-timeout",
            str(STAGE_INIT_TIMEOUT_S),
            "--trust-remote-code",
            "--enforce-eager",
            "--disable-log-stats",
        ],
    ) as server:
        yield server

    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestSpeechBatchSize2:
    """E2E tests with max_batch_size=2 to verify true batched inference."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch2_produces_valid_audio(self, omni_server_batch2) -> None:
        """Batch of 2 items with batched engine produces valid audio."""
        items = [
            {"input": "Hello, this is the first sentence."},
            {"input": "And this is the second sentence."},
        ]
        response = make_batch_request(
            host=omni_server_batch2.host,
            port=omni_server_batch2.port,
            items=items,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()

        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0

        # Save audio files for inspection
        output_dir = Path(tempfile.mkdtemp(prefix="tts_batch2_output_"))
        for result in data["results"]:
            assert result["status"] == "success"
            audio_bytes = base64.b64decode(result["audio_data"])
            assert verify_wav_bytes(audio_bytes), f"Item {result['index']}: invalid WAV"
            assert len(audio_bytes) > MIN_AUDIO_BYTES

            wav_path = output_dir / f"batch_item_{result['index']}.wav"
            wav_path.write_bytes(audio_bytes)
            print(f"  Item {result['index']}: {len(audio_bytes)} bytes -> {wav_path}")

        print(f"\nBatch audio saved to: {output_dir}")

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_batch2_vs_sequential_timing(self, omni_server_batch2) -> None:
        """Compare batch-of-2 vs 2 sequential single requests.

        With max_batch_size=2 the engine can process both items
        concurrently, so the batch call should be faster than two
        sequential single calls (or at least comparable).
        """
        text_a = "The quick brown fox jumps over the lazy dog."
        text_b = "A journey of a thousand miles begins with a single step."

        host = omni_server_batch2.host
        port = omni_server_batch2.port

        # --- Sequential: 2 single /v1/audio/speech calls ---
        t0 = time.perf_counter()
        resp_a = make_single_request(host, port, text_a)
        resp_b = make_single_request(host, port, text_b)
        sequential_time = time.perf_counter() - t0

        assert resp_a.status_code == 200, f"Single A failed: {resp_a.text}"
        assert resp_b.status_code == 200, f"Single B failed: {resp_b.text}"

        # --- Batch: 1 /v1/audio/speech/batch call with 2 items ---
        items = [{"input": text_a}, {"input": text_b}]
        t0 = time.perf_counter()
        resp_batch = make_batch_request(host, port, items)
        batch_time = time.perf_counter() - t0

        assert resp_batch.status_code == 200, f"Batch failed: {resp_batch.text}"
        batch_data = resp_batch.json()
        assert batch_data["succeeded"] == 2

        # Verify both produced valid audio
        for result in batch_data["results"]:
            audio_bytes = base64.b64decode(result["audio_data"])
            assert verify_wav_bytes(audio_bytes)
            assert len(audio_bytes) > MIN_AUDIO_BYTES

        print(f"\n  Sequential (2 singles): {sequential_time:.2f}s")
        print(f"  Batch (2 items):        {batch_time:.2f}s")
        print(f"  Speedup:                {sequential_time / batch_time:.2f}x")
