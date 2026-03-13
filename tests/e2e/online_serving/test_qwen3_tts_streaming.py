# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS streaming text input via WebSocket.

These tests verify the /v1/audio/speech/stream endpoint works correctly
with actual model inference, sending text incrementally and receiving
progressive audio output.
"""

import asyncio
import base64
import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import httpx
import pytest
import websockets

from tests.conftest import OmniServer
from tests.utils import hardware_test

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Minimum expected audio size for a short sentence (~1 second of 24kHz 16-bit mono)
MIN_AUDIO_BYTES = 10000


def get_stage_config():
    return str(
        Path(__file__).parent.parent.parent.parent
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / "qwen3_tts.yaml"
    )


def verify_pcm_audio(chunks: list[bytes]) -> bool:
    """Verify that audio chunks contain valid PCM data.

    Checks:
    - At least one chunk received
    - Total size above minimum threshold
    - Each chunk has even byte count (int16 alignment)
    - Audio data is not all zeros
    """
    if not chunks:
        return False
    total = sum(len(c) for c in chunks)
    if total < MIN_AUDIO_BYTES:
        return False
    # int16 PCM requires even byte count per chunk
    if any(len(c) % 2 != 0 for c in chunks):
        return False
    # At least some non-zero audio data
    all_zero = all(all(b == 0 for b in c) for c in chunks)
    return not all_zero


@pytest.fixture(scope="module")
def omni_server():
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


async def streaming_speech_request(
    host: str,
    port: int,
    initial_text: str,
    streaming_chunks: list[str] | None = None,
    voice: str = "vivian",
    chunk_delay: float = 0.03,
    timeout: float = 60.0,
) -> tuple[int, list[bytes], str | None]:
    """Send a streaming TTS request via WebSocket.

    Returns (total_audio_bytes, list_of_audio_chunks, error_message_or_none).
    """
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"
    audio_chunks: list[bytes] = []
    error_msg = None

    async with asyncio.timeout(timeout):
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "text": initial_text,
                "voice": voice,
            }))

            async def send_text():
                if streaming_chunks:
                    for chunk in streaming_chunks:
                        await asyncio.sleep(chunk_delay)
                        await ws.send(json.dumps({
                            "type": "text",
                            "content": chunk,
                        }))
                await ws.send(json.dumps({"type": "end"}))

            async def recv_audio():
                nonlocal error_msg
                async for msg in ws:
                    data = json.loads(msg)
                    if data["type"] == "started":
                        continue
                    elif data["type"] == "audio":
                        audio_chunks.append(base64.b64decode(data["data"]))
                    elif data["type"] == "error":
                        error_msg = data.get("message", "unknown error")
                        break
                    elif data["type"] == "done":
                        break

            await asyncio.gather(send_text(), recv_audio())

    total = sum(len(c) for c in audio_chunks)
    return total, audio_chunks, error_msg


def make_baseline_request(host: str, port: int, text: str, voice: str = "vivian") -> int:
    """Non-streaming baseline for comparison. Returns audio size in bytes."""
    url = f"http://{host}:{port}/v1/audio/speech"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json={
            "input": text,
            "voice": voice,
            "non_streaming_mode": False,
        })
    assert resp.status_code == 200
    return len(resp.content)


class TestQwen3TTSStreaming:
    """E2E tests for streaming TTS text input via WebSocket."""

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_all_text_in_initial(self, omni_server) -> None:
        """All text in initial message, no streaming chunks. Should produce
        valid audio comparable to the non-streaming baseline."""
        text = "Hello, how are you today?"
        baseline_bytes = make_baseline_request(omni_server.host, omni_server.port, text)

        total, chunks, error = asyncio.run(streaming_speech_request(
            omni_server.host, omni_server.port,
            initial_text=text,
        ))

        assert error is None, f"Server returned error: {error}"
        assert verify_pcm_audio(chunks), "Invalid PCM audio data"
        assert len(chunks) >= 1, "Expected at least 1 audio chunk"
        assert total < baseline_bytes * 2.5, (
            f"Streaming audio ({total}) much larger than baseline ({baseline_bytes})"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_chunked_text(self, omni_server) -> None:
        """Text split into initial + streaming chunks at typical LLM rate."""
        text = "Hello, I am going to tell you a story. Once upon a time."
        words = text.split()
        initial = " ".join(words[:4])
        remaining = [" " + w for w in words[4:]]

        baseline_bytes = make_baseline_request(omni_server.host, omni_server.port, text)

        total, chunks, error = asyncio.run(streaming_speech_request(
            omni_server.host, omni_server.port,
            initial_text=initial,
            streaming_chunks=remaining,
            chunk_delay=0.03,
        ))

        assert error is None, f"Server returned error: {error}"
        assert verify_pcm_audio(chunks), "Invalid PCM audio data"
        assert len(chunks) > 1, "Expected multiple progressive audio chunks"
        assert total < baseline_bytes * 2.5, (
            f"Streaming audio ({total}) much larger than baseline ({baseline_bytes})"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_slow_delivery(self, omni_server) -> None:
        """Text delivered slowly (100ms per word). Scheduler pausing should
        prevent pad steps and the model should still stop naturally."""
        text = "Hello, this is a slow delivery test."
        words = text.split()
        initial = " ".join(words[:3])
        remaining = [" " + w for w in words[3:]]

        total, chunks, error = asyncio.run(streaming_speech_request(
            omni_server.host, omni_server.port,
            initial_text=initial,
            streaming_chunks=remaining,
            chunk_delay=0.1,
        ))

        assert error is None, f"Server returned error: {error}"
        assert verify_pcm_audio(chunks), "Invalid PCM audio data"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_sequential_requests(self, omni_server) -> None:
        """Multiple sequential streaming requests should all complete
        without hangs or state leaks between requests."""
        text = "Hello test."
        for i in range(3):
            total, chunks, error = asyncio.run(streaming_speech_request(
                omni_server.host, omni_server.port,
                initial_text=text,
            ))
            assert error is None, f"Request {i+1} returned error: {error}"
            assert verify_pcm_audio(chunks), f"Request {i+1}: invalid PCM audio"

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_streaming_response_is_audio_not_error(self, omni_server) -> None:
        """Regression test: verify streaming returns binary audio data,
        not JSON error messages disguised as audio chunks."""
        total, chunks, error = asyncio.run(streaming_speech_request(
            omni_server.host, omni_server.port,
            initial_text="This should return audio, not an error.",
        ))

        assert error is None, f"Server returned error: {error}"
        assert len(chunks) > 0, "No audio chunks received"

        # Verify chunks are binary audio, not JSON error strings
        for i, chunk in enumerate(chunks):
            try:
                text = chunk.decode("utf-8")
                assert not text.startswith("{"), (
                    f"Chunk {i} appears to be JSON, not audio: {text[:100]}"
                )
            except UnicodeDecodeError:
                pass  # Expected — binary audio can't be decoded as UTF-8

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_non_streaming_still_works(self, omni_server) -> None:
        """Non-streaming /v1/audio/speech endpoint should still work
        correctly after streaming requests."""
        # Do a streaming request first
        asyncio.run(streaming_speech_request(
            omni_server.host, omni_server.port,
            initial_text="Streaming first.",
        ))

        # Then verify non-streaming still works
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json={
                "input": "Non-streaming after streaming.",
                "voice": "vivian",
            })

        assert resp.status_code == 200, f"Request failed: {resp.text}"
        assert resp.headers.get("content-type") == "audio/wav"
        from tests.e2e.online_serving.test_qwen3_tts import verify_wav_audio
        assert verify_wav_audio(resp.content), "Response is not valid WAV audio"
        assert len(resp.content) > MIN_AUDIO_BYTES
