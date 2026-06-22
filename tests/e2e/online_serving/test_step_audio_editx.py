# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online serving tests for StepAudioEditX."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import socket
import subprocess
import sys
import time
import wave

import aiohttp
import numpy as np
import pytest
from vllm.utils.network_utils import get_open_port

from tests.helpers.mark import hardware_test

MODEL = "stepfun-ai/Step-Audio-EditX"
AUDIO_TOKENIZER = "stepfun-ai/Step-Audio-Tokenizer"
STAGE_CONFIG = "vllm_omni/deploy/step_audio_editx.yaml"
OUTPUT_SAMPLE_RATE = 24000


class OmniServer:
    """Omni server context manager for opt-in StepAudioEditX online tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict or {}
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        self.proc = subprocess.Popen(cmd, env=env)

        max_wait = 600
        start_time = time.time()
        while time.time() - start_time < max_wait:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    return
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def create_dummy_audio_base64(duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_ref_audio_data_url(duration_sec: float = 2.0) -> str:
    return f"data:audio/wav;base64,{create_dummy_audio_base64(duration_sec=duration_sec)}"


def assert_valid_wav_response(body: bytes) -> None:
    assert len(body) > 44
    with wave.open(io.BytesIO(body), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == OUTPUT_SAMPLE_RATE
        assert wav_file.getnframes() > 0


def assert_valid_pcm_response(body: bytes) -> None:
    assert len(body) > 0
    assert len(body) % 2 == 0


def create_speech_request(
    *,
    edit_type: str = "clone",
    edit_info: str | None = None,
    text: str | None = None,
    stream: bool = False,
) -> dict:
    if text is None:
        text = "Please review the document before we begin." if edit_type in {"clone", "paralinguistic"} else ""
    payload = {
        "model": MODEL,
        "input": text,
        "voice": "step_audio_editx",
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "ref_audio": create_ref_audio_data_url(),
        "ref_text": "Good one. Okay, fine, I'm just gonna leave this here. Goodbye.",
        "max_new_tokens": 256,
        "extra_params": {"edit_type": edit_type},
    }
    if edit_info is not None:
        payload["extra_params"]["edit_info"] = edit_info
    return payload


async def send_speech_request(session: aiohttp.ClientSession, url: str, payload: dict) -> tuple[int, bytes]:
    async with session.post(url, json=payload) as response:
        body = await response.read()
        return response.status, body


@pytest.fixture(scope="class")
def step_audio_editx_server():
    with OmniServer(
        MODEL,
        ["--stage-configs-path", STAGE_CONFIG],
        env_dict={"STEP_AUDIO_TOKENIZER_PATH": AUDIO_TOKENIZER},
    ) as omni_server:
        yield f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestStepAudioEditxOnlineServing:
    @pytest.mark.asyncio
    async def test_single_clone_request(self, step_audio_editx_server: str) -> None:
        payload = create_speech_request(edit_type="clone")

        async with aiohttp.ClientSession() as session:
            status, body = await send_speech_request(session, step_audio_editx_server, payload)

        assert status == 200
        assert_valid_wav_response(body)

    @pytest.mark.asyncio
    async def test_single_edit_request(self, step_audio_editx_server: str) -> None:
        payload = create_speech_request(edit_type="emotion", edit_info="angry")

        async with aiohttp.ClientSession() as session:
            status, body = await send_speech_request(session, step_audio_editx_server, payload)

        assert status == 200
        assert_valid_wav_response(body)

    @pytest.mark.asyncio
    async def test_single_clone_stream_request(self, step_audio_editx_server: str) -> None:
        payload = create_speech_request(edit_type="clone", stream=True)

        async with aiohttp.ClientSession() as session:
            status, body = await send_speech_request(session, step_audio_editx_server, payload)

        assert status == 200
        assert_valid_pcm_response(body)


async def run_throughput_benchmark(
    server_url: str,
    audio_duration: float = 2.0,
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 4,
) -> list[dict]:
    concurrency_levels = concurrency_levels or [1, 2, 4]
    results = []

    async with aiohttp.ClientSession() as session:
        for num_concurrent in concurrency_levels:
            payload = create_speech_request()
            payload["ref_audio"] = create_ref_audio_data_url(duration_sec=audio_duration)
            semaphore = asyncio.Semaphore(num_concurrent)

            async def bounded_request():
                async with semaphore:
                    start = time.perf_counter()
                    status, body = await send_speech_request(session, server_url, payload)
                    return time.perf_counter() - start, status, len(body)

            start = time.perf_counter()
            batch = await asyncio.gather(*(bounded_request() for _ in range(requests_per_level)))
            wall_time = time.perf_counter() - start
            latencies = np.array([item[0] for item in batch], dtype=np.float32)
            results.append(
                {
                    "num_concurrent": num_concurrent,
                    "num_requests": requests_per_level,
                    "successful_requests": sum(1 for _, status, _ in batch if status == 200),
                    "wall_time": wall_time,
                    "p50_latency": float(np.percentile(latencies, 50)),
                    "p95_latency": float(np.percentile(latencies, 95)),
                }
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StepAudioEditX online serving benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000/v1/audio/speech")
    parser.add_argument("--audio-duration", type=float, default=2.0)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-level", type=int, default=4)
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    print(
        asyncio.run(
            run_throughput_benchmark(
                args.server_url,
                audio_duration=args.audio_duration,
                concurrency_levels=levels,
                requests_per_level=args.requests_per_level,
            )
        )
    )
