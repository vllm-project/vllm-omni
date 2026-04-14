# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for Qwen3-TTS WebSocket streaming speech.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import websockets

from tests.conftest import OmniServer
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
STAGE_INIT_TIMEOUT_S = 120


def get_stage_config() -> str:
    return str(
        Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs" / "qwen3_tts.yaml"
    )


@pytest.fixture(scope="module")
def omni_server():
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
        env_dict={"VLLM_DISABLE_COMPILE_CACHE": "1"},
    ) as server:
        yield server


async def _run_ws_session(host: str, port: int) -> dict:
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"
    starts: list[dict] = []
    dones: list[dict] = []
    chunk_lengths: dict[int, list[int]] = {}
    session_done: dict | None = None

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": MODEL,
                    "voice": "vivian",
                    "language": "English",
                    "response_format": "pcm",
                    "stream_audio": True,
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "input.text",
                    "text": (
                        "Hello, this is a websocket streaming test for Qwen three TTS, "
                        "and this sentence is intentionally long enough to produce audio chunks. "
                        "This is the second sentence."
                    ),
                }
            )
        )
        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=180)
            if isinstance(message, bytes):
                if not starts:
                    raise AssertionError("Received audio bytes before audio.start")
                sentence_index = starts[-1]["sentence_index"]
                chunk_lengths.setdefault(sentence_index, []).append(len(message))
                continue

            payload = json.loads(message)
            msg_type = payload.get("type")
            if msg_type == "audio.start":
                starts.append(payload)
                chunk_lengths.setdefault(payload["sentence_index"], [])
            elif msg_type == "audio.done":
                dones.append(payload)
            elif msg_type == "session.done":
                session_done = payload
                break
            elif msg_type == "error":
                raise AssertionError(f"WebSocket error: {payload['message']}")
            else:
                raise AssertionError(f"Unexpected WebSocket message: {payload}")

    return {
        "starts": starts,
        "dones": dones,
        "chunk_lengths": chunk_lengths,
        "session_done": session_done,
    }


class TestQwen3TTSWebSocket:
    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "L4"}, num_cards=4)
    def test_streaming_pcm_output(self, omni_server) -> None:
        result = asyncio.run(_run_ws_session(omni_server.host, omni_server.port))

        starts = result["starts"]
        dones = result["dones"]
        chunk_lengths = result["chunk_lengths"]
        session_done = result["session_done"]

        assert session_done is not None
        assert session_done["total_sentences"] == 2
        assert len(starts) == 2
        assert len(dones) == 2

        for idx, start in enumerate(starts):
            assert start["type"] == "audio.start"
            assert start["sentence_index"] == idx
            assert start["format"] == "pcm"
            assert start["sample_rate"] == 24000
            assert start["sentence_text"]

        for done in dones:
            sentence_index = done["sentence_index"]
            total_bytes = done["total_bytes"]
            assert done["error"] is False
            assert total_bytes > 0
            assert chunk_lengths[sentence_index], f"Expected binary PCM frames for sentence {sentence_index}"
            assert sum(chunk_lengths[sentence_index]) == total_bytes
